import mlflow
import torch
from torch.optim.lr_scheduler import (
    LRScheduler,
    LinearLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    SequentialLR,
)
from torch.utils.tensorboard import SummaryWriter
import time
import tiktoken
from functools import partial
from data import InstructionDataset, custom_collate_fn, format_input
from torch.utils.data import DataLoader
from transformer import GPTModel
from utils import (
    load_weights_into_gpt,
    download_and_load_gpt2,
    load_dataset,
    text_to_token_ids,
    token_ids_to_text,
)
from typing import List
from argparse import ArgumentParser
from tqdm import tqdm
from eval import score_response, get_stats


# TODO: refactor function to only define necessary params, the rest can be passed through kwargs
def train(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler | None,
    device: str,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    tokenizer,
    writer: SummaryWriter,
    eval_samples=3,
    warmup_steps=0,
) -> (List[float], List[float], List[float]):

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = batch_loss(input_batch, target_batch, model, device)
            loss.backward()

            # update lr scheduler if present
            if scheduler is not None:
                scheduler.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if warmup_steps > 0 and global_step > warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                # metric logging
                writer.add_scalar("Loss/train", train_loss, global_step)
                writer.add_scalar("Loss/val", val_loss, global_step)

                if scheduler is not None:
                    writer.add_scalar(
                        "LR Scheduler", scheduler.get_last_lr()[0], global_step
                    )

                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )
        # eval model on first eval_samples instructions
        for i in range(eval_samples):
            instruction = format_input(val_data[i])
            response = generate_response(
                model, instruction=instruction, tokenizer=tokenizer
            )
            writer.add_text(f"Model response [{i+1}]", response, epoch)

    return train_losses, val_losses, track_tokens_seen


def evaluate(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = epoch_loss(train_loader, model, device, num_batches=eval_iter)
        val_loss = epoch_loss(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


# TODO: modify temp scaling and add top-k sampling
def generate_response(
    model, instruction, tokenizer, max_new_tokens=50, temperature=1, eos_idx=50256
):

    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    idx = text_to_token_ids(instruction, tokenizer).to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        scaled_logits = logits / temperature
        probas = torch.softmax(scaled_logits, dim=-1)
        idx_next = torch.multinomial(probas, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

        if idx_next == eos_idx:
            break

    decoded_text = token_ids_to_text(idx, tokenizer)
    model.train()

    return decoded_text


def batch_loss(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def epoch_loss(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = batch_loss(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/instruction-data-small.json",
        help="Absolute path to dataset",
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        choices=["linear", "cosine", "cosine_warm_restart", "none"],
        default="none",
        help="Type of learning rate scheduler to use",
    )
    parser.add_argument("--warmup_steps", action="store_true", default=False)
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt_alpaca_small",
        help="Name used to identify model.",
    )
    parser.add_argument("--grad_clip", action="store_true", default=False)
    args = parser.parse_args()

    """
    Experiment level config
    """
    BASE_CONFIG = {
        "vocab_size": 50257,  # Vocabulary size
        "ctx_len": 1024,  # Context length
        "drop_rate": 0.0,  # Dropout rate
        "qkv_bias": True,  # Query-key-value bias
        "emb_dim": 1024,
        "n_layers": 24,
        "n_heads": 16,
    }

    exp_params = {
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "lr": args.lr,
        "lr_scheduler": args.lr_scheduler,
        "bells_whistles": "gradient_clipping" if args.grad_clip else "",
    }
    exp_params.update(BASE_CONFIG)

    # set seed
    torch.manual_seed(123)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    """
    Load train/val datasets
    """
    # TODO: load dataset from huggingface
    data = load_dataset(file_path=args.dataset_path)

    train_portion = int(len(data) * 0.85)  # 85% for training
    test_portion = int(len(data) * 0.1)  # 10% for testing

    train_data = data[:train_portion]
    test_data = data[train_portion : train_portion + test_portion]

    # Use 20% of training iterations as warmup
    warmup_steps = 0
    if args.warmup_steps:
        warmup_steps = int((len(train_data) * args.num_epochs / args.batch_size) * 0.2)

    print(f"Warmup steps: {warmup_steps}")
    print("Training set length:", len(train_data))
    print("Test set length:", len(test_data))

    tokenizer = tiktoken.get_encoding("gpt2")
    customized_collate_fn = partial(
        custom_collate_fn, device=device, allowed_max_length=512
    )

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=exp_params["batch_size"],
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=exp_params["num_workers"],
    )

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=exp_params["batch_size"],
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=exp_params["num_workers"],
    )

    """
    Model init
    """
    settings, params = download_and_load_gpt2(model_size="355M", models_dir="gpt2")

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)

    model.to(device)

    # TODO: use early stopping with model checkpointing
    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    exp_params["optimizer"] = type(optimizer)

    # select learning rate scheduler
    scheduler = None

    if args.lr_scheduler == "linear":
        scheduler = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
    elif args.lr_scheduler == "cosine":
        linear_warmup = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=185, eta_min=args.lr * 0.1
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[linear_warmup, cosine_scheduler],
            milestones=[warmup_steps],
        )
    elif args.lr_scheduler == "cosine_warm_restart":
        linear_warmup = LinearLR(
            optimizer=optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer, T_0=80, eta_min=args.lr * 0.1
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[linear_warmup, cosine_scheduler],
            milestones=[warmup_steps],
        )

    """
    Training
    """
    mlflow.set_experiment("Instruction fine-tuning")

    with mlflow.start_run() as run:

        writer = SummaryWriter(log_dir=f"./runs/{run.info.run_name}")

        train_losses, val_losses, tokens_seen = train(
            model,
            train_loader,
            test_loader,
            optimizer,
            scheduler,
            device,
            num_epochs=exp_params["num_epochs"],
            eval_freq=5,
            eval_iter=5,
            tokenizer=tokenizer,
            writer=writer,
            warmup_steps=warmup_steps,
        )

        end_time = time.time()
        execution_time_minutes = (end_time - start_time) / 60
        print(f"Training completed in {execution_time_minutes:.2f} minutes.")

        mlflow.set_tracking_uri("http://127.0.0.1:8080")

        mlflow.log_params(exp_params)
        mlflow.set_tag("Training time", f"{execution_time_minutes:.2f} mins")
        # TODO: only checkpoint models that have outperformed previous models
        # mlflow.pytorch.log_model(
        #     model,
        #     artifact_path="mlruns/models",
        #     registered_model_name=f"{args.model_name}",
        # )
        # TODO: Use mlflow api to automatically remove unsuccesful runs or experiments that terminate early
        # TODO: name model by parameter count

        """
        Evaluate model on validation set using Llama3
        """
        val_data = data[train_portion + test_portion :]
        scores = []

        for i, entry in tqdm(enumerate(val_data), total=len(val_data)):
            instruction = format_input(entry)

            response = generate_response(
                model, instruction=instruction, tokenizer=tokenizer
            )
            response = response[len(instruction) :].replace("### Response:", "").strip()
            val_data[i]["model_response"] = response

            score = score_response(entry)
            scores.append(score)

        stats = get_stats(scores)

        # log metrics with MLFlow
        for k, v in stats.items():

            if k not in ["hist", "bins"]:
                mlflow.log_metric(key=k, value=v)
            elif k == "hist":
                for value, step in zip(stats["hist"], stats["bins"]):
                    mlflow.log_metric("Density hist.", value * 100, step=step)
