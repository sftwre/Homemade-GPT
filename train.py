import mlflow
import torch
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


def train(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    tokenizer,
    writer: SummaryWriter,
) -> (List[float], List[float], List[float]):

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = batch_loss(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

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

                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )
        # TODO: randomly sample instructions from the val set and evaluate the model.
        instruction = format_input(val_data[0])
        response = generate_response(
            model, instruction=instruction, tokenizer=tokenizer
        )
        writer.add_text("Model response", response, epoch)

    return train_losses, val_losses, track_tokens_seen


def evaluate(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = epoch_loss(train_loader, model, device, num_batches=eval_iter)
        val_loss = epoch_loss(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_response(model, instruction, tokenizer, max_new_tokens=50, temperature=1):

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
        "--dataset_path", type=str, required=True, help="Absolute path to dataset"
    )
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
    }
    exp_params.update(BASE_CONFIG)

    # set seed
    torch.manual_seed(123)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    """
    Load train/val datasets
    """

    data = load_dataset(file_path=args.dataset_path)

    train_portion = int(len(data) * 0.85)  # 85% for training
    test_portion = int(len(data) * 0.1)  # 10% for testing
    val_portion = (
        len(data) - train_portion - test_portion
    )  # Remaining 5% for validation

    train_data = data[:train_portion]
    test_data = data[train_portion : train_portion + test_portion]
    val_data = data[train_portion + test_portion :]

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
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

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=exp_params["batch_size"],
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
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

    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    """
    Training
    """
    with mlflow.start_run() as run:

        writer = SummaryWriter(log_dir=f"./runs/{run.info.run_name}")

        train_losses, val_losses, tokens_seen = train(
            model,
            train_loader,
            test_loader,
            optimizer,
            device,
            num_epochs=exp_params["num_epochs"],
            eval_freq=5,
            eval_iter=5,
            tokenizer=tokenizer,
            writer=writer,
        )

        end_time = time.time()
        execution_time_minutes = (end_time - start_time) / 60
        print(f"Training completed in {execution_time_minutes:.2f} minutes.")

        mlflow.set_tracking_uri("http://127.0.0.1:8080")
        mlflow.set_experiment("Instruction fine-tuning")

        mlflow.log_params(exp_params)
        mlflow.set_tag("Training time", f"{execution_time_minutes:.2f} mins")

        mlflow.pytorch.log_model(
            model,
            artifact_path="mlruns/models",
            registered_model_name=f"gpt_alpaca",
        )
