import numpy as np
from ollama import chat
from typing import List

SYS_PROMPT = """
    You're a helpful grading agent that will assist in scoring responses from a smaller language model.
    Follow the instructions provided by the user. You must score responses on a scale from 0-100,
    where 0 is the worst score and 100 is the best. Grade responses based on how well the model answers the prompt.
    Use the following criteria to score the responses:
    - Correctness: Does the response answer the prompt correctly?
    - Coherence: Is the response coherent and easy to understand?
    - Completeness: Does the response provide a complete answer to the prompt?

    The above criteria all hold the same weight in scoring the responses.
    You must only return an integer score between 0 and 100.
    Do your best to score the response, even if you're unsure how to score the response or if the response is poor, and always provide a score.
"""


def score_response(entry: dict, model="llama3") -> int:
    """
    Queries llama3 model running on ollama to score model response on 0 - 100 scale.

    entry: entry in validation set
    model: ollama model tag to use
    :returns: integer score
    """

    prompt = (
        f"Given the input `{entry['instruction']}` "
        f"and correct output `{entry['output']}`, "
        f"score the model response `{entry['model_response']}`"
        f" on a scale from 0 to 100."
        f"Respond with the integer number only."
    )

    messages = [
        {"role": "assistant", "content": SYS_PROMPT},
        {"role": "user", "content": prompt},
    ]
    response = chat(
        model,
        messages=messages,
        options={"temperature": 0, "seed": 123},
    )

    score = int(response.message.content)
    return score


def get_stats(scores: List[int]) -> dict:
    """
    Computes summary statistics on scored responses from a trained model.
    :returns: dictionary of summary statistics along with histogram vector.
    """

    scores = np.array(scores)
    bins = np.arange(0, 110, 10)
    stats = {}

    # get summary stats
    stats["mean_score"] = scores.mean()
    stats["median_score"] = np.median(scores)
    stats["min_score"] = scores.min()
    stats["max_score"] = scores.max()
    stats["std_dev_score"] = np.std(scores)

    # get histogram data
    density_hist, _ = np.histogram(scores, bins=bins, density=True)
    stats["density_hist"] = density_hist.tolist()
    stats["bins"] = bins.tolist()[1:]

    # compute cdf
    hist, _ = np.histogram(scores, bins=bins)
    cdf = np.cumsum(hist) / scores.shape[0]
    stats["cdf"] = cdf.tolist()
    stats["hist"] = hist.tolist()

    return stats
