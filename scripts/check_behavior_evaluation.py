"""Run a focused smoke check of forced-choice behavior evaluation.

This diagnostic script is intended for validating scoring semantics on a small
example before launching full experiment sweeps.
"""

import json

import torch

from steerfrontier.evaluation import evaluate_behavior
from steerfrontier.models import load_model

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
BEHAVIOR = "survival_instinct"
LAYER = 13
DEVICE = "cuda"


# =============================================================================
# Main
# =============================================================================


def main():
    with open(f"data/{BEHAVIOR}/vector.json") as f:
        data = json.load(f)

    # Prefer an example whose behavior-matching answer is not A/B.
    item = next(x for x in data if x["answer_matching_behavior"] not in {"(A)", "(B)"})

    print("Matching:    ", item["answer_matching_behavior"])
    print("Nonmatching: ", item["answer_not_matching_behavior"])
    print()
    print(item["question"][:500])
    print()

    model, tokenizer = load_model(
        MODEL_NAME,
        device=DEVICE,
        dtype=torch.float32,
    )

    hidden_size = model.config.hidden_size

    # α=0, so this vector should have no effect.
    zero_vector = torch.zeros(
        hidden_size,
        dtype=torch.float32,
        device=DEVICE,
    )

    result = evaluate_behavior(
        model=model,
        tokenizer=tokenizer,
        dataset=[item],
        layer_idx=LAYER,
        vector=zero_vector,
        coefficient=0.0,
        device=DEVICE,
        prompt_family="llama2-chat",
    )

    print()
    print("RESULT")
    print(result)


if __name__ == "__main__":
    main()
