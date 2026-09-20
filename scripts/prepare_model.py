"""Load one configured model and verify its prompt-adapter setup.

This lightweight preparation step checks model metadata and adapter resolution
before expensive vector extraction or evaluation jobs.
"""

import argparse

from steerfrontier.config import load_experiment_config
from steerfrontier.models import load_model
from steerfrontier.prompts import get_prompt_adapter
from steerfrontier.workflow import (
    build_shared_normalization,
    extract_or_load_raw_vectors,
)

# =============================================================================
# Command-line interface
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract CAA vectors and build shared "
            "per-layer normalization for one model."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--extract-mode",
        choices=["auto", "always", "never"],
        default="auto",
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()
    config = load_experiment_config(args.config)

    print("=" * 76)
    print("MODEL PREPARATION")
    print("=" * 76)
    print("Model:", config.model_label)
    print("HF model:", config.model_name)
    print("Behaviors:", config.behaviors)
    print("Device:", args.device)
    print()

    model, tokenizer = load_model(
        model_name=config.model_name,
        device=args.device,
        dtype=config.torch_dtype,
    )

    adapter = get_prompt_adapter(
        model=model,
        tokenizer=tokenizer,
        model_name=config.model_name,
        prompt_family=config.prompt_family,
    )

    print("Prompt family:", adapter.family)

    raw_vectors = extract_or_load_raw_vectors(
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        config=config,
        extract_mode=args.extract_mode,
        device=args.device,
    )

    reference_norms, normalized_vectors = build_shared_normalization(
        raw_vectors=raw_vectors,
        config=config,
    )

    print()
    print("=" * 76)
    print("MODEL PREPARATION COMPLETE")
    print("=" * 76)

    for behavior in config.behaviors:
        raw = raw_vectors[behavior]
        normalized = normalized_vectors[behavior]

        print(
            f"{behavior:18s} "
            f"raw={tuple(raw.shape)} "
            f"normalized={tuple(normalized.shape)}"
        )

    print()
    print(
        "Reference layer norms:",
        tuple(reference_norms.shape),
    )


if __name__ == "__main__":
    main()
