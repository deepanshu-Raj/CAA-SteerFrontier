import argparse
import csv
import json
from pathlib import Path

import torch

from steerfrontier.evaluation import evaluate_mmlu_detailed
from steerfrontier.experiment import (
    DEFAULT_ALPHAS,
    normalized_vectors_path,
    results_dir,
)
from steerfrontier.models import load_model

MMLU_PATH = Path("data/mmlu/test.json")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--behavior",
        required=True,
    )

    parser.add_argument(
        "--layer",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=DEFAULT_ALPHAS,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    behavior = args.behavior
    layer = args.layer
    alphas = args.alphas

    vector_path = normalized_vectors_path(behavior)

    output_dir = results_dir(behavior)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = output_dir / f"mmlu_layer{layer}.csv"

    items_path = output_dir / f"mmlu_layer{layer}_items.csv"

    # ------------------------------------------------------------
    # Print experiment configuration BEFORE loading model.
    # This makes accidental hardcoding obvious immediately.
    # ------------------------------------------------------------
    print("Behavior:", behavior)
    print("Layer:", layer)
    print("MMLU dataset:", MMLU_PATH)
    print("Vector file:", vector_path)
    print("Summary output:", summary_path)
    print("Items output:", items_path)
    print("Alphas:", alphas)
    print()

    # ------------------------------------------------------------
    # Load MMLU
    # ------------------------------------------------------------
    with MMLU_PATH.open() as f:
        dataset = json.load(f)

    print("MMLU examples:", len(dataset))

    # ------------------------------------------------------------
    # Load behavior vectors
    # ------------------------------------------------------------
    vectors = torch.load(
        vector_path,
        map_location="cpu",
    ).float()

    if layer < 0 or layer >= vectors.shape[0]:
        raise ValueError(
            f"Invalid layer {layer}; vectors contain {vectors.shape[0]} layers."
        )

    vector = vectors[layer]

    print(f"Vector norm: {vector.norm().item():.6f}")

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------
    print("\nLoading model...")

    model, tokenizer = load_model(
        dtype=torch.float32,
    )

    summary_rows = []
    item_rows = []

    # ------------------------------------------------------------
    # Run MMLU sweep
    # ------------------------------------------------------------
    for alpha in alphas:
        result = evaluate_mmlu_detailed(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            layer_idx=layer,
            vector=vector,
            coefficient=alpha,
        )

        capability = result["capability_score"]
        accuracy = result["accuracy"]

        # Support either key name in case evaluation.py uses
        # "records" or "items".
        records = result.get("records")

        if records is None:
            records = result.get("items")

        if records is None:
            raise KeyError(
                "evaluate_mmlu_detailed() returned neither "
                "'records' nor 'items'. "
                f"Available keys: {list(result.keys())}"
            )

        print(
            f"alpha={alpha:+.1f} | C(alpha)={capability:.6f} | accuracy={accuracy:.4f}"
        )

        summary_rows.append(
            {
                "behavior": behavior,
                "layer": layer,
                "alpha": alpha,
                "capability_score": capability,
                "accuracy": accuracy,
            }
        )

        for index, record in enumerate(records):
            item_rows.append(
                {
                    "behavior": behavior,
                    "layer": layer,
                    "alpha": alpha,
                    "item_index": index,
                    **record,
                }
            )

    # ------------------------------------------------------------
    # Calculate capability degradation relative to alpha=0.
    # ------------------------------------------------------------
    baseline_rows = [row for row in summary_rows if row["alpha"] == 0.0]

    if len(baseline_rows) != 1:
        raise ValueError("Alpha sweep must contain exactly one alpha=0 baseline.")

    baseline = baseline_rows[0]["capability_score"]

    for row in summary_rows:
        absolute_drop = baseline - row["capability_score"]

        relative_drop = absolute_drop / baseline

        row["baseline"] = baseline
        row["absolute_drop"] = absolute_drop
        row["relative_drop"] = relative_drop

    # ------------------------------------------------------------
    # Save summary CSV
    # ------------------------------------------------------------
    with summary_path.open(
        "w",
        newline="",
    ) as f:
        fieldnames = [
            "behavior",
            "layer",
            "alpha",
            "capability_score",
            "accuracy",
            "baseline",
            "absolute_drop",
            "relative_drop",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    # ------------------------------------------------------------
    # Save per-example CSV
    # ------------------------------------------------------------
    if item_rows:
        fieldnames = []

        for row in item_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        with items_path.open(
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(item_rows)

    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------
    print("\nSaved:", summary_path)
    print("Saved:", items_path)

    print("\nSummary:")

    for row in summary_rows:
        print(
            f"alpha={row['alpha']:+.1f} | "
            f"C={row['capability_score']:.4f} | "
            f"drop={row['absolute_drop']:+.4f} | "
            f"relative={100 * row['relative_drop']:+.2f}%"
        )


if __name__ == "__main__":
    main()
