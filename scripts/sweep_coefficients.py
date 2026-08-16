import argparse
import csv

import torch

from steerfrontier.data import load_ab_test_data
from steerfrontier.evaluation import evaluate_ab
from steerfrontier.experiment import (
    DEFAULT_ALPHAS,
    normalized_vectors_path,
    results_dir,
    test_ab_path,
)
from steerfrontier.models import load_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--behavior",
        required=True,
    )

    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        required=True,
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
    layers = args.layers
    alphas = args.alphas

    output_path = results_dir(behavior) / "coefficient_layer_sweep.csv"

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Behavior:", behavior)
    print("Layers:", layers)
    print("Alphas:", alphas)

    print("\nLoading model...")
    model, tokenizer = load_model(
        dtype=torch.float32,
    )

    dataset = load_ab_test_data(test_ab_path(behavior))

    vectors = torch.load(
        normalized_vectors_path(behavior),
        map_location="cpu",
    ).float()

    # ------------------------------------------------------------
    # Baseline only needs to be calculated once.
    # ------------------------------------------------------------
    baseline_result = evaluate_ab(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        layer_idx=layers[0],
        vector=vectors[layers[0]],
        coefficient=0.0,
    )

    baseline = baseline_result["behavior_score"]

    print(f"\nBaseline P(behavior) = {baseline:.6f}")

    rows = []

    for layer in layers:
        print(f"\n=== Layer {layer} ===")

        for alpha in alphas:
            if alpha == 0.0:
                score = baseline

            else:
                result = evaluate_ab(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    layer_idx=layer,
                    vector=vectors[layer],
                    coefficient=alpha,
                )

                score = result["behavior_score"]

            delta = score - baseline

            rows.append(
                {
                    "behavior": behavior,
                    "layer": layer,
                    "alpha": alpha,
                    "behavior_score": score,
                    "baseline": baseline,
                    "delta_from_baseline": delta,
                }
            )

            print(
                f"L{layer:02d} | "
                f"alpha={alpha:+.1f} | "
                f"P(behavior)={score:.4f} | "
                f"delta={delta:+.4f}"
            )

    with output_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "behavior",
                "layer",
                "alpha",
                "behavior_score",
                "baseline",
                "delta_from_baseline",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
