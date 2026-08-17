import argparse
from pathlib import Path

import pandas as pd
import torch

from steerfrontier.data import load_ab_test_data
from steerfrontier.evaluation import evaluate_ab
from steerfrontier.experiment import (
    normalized_vectors_path,
    test_ab_path,
)
from steerfrontier.models import load_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--behaviors",
        nargs="+",
        default=[
            "sycophancy",
            "refusal",
            "hallucination",
        ],
    )

    parser.add_argument(
        "--layer",
        type=int,
        default=13,
    )

    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[
            -1.0,
            0.0,
            1.0,
        ],
    )

    return parser.parse_args()


def main():
    args = parse_args()

    behaviors = args.behaviors
    layer = args.layer
    alphas = args.alphas

    print("Behaviors:", behaviors)
    print("Common layer:", layer)
    print("Alphas:", alphas)

    # ------------------------------------------------------------
    # Load vectors
    # ------------------------------------------------------------
    vectors = {}

    for behavior in behaviors:
        all_vectors = torch.load(
            normalized_vectors_path(behavior),
            map_location="cpu",
        ).float()

        vectors[behavior] = all_vectors[layer]

        print(f"{behavior:15s} | L{layer} norm={vectors[behavior].norm().item():.6f}")

    # ------------------------------------------------------------
    # Load target behavior datasets
    # ------------------------------------------------------------
    datasets = {
        behavior: load_ab_test_data(test_ab_path(behavior)) for behavior in behaviors
    }

    print("\nLoading model...")

    model, tokenizer = load_model(
        dtype=torch.float32,
    )

    # ------------------------------------------------------------
    # Compute one alpha=0 baseline per TARGET behavior.
    # Baseline is independent of source vector.
    # ------------------------------------------------------------
    baselines = {}

    reference_vector = vectors[behaviors[0]]

    print("\n=== Baselines ===")

    for target in behaviors:
        result = evaluate_ab(
            model=model,
            tokenizer=tokenizer,
            dataset=datasets[target],
            layer_idx=layer,
            vector=reference_vector,
            coefficient=0.0,
        )

        baselines[target] = result["behavior_score"]

        print(f"{target:15s} | B(0)={baselines[target]:.4f}")

    # ------------------------------------------------------------
    # Source-vector × target-behavior sweep
    # ------------------------------------------------------------
    rows = []

    for source in behaviors:
        print(f"\n=== Steering with {source} vector ===")

        for target in behaviors:
            baseline = baselines[target]

            for alpha in alphas:
                if alpha == 0.0:
                    score = baseline

                else:
                    result = evaluate_ab(
                        model=model,
                        tokenizer=tokenizer,
                        dataset=datasets[target],
                        layer_idx=layer,
                        vector=vectors[source],
                        coefficient=alpha,
                    )

                    score = result["behavior_score"]

                delta = score - baseline

                rows.append(
                    {
                        "source_behavior": source,
                        "target_behavior": target,
                        "layer": layer,
                        "alpha": alpha,
                        "target_behavior_score": score,
                        "baseline": baseline,
                        "delta_from_baseline": delta,
                    }
                )

                print(
                    f"{source:15s} -> "
                    f"{target:15s} | "
                    f"a={alpha:+.1f} | "
                    f"B={score:.4f} | "
                    f"d={delta:+.4f}"
                )

    df = pd.DataFrame(rows)

    output_dir = Path("results/comparison")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / f"cross_behavior_steering_layer{layer}.csv"

    df.to_csv(
        output_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Compact +/-1 delta matrices
    # ------------------------------------------------------------
    for alpha in [-1.0, 1.0]:
        sub = df[df["alpha"] == alpha]

        matrix = sub.pivot(
            index="source_behavior",
            columns="target_behavior",
            values="delta_from_baseline",
        )

        print(f"\n=== Delta behavior at alpha={alpha:+.1f} ===")

        print(matrix.to_string(float_format=lambda x: f"{x:+.4f}"))

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
