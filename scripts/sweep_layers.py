import argparse
import csv

import torch

from steerfrontier.data import load_ab_test_data
from steerfrontier.evaluation import evaluate_ab
from steerfrontier.experiment import (
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

    return parser.parse_args()


def main():
    args = parse_args()
    behavior = args.behavior

    data_path = test_ab_path(behavior)
    vector_path = normalized_vectors_path(behavior)
    output_path = results_dir(behavior) / "layer_sweep.csv"

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Behavior:", behavior)
    print("Loading model...")

    model, tokenizer = load_model(
        dtype=torch.float32,
    )

    dataset = load_ab_test_data(data_path)

    vectors = torch.load(
        vector_path,
        map_location="cpu",
    ).float()

    print("Test examples:", len(dataset))
    print("Vectors:", vectors.shape)

    # ------------------------------------------------------------
    # alpha=0 is independent of intervention layer.
    # ------------------------------------------------------------
    print("\nEvaluating baseline...")

    baseline_result = evaluate_ab(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        layer_idx=0,
        vector=vectors[0],
        coefficient=0.0,
    )

    baseline = baseline_result["behavior_score"]

    print(f"Baseline P(behavior) = {baseline:.4f}")

    rows = []

    # ------------------------------------------------------------
    # Full layer sweep at alpha = +/-1.
    # ------------------------------------------------------------
    for layer in range(vectors.shape[0]):
        scores = {}

        for alpha in [-1.0, 1.0]:
            result = evaluate_ab(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                layer_idx=layer,
                vector=vectors[layer],
                coefficient=alpha,
            )

            score = result["behavior_score"]
            scores[alpha] = score

            rows.append(
                {
                    "behavior": behavior,
                    "layer": layer,
                    "alpha": alpha,
                    "behavior_score": score,
                    "baseline": baseline,
                    "delta_from_baseline": (score - baseline),
                }
            )

        span = scores[1.0] - scores[-1.0]

        print(
            f"L{layer:02d} | "
            f"-1={scores[-1.0]:.4f} | "
            f"0={baseline:.4f} | "
            f"+1={scores[1.0]:.4f} | "
            f"span={span:+.4f}"
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

    # ------------------------------------------------------------
    # Rank by +/- separation.
    # ------------------------------------------------------------
    ranked = []

    for layer in range(vectors.shape[0]):
        layer_rows = [row for row in rows if row["layer"] == layer]

        scores = {row["alpha"]: row["behavior_score"] for row in layer_rows}

        span = scores[1.0] - scores[-1.0]

        ranked.append(
            (
                span,
                layer,
                scores[-1.0],
                scores[1.0],
            )
        )

    ranked.sort(reverse=True)

    print("\nTop layers by steering span:")

    for span, layer, neg, pos in ranked[:10]:
        print(f"L{layer:02d} | span={span:+.4f} | -1={neg:.4f} | +1={pos:.4f}")

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
