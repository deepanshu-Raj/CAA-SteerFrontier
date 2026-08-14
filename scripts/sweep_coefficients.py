import csv
from pathlib import Path

import torch

from steerfrontier.data import load_ab_test_data
from steerfrontier.evaluation import evaluate_ab
from steerfrontier.models import load_model

DATA_PATH = "data/sycophancy/test_ab.json"
VECTOR_PATH = "vectors/sycophancy/normalized_vectors.pt"

OUTPUT_PATH = Path("results/sycophancy/coefficient_layer_sweep.csv")

LAYERS = list(range(10, 19))

ALPHAS = [
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
]


def main():
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading model...")

    model, tokenizer = load_model(
        dtype=torch.float32,
    )

    dataset = load_ab_test_data(DATA_PATH)

    vectors = torch.load(
        VECTOR_PATH,
        map_location="cpu",
    ).float()

    # ------------------------------------------------------------
    # Compute alpha=0 once.
    # ------------------------------------------------------------
    print("\nEvaluating baseline...")

    baseline_result = evaluate_ab(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        layer_idx=10,
        vector=vectors[10],
        coefficient=0.0,
    )

    baseline = baseline_result["behavior_score"]

    print(f"Baseline P(behavior) = {baseline:.6f}")

    rows = []

    # ------------------------------------------------------------
    # Layer x coefficient sweep
    # ------------------------------------------------------------
    for layer in LAYERS:
        print(f"\n=== Layer {layer} ===")

        for alpha in ALPHAS:
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

    # ------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------
    with OUTPUT_PATH.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "alpha",
                "behavior_score",
                "baseline",
                "delta_from_baseline",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
