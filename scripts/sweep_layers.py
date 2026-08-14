import csv
from pathlib import Path

import torch

from steerfrontier.data import load_ab_test_data
from steerfrontier.evaluation import evaluate_ab
from steerfrontier.models import load_model

DATA_PATH = "data/sycophancy/test_ab.json"
VECTOR_PATH = "vectors/sycophancy/normalized_vectors.pt"
OUTPUT_PATH = Path("results/sycophancy/layer_sweep.csv")


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model, tokenizer = load_model(dtype=torch.float32)

    dataset = load_ab_test_data(DATA_PATH)

    vectors = torch.load(
        VECTOR_PATH,
        map_location="cpu",
    ).float()

    print("Vectors:", vectors.shape)
    print("Test examples:", len(dataset))

    # ------------------------------------------------------------
    # Baseline: alpha = 0 is independent of layer.
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

    print(f"\nBaseline P(behavior) = {baseline:.4f}")

    rows = []

    # ------------------------------------------------------------
    # Layer sweep
    # ------------------------------------------------------------
    for layer in range(vectors.shape[0]):
        layer_scores = {}

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
            layer_scores[alpha] = score

            rows.append(
                {
                    "layer": layer,
                    "alpha": alpha,
                    "behavior_score": score,
                    "baseline": baseline,
                    "delta_from_baseline": score - baseline,
                }
            )

        negative_score = layer_scores[-1.0]
        positive_score = layer_scores[1.0]

        span = positive_score - negative_score

        print(
            f"L{layer:02d} | "
            f"-1={negative_score:.4f} | "
            f"0={baseline:.4f} | "
            f"+1={positive_score:.4f} | "
            f"span={span:+.4f}"
        )

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------
    with OUTPUT_PATH.open("w", newline="") as f:
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

    # ------------------------------------------------------------
    # Rank layers by -1 / +1 separation.
    # ------------------------------------------------------------
    summary = {}

    for row in rows:
        layer = row["layer"]
        summary.setdefault(layer, {})
        summary[layer][row["alpha"]] = row["behavior_score"]

    ranked = []

    for layer, scores in summary.items():
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


if __name__ == "__main__":
    main()
