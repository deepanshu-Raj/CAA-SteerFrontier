import argparse
from pathlib import Path

import pandas as pd
import torch


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
        "--selected-layers",
        nargs="+",
        type=int,
        default=[
            16,
            13,
            13,
        ],
        help="Selected layer corresponding to each behavior.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    behaviors = args.behaviors
    selected_layers = args.selected_layers

    if len(behaviors) != len(selected_layers):
        raise ValueError("--behaviors and --selected-layers must have equal length.")

    # Load OUR raw vectors.
    raw_vectors = {}

    for behavior in behaviors:
        path = Path("vectors") / behavior / "raw_vectors.pt"

        raw_vectors[behavior] = torch.load(
            path,
            map_location="cpu",
        ).float()

        print(f"{behavior:15s} {tuple(raw_vectors[behavior].shape)}")

    shapes = {tuple(v.shape) for v in raw_vectors.values()}

    if len(shapes) != 1:
        raise ValueError(f"Raw vector shapes differ: {shapes}")

    num_layers = next(iter(raw_vectors.values())).shape[0]

    # Compute OUR 3-behavior reference norm:
    # m_3[l] = mean_b ||v_b,l||
    raw_norms = torch.stack(
        [raw_vectors[b].norm(dim=1) for b in behaviors],
        dim=0,
    )

    reference_3 = raw_norms.mean(dim=0)

    normalization_dir = Path("vectors") / "normalization"

    normalization_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_3_path = normalization_dir / "three_behavior_reference_layer_norms.pt"

    torch.save(
        reference_3,
        reference_3_path,
    )

    # Build new normalized vectors
    for behavior in behaviors:
        raw = raw_vectors[behavior]
        norms = raw.norm(
            dim=1,
            keepdim=True,
        )

        normalized_3 = raw * reference_3.unsqueeze(1) / norms

        output = Path("vectors") / behavior / "normalized_vectors_3beh.pt"

        torch.save(
            normalized_3,
            output,
        )

        print(f"Saved {behavior:15s}: {output}")

    # Compare OUR 3-beh norm against current CAA-style reference.
    # Each existing behavior folder contains the exact reference
    # norms used when that behavior's normalized vector was built.
    rows = []

    for behavior, selected_layer in zip(
        behaviors,
        selected_layers,
    ):
        old_reference_path = Path("vectors") / behavior / "reference_layer_norms.pt"

        reference_7 = torch.load(
            old_reference_path,
            map_location="cpu",
        ).float()

        raw = raw_vectors[behavior]
        raw_norm = raw.norm(dim=1)

        for layer in range(num_layers):
            old_target = reference_7[layer].item()

            new_target = reference_3[layer].item()

            ratio = new_target / old_target

            rows.append(
                {
                    "behavior": behavior,
                    "layer": layer,
                    "selected_layer": (layer == selected_layer),
                    "raw_norm": (raw_norm[layer].item()),
                    "reference_norm_7beh": (old_target),
                    "reference_norm_3beh": (new_target),
                    "ratio_3_to_7": (ratio),
                    "relative_change": (ratio - 1.0),
                    "percent_change": (100.0 * (ratio - 1.0)),
                }
            )

    df = pd.DataFrame(rows)

    output_dir = Path("results/comparison")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / "normalization_robustness.csv"

    df.to_csv(
        output_path,
        index=False,
    )

    # Print selected-layer comparison.
    print("\n=== Selected-layer normalization comparison ===")

    selected = df[df["selected_layer"]][
        [
            "behavior",
            "layer",
            "raw_norm",
            "reference_norm_7beh",
            "reference_norm_3beh",
            "ratio_3_to_7",
            "percent_change",
        ]
    ]

    print(
        selected.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # Global summary across all layers.
    print("\n=== Ratio summary across all layers ===")

    for behavior in behaviors:
        sub = df[df["behavior"] == behavior]

        print(
            f"{behavior:15s} | "
            f"min={sub['ratio_3_to_7'].min():.4f} | "
            f"max={sub['ratio_3_to_7'].max():.4f} | "
            f"mean={sub['ratio_3_to_7'].mean():.4f}"
        )

    # Verify direction numerically for the newly normalized vectors.
    print("\n=== Direction preservation check ===")

    for behavior in behaviors:
        old = torch.load(
            Path("vectors") / behavior / "normalized_vectors.pt",
            map_location="cpu",
        ).float()

        new = torch.load(
            Path("vectors") / behavior / "normalized_vectors_3beh.pt",
            map_location="cpu",
        ).float()

        cosine = torch.nn.functional.cosine_similarity(
            old,
            new,
            dim=1,
        )

        print(f"{behavior:15s} | min cosine={cosine.min().item():.9f}")

    print()
    print(
        "Saved 3-beh reference:",
        reference_3_path,
    )

    print(
        "Saved comparison:",
        output_path,
    )


if __name__ == "__main__":
    main()
