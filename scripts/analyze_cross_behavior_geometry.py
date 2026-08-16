import argparse
import itertools
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from steerfrontier.experiment import raw_vectors_path


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

    return parser.parse_args()


def main():
    args = parse_args()
    behaviors = args.behaviors

    vectors = {}

    for behavior in behaviors:
        path = raw_vectors_path(behavior)

        vectors[behavior] = torch.load(
            path,
            map_location="cpu",
        ).float()

        print(f"{behavior:15s} {tuple(vectors[behavior].shape)}")

    shapes = {tuple(v.shape) for v in vectors.values()}

    if len(shapes) != 1:
        raise ValueError(f"Vector shapes differ: {shapes}")

    num_layers = next(iter(vectors.values())).shape[0]

    rows = []

    for behavior_a, behavior_b in itertools.combinations(
        behaviors,
        2,
    ):
        cosine = F.cosine_similarity(
            vectors[behavior_a],
            vectors[behavior_b],
            dim=1,
        )

        for layer in range(num_layers):
            rows.append(
                {
                    "behavior_a": behavior_a,
                    "behavior_b": behavior_b,
                    "layer": layer,
                    "cosine_similarity": cosine[layer].item(),
                }
            )

    df = pd.DataFrame(rows)

    output_dir = Path("results/comparison")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / "cross_behavior_vector_cosine.csv"

    df.to_csv(
        output_path,
        index=False,
    )

    print("\n=== Layers 10-16 ===")

    subset = df[df["layer"].between(10, 16)]

    print(
        subset.pivot(
            index="layer",
            columns=[
                "behavior_a",
                "behavior_b",
            ],
            values="cosine_similarity",
        ).to_string(float_format=lambda x: f"{x:+.4f}")
    )

    print("\n=== Layer 13 ===")

    print(
        df[df["layer"] == 13][
            [
                "behavior_a",
                "behavior_b",
                "cosine_similarity",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:+.4f}",
        )
    )

    print("\n=== Layer 16 ===")

    print(
        df[df["layer"] == 16][
            [
                "behavior_a",
                "behavior_b",
                "cosine_similarity",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:+.4f}",
        )
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
