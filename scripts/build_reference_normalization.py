import argparse
from pathlib import Path

import torch

from steerfrontier.experiment import (
    BEHAVIORS,
    normalized_vectors_path,
    raw_vectors_path,
)
from steerfrontier.normalization import (
    compute_mean_layer_norms,
    normalize_to_layer_norms,
)

CAA_ROOT = Path("../CAA")
MODEL_SUFFIX = "Llama-2-7b-chat-hf"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--behavior",
        required=True,
        choices=BEHAVIORS,
    )

    return parser.parse_args()


def load_upstream_behavior(behavior):
    vectors = []

    for layer in range(32):
        path = CAA_ROOT / "vectors" / behavior / f"vec_layer_{layer}_{MODEL_SUFFIX}.pt"

        vectors.append(
            torch.load(
                path,
                map_location="cpu",
            ).float()
        )

    return torch.stack(vectors)


def main():
    args = parse_args()

    target_behavior = args.behavior

    ours = torch.load(
        raw_vectors_path(target_behavior),
        map_location="cpu",
    ).float()

    behavior_vectors = {}

    for behavior in BEHAVIORS:
        if behavior == target_behavior:
            behavior_vectors[behavior] = ours
        else:
            behavior_vectors[behavior] = load_upstream_behavior(behavior)

    target_norms = compute_mean_layer_norms(behavior_vectors)

    normalized = normalize_to_layer_norms(
        ours,
        target_norms,
    )

    output = normalized_vectors_path(target_behavior)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        normalized,
        output,
    )

    torch.save(
        target_norms,
        output.parent / "reference_layer_norms.pt",
    )

    print("Behavior:", target_behavior)
    print("Saved:", output)
    print("Shape:", normalized.shape)

    print("\nSelected norms:")

    for layer in [0, 5, 10, 13, 16, 20, 31]:
        print(
            f"L{layer:02d} | "
            f"raw={ours[layer].norm().item():.6f} | "
            f"target={target_norms[layer].item():.6f} | "
            f"normalized={normalized[layer].norm().item():.6f}"
        )


if __name__ == "__main__":
    main()
