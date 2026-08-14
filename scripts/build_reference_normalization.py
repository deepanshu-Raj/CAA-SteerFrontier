from pathlib import Path

import torch

from steerfrontier.normalization import (
    compute_mean_layer_norms,
    normalize_to_layer_norms,
)

UPSTREAM_BEHAVIORS = [
    "coordinate-other-ais",
    "corrigible-neutral-HHH",
    "hallucination",
    "myopic-reward",
    "refusal",
    "survival-instinct",
]

CAA_ROOT = Path("../CAA")
MODEL_SUFFIX = "Llama-2-7b-chat-hf"


def load_upstream_behavior(behavior):
    vectors = []

    for layer in range(32):
        path = CAA_ROOT / "vectors" / behavior / f"vec_layer_{layer}_{MODEL_SUFFIX}.pt"

        vectors.append(torch.load(path, map_location="cpu").float())

    return torch.stack(vectors)


def main():
    # Six non-sycophancy behaviors from upstream CAA.
    behavior_vectors = {
        behavior: load_upstream_behavior(behavior) for behavior in UPSTREAM_BEHAVIORS
    }

    # Our independently extracted sycophancy vectors.
    ours = torch.load(
        "vectors/sycophancy/raw_vectors.pt",
        map_location="cpu",
    ).float()

    behavior_vectors["sycophancy"] = ours

    # Mean norm across the seven behaviors at each layer.
    target_norms = compute_mean_layer_norms(behavior_vectors)

    normalized = normalize_to_layer_norms(
        ours,
        target_norms,
    )

    torch.save(
        target_norms,
        "vectors/reference_layer_norms.pt",
    )

    torch.save(
        normalized,
        "vectors/sycophancy/normalized_vectors.pt",
    )

    print("Saved normalized sycophancy vectors.")
    print("Shape:", normalized.shape)

    print("\nSelected norms:")
    for layer in [0, 5, 10, 13, 20, 31]:
        print(
            f"Layer {layer:2d} | "
            f"raw={ours[layer].norm().item():.6f} | "
            f"target={target_norms[layer].item():.6f} | "
            f"normalized={normalized[layer].norm().item():.6f}"
        )


if __name__ == "__main__":
    main()
