from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


VECTOR_PATH = "vectors/sycophancy/raw_vectors.pt"
OUTPUT = Path("plots/sycophancy/adjacent_layer_similarity.pdf")


def main():
    vectors = torch.load(
        VECTOR_PATH,
        map_location="cpu",
    ).float()

    vectors = F.normalize(vectors, dim=1)

    # cos(v_l, v_{l+1}) for l = 0 ... 30
    similarities = torch.sum(
        vectors[:-1] * vectors[1:],
        dim=1,
    )

    layers = list(range(len(similarities)))

    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    ax.plot(
        layers,
        similarities.numpy(),
        marker="o",
        markersize=4,
        linewidth=1.6,
    )

    ax.set_xlabel(r"Layer pair $(l, l+1)$")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(
        "Adjacent-layer CAA vector similarity (sycophancy)"
    )

    ax.set_xticks(range(0, 31, 2))
    ax.set_xlim(0, 30)
    ax.set_ylim(0.0, 1.0)

    ax.grid(alpha=0.2)

    fig.tight_layout()

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        OUTPUT,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved {OUTPUT}")
    print(f"Saved {OUTPUT.with_suffix('.png')}")

    print("\nAdjacent-layer similarities:")
    for layer, similarity in zip(layers, similarities):
        print(
            f"L{layer:02d} -> L{layer + 1:02d}: "
            f"{similarity.item():.4f}"
        )


if __name__ == "__main__":
    main()
