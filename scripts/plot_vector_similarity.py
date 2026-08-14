from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

VECTOR_PATH = "vectors/sycophancy/raw_vectors.pt"
OUTPUT = Path("plots/sycophancy/vector_cosine_similarity.pdf")


def main():
    vectors = torch.load(
        VECTOR_PATH,
        map_location="cpu",
    ).float()

    normalized = F.normalize(vectors, dim=1)
    similarity = normalized @ normalized.T

    fig, ax = plt.subplots(figsize=(6.2, 5.2))

    image = ax.imshow(
        similarity.numpy(),
        origin="lower",
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )

    ax.set_xlabel("Layer #")
    ax.set_ylabel("Layer #")
    ax.set_title("CAA vector similarity across layers (sycophancy)")

    major_ticks = range(0, 32, 4)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)

    # light cell grid
    ax.set_xticks([x - 0.5 for x in range(1, 32)], minor=True)
    ax.set_yticks([x - 0.5 for x in range(1, 32)], minor=True)

    ax.grid(which="minor", linewidth=0.2, alpha=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Cosine similarity")

    fig.tight_layout()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")

    print(f"Saved {OUTPUT}")
    print(f"Saved {OUTPUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
