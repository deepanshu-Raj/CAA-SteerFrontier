from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

INPUT = "results/sycophancy/layer_sweep.csv"
OUTPUT = Path("plots/sycophancy/layer_sweep.pdf")


def main():
    df = pd.read_csv(INPUT)

    baseline = df["baseline"].iloc[0]

    neg = df[df["alpha"] == -1.0].sort_values("layer")

    pos = df[df["alpha"] == 1.0].sort_values("layer")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    ax.plot(
        neg["layer"],
        neg["behavior_score"],
        marker="o",
        markersize=3,
        linewidth=1.6,
        label=r"$\alpha=-1$",
    )

    ax.plot(
        pos["layer"],
        pos["behavior_score"],
        marker="o",
        markersize=3,
        linewidth=1.6,
        label=r"$\alpha=+1$",
    )

    ax.axhline(
        baseline,
        linestyle="--",
        linewidth=1.3,
        label=r"$\alpha=0$",
    )

    ax.set_xlabel("Layer#")
    ax.set_ylabel("P(answer matching behavior)")
    ax.set_title("Per-layer CAA effect (sycophancy): Llama 2 7B Chat")

    ax.set_xticks(range(0, 32, 2))
    ax.set_xlim(0, 31)
    ax.set_ylim(0.4, 0.76)

    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    fig.tight_layout()

    fig.savefig(OUTPUT, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")

    print(f"Saved {OUTPUT}")
    print(f"Saved {OUTPUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
