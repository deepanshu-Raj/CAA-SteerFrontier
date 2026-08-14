from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT = "results/sycophancy/coefficient_layer_sweep.csv"
OUTPUT = Path("plots/sycophancy/coefficient_curves.pdf")


def main():
    df = pd.read_csv(INPUT)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    for layer in sorted(df["layer"].unique()):
        sub = (
            df[df["layer"] == layer]
            .sort_values("alpha")
        )

        ax.plot(
            sub["alpha"],
            sub["behavior_score"],
            marker="o",
            markersize=3,
            linewidth=1.3,
            label=f"L{layer}",
        )

    baseline = df["baseline"].iloc[0]

    ax.axhline(
        baseline,
        linestyle="--",
        linewidth=1.2,
        label=r"$\alpha=0$ baseline",
    )

    ax.set_xlabel(r"Steering coefficient $\alpha$")
    ax.set_ylabel("P(answer matching behavior)")
    ax.set_title("CAA coefficient response across layers (sycophancy)")

    ax.set_xticks(sorted(df["alpha"].unique()))
    ax.grid(alpha=0.2)

    ax.legend(
        frameon=False,
        ncol=2,
        fontsize=8,
    )

    fig.tight_layout()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(OUTPUT, bbox_inches="tight")
    fig.savefig(
        OUTPUT.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
