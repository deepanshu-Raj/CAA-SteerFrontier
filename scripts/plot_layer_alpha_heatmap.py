from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT = "results/sycophancy/coefficient_layer_sweep.csv"
OUTPUT = Path("plots/sycophancy/layer_alpha_heatmap.pdf")


def main():
    df = pd.read_csv(INPUT)

    pivot = df.pivot(
        index="layer",
        columns="alpha",
        values="behavior_score",
    )

    layers = list(pivot.index)
    alphas = list(pivot.columns)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    image = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=df["behavior_score"].min(),
        vmax=df["behavior_score"].max(),
    )

    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels(
        [f"{alpha:g}" for alpha in alphas]
    )

    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)

    ax.set_xlabel(r"Steering coefficient $\alpha$")
    ax.set_ylabel("Layer #")
    ax.set_title(
        "Layer–coefficient CAA response (sycophancy)"
    )

    # Cell boundaries.
    ax.set_xticks(
        [x - 0.5 for x in range(1, len(alphas))],
        minor=True,
    )
    ax.set_yticks(
        [y - 0.5 for y in range(1, len(layers))],
        minor=True,
    )

    ax.grid(
        which="minor",
        linewidth=0.3,
        alpha=0.25,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("P(answer matching behavior)")

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
