from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT = "results/sycophancy/layer_selection_metrics.csv"
OUTPUT = Path("plots/sycophancy/layer_steering_gains.pdf")


def main():
    df = pd.read_csv(INPUT)

    layers = df["layer"].to_numpy()
    x = np.arange(len(layers))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    ax.bar(
        x - width / 2,
        df["negative_gain"],
        width,
        label="Counter-steering gain",
    )

    ax.bar(
        x + width / 2,
        df["positive_gain"],
        width,
        label="In-direction gain",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(layers)

    ax.set_xlabel("Layer #")
    ax.set_ylabel("Behavior probability gain")
    ax.set_title("Directional CAA steering gains (sycophancy)")

    ax.legend(frameon=False)
    ax.grid(
        axis="y",
        alpha=0.2,
    )

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


if __name__ == "__main__":
    main()
