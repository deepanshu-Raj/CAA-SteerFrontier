from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


INPUT = "results/sycophancy/coefficient_layer_sweep.csv"
OUTPUT = Path("results/sycophancy/layer_selection_metrics.csv")


def main():
    df = pd.read_csv(INPUT)

    rows = []

    for layer in sorted(df["layer"].unique()):
        sub = (
            df[df["layer"] == layer]
            .sort_values("alpha")
            .reset_index(drop=True)
        )

        baseline = sub["baseline"].iloc[0]

        neg = sub[sub["alpha"] < 0]
        pos = sub[sub["alpha"] > 0]

        # --------------------------------------------------------
        # Maximum achievable directional gains.
        # --------------------------------------------------------
        negative_gain = (
            baseline - neg["behavior_score"].min()
        )

        positive_gain = (
            pos["behavior_score"].max() - baseline
        )

        bidirectional_gain = (
            negative_gain + positive_gain
        )

        # --------------------------------------------------------
        # Average gain over each side of the coefficient sweep.
        #
        # This rewards layers whose steering works consistently,
        # rather than only at one lucky alpha.
        # --------------------------------------------------------
        mean_negative_gain = (
            baseline - neg["behavior_score"]
        ).mean()

        mean_positive_gain = (
            pos["behavior_score"] - baseline
        ).mean()

        # --------------------------------------------------------
        # Fraction of coefficients steering in expected direction.
        # --------------------------------------------------------
        negative_direction_fraction = (
            neg["behavior_score"] < baseline
        ).mean()

        positive_direction_fraction = (
            pos["behavior_score"] > baseline
        ).mean()

        # --------------------------------------------------------
        # Monotonicity:
        # ideal behavior is B(alpha) increasing with alpha.
        # --------------------------------------------------------
        rho, _ = spearmanr(
            sub["alpha"],
            sub["behavior_score"],
        )

        scores = sub["behavior_score"].to_numpy()

        monotonic_violations = int(
            np.sum(np.diff(scores) < -1e-6)
        )

        # --------------------------------------------------------
        # Full observed behavior range.
        # --------------------------------------------------------
        total_range = (
            sub["behavior_score"].max()
            - sub["behavior_score"].min()
        )

        rows.append({
            "layer": layer,
            "spearman_rho": rho,
            "monotonic_violations": monotonic_violations,
            "negative_gain": negative_gain,
            "positive_gain": positive_gain,
            "bidirectional_gain": bidirectional_gain,
            "mean_negative_gain": mean_negative_gain,
            "mean_positive_gain": mean_positive_gain,
            "negative_direction_fraction": negative_direction_fraction,
            "positive_direction_fraction": positive_direction_fraction,
            "total_range": total_range,
        })

    result = pd.DataFrame(rows)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT,
        index=False,
    )

    display_cols = [
        "layer",
        "spearman_rho",
        "monotonic_violations",
        "negative_gain",
        "positive_gain",
        "bidirectional_gain",
        "mean_negative_gain",
        "mean_positive_gain",
        "negative_direction_fraction",
        "positive_direction_fraction",
    ]

    print(
        result[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
