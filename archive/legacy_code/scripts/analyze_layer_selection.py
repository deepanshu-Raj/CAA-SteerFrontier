import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from steerfrontier.experiment import results_dir


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--behavior",
        required=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    behavior = args.behavior

    input_path = results_dir(behavior) / "coefficient_layer_sweep.csv"

    output_path = results_dir(behavior) / "layer_selection_metrics.csv"

    df = pd.read_csv(input_path)

    rows = []

    for layer in sorted(df["layer"].unique()):
        sub = df[df["layer"] == layer].sort_values("alpha").reset_index(drop=True)

        baseline = sub["baseline"].iloc[0]

        neg = sub[sub["alpha"] < 0]
        pos = sub[sub["alpha"] > 0]

        # Maximum directional gains
        negative_gain = baseline - neg["behavior_score"].min()

        positive_gain = pos["behavior_score"].max() - baseline

        bidirectional_gain = negative_gain + positive_gain

        # Average directional gain over the sweep
        mean_negative_gain = (baseline - neg["behavior_score"]).mean()

        mean_positive_gain = (pos["behavior_score"] - baseline).mean()

        # Fraction of coefficients that steer correctly
        negative_direction_fraction = (neg["behavior_score"] < baseline).mean()

        positive_direction_fraction = (pos["behavior_score"] > baseline).mean()

        # Global monotonicity
        rho, _ = spearmanr(
            sub["alpha"],
            sub["behavior_score"],
        )

        scores = sub["behavior_score"].to_numpy()

        monotonic_violations = int(np.sum(np.diff(scores) < -1e-6))

        # Observed total behavior range
        total_range = sub["behavior_score"].max() - sub["behavior_score"].min()

        rows.append(
            {
                "behavior": behavior,
                "layer": layer,
                "spearman_rho": rho,
                "monotonic_violations": monotonic_violations,
                "negative_gain": negative_gain,
                "positive_gain": positive_gain,
                "bidirectional_gain": bidirectional_gain,
                "mean_negative_gain": mean_negative_gain,
                "mean_positive_gain": mean_positive_gain,
                "negative_direction_fraction": (negative_direction_fraction),
                "positive_direction_fraction": (positive_direction_fraction),
                "total_range": total_range,
            }
        )

    result = pd.DataFrame(rows)

    result.to_csv(
        output_path,
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
        "total_range",
    ]

    print(
        result[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
