import argparse
import json
from pathlib import Path

import pandas as pd

from steerfrontier.experiment import results_dir


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--behavior",
        required=True,
    )

    parser.add_argument(
        "--selected-layer",
        required=True,
        type=int,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    directory = results_dir(args.behavior)

    coefficient = pd.read_csv(directory / "coefficient_layer_sweep.csv")

    selection = pd.read_csv(directory / "layer_selection_metrics.csv")

    mmlu = pd.read_csv(directory / f"mmlu_layer{args.selected_layer}.csv")

    layer_data = coefficient[coefficient["layer"] == args.selected_layer]

    selection_row = selection[selection["layer"] == args.selected_layer].iloc[0]

    baseline_behavior = layer_data[layer_data["alpha"] == 0]["behavior_score"].iloc[0]

    baseline_capability = mmlu[mmlu["alpha"] == 0]["capability_score"].iloc[0]

    summary = {
        "behavior": args.behavior,
        "selected_layer": args.selected_layer,
        "baseline_behavior": float(baseline_behavior),
        "baseline_capability": float(baseline_capability),
        "alphas": sorted(coefficient["alpha"].unique().tolist()),
        "layer_selection": {
            "spearman_rho": float(selection_row["spearman_rho"]),
            "monotonic_violations": int(selection_row["monotonic_violations"]),
            "negative_gain": float(selection_row["negative_gain"]),
            "positive_gain": float(selection_row["positive_gain"]),
            "bidirectional_gain": float(selection_row["bidirectional_gain"]),
            "negative_direction_fraction": float(
                selection_row["negative_direction_fraction"]
            ),
            "positive_direction_fraction": float(
                selection_row["positive_direction_fraction"]
            ),
        },
    }

    output = directory / "summary.json"

    with output.open("w") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print(json.dumps(summary, indent=2))
    print("\nSaved:", output)


if __name__ == "__main__":
    main()
