"""Select and freeze capability-constrained steering budgets.

Behavior-selected layers are fixed upstream, feasibility is determined only on
capability-selection data, and holdout results are never used here.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

TOLERANCES = {
    "01": 0.01,
    "03": 0.03,
    "05": 0.05,
}

CORE_BENCHMARKS = {
    "mmlu",
    "arc_challenge",
    "hellaswag",
}


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select and freeze steering "
            "frontiers from behavior-validation and "
            "capability-selection results."
        )
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def load_selected_layers(
    results_root: Path,
):
    path = results_root / "selected_layers.json"

    with path.open("r") as f:
        payload = json.load(f)

    selected = payload.get("selected_layers")

    if not selected:
        raise ValueError(f"No selected_layers found in {path}")

    return {str(behavior): int(layer) for behavior, layer in selected.items()}


def load_capability_summary(
    results_root: Path,
):
    path = results_root / "capability_selection" / "summary.csv"

    df = pd.read_csv(path)

    required = {
        "behavior",
        "layer",
        "benchmark",
        "alpha",
        "baseline_accuracy",
        "accuracy",
        "delta_accuracy",
        "delta_ci_low",
        "delta_ci_high",
        "feasible_eps_01",
        "feasible_eps_03",
        "feasible_eps_05",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    return df


def load_behavior_sweep(
    results_root: Path,
    behavior: str,
    selected_layer: int,
):
    path = results_root / behavior / "coefficient_layer_sweep.csv"

    df = pd.read_csv(path)

    required = {
        "behavior",
        "layer",
        "alpha",
        "behavior_score",
        "baseline",
        "delta_from_baseline",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = df[df["layer"].astype(int) == int(selected_layer)].copy()

    if df.empty:
        raise ValueError(
            f"No coefficient-sweep rows for "
            f"{behavior} at selected layer "
            f"{selected_layer}."
        )

    if df["alpha"].duplicated().any():
        duplicates = df.loc[
            df["alpha"].duplicated(keep=False),
            "alpha",
        ].tolist()

        raise ValueError(
            f"Duplicate alpha rows for {behavior} at L{selected_layer}: {duplicates}"
        )

    return df


def build_capability_feasibility(
    capability_df: pd.DataFrame,
    behavior: str,
    selected_layer: int,
    epsilon_key: str,
):
    column = f"feasible_eps_{epsilon_key}"

    subset = capability_df[
        (capability_df["behavior"] == behavior)
        & (capability_df["layer"].astype(int) == int(selected_layer))
    ].copy()

    if subset.empty:
        raise ValueError(f"No capability results for {behavior} at L{selected_layer}.")

    benchmarks = set(subset["benchmark"].unique())

    missing_benchmarks = CORE_BENCHMARKS - benchmarks

    if missing_benchmarks:
        raise ValueError(
            f"Capability summary for "
            f"{behavior} is missing benchmarks: "
            f"{sorted(missing_benchmarks)}"
        )

    rows = []

    for alpha, group in subset.groupby(
        "alpha",
        sort=True,
    ):
        group = group[group["benchmark"].isin(CORE_BENCHMARKS)]

        if len(group) != len(CORE_BENCHMARKS):
            raise ValueError(
                f"{behavior}, alpha={alpha}: "
                "expected exactly one row per "
                "core benchmark."
            )

        feasible = bool(group[column].astype(bool).all())

        worst_row = group.sort_values(
            "delta_ci_low",
            ascending=True,
        ).iloc[0]

        rows.append(
            {
                "alpha": float(alpha),
                "capability_feasible": (feasible),
                "worst_capability_lcb": float(worst_row["delta_ci_low"]),
                "limiting_benchmark": str(worst_row["benchmark"]),
            }
        )

    return pd.DataFrame(rows)


def steering_eta(
    row,
):
    if "signed_eta" in row.index:
        value = row["signed_eta"]

        if pd.notna(value):
            return float(value)

    if "eta" in row.index:
        value = row["eta"]

        if pd.notna(value):
            magnitude = float(value)

            alpha = float(row["alpha"])

            if alpha < 0:
                return -abs(magnitude)

            return abs(magnitude)

    return None


def select_direction(
    merged: pd.DataFrame,
    direction: str,
):
    if direction not in {
        "negative",
        "positive",
    }:
        raise ValueError(f"Unknown direction: {direction}")

    if direction == "negative":
        candidates = merged[
            (merged["alpha"] < 0) & (merged["capability_feasible"])
        ].copy()

        candidates["behavior_gain"] = (
            candidates["baseline"] - candidates["behavior_score"]
        )

    else:
        candidates = merged[
            (merged["alpha"] > 0) & (merged["capability_feasible"])
        ].copy()

        candidates["behavior_gain"] = (
            candidates["behavior_score"] - candidates["baseline"]
        )

    # No usable nonzero steering is better than selecting
    # a coefficient that moves behavior in the wrong direction.
    candidates = candidates[candidates["behavior_gain"] > 0].copy()

    if candidates.empty:
        zero = merged[merged["alpha"] == 0]

        if len(zero) != 1:
            raise ValueError("Expected exactly one alpha=0 row.")

        row = zero.iloc[0]

        return {
            "alpha_star": 0.0,
            "eta_star": 0.0,
            "behavior_score": float(row["behavior_score"]),
            "behavior_baseline": float(row["baseline"]),
            "behavior_gain": 0.0,
            "worst_capability_lcb": 0.0,
            "limiting_benchmark": None,
            "nonzero_feasible": False,
        }

    # Deterministic conservative tie-break:
    #
    # 1. maximize behavior gain
    # 2. among exact ties, choose smaller |alpha|
    candidates["abs_alpha"] = candidates["alpha"].abs()

    candidates = candidates.sort_values(
        by=[
            "behavior_gain",
            "abs_alpha",
        ],
        ascending=[
            False,
            True,
        ],
    )

    row = candidates.iloc[0]

    eta = steering_eta(row)

    return {
        "alpha_star": float(row["alpha"]),
        "eta_star": (eta),
        "behavior_score": float(row["behavior_score"]),
        "behavior_baseline": float(row["baseline"]),
        "behavior_gain": float(row["behavior_gain"]),
        "worst_capability_lcb": float(row["worst_capability_lcb"]),
        "limiting_benchmark": str(row["limiting_benchmark"]),
        "nonzero_feasible": True,
    }


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    results_root = args.results_root

    selected_layers = load_selected_layers(results_root)

    capability_df = load_capability_summary(results_root)

    output_rows = []

    frozen = {
        "protocol": ("steering_frontiers"),
        "selection_split": ("behavior_validation + capability_selection"),
        "selection_rule": (
            "For each behavior, direction, and "
            "epsilon: maximize directional "
            "behavior-validation gain among "
            "coefficients satisfying the capability "
            "constraint on every core benchmark. "
            "If none produces positive directional "
            "gain, select alpha=0. Exact gain ties "
            "prefer smaller absolute alpha."
        ),
        "capability_constraint": (
            "LCB95(delta_accuracy) >= -epsilon for every core benchmark"
        ),
        "core_benchmarks": sorted(CORE_BENCHMARKS),
        "tolerances": {key: value for key, value in TOLERANCES.items()},
        "behaviors": {},
    }

    for behavior, layer in selected_layers.items():
        behavior_df = load_behavior_sweep(
            results_root=(results_root),
            behavior=behavior,
            selected_layer=layer,
        )

        frozen["behaviors"][behavior] = {
            "layer": int(layer),
            "tolerances": {},
        }

        for epsilon_key, epsilon in TOLERANCES.items():
            capability = build_capability_feasibility(
                capability_df=(capability_df),
                behavior=behavior,
                selected_layer=layer,
                epsilon_key=(epsilon_key),
            )

            merged = behavior_df.merge(
                capability,
                on="alpha",
                how="inner",
                validate="one_to_one",
            )

            if len(merged) != len(behavior_df):
                missing = sorted(
                    set(behavior_df["alpha"].astype(float))
                    - set(merged["alpha"].astype(float))
                )

                raise ValueError(
                    f"{behavior}: capability results missing alphas: {missing}"
                )

            tolerance_payload = {}

            for direction in (
                "negative",
                "positive",
            ):
                selected = select_direction(
                    merged=merged,
                    direction=direction,
                )

                output_rows.append(
                    {
                        "behavior": (behavior),
                        "layer": int(layer),
                        "epsilon": float(epsilon),
                        "direction": (direction),
                        "alpha_star": (selected["alpha_star"]),
                        "eta_star": (selected["eta_star"]),
                        "behavior_baseline": (selected["behavior_baseline"]),
                        "behavior_score": (selected["behavior_score"]),
                        "behavior_gain": (selected["behavior_gain"]),
                        "worst_capability_lcb": (selected["worst_capability_lcb"]),
                        "limiting_benchmark": (selected["limiting_benchmark"]),
                        "nonzero_feasible": (selected["nonzero_feasible"]),
                    }
                )

                tolerance_payload[direction] = {
                    "alpha": (selected["alpha_star"]),
                    "eta": (selected["eta_star"]),
                    "behavior_gain": (selected["behavior_gain"]),
                    "behavior_score": (selected["behavior_score"]),
                    "behavior_baseline": (selected["behavior_baseline"]),
                    "worst_capability_lcb": (selected["worst_capability_lcb"]),
                    "limiting_benchmark": (selected["limiting_benchmark"]),
                }

            frozen["behaviors"][behavior]["tolerances"][epsilon_key] = tolerance_payload

    frontier_df = pd.DataFrame(output_rows)

    frontier_path = results_root / "steering_frontiers.csv"

    frontier_df.to_csv(
        frontier_path,
        index=False,
    )

    frozen_path = results_root / "selected_coefficients.json"

    with frozen_path.open("w") as f:
        json.dump(
            frozen,
            f,
            indent=2,
        )

    print("=" * 76)

    print("STEERING FRONTIERS FROZEN")

    print("=" * 76)

    print()

    display = frontier_df[
        [
            "behavior",
            "epsilon",
            "direction",
            "alpha_star",
            "eta_star",
            "behavior_gain",
            "worst_capability_lcb",
            "limiting_benchmark",
        ]
    ]

    print(display.to_string(index=False))

    print()

    print(
        "Frontiers ->",
        frontier_path,
    )

    print(
        "Frozen coefficients ->",
        frozen_path,
    )


if __name__ == "__main__":
    main()
