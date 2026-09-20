"""Audit and aggregate frozen per-model CAA-SteerFrontier results.

This module performs no model evaluation and no new intervention selection; it
only validates and summarizes already-frozen artifacts across models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

MODEL_SPECS = [
    {
        "slug": "llama2-7b-chat",
        "label": "Llama-2 7B Chat",
        "num_layers": 32,
    },
    {
        "slug": "llama3.1-8b-instruct",
        "label": "Llama-3.1 8B Instruct",
        "num_layers": 32,
    },
    {
        "slug": "mistral-7b-instruct-v0.3",
        "label": "Mistral 7B Instruct v0.3",
        "num_layers": 32,
    },
    {
        "slug": "qwen2.5-7b-instruct",
        "label": "Qwen2.5 7B Instruct",
        "num_layers": 28,
    },
]


BEHAVIOR_ORDER = [
    "sycophancy",
    "refusal",
    "hallucination",
    "corrigibility",
    "survival_instinct",
    "myopic_reward",
    "ai_coordination",
]


EPSILONS = [
    0.01,
    0.03,
    0.05,
]


DIRECTIONS = [
    "negative",
    "positive",
]


BENCHMARK_ORDER = [
    "mmlu",
    "arc_challenge",
    "hellaswag",
]


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit and aggregate frozen CAA-SteerFrontier results "
            "across models. No plotting and no new selection."
        )
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/cross_model"),
    )

    return parser.parse_args()


# =============================================================================
# Utilities
# =============================================================================


def require_file(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")


def to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "True": True,
        "False": False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    converted = series.map(mapping)

    if converted.isna().any():
        bad = series[converted.isna()].unique().tolist()

        raise ValueError(f"Could not parse boolean values: {bad}")

    return converted.astype(bool)


def assert_close(
    left,
    right,
    name,
    atol=1e-10,
):
    left_values = left.to_numpy(dtype=float)

    right_values = right.to_numpy(dtype=float)

    if not np.allclose(
        left_values,
        right_values,
        atol=atol,
        rtol=0.0,
        equal_nan=True,
    ):
        diff = np.abs(left_values - right_values)

        max_diff = np.nanmax(diff)

        raise ValueError(
            f"Frozen/final audit failed for {name}; max abs diff={max_diff}"
        )


def read_selected_layers(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        payload = json.load(f)

    raw = payload.get(
        "selected_layers",
        payload,
    )

    selected = {}

    for behavior in BEHAVIOR_ORDER:
        if behavior not in raw:
            raise ValueError(f"Missing selected layer for {behavior!r} in {path}")

        value = raw[behavior]

        if isinstance(
            value,
            dict,
        ):
            layer = None

            for key in (
                "layer",
                "selected_layer",
                "layer_star",
            ):
                if key in value:
                    layer = value[key]
                    break

            if layer is None:
                raise ValueError(
                    f"Could not determine layer for {behavior!r} in {path}"
                )

            value = layer

        selected[behavior] = int(value)

    return selected


# =============================================================================
# Per-model artifact loading + frozen/final audit
# =============================================================================


def load_model_artifacts(
    results_root: Path,
    spec,
):
    slug = spec["slug"]

    label = spec["label"]

    num_layers = int(spec["num_layers"])

    model_root = results_root / slug

    selected_layers_path = model_root / "selected_layers.json"

    frozen_path = model_root / "steering_frontiers.csv"

    final_root = model_root / "final_evaluation"

    final_path = final_root / "frontier_test_results.csv"

    behavior_summary_path = final_root / "behavior_test_summary.csv"

    capability_summary_path = final_root / "capability_holdout_summary.csv"

    for path in (
        selected_layers_path,
        frozen_path,
        final_path,
        behavior_summary_path,
        capability_summary_path,
    ):
        require_file(path)

    selected_layers = read_selected_layers(selected_layers_path)

    frozen = pd.read_csv(frozen_path)

    final = pd.read_csv(final_path)

    behavior_summary = pd.read_csv(behavior_summary_path)

    capability_summary = pd.read_csv(capability_summary_path)

    expected_rows = len(BEHAVIOR_ORDER) * len(EPSILONS) * len(DIRECTIONS)

    if len(frozen) != expected_rows:
        raise ValueError(
            f"{slug}: expected {expected_rows} frozen frontier rows, got {len(frozen)}"
        )

    if len(final) != expected_rows:
        raise ValueError(
            f"{slug}: expected {expected_rows} final frontier rows, got {len(final)}"
        )

    for column in (
        "test_direction_supported",
        "capability_holdout_pass",
    ):
        final[column] = to_bool(final[column])

    key = [
        "behavior",
        "layer",
        "epsilon",
        "direction",
    ]

    if frozen.duplicated(key).any():
        raise ValueError(f"{slug}: duplicate keys in steering_frontiers.csv")

    if final.duplicated(key).any():
        raise ValueError(f"{slug}: duplicate keys in frontier_test_results.csv")

    audit = final.merge(
        frozen,
        on=key,
        how="outer",
        validate="one_to_one",
        suffixes=(
            "_final",
            "_frozen",
        ),
        indicator=True,
    )

    if not (audit["_merge"] == "both").all():
        raise ValueError(f"{slug}: frozen and final frontier keys do not match exactly")

    assert_close(
        audit["alpha_star_final"],
        audit["alpha_star_frozen"],
        f"{slug}: alpha_star",
    )

    assert_close(
        audit["eta_star_final"],
        audit["eta_star_frozen"],
        f"{slug}: eta_star",
    )

    assert_close(
        audit["selection_behavior_gain"],
        audit["behavior_gain"],
        f"{slug}: selection behavior gain",
    )

    for behavior, layer in selected_layers.items():
        observed = set(
            final.loc[
                final["behavior"] == behavior,
                "layer",
            ]
            .astype(int)
            .tolist()
        )

        if observed != {layer}:
            raise ValueError(
                f"{slug}/{behavior}: "
                f"selected layer {layer}, "
                f"but final artifact contains "
                f"{sorted(observed)}"
            )

    layer_rows = []

    for behavior in BEHAVIOR_ORDER:
        layer = selected_layers[behavior]

        layer_rows.append(
            {
                "model_slug": slug,
                "model_label": label,
                "behavior": behavior,
                "selected_layer": layer,
                "num_hidden_layers": num_layers,
                "fractional_depth": (layer / (num_layers - 1)),
            }
        )

    layer_df = pd.DataFrame(layer_rows)

    for frame in (
        frozen,
        final,
        behavior_summary,
        capability_summary,
    ):
        frame.insert(
            0,
            "model_label",
            label,
        )

        frame.insert(
            0,
            "model_slug",
            slug,
        )

    return {
        "final": final,
        "frozen": frozen,
        "behavior_summary": (behavior_summary),
        "capability_summary": (capability_summary),
        "selected_layers": (layer_df),
    }


# =============================================================================
# Unique frozen interventions
# =============================================================================


def unique_nonzero_interventions(
    combined,
):
    nonzero = combined.loc[combined["alpha_star"].abs() > 1e-12].copy()

    intervention_key = [
        "model_slug",
        "model_label",
        "behavior",
        "layer",
        "direction",
        "alpha_star",
        "eta_star",
    ]

    consistency_columns = [
        "selection_behavior_gain",
        "test_behavior_baseline",
        "test_behavior_score",
        "test_behavior_delta",
        "test_behavior_delta_ci_low",
        "test_behavior_delta_ci_high",
        "test_directional_gain",
        "test_directional_gain_ci_low",
        "test_directional_gain_ci_high",
        "test_direction_supported",
        "worst_holdout_lcb",
        "limiting_holdout_benchmark",
    ]

    for _, group in nonzero.groupby(
        intervention_key,
        dropna=False,
    ):
        for column in consistency_columns:
            if column not in (group.columns):
                continue

            if pd.api.types.is_numeric_dtype(group[column]):
                values = group[column].to_numpy(dtype=float)

                if not np.allclose(
                    values,
                    values[0],
                    atol=1e-10,
                    rtol=0.0,
                    equal_nan=True,
                ):
                    raise ValueError(
                        f"Repeated frozen intervention has inconsistent {column}"
                    )

            else:
                if group[column].nunique(dropna=False) != 1:
                    raise ValueError(
                        f"Repeated frozen intervention has inconsistent {column}"
                    )

    deduplicated = (
        nonzero.sort_values("epsilon")
        .drop_duplicates(
            intervention_key,
            keep="first",
        )
        .reset_index(drop=True)
    )

    return deduplicated


# =============================================================================
# Validation -> test generalization
# =============================================================================


def correlation_row(
    scope,
    frame,
    model_slug=None,
):
    x = frame["selection_behavior_gain"].astype(float)

    y = frame["test_directional_gain"].astype(float)

    if len(frame) >= 2:
        pearson_r = float(
            x.corr(
                y,
                method="pearson",
            )
        )

        spearman_r = float(
            x.corr(
                y,
                method="spearman",
            )
        )

    else:
        pearson_r = np.nan
        spearman_r = np.nan

    if len(frame) >= 2 and float(x.std(ddof=0)) > 0.0:
        slope, intercept = np.polyfit(
            x.to_numpy(),
            y.to_numpy(),
            deg=1,
        )

    else:
        slope = np.nan
        intercept = np.nan

    return {
        "scope": scope,
        "model_slug": model_slug,
        "n_unique_nonzero_interventions": int(len(frame)),
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "linear_slope": float(slope),
        "linear_intercept": float(intercept),
        "mean_selection_gain": (float(x.mean()) if len(frame) else np.nan),
        "mean_test_gain": (float(y.mean()) if len(frame) else np.nan),
        "mean_gain_difference": (float((y - x).mean()) if len(frame) else np.nan),
        "mean_absolute_gain_difference": (
            float((y - x).abs().mean()) if len(frame) else np.nan
        ),
    }


def build_correlation_summary(
    unique_nonzero,
):
    rows = [
        correlation_row(
            "all_models",
            unique_nonzero,
        )
    ]

    for spec in MODEL_SPECS:
        group = unique_nonzero.loc[unique_nonzero["model_slug"] == spec["slug"]]

        rows.append(
            correlation_row(
                spec["label"],
                group,
                model_slug=spec["slug"],
            )
        )

    return pd.DataFrame(rows)


# =============================================================================
# Frontier survival
# =============================================================================


def frontier_survival_row(
    group,
):
    nonzero = group.loc[group["alpha_star"].abs() > 1e-12]

    n_nonzero = int(len(nonzero))

    behavior_supported = nonzero["test_direction_supported"]

    capability_pass = nonzero["capability_holdout_pass"]

    joint = behavior_supported & capability_pass

    return {
        "total_frontier_claims": int(len(group)),
        "zero_frontiers": int(len(group) - n_nonzero),
        "nonzero_frontiers": n_nonzero,
        "behavior_supported": int(behavior_supported.sum()),
        "capability_holdout_pass": int(capability_pass.sum()),
        "joint_pass": int(joint.sum()),
        "behavior_support_rate": (
            float(behavior_supported.mean()) if n_nonzero else np.nan
        ),
        "capability_pass_rate": (
            float(capability_pass.mean()) if n_nonzero else np.nan
        ),
        "joint_pass_rate": (float(joint.mean()) if n_nonzero else np.nan),
    }


def build_frontier_survival(
    combined,
):
    model_rows = []

    for (
        slug,
        label,
        epsilon,
    ), group in combined.groupby(
        [
            "model_slug",
            "model_label",
            "epsilon",
        ],
        sort=False,
    ):
        row = {
            "model_slug": slug,
            "model_label": label,
            "epsilon": float(epsilon),
        }

        row.update(frontier_survival_row(group))

        model_rows.append(row)

    overall_rows = []

    for epsilon, group in combined.groupby(
        "epsilon",
        sort=True,
    ):
        row = {"epsilon": float(epsilon)}

        row.update(frontier_survival_row(group))

        overall_rows.append(row)

    return (
        pd.DataFrame(model_rows),
        pd.DataFrame(overall_rows),
    )


def build_behavior_survival(
    combined,
):
    rows = []

    for (
        behavior,
        epsilon,
    ), group in combined.groupby(
        [
            "behavior",
            "epsilon",
        ],
        sort=False,
    ):
        row = {
            "behavior": behavior,
            "epsilon": float(epsilon),
        }

        row.update(frontier_survival_row(group))

        rows.append(row)

    return pd.DataFrame(rows)


def build_model_behavior_survival(
    combined,
):
    rows = []

    for (
        slug,
        label,
        behavior,
        epsilon,
    ), group in combined.groupby(
        [
            "model_slug",
            "model_label",
            "behavior",
            "epsilon",
        ],
        sort=False,
    ):
        row = {
            "model_slug": slug,
            "model_label": label,
            "behavior": behavior,
            "epsilon": float(epsilon),
        }

        row.update(frontier_survival_row(group))

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Directional analysis
# =============================================================================


def build_directional_summary(
    combined,
):
    nonzero = combined.loc[combined["alpha_star"].abs() > 1e-12].copy()

    rows = []

    for (
        slug,
        label,
        epsilon,
        direction,
    ), group in nonzero.groupby(
        [
            "model_slug",
            "model_label",
            "epsilon",
            "direction",
        ],
        sort=False,
    ):
        joint = group["test_direction_supported"] & group["capability_holdout_pass"]

        rows.append(
            {
                "model_slug": slug,
                "model_label": label,
                "epsilon": float(epsilon),
                "direction": direction,
                "nonzero_frontiers": int(len(group)),
                "mean_selection_gain": float(group["selection_behavior_gain"].mean()),
                "median_selection_gain": float(
                    group["selection_behavior_gain"].median()
                ),
                "mean_test_gain": float(group["test_directional_gain"].mean()),
                "median_test_gain": float(group["test_directional_gain"].median()),
                "behavior_support_rate": float(
                    group["test_direction_supported"].mean()
                ),
                "capability_pass_rate": float(group["capability_holdout_pass"].mean()),
                "joint_pass_rate": float(joint.mean()),
            }
        )

    return pd.DataFrame(rows)


def build_directional_asymmetry(
    combined,
):
    records = []

    for (
        slug,
        label,
        behavior,
        epsilon,
    ), group in combined.groupby(
        [
            "model_slug",
            "model_label",
            "behavior",
            "epsilon",
        ],
        sort=False,
    ):
        by_direction = {row["direction"]: row for _, row in (group.iterrows())}

        if set(by_direction) != set(DIRECTIONS):
            raise ValueError(
                f"Expected both directions for {slug}/{behavior}/epsilon={epsilon}"
            )

        negative = by_direction["negative"]

        positive = by_direction["positive"]

        negative_gain = float(negative["test_directional_gain"])

        positive_gain = float(positive["test_directional_gain"])

        denominator = abs(positive_gain) + abs(negative_gain)

        records.append(
            {
                "model_slug": slug,
                "model_label": label,
                "behavior": behavior,
                "epsilon": float(epsilon),
                "negative_alpha": float(negative["alpha_star"]),
                "positive_alpha": float(positive["alpha_star"]),
                "negative_eta": float(negative["eta_star"]),
                "positive_eta": float(positive["eta_star"]),
                "negative_test_gain": (negative_gain),
                "positive_test_gain": (positive_gain),
                "positive_minus_negative_test_gain": (positive_gain - negative_gain),
                "absolute_test_gain_asymmetry": abs(positive_gain - negative_gain),
                "normalized_test_gain_asymmetry": (
                    (positive_gain - negative_gain) / denominator
                    if denominator > 0
                    else 0.0
                ),
                "negative_behavior_supported": bool(
                    negative["test_direction_supported"]
                ),
                "positive_behavior_supported": bool(
                    positive["test_direction_supported"]
                ),
                "negative_capability_pass": bool(negative["capability_holdout_pass"]),
                "positive_capability_pass": bool(positive["capability_holdout_pass"]),
                "negative_joint_pass": bool(
                    negative["test_direction_supported"]
                    and negative["capability_holdout_pass"]
                ),
                "positive_joint_pass": bool(
                    positive["test_direction_supported"]
                    and positive["capability_holdout_pass"]
                ),
            }
        )

    return pd.DataFrame(records)


# =============================================================================
# Capability bottlenecks
# =============================================================================


def build_limiting_benchmark_tables(
    combined,
    unique_nonzero,
):
    unique_counts = (
        unique_nonzero.groupby(
            [
                "model_slug",
                "model_label",
                "limiting_holdout_benchmark",
            ],
            dropna=False,
        )
        .size()
        .rename("unique_interventions")
        .reset_index()
    )

    totals = unique_counts.groupby(
        [
            "model_slug",
            "model_label",
        ]
    )["unique_interventions"].transform("sum")

    unique_counts["fraction_with_benchmark_limiting"] = (
        unique_counts["unique_interventions"] / totals
    )

    overall_unique = (
        unique_nonzero.groupby(
            "limiting_holdout_benchmark",
            dropna=False,
        )
        .size()
        .rename("unique_interventions")
        .reset_index()
    )

    overall_total = int(len(unique_nonzero))

    overall_unique["fraction_with_benchmark_limiting"] = (
        overall_unique["unique_interventions"] / overall_total
    )

    failures = combined.loc[
        (combined["alpha_star"].abs() > 1e-12) & (~combined["capability_holdout_pass"])
    ].copy()

    if failures.empty:
        failure_counts = pd.DataFrame(
            columns=[
                "model_slug",
                "model_label",
                "epsilon",
                "limiting_holdout_benchmark",
                "failed_frontiers",
            ]
        )

        overall_failures = pd.DataFrame(
            columns=[
                "epsilon",
                "limiting_holdout_benchmark",
                "failed_frontiers",
            ]
        )

    else:
        failure_counts = (
            failures.groupby(
                [
                    "model_slug",
                    "model_label",
                    "epsilon",
                    "limiting_holdout_benchmark",
                ],
                dropna=False,
            )
            .size()
            .rename("failed_frontiers")
            .reset_index()
        )

        overall_failures = (
            failures.groupby(
                [
                    "epsilon",
                    "limiting_holdout_benchmark",
                ],
                dropna=False,
            )
            .size()
            .rename("failed_frontiers")
            .reset_index()
        )

    return (
        unique_counts,
        overall_unique,
        failure_counts,
        overall_failures,
    )


# =============================================================================
# Steering magnitude
# =============================================================================


def build_eta_summary(
    combined,
):
    nonzero = combined.loc[combined["alpha_star"].abs() > 1e-12].copy()

    nonzero["abs_eta_star"] = nonzero["eta_star"].abs()

    rows = []

    for (
        slug,
        label,
        epsilon,
    ), group in nonzero.groupby(
        [
            "model_slug",
            "model_label",
            "epsilon",
        ],
        sort=False,
    ):
        rows.append(
            {
                "model_slug": slug,
                "model_label": label,
                "epsilon": float(epsilon),
                "n": int(len(group)),
                "mean_abs_eta": float(group["abs_eta_star"].mean()),
                "median_abs_eta": float(group["abs_eta_star"].median()),
                "min_abs_eta": float(group["abs_eta_star"].min()),
                "max_abs_eta": float(group["abs_eta_star"].max()),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Model × behavior summary
# =============================================================================


def build_model_behavior_summary(
    selected_layers,
    unique_nonzero,
    combined,
):
    behavior_metrics = (
        unique_nonzero.groupby(
            [
                "model_slug",
                "model_label",
                "behavior",
            ]
        )
        .agg(
            n_unique_nonzero_interventions=(
                "alpha_star",
                "size",
            ),
            mean_selection_gain=(
                "selection_behavior_gain",
                "mean",
            ),
            mean_test_gain=(
                "test_directional_gain",
                "mean",
            ),
            median_test_gain=(
                "test_directional_gain",
                "median",
            ),
            behavior_support_rate_unique=(
                "test_direction_supported",
                "mean",
            ),
            mean_abs_eta=(
                "eta_star",
                lambda x: np.abs(x).mean(),
            ),
        )
        .reset_index()
    )

    eps05 = combined.loc[
        np.isclose(
            combined["epsilon"],
            0.05,
        )
    ].copy()

    eps05_rows = []

    for (
        slug,
        label,
        behavior,
    ), group in eps05.groupby(
        [
            "model_slug",
            "model_label",
            "behavior",
        ],
        sort=False,
    ):
        nonzero = group.loc[group["alpha_star"].abs() > 1e-12]

        joint = nonzero["test_direction_supported"] & nonzero["capability_holdout_pass"]

        eps05_rows.append(
            {
                "model_slug": slug,
                "model_label": label,
                "behavior": behavior,
                "eps05_nonzero_frontiers": int(len(nonzero)),
                "eps05_behavior_supported": int(
                    nonzero["test_direction_supported"].sum()
                ),
                "eps05_capability_pass": int(nonzero["capability_holdout_pass"].sum()),
                "eps05_joint_pass": int(joint.sum()),
                "eps05_joint_pass_rate": (
                    float(joint.mean()) if len(nonzero) else np.nan
                ),
            }
        )

    eps05_df = pd.DataFrame(eps05_rows)

    return selected_layers.merge(
        behavior_metrics,
        on=[
            "model_slug",
            "model_label",
            "behavior",
        ],
        how="left",
        validate="one_to_one",
    ).merge(
        eps05_df,
        on=[
            "model_slug",
            "model_label",
            "behavior",
        ],
        how="left",
        validate="one_to_one",
    )


# =============================================================================
# Machine-readable project summary
# =============================================================================


def build_summary_json(
    combined,
    unique_nonzero,
    correlation,
    survival_overall,
):
    overall_corr = correlation.loc[correlation["scope"] == "all_models"].iloc[0]

    survival = {}

    for row in survival_overall.itertuples(index=False):
        key = f"{100 * row.epsilon:.0f}%"

        survival[key] = {
            "nonzero_frontiers": int(row.nonzero_frontiers),
            "behavior_supported": int(row.behavior_supported),
            "capability_holdout_pass": int(row.capability_holdout_pass),
            "joint_pass": int(row.joint_pass),
            "behavior_support_rate": float(row.behavior_support_rate),
            "capability_pass_rate": float(row.capability_pass_rate),
            "joint_pass_rate": float(row.joint_pass_rate),
        }

    return {
        "models": len(MODEL_SPECS),
        "behaviors": len(BEHAVIOR_ORDER),
        "model_behavior_pairs": (len(MODEL_SPECS) * len(BEHAVIOR_ORDER)),
        "frozen_frontier_claims": int(len(combined)),
        "unique_nonzero_interventions": int(len(unique_nonzero)),
        "validation_test_generalization": {
            "pearson_r": float(overall_corr["pearson_r"]),
            "spearman_r": float(overall_corr["spearman_r"]),
            "mean_gain_difference": float(overall_corr["mean_gain_difference"]),
            "mean_absolute_gain_difference": float(
                overall_corr["mean_absolute_gain_difference"]
            ),
        },
        "frontier_survival": survival,
    }


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    finals = []
    frozens = []
    behavior_summaries = []
    capability_summaries = []
    selected_layer_frames = []

    print("=" * 76)

    print("AUDITING MODEL ARTIFACTS")

    print("=" * 76)

    for spec in MODEL_SPECS:
        artifacts = load_model_artifacts(
            args.results_root,
            spec,
        )

        finals.append(artifacts["final"])

        frozens.append(artifacts["frozen"])

        behavior_summaries.append(artifacts["behavior_summary"])

        capability_summaries.append(artifacts["capability_summary"])

        selected_layer_frames.append(artifacts["selected_layers"])

        print(f"[PASS] {spec['label']}")

    combined = pd.concat(
        finals,
        ignore_index=True,
    )

    combined_frozen = pd.concat(
        frozens,
        ignore_index=True,
    )

    combined_behavior_summary = pd.concat(
        behavior_summaries,
        ignore_index=True,
    )

    combined_capability_summary = pd.concat(
        capability_summaries,
        ignore_index=True,
    )

    selected_layers = pd.concat(
        selected_layer_frames,
        ignore_index=True,
    )

    unique_nonzero = unique_nonzero_interventions(combined)

    correlation = build_correlation_summary(unique_nonzero)

    (
        survival_by_model,
        survival_overall,
    ) = build_frontier_survival(combined)

    behavior_survival = build_behavior_survival(combined)

    model_behavior_survival = build_model_behavior_survival(combined)

    directional_summary = build_directional_summary(combined)

    directional_asymmetry = build_directional_asymmetry(combined)

    (
        limiting_by_model,
        limiting_overall,
        failures_by_model,
        failures_overall,
    ) = build_limiting_benchmark_tables(
        combined,
        unique_nonzero,
    )

    eta_summary = build_eta_summary(combined)

    model_behavior_summary = build_model_behavior_summary(
        selected_layers,
        unique_nonzero,
        combined,
    )

    tables = {
        "combined_frontier_test_results.csv": (combined),
        "combined_frozen_frontiers.csv": (combined_frozen),
        "combined_behavior_test_summary.csv": (combined_behavior_summary),
        "combined_capability_holdout_summary.csv": (combined_capability_summary),
        "selected_layers_fractional_depth.csv": (selected_layers),
        "unique_nonzero_interventions.csv": (unique_nonzero),
        "validation_test_correlation.csv": (correlation),
        "frontier_survival_by_model.csv": (survival_by_model),
        "frontier_survival_overall.csv": (survival_overall),
        "frontier_survival_by_behavior.csv": (behavior_survival),
        "frontier_survival_by_model_behavior.csv": (model_behavior_survival),
        "directional_summary.csv": (directional_summary),
        "directional_asymmetry.csv": (directional_asymmetry),
        "limiting_benchmark_by_model.csv": (limiting_by_model),
        "limiting_benchmark_overall.csv": (limiting_overall),
        "capability_failures_by_model.csv": (failures_by_model),
        "capability_failures_overall.csv": (failures_overall),
        "steering_magnitude_summary.csv": (eta_summary),
        "model_behavior_summary.csv": (model_behavior_summary),
    }

    for filename, frame in tables.items():
        frame.to_csv(
            args.output_root / filename,
            index=False,
        )

    summary = build_summary_json(
        combined,
        unique_nonzero,
        correlation,
        survival_overall,
    )

    with (args.output_root / "cross_model_summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()

    print("=" * 76)

    print("CAA-STEERFRONTIER CROSS-MODEL ANALYSIS COMPLETE")

    print("=" * 76)

    print(f"Models:                       {len(MODEL_SPECS)}")

    print(f"Behaviors:                    {len(BEHAVIOR_ORDER)}")

    print(f"Model-behavior pairs:         {len(selected_layers)}")

    print(f"Frozen frontier claims:       {len(combined)}")

    print(f"Unique nonzero interventions: {len(unique_nonzero)}")

    print()

    print("Validation -> held-out test")

    print(
        correlation[
            [
                "scope",
                "n_unique_nonzero_interventions",
                "pearson_r",
                "spearman_r",
            ]
        ].to_string(index=False)
    )

    print()

    print("Overall frontier survival")

    print(
        survival_overall[
            [
                "epsilon",
                "nonzero_frontiers",
                "behavior_supported",
                "capability_holdout_pass",
                "joint_pass",
                "joint_pass_rate",
            ]
        ].to_string(index=False)
    )

    print()

    print(f"Artifacts -> {args.output_root}")


if __name__ == "__main__":
    main()
