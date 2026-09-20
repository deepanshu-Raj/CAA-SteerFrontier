"""Analyze WikiText-2 perplexity as a secondary capability measure.

The script summarizes frozen intervention results and relates perturbation
magnitude to NLL degradation without re-selecting operating points.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

MODEL_ORDER = [
    "llama2-7b-chat",
    "llama3.1-8b-instruct",
    "mistral-7b-instruct-v0.3",
    "qwen2.5-7b-instruct",
]

EXPECTED_POINTS_PER_MODEL = 14
EXPECTED_TOTAL_POINTS = 56

INPUT_RELATIVE_PATH = (
    Path("secondary_capability") / "perplexity" / "perplexity_summary.csv"
)

OUTPUT_ROOT = Path("results") / "cross_model" / "secondary_capability" / "perplexity"


# =============================================================================
# Helpers
# =============================================================================


def write_csv(
    frame: pd.DataFrame,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(path.suffix + ".tmp")

    frame.to_csv(
        temporary,
        index=False,
    )

    temporary.replace(path)


def pearson_corr(
    x: np.ndarray,
    y: np.ndarray,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    if len(x) < 2:
        return float("nan")

    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")

    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )


def spearman_corr(
    x: np.ndarray,
    y: np.ndarray,
):
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)

    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)

    return pearson_corr(
        x_rank,
        y_rank,
    )


# =============================================================================
# Loading / validation
# =============================================================================


def load_results():
    frames = []

    for slug in MODEL_ORDER:
        path = Path("results") / slug / INPUT_RELATIVE_PATH

        if not path.is_file():
            raise FileNotFoundError(path)

        frame = pd.read_csv(path)

        if len(frame) != EXPECTED_POINTS_PER_MODEL:
            raise RuntimeError(
                f"{slug}: expected "
                f"{EXPECTED_POINTS_PER_MODEL} "
                f"points, found "
                f"{len(frame)}."
            )

        frame = frame.copy()

        if not (frame["model_slug"] == slug).all():
            raise RuntimeError(f"model_slug mismatch in {path}.")

        frames.append(frame)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    if len(combined) != EXPECTED_TOTAL_POINTS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_POINTS} total points, found {len(combined)}."
        )

    duplicate = combined.duplicated(
        [
            "model_slug",
            "behavior",
            "direction",
        ]
    )

    if duplicate.any():
        bad = combined.loc[
            duplicate,
            [
                "model_slug",
                "behavior",
                "direction",
            ],
        ]

        raise RuntimeError(f"Duplicate operating points:\n{bad}")

    return combined


# =============================================================================
# Derived columns
# =============================================================================


def add_derived_columns(
    frame: pd.DataFrame,
):
    result = frame.copy()

    result["abs_eta"] = result["eta_star"].astype(float).abs()

    result["is_nonzero_intervention"] = result["alpha_star"].astype(float).abs() > 1e-12

    result["ci_supported_nll_increase"] = result["delta_nll_ci_low"].astype(float) > 0.0

    result["ci_supported_nll_decrease"] = (
        result["delta_nll_ci_high"].astype(float) < 0.0
    )

    result["ci_overlaps_zero"] = ~(
        result["ci_supported_nll_increase"] | result["ci_supported_nll_decrease"]
    )

    result["nll_effect_class"] = np.select(
        [
            result["ci_supported_nll_increase"],
            result["ci_supported_nll_decrease"],
        ],
        [
            "increase",
            "decrease",
        ],
        default="overlaps_zero",
    )

    return result


# =============================================================================
# Summaries
# =============================================================================


def summarize_group(
    frame: pd.DataFrame,
    *,
    model_slug: str,
    model_label: str,
):
    nonzero = frame[frame["is_nonzero_intervention"]]

    baseline_nll_values = frame["baseline_mean_nll"].astype(float).to_numpy()

    baseline_ppl_values = frame["baseline_perplexity"].astype(float).to_numpy()

    if not np.allclose(
        baseline_nll_values,
        baseline_nll_values[0],
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError(f"{model_slug}: baseline NLL is not constant.")

    if not np.allclose(
        baseline_ppl_values,
        baseline_ppl_values[0],
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError(f"{model_slug}: baseline PPL is not constant.")

    return {
        "model_slug": model_slug,
        "model_label": model_label,
        "num_points": int(len(frame)),
        "num_nonzero_interventions": int(len(nonzero)),
        "baseline_mean_nll": float(baseline_nll_values[0]),
        "baseline_perplexity": float(baseline_ppl_values[0]),
        "mean_delta_nll": float(frame["delta_nll"].mean()),
        "median_delta_nll": float(frame["delta_nll"].median()),
        "min_delta_nll": float(frame["delta_nll"].min()),
        "max_delta_nll": float(frame["delta_nll"].max()),
        "mean_perplexity_ratio": float(frame["perplexity_ratio"].mean()),
        "mean_perplexity_percent_change": float(
            frame["perplexity_percent_change"].mean()
        ),
        "num_ci_supported_increases": int(frame["ci_supported_nll_increase"].sum()),
        "num_ci_supported_decreases": int(frame["ci_supported_nll_decrease"].sum()),
        "num_ci_overlaps_zero": int(frame["ci_overlaps_zero"].sum()),
        "nonzero_mean_delta_nll": float(nonzero["delta_nll"].mean()),
        "nonzero_mean_perplexity_percent_change": float(
            nonzero["perplexity_percent_change"].mean()
        ),
        "nonzero_num_ci_supported_increases": int(
            nonzero["ci_supported_nll_increase"].sum()
        ),
        "nonzero_num_ci_supported_decreases": int(
            nonzero["ci_supported_nll_decrease"].sum()
        ),
        "nonzero_num_ci_overlaps_zero": int(nonzero["ci_overlaps_zero"].sum()),
    }


def build_model_summary(
    frame: pd.DataFrame,
):
    rows = []

    for slug in MODEL_ORDER:
        group = frame[frame["model_slug"] == slug].copy()

        if group.empty:
            raise RuntimeError(f"No rows for {slug}.")

        rows.append(
            summarize_group(
                group,
                model_slug=slug,
                model_label=str(group["model_label"].iloc[0]),
            )
        )

    return pd.DataFrame(rows)


def build_overall_summary(
    frame: pd.DataFrame,
):
    nonzero = frame[frame["is_nonzero_intervention"]]

    return pd.DataFrame(
        [
            {
                "num_models": int(frame["model_slug"].nunique()),
                "num_points": int(len(frame)),
                "num_nonzero_interventions": int(len(nonzero)),
                "mean_delta_nll": float(frame["delta_nll"].mean()),
                "median_delta_nll": float(frame["delta_nll"].median()),
                "min_delta_nll": float(frame["delta_nll"].min()),
                "max_delta_nll": float(frame["delta_nll"].max()),
                "mean_perplexity_ratio": float(frame["perplexity_ratio"].mean()),
                "mean_perplexity_percent_change": float(
                    frame["perplexity_percent_change"].mean()
                ),
                "num_ci_supported_increases": int(
                    frame["ci_supported_nll_increase"].sum()
                ),
                "num_ci_supported_decreases": int(
                    frame["ci_supported_nll_decrease"].sum()
                ),
                "num_ci_overlaps_zero": int(frame["ci_overlaps_zero"].sum()),
                "nonzero_mean_delta_nll": float(nonzero["delta_nll"].mean()),
                "nonzero_median_delta_nll": float(nonzero["delta_nll"].median()),
                "nonzero_mean_perplexity_ratio": float(
                    nonzero["perplexity_ratio"].mean()
                ),
                "nonzero_mean_perplexity_percent_change": float(
                    nonzero["perplexity_percent_change"].mean()
                ),
                "nonzero_num_ci_supported_increases": int(
                    nonzero["ci_supported_nll_increase"].sum()
                ),
                "nonzero_num_ci_supported_decreases": int(
                    nonzero["ci_supported_nll_decrease"].sum()
                ),
                "nonzero_num_ci_overlaps_zero": int(nonzero["ci_overlaps_zero"].sum()),
            }
        ]
    )


# =============================================================================
# Eta / degradation association
# =============================================================================


def association_row(
    frame: pd.DataFrame,
    *,
    scope: str,
    model_slug: str,
    model_label: str,
):
    x = frame["abs_eta"].astype(float).to_numpy()

    y = frame["delta_nll"].astype(float).to_numpy()

    return {
        "scope": scope,
        "model_slug": model_slug,
        "model_label": model_label,
        "num_points": int(len(frame)),
        "pearson_abs_eta_vs_delta_nll": (
            pearson_corr(
                x,
                y,
            )
        ),
        "spearman_abs_eta_vs_delta_nll": (
            spearman_corr(
                x,
                y,
            )
        ),
    }


def build_association_summary(
    frame: pd.DataFrame,
):
    rows = []

    # ------------------------------------------------------------
    # Overall: all 56 points
    # ------------------------------------------------------------

    rows.append(
        association_row(
            frame,
            scope="all_points",
            model_slug="ALL",
            model_label="All models",
        )
    )

    # ------------------------------------------------------------
    # Overall: nonzero interventions only
    # ------------------------------------------------------------

    nonzero = frame[frame["is_nonzero_intervention"]]

    rows.append(
        association_row(
            nonzero,
            scope="nonzero_only",
            model_slug="ALL",
            model_label="All models",
        )
    )

    # ------------------------------------------------------------
    # Per model
    # ------------------------------------------------------------

    for slug in MODEL_ORDER:
        group = frame[frame["model_slug"] == slug]

        label = str(group["model_label"].iloc[0])

        rows.append(
            association_row(
                group,
                scope="all_points",
                model_slug=slug,
                model_label=label,
            )
        )

        group_nonzero = group[group["is_nonzero_intervention"]]

        rows.append(
            association_row(
                group_nonzero,
                scope="nonzero_only",
                model_slug=slug,
                model_label=label,
            )
        )

    return pd.DataFrame(rows)


# =============================================================================
# Behavior-level descriptive summary
# =============================================================================


def build_behavior_summary(
    frame: pd.DataFrame,
):
    rows = []

    for behavior, group in frame.groupby(
        "behavior",
        sort=True,
    ):
        nonzero = group[group["is_nonzero_intervention"]]

        rows.append(
            {
                "behavior": behavior,
                "num_points": int(len(group)),
                "num_nonzero_interventions": int(len(nonzero)),
                "mean_abs_eta": float(group["abs_eta"].mean()),
                "mean_delta_nll": float(group["delta_nll"].mean()),
                "median_delta_nll": float(group["delta_nll"].median()),
                "mean_perplexity_percent_change": float(
                    group["perplexity_percent_change"].mean()
                ),
                "num_ci_supported_increases": int(
                    group["ci_supported_nll_increase"].sum()
                ),
                "num_ci_supported_decreases": int(
                    group["ci_supported_nll_decrease"].sum()
                ),
                "num_ci_overlaps_zero": int(group["ci_overlaps_zero"].sum()),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================


def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = load_results()

    combined = add_derived_columns(raw)

    nonzero = combined[combined["is_nonzero_intervention"]].copy()

    if len(nonzero) != 55:
        raise RuntimeError(f"Expected 55 nonzero interventions, found {len(nonzero)}.")

    model_summary = build_model_summary(combined)

    overall_summary = build_overall_summary(combined)

    association_summary = build_association_summary(combined)

    behavior_summary = build_behavior_summary(combined)

    # ----------------------------------------------------------------
    # Persist
    # ----------------------------------------------------------------

    write_csv(
        combined,
        OUTPUT_ROOT / "combined_results.csv",
    )

    write_csv(
        nonzero,
        OUTPUT_ROOT / "nonzero_interventions.csv",
    )

    write_csv(
        model_summary,
        OUTPUT_ROOT / "model_summary.csv",
    )

    write_csv(
        overall_summary,
        OUTPUT_ROOT / "overall_summary.csv",
    )

    write_csv(
        association_summary,
        OUTPUT_ROOT / "eta_nll_association.csv",
    )

    write_csv(
        behavior_summary,
        OUTPUT_ROOT / "behavior_summary.csv",
    )

    # ----------------------------------------------------------------
    # Console report
    # ----------------------------------------------------------------

    print("=" * 96)
    print("WIKITEXT-2 PERPLEXITY SECONDARY CAPABILITY")
    print("=" * 96)

    print()
    print("PER-MODEL SUMMARY")
    print(
        model_summary[
            [
                "model_slug",
                "num_points",
                "num_nonzero_interventions",
                "baseline_mean_nll",
                "baseline_perplexity",
                "mean_delta_nll",
                "mean_perplexity_percent_change",
                "num_ci_supported_increases",
                "num_ci_supported_decreases",
                "num_ci_overlaps_zero",
            ]
        ].to_string(index=False)
    )

    overall = overall_summary.iloc[0]

    print()
    print("=" * 96)
    print("OVERALL")
    print("=" * 96)

    print(f"All points:                 {int(overall['num_points'])}")

    print(f"Nonzero interventions:      {int(overall['num_nonzero_interventions'])}")

    print(f"Mean delta NLL:             {overall['mean_delta_nll']:+.6f}")

    print(f"Median delta NLL:           {overall['median_delta_nll']:+.6f}")

    print(
        f"Mean PPL change:            {overall['mean_perplexity_percent_change']:+.3f}%"
    )

    print(
        f"Supported NLL increase:     "
        f"{int(overall['num_ci_supported_increases'])}/"
        f"{int(overall['num_points'])}"
    )

    print(
        f"Supported NLL decrease:     "
        f"{int(overall['num_ci_supported_decreases'])}/"
        f"{int(overall['num_points'])}"
    )

    print()
    print("NONZERO INTERVENTIONS")

    print(f"Mean delta NLL:             {overall['nonzero_mean_delta_nll']:+.6f}")

    print(f"Median delta NLL:           {overall['nonzero_median_delta_nll']:+.6f}")

    print(
        f"Mean PPL change:            "
        f"{overall['nonzero_mean_perplexity_percent_change']:+.3f}%"
    )

    print(
        f"Supported NLL increase:     "
        f"{int(overall['nonzero_num_ci_supported_increases'])}/"
        f"{int(overall['num_nonzero_interventions'])}"
    )

    print(
        f"Supported NLL decrease:     "
        f"{int(overall['nonzero_num_ci_supported_decreases'])}/"
        f"{int(overall['num_nonzero_interventions'])}"
    )

    print()
    print("=" * 96)
    print("|ETA| vs DELTA NLL")
    print("=" * 96)

    print(
        association_summary[
            [
                "scope",
                "model_slug",
                "num_points",
                "pearson_abs_eta_vs_delta_nll",
                "spearman_abs_eta_vs_delta_nll",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 96)
    print("BEHAVIOR SUMMARY")
    print("=" * 96)

    print(
        behavior_summary[
            [
                "behavior",
                "num_points",
                "mean_abs_eta",
                "mean_delta_nll",
                "mean_perplexity_percent_change",
                "num_ci_supported_increases",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
