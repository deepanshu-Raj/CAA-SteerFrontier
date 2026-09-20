"""Analyze off-target-KL-matched random-direction specificity controls.

Only successfully KL-matched controls contribute to matched-effect summaries;
match quality is reported separately from target-behavior effects.
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

INPUT_RELATIVE_PATH = (
    Path("controls") / "off_target_kl_matched" / "kl_matched_results.csv"
)

OUTPUT_ROOT = Path("results") / "cross_model" / "controls" / "off_target_kl_matched"

POINT_KEY = [
    "model_slug",
    "behavior",
    "layer",
    "direction",
    "alpha_star",
    "eta_star",
]


# =============================================================================
# Helpers
# =============================================================================


def as_bool(
    series: pd.Series,
):
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
        .astype(bool)
    )


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


def load_results():
    frames = []

    for slug in MODEL_ORDER:
        path = Path("results") / slug / INPUT_RELATIVE_PATH

        if not path.is_file():
            raise FileNotFoundError(path)

        frame = pd.read_csv(path)

        if frame.empty:
            raise ValueError(f"{path} is empty.")

        frames.append(frame)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined["kl_match_success"] = as_bool(combined["kl_match_success"])

    combined["matched_direction_supported"] = as_bool(
        combined["matched_direction_supported"]
    )

    combined["caa_test_direction_supported"] = as_bool(
        combined["caa_test_direction_supported"]
    )

    return combined


# =============================================================================
# Match-quality diagnostics
# =============================================================================


def build_match_quality(
    frame: pd.DataFrame,
):
    rows = []

    for slug in MODEL_ORDER:
        group = frame[frame["model_slug"] == slug].copy()

        if group.empty:
            raise ValueError(f"No rows for {slug}.")

        successful = group[group["kl_match_success"]]

        rows.append(
            {
                "model_slug": slug,
                "model_label": str(group["model_label"].iloc[0]),
                "num_control_evaluations": int(len(group)),
                "num_successful_matches": int(len(successful)),
                "num_failed_matches": int(len(group) - len(successful)),
                "kl_match_success_rate": float(
                    group["kl_match_success"].astype(float).mean()
                ),
                "mean_relative_kl_error_all": float(group["kl_match_rel_error"].mean()),
                "median_relative_kl_error_all": float(
                    group["kl_match_rel_error"].median()
                ),
                "max_relative_kl_error_all": float(group["kl_match_rel_error"].max()),
                "mean_relative_kl_error_successful": float(
                    successful["kl_match_rel_error"].mean()
                ),
                "max_relative_kl_error_successful": float(
                    successful["kl_match_rel_error"].max()
                ),
            }
        )

    overall_successful = frame[frame["kl_match_success"]]

    rows.append(
        {
            "model_slug": "ALL",
            "model_label": "All models",
            "num_control_evaluations": int(len(frame)),
            "num_successful_matches": int(len(overall_successful)),
            "num_failed_matches": int(len(frame) - len(overall_successful)),
            "kl_match_success_rate": float(
                frame["kl_match_success"].astype(float).mean()
            ),
            "mean_relative_kl_error_all": float(frame["kl_match_rel_error"].mean()),
            "median_relative_kl_error_all": float(frame["kl_match_rel_error"].median()),
            "max_relative_kl_error_all": float(frame["kl_match_rel_error"].max()),
            "mean_relative_kl_error_successful": float(
                overall_successful["kl_match_rel_error"].mean()
            ),
            "max_relative_kl_error_successful": float(
                overall_successful["kl_match_rel_error"].max()
            ),
        }
    )

    return pd.DataFrame(rows)


# =============================================================================
# Intervention-point analysis
# =============================================================================


def build_point_summary(
    frame: pd.DataFrame,
):
    rows = []

    grouped = frame.groupby(
        POINT_KEY,
        sort=False,
        dropna=False,
    )

    for key, group in grouped:
        successful = group[group["kl_match_success"]].copy()

        if successful.empty:
            raise RuntimeError(
                f"An intervention point has zero successful KL matches: {key}"
            )

        caa_gains = group["caa_test_directional_gain"].astype(float).unique()

        if len(caa_gains) != 1:
            raise RuntimeError(
                f"CAA directional gain is not constant within point {key}."
            )

        caa_gain = float(caa_gains[0])

        random_gains = successful["matched_directional_gain"].astype(float).to_numpy()

        random_mean = float(random_gains.mean())

        random_median = float(np.median(random_gains))

        q025 = float(
            np.quantile(
                random_gains,
                0.025,
            )
        )

        q975 = float(
            np.quantile(
                random_gains,
                0.975,
            )
        )

        num_ge_caa = int(np.sum(random_gains >= caa_gain))

        empirical_p = float((1 + num_ge_caa) / (len(random_gains) + 1))

        percentile = float(np.mean(random_gains < caa_gain))

        row = {
            **dict(
                zip(
                    POINT_KEY,
                    key,
                )
            ),
            "model_label": str(group["model_label"].iloc[0]),
            "frontier_epsilons": str(group["frontier_epsilons"].iloc[0]),
            "num_random_controls": int(len(group)),
            "num_successful_matches": int(len(successful)),
            "num_failed_matches": int(len(group) - len(successful)),
            "kl_match_success_rate": float(
                group["kl_match_success"].astype(float).mean()
            ),
            "caa_test_directional_gain": (caa_gain),
            "caa_test_direction_supported": bool(
                group["caa_test_direction_supported"].iloc[0]
            ),
            "caa_offtarget_kl_mean": float(group["caa_offtarget_kl_mean"].iloc[0]),
            "matched_random_gain_mean": (random_mean),
            "matched_random_gain_std": float(
                np.std(
                    random_gains,
                    ddof=1,
                )
                if len(random_gains) > 1
                else 0.0
            ),
            "matched_random_gain_q025": (q025),
            "matched_random_gain_median": (random_median),
            "matched_random_gain_q975": (q975),
            "caa_minus_matched_mean": float(caa_gain - random_mean),
            "caa_gain_matched_percentile": (percentile),
            "empirical_p_matched_ge_caa": (empirical_p),
            "caa_exceeds_all_successful_random": bool(
                caa_gain > float(random_gains.max())
            ),
            "matched_direction_support_rate": float(
                successful["matched_direction_supported"].astype(float).mean()
            ),
            "achieved_offtarget_kl_mean": float(
                successful["matched_offtarget_kl_mean"].mean()
            ),
            "mean_abs_kl_error_successful": float(
                successful["kl_match_abs_error"].mean()
            ),
            "mean_relative_kl_error_successful": float(
                successful["kl_match_rel_error"].mean()
            ),
            "max_relative_kl_error_successful": float(
                successful["kl_match_rel_error"].max()
            ),
            "mean_abs_beta_star_successful": float(
                successful["beta_star"].abs().mean()
            ),
            "mean_abs_cosine_with_caa": float(
                successful["cosine_with_caa"].abs().mean()
            ),
            "max_abs_cosine_with_caa": float(successful["cosine_with_caa"].abs().max()),
        }

        rows.append(row)

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        [
            "model_slug",
            "behavior",
            "direction",
            "alpha_star",
        ]
    ).reset_index(drop=True)

    return summary


# =============================================================================
# Model / overall summaries
# =============================================================================


def summarize_points(
    points: pd.DataFrame,
    *,
    model_slug: str,
    model_label: str,
):
    gaps = points["caa_minus_matched_mean"].astype(float)

    return {
        "model_slug": (model_slug),
        "model_label": (model_label),
        "num_intervention_points": int(len(points)),
        "mean_caa_directional_gain": float(points["caa_test_directional_gain"].mean()),
        "mean_kl_matched_random_gain": float(points["matched_random_gain_mean"].mean()),
        "mean_caa_minus_matched_gap": float(gaps.mean()),
        "median_caa_minus_matched_gap": float(gaps.median()),
        "min_caa_minus_matched_gap": float(gaps.min()),
        "max_caa_minus_matched_gap": float(gaps.max()),
        "num_positive_gaps": int((gaps > 0).sum()),
        "positive_gap_rate": float((gaps > 0).astype(float).mean()),
        "num_caa_exceeds_all_successful_random": int(
            points["caa_exceeds_all_successful_random"].sum()
        ),
        "rate_caa_exceeds_all_successful_random": float(
            points["caa_exceeds_all_successful_random"].astype(float).mean()
        ),
        "mean_successful_matches_per_point": float(
            points["num_successful_matches"].mean()
        ),
        "mean_point_kl_match_success_rate": float(
            points["kl_match_success_rate"].mean()
        ),
        "mean_relative_kl_error_successful": float(
            points["mean_relative_kl_error_successful"].mean()
        ),
        "mean_matched_direction_support_rate": float(
            points["matched_direction_support_rate"].mean()
        ),
    }


def build_model_summary(
    points: pd.DataFrame,
):
    rows = []

    for slug in MODEL_ORDER:
        group = points[points["model_slug"] == slug]

        if group.empty:
            raise ValueError(f"No intervention points for {slug}.")

        rows.append(
            summarize_points(
                group,
                model_slug=(slug),
                model_label=str(group["model_label"].iloc[0]),
            )
        )

    return pd.DataFrame(rows)


def build_overall_summary(
    points: pd.DataFrame,
    raw: pd.DataFrame,
):
    summary = summarize_points(
        points,
        model_slug="ALL",
        model_label="All models",
    )

    summary.update(
        {
            "num_raw_control_evaluations": int(len(raw)),
            "num_successful_kl_matches": int(raw["kl_match_success"].sum()),
            "num_failed_kl_matches": int((~raw["kl_match_success"]).sum()),
            "raw_kl_match_success_rate": float(
                raw["kl_match_success"].astype(float).mean()
            ),
        }
    )

    return pd.DataFrame([summary])


# =============================================================================
# Main
# =============================================================================


def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = load_results()

    expected_rows = {
        "llama2-7b-chat": 460,
        "llama3.1-8b-instruct": 640,
        "mistral-7b-instruct-v0.3": 600,
        "qwen2.5-7b-instruct": 480,
    }

    for slug, expected in expected_rows.items():
        observed = int((raw["model_slug"] == slug).sum())

        if observed != expected:
            raise RuntimeError(f"{slug}: expected {expected} rows, found {observed}.")

    if len(raw) != 2180:
        raise RuntimeError(
            f"Expected 2180 total control evaluations, found {len(raw)}."
        )

    successful = raw[raw["kl_match_success"]].copy()

    failed = raw[~raw["kl_match_success"]].copy()

    match_quality = build_match_quality(raw)

    points = build_point_summary(raw)

    if len(points) != 109:
        raise RuntimeError(
            f"Expected 109 unique nonzero intervention points, found {len(points)}."
        )

    model_summary = build_model_summary(points)

    overall_summary = build_overall_summary(
        points,
        raw,
    )

    # ----------------------------------------------------------------
    # Persist
    # ----------------------------------------------------------------

    write_csv(
        raw,
        OUTPUT_ROOT / "combined_results.csv",
    )

    write_csv(
        successful,
        OUTPUT_ROOT / "successful_matches.csv",
    )

    write_csv(
        failed,
        OUTPUT_ROOT / "failed_matches.csv",
    )

    write_csv(
        match_quality,
        OUTPUT_ROOT / "match_quality_by_model.csv",
    )

    write_csv(
        points,
        OUTPUT_ROOT / "intervention_point_summary.csv",
    )

    write_csv(
        model_summary,
        OUTPUT_ROOT / "model_summary.csv",
    )

    write_csv(
        overall_summary,
        OUTPUT_ROOT / "overall_summary.csv",
    )

    # ----------------------------------------------------------------
    # Console report
    # ----------------------------------------------------------------

    print("=" * 88)
    print("OFF-TARGET-KL MATCHED CONTROL")
    print("=" * 88)

    print()
    print("MATCH QUALITY")
    print(
        match_quality[
            [
                "model_slug",
                "num_control_evaluations",
                "num_successful_matches",
                "kl_match_success_rate",
                "mean_relative_kl_error_successful",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 88)
    print("SUCCESSFUL-MATCH POINT SUMMARY")
    print("=" * 88)

    print(
        model_summary[
            [
                "model_slug",
                "num_intervention_points",
                "mean_caa_directional_gain",
                "mean_kl_matched_random_gain",
                "mean_caa_minus_matched_gap",
                "num_positive_gaps",
                "num_caa_exceeds_all_successful_random",
                "mean_successful_matches_per_point",
                "mean_relative_kl_error_successful",
            ]
        ].to_string(index=False)
    )

    overall = overall_summary.iloc[0]

    print()
    print("=" * 88)
    print("OVERALL")
    print("=" * 88)

    print(f"Raw controls:             {int(overall['num_raw_control_evaluations'])}")

    print(
        f"Successful KL matches:    "
        f"{int(overall['num_successful_kl_matches'])}/"
        f"{int(overall['num_raw_control_evaluations'])} "
        f"("
        f"{overall['raw_kl_match_success_rate']:.3%}"
        f")"
    )

    print(f"Intervention points:      {int(overall['num_intervention_points'])}")

    print(f"Mean CAA gain:            {overall['mean_caa_directional_gain']:.6f}")

    print(f"Mean KL-matched gain:     {overall['mean_kl_matched_random_gain']:.6f}")

    print(f"Mean CAA - matched gap:   {overall['mean_caa_minus_matched_gap']:+.6f}")

    print(
        f"Positive point gaps:      "
        f"{int(overall['num_positive_gaps'])}/"
        f"{int(overall['num_intervention_points'])} "
        f"("
        f"{overall['positive_gap_rate']:.3%}"
        f")"
    )

    print(
        f"CAA > all matched draws:  "
        f"{int(overall['num_caa_exceeds_all_successful_random'])}/"
        f"{int(overall['num_intervention_points'])} "
        f"("
        f"{overall['rate_caa_exceeds_all_successful_random']:.3%}"
        f")"
    )

    print(
        f"Mean successful-match "
        f"KL rel. error:            "
        f"{overall['mean_relative_kl_error_successful']:.4f}"
    )

    print()
    print(f"Saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
