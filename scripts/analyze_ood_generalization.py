"""Aggregate out-of-distribution steering and generalization results.

The analysis covers sycophancy cross-domain transfer, steering-vector geometry,
and hallucination transfer to TruthfulQA using frozen evaluation artifacts.
"""

from __future__ import annotations

import json
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

EXPECTED_ROWS_PER_MODEL = 10
EXPECTED_TOTAL_ROWS = 40

OUTPUT_ROOT = Path("results") / "cross_model" / "ood_generalization"


def write_csv(
    frame: pd.DataFrame,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        path,
        index=False,
    )


def write_json(
    payload,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
        )
        f.write("\n")


def load_all_results():
    frames = []

    for model_slug in MODEL_ORDER:
        path = Path("results") / model_slug / "ood_generalization" / "ood_results.csv"

        if not path.is_file():
            raise FileNotFoundError(path)

        frame = pd.read_csv(path)

        if len(frame) != EXPECTED_ROWS_PER_MODEL:
            raise RuntimeError(
                f"{model_slug}: expected "
                f"{EXPECTED_ROWS_PER_MODEL} rows, "
                f"found {len(frame)}."
            )

        required = {
            "experiment",
            "model_slug",
            "behavior",
            "source_domain",
            "test_domain",
            "transfer_type",
            "direction",
            "alpha_star",
            "eta_star",
            "baseline_behavior_score",
            "behavior_score",
            "behavior_delta",
            "behavior_delta_ci_low",
            "behavior_delta_ci_high",
            "directional_gain",
            "directional_gain_ci_low",
            "directional_gain_ci_high",
            "direction_supported",
            "num_examples",
        }

        missing = required - set(frame.columns)

        if missing:
            raise RuntimeError(f"{path} missing columns: {sorted(missing)}")

        frames.append(frame)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    if len(combined) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_ROWS} rows total, found {len(combined)}."
        )

    duplicate_key = [
        "model_slug",
        "experiment",
        "source_domain",
        "test_domain",
        "direction",
    ]

    if combined[duplicate_key].duplicated().any():
        raise RuntimeError("Duplicate OOD result keys found.")

    return combined


def load_vector_geometry():
    rows = []

    for model_slug in MODEL_ORDER:
        path = Path("vectors") / model_slug / "ood" / "sycophancy" / "manifest.json"

        if not path.is_file():
            raise FileNotFoundError(path)

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            payload = json.load(f)

        domains = payload["domains"]

        geometry = payload["nlp_vs_political_geometry"]

        rows.append(
            {
                "model_slug": (model_slug),
                "frozen_layer": int(payload["frozen_layer"]),
                "nlp_vs_political_cosine": float(geometry["cosine_at_frozen_layer"]),
                "nlp_vs_canonical_cosine": float(
                    domains["nlp"]["canonical_cosine_at_frozen_layer"]
                ),
                "political_vs_canonical_cosine": float(
                    domains["political"]["canonical_cosine_at_frozen_layer"]
                ),
            }
        )

    return pd.DataFrame(rows)


def build_sycophancy_pairs(
    combined: pd.DataFrame,
):
    syc = combined[combined["experiment"] == "sycophancy_cross_domain"].copy()

    if len(syc) != 32:
        raise RuntimeError(
            "Expected 32 sycophancy rows "
            "(4 models × 2 source domains × "
            "2 test domains × 2 directions)."
        )

    rows = []

    for (
        model_slug,
        source_domain,
        direction,
    ), group in syc.groupby(
        [
            "model_slug",
            "source_domain",
            "direction",
        ],
        sort=False,
    ):
        if len(group) != 2:
            raise RuntimeError(
                f"{model_slug} / "
                f"{source_domain} / "
                f"{direction}: expected "
                "one ID and one OOD row."
            )

        id_rows = group[group["transfer_type"] == "id"]

        ood_rows = group[group["transfer_type"] == "ood"]

        if len(id_rows) != 1 or len(ood_rows) != 1:
            raise RuntimeError(
                f"{model_slug} / {source_domain} / {direction}: ID/OOD row mismatch."
            )

        id_row = id_rows.iloc[0]

        ood_row = ood_rows.iloc[0]

        id_gain = float(id_row["directional_gain"])

        ood_gain = float(ood_row["directional_gain"])

        id_supported = bool(float(id_row["directional_gain_ci_low"]) > 0.0)

        ood_supported = bool(float(ood_row["directional_gain_ci_low"]) > 0.0)

        # Ratio is intentionally secondary.
        #
        # Only compute it when the in-domain
        # effect is itself CI-supported positive.
        if id_supported and id_gain > 0.0:
            retention_ratio = ood_gain / id_gain
        else:
            retention_ratio = np.nan

        rows.append(
            {
                "model_slug": (model_slug),
                "model_label": (id_row["model_label"]),
                "source_domain": (source_domain),
                "id_test_domain": (id_row["test_domain"]),
                "ood_test_domain": (ood_row["test_domain"]),
                "direction": (direction),
                "layer": int(id_row["layer"]),
                "alpha_star": float(id_row["alpha_star"]),
                "eta_star": float(id_row["eta_star"]),
                "id_baseline_score": float(id_row["baseline_behavior_score"]),
                "id_behavior_score": float(id_row["behavior_score"]),
                "id_directional_gain": (id_gain),
                "id_ci_low": float(id_row["directional_gain_ci_low"]),
                "id_ci_high": float(id_row["directional_gain_ci_high"]),
                "id_supported": (id_supported),
                "ood_baseline_score": float(ood_row["baseline_behavior_score"]),
                "ood_behavior_score": float(ood_row["behavior_score"]),
                "ood_directional_gain": (ood_gain),
                "ood_ci_low": float(ood_row["directional_gain_ci_low"]),
                "ood_ci_high": float(ood_row["directional_gain_ci_high"]),
                "ood_supported": (ood_supported),
                "ood_minus_id_gain": float(ood_gain - id_gain),
                "retention_ratio": (retention_ratio),
                "num_id_examples": int(id_row["num_examples"]),
                "num_ood_examples": int(ood_row["num_examples"]),
            }
        )

    result = pd.DataFrame(rows)

    if len(result) != 16:
        raise RuntimeError(
            f"Expected 16 paired sycophancy transfers, found {len(result)}."
        )

    return result.sort_values(
        [
            "model_slug",
            "source_domain",
            "direction",
        ]
    ).reset_index(drop=True)


def build_sycophancy_model_summary(
    pairs: pd.DataFrame,
):
    rows = []

    for model_slug in MODEL_ORDER:
        frame = pairs[pairs["model_slug"] == model_slug]

        valid_retention = frame["retention_ratio"].dropna()

        rows.append(
            {
                "model_slug": (model_slug),
                "model_label": (frame["model_label"].iloc[0]),
                "num_transfer_pairs": int(len(frame)),
                "mean_id_gain": float(frame["id_directional_gain"].mean()),
                "median_id_gain": float(frame["id_directional_gain"].median()),
                "mean_ood_gain": float(frame["ood_directional_gain"].mean()),
                "median_ood_gain": float(frame["ood_directional_gain"].median()),
                "mean_ood_minus_id": float(frame["ood_minus_id_gain"].mean()),
                "positive_id_gain_count": int((frame["id_directional_gain"] > 0).sum()),
                "supported_id_gain_count": int(frame["id_supported"].sum()),
                "positive_ood_gain_count": int(
                    (frame["ood_directional_gain"] > 0).sum()
                ),
                "supported_ood_gain_count": int(frame["ood_supported"].sum()),
                "retention_ratio_count": int(len(valid_retention)),
                "mean_retention_ratio": (
                    float(valid_retention.mean()) if len(valid_retention) else np.nan
                ),
                "median_retention_ratio": (
                    float(valid_retention.median()) if len(valid_retention) else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def build_sycophancy_overall_summary(
    pairs: pd.DataFrame,
):
    valid_retention = pairs["retention_ratio"].dropna()

    return pd.DataFrame(
        [
            {
                "num_models": int(pairs["model_slug"].nunique()),
                "num_transfer_pairs": int(len(pairs)),
                "mean_id_gain": float(pairs["id_directional_gain"].mean()),
                "median_id_gain": float(pairs["id_directional_gain"].median()),
                "mean_ood_gain": float(pairs["ood_directional_gain"].mean()),
                "median_ood_gain": float(pairs["ood_directional_gain"].median()),
                "mean_ood_minus_id": float(pairs["ood_minus_id_gain"].mean()),
                "median_ood_minus_id": float(pairs["ood_minus_id_gain"].median()),
                "positive_id_gain_count": int((pairs["id_directional_gain"] > 0).sum()),
                "supported_id_gain_count": int(pairs["id_supported"].sum()),
                "positive_ood_gain_count": int(
                    (pairs["ood_directional_gain"] > 0).sum()
                ),
                "supported_ood_gain_count": int(pairs["ood_supported"].sum()),
                "retention_ratio_count": int(len(valid_retention)),
                "mean_retention_ratio": (
                    float(valid_retention.mean()) if len(valid_retention) else np.nan
                ),
                "median_retention_ratio": (
                    float(valid_retention.median()) if len(valid_retention) else np.nan
                ),
            }
        ]
    )


def build_hallucination_summary(
    combined: pd.DataFrame,
):
    hall = combined[combined["experiment"] == "hallucination_truthfulqa"].copy()

    if len(hall) != 8:
        raise RuntimeError("Expected 8 TruthfulQA rows (4 models × 2 directions).")

    hall["positive_directional_gain"] = hall["directional_gain"] > 0

    hall["ci_supported_directional_gain"] = hall["directional_gain_ci_low"] > 0

    return hall.sort_values(
        [
            "model_slug",
            "direction",
        ]
    ).reset_index(drop=True)


def build_hallucination_model_summary(
    hall: pd.DataFrame,
):
    rows = []

    for model_slug in MODEL_ORDER:
        frame = hall[hall["model_slug"] == model_slug]

        if len(frame) != 2:
            raise RuntimeError(f"{model_slug}: expected two TruthfulQA directions.")

        rows.append(
            {
                "model_slug": (model_slug),
                "model_label": (frame["model_label"].iloc[0]),
                "baseline_behavior_score": float(
                    frame["baseline_behavior_score"].iloc[0]
                ),
                "mean_directional_gain": float(frame["directional_gain"].mean()),
                "min_directional_gain": float(frame["directional_gain"].min()),
                "max_directional_gain": float(frame["directional_gain"].max()),
                "positive_gain_count": int(frame["positive_directional_gain"].sum()),
                "supported_gain_count": int(
                    frame["ci_supported_directional_gain"].sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_hallucination_overall_summary(
    hall: pd.DataFrame,
):
    return pd.DataFrame(
        [
            {
                "num_models": int(hall["model_slug"].nunique()),
                "num_directional_conditions": int(len(hall)),
                "mean_directional_gain": float(hall["directional_gain"].mean()),
                "median_directional_gain": float(hall["directional_gain"].median()),
                "min_directional_gain": float(hall["directional_gain"].min()),
                "max_directional_gain": float(hall["directional_gain"].max()),
                "positive_gain_count": int((hall["directional_gain"] > 0).sum()),
                "supported_gain_count": int(
                    (hall["directional_gain_ci_low"] > 0).sum()
                ),
            }
        ]
    )


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 100)
    print("OOD / GENERALIZATION ANALYSIS")
    print("=" * 100)

    combined = load_all_results()

    syc_pairs = build_sycophancy_pairs(combined)

    syc_model = build_sycophancy_model_summary(syc_pairs)

    syc_overall = build_sycophancy_overall_summary(syc_pairs)

    hall = build_hallucination_summary(combined)

    hall_model = build_hallucination_model_summary(hall)

    hall_overall = build_hallucination_overall_summary(hall)

    geometry = load_vector_geometry()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        combined,
        OUTPUT_ROOT / "combined_results.csv",
    )

    write_csv(
        syc_pairs,
        OUTPUT_ROOT / "sycophancy_transfer_pairs.csv",
    )

    write_csv(
        syc_model,
        OUTPUT_ROOT / "sycophancy_model_summary.csv",
    )

    write_csv(
        syc_overall,
        OUTPUT_ROOT / "sycophancy_overall_summary.csv",
    )

    write_csv(
        hall,
        OUTPUT_ROOT / "hallucination_truthfulqa_results.csv",
    )

    write_csv(
        hall_model,
        OUTPUT_ROOT / "hallucination_model_summary.csv",
    )

    write_csv(
        hall_overall,
        OUTPUT_ROOT / "hallucination_overall_summary.csv",
    )

    write_csv(
        geometry,
        OUTPUT_ROOT / "sycophancy_vector_geometry.csv",
    )

    manifest = {
        "protocol": ("ood_generalization_analysis_v1"),
        "models": (MODEL_ORDER),
        "num_raw_rows": int(len(combined)),
        "num_sycophancy_transfer_pairs": int(len(syc_pairs)),
        "num_truthfulqa_conditions": int(len(hall)),
        "primary_sycophancy_metric": ("Raw cross-domain directional gain."),
        "retention_policy": (
            "OOD / ID retention ratio is secondary "
            "and is computed only when the "
            "corresponding in-domain directional "
            "gain has a 95% CI lower bound > 0."
        ),
        "truthfulqa_metric": (
            "Directional gain using the frozen canonical hallucination intervention."
        ),
        "no_posthoc_selection": True,
    }

    write_json(
        manifest,
        OUTPUT_ROOT / "manifest.json",
    )

    # ================================================================
    # Console report
    # ================================================================

    print()
    print("=" * 100)
    print("SYCOPHANCY: PAIRED ID -> OOD TRANSFER")
    print("=" * 100)

    display = syc_pairs[
        [
            "model_slug",
            "source_domain",
            "direction",
            "eta_star",
            "id_directional_gain",
            "id_ci_low",
            "id_ci_high",
            "ood_directional_gain",
            "ood_ci_low",
            "ood_ci_high",
            "id_supported",
            "ood_supported",
            "retention_ratio",
        ]
    ].copy()

    print(display.to_string(index=False))

    print()
    print("=" * 100)
    print("SYCOPHANCY: MODEL SUMMARY")
    print("=" * 100)

    print(
        syc_model[
            [
                "model_slug",
                "mean_id_gain",
                "mean_ood_gain",
                "mean_ood_minus_id",
                "supported_id_gain_count",
                "supported_ood_gain_count",
                "retention_ratio_count",
                "median_retention_ratio",
            ]
        ].to_string(index=False)
    )

    overall = syc_overall.iloc[0]

    print()
    print("Sycophancy overall:")

    print(f"  transfer pairs:            {int(overall['num_transfer_pairs'])}")

    print(f"  mean ID gain:              {overall['mean_id_gain']:+.6f}")

    print(f"  mean OOD gain:             {overall['mean_ood_gain']:+.6f}")

    print(f"  median OOD gain:           {overall['median_ood_gain']:+.6f}")

    print(f"  mean OOD - ID:             {overall['mean_ood_minus_id']:+.6f}")

    print(
        f"  positive OOD gains:        "
        f"{int(overall['positive_ood_gain_count'])}/"
        f"{int(overall['num_transfer_pairs'])}"
    )

    print(
        f"  CI-supported OOD gains:    "
        f"{int(overall['supported_ood_gain_count'])}/"
        f"{int(overall['num_transfer_pairs'])}"
    )

    print(
        f"  valid retention ratios:    "
        f"{int(overall['retention_ratio_count'])}/"
        f"{int(overall['num_transfer_pairs'])}"
    )

    if not np.isnan(overall["median_retention_ratio"]):
        print(f"  median retention ratio:    {overall['median_retention_ratio']:.6f}")

    print()
    print("=" * 100)
    print("SYCOPHANCY VECTOR GEOMETRY")
    print("=" * 100)

    print(geometry.to_string(index=False))

    print()
    print("=" * 100)
    print("HALLUCINATION -> TRUTHFULQA")
    print("=" * 100)

    print(
        hall[
            [
                "model_slug",
                "direction",
                "eta_star",
                "baseline_behavior_score",
                "behavior_score",
                "directional_gain",
                "directional_gain_ci_low",
                "directional_gain_ci_high",
                "ci_supported_directional_gain",
            ]
        ].to_string(index=False)
    )

    hall_o = hall_overall.iloc[0]

    print()
    print("TruthfulQA overall:")

    print(f"  directional conditions:    {int(hall_o['num_directional_conditions'])}")

    print(f"  mean directional gain:     {hall_o['mean_directional_gain']:+.6f}")

    print(f"  median directional gain:   {hall_o['median_directional_gain']:+.6f}")

    print(
        f"  positive gains:            "
        f"{int(hall_o['positive_gain_count'])}/"
        f"{int(hall_o['num_directional_conditions'])}"
    )

    print(
        f"  CI-supported gains:        "
        f"{int(hall_o['supported_gain_count'])}/"
        f"{int(hall_o['num_directional_conditions'])}"
    )

    print()
    print(f"Saved analysis to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
