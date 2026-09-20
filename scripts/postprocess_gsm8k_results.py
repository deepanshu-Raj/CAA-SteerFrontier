"""Postprocess GSM8K generations and build frozen cross-model capability summaries.

Flexible numeric accuracy remains the primary metric; strict final-marker
parsing is retained only as a corrected diagnostic.
"""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# Frozen protocol
# =============================================================================

MODEL_ORDER = [
    "llama2-7b-chat",
    "llama3.1-8b-instruct",
    "mistral-7b-instruct-v0.3",
    "qwen2.5-7b-instruct",
]

EXPECTED_CONDITIONS_PER_MODEL = 14
EXPECTED_ITEMS_PER_CONDITION = 64
EXPECTED_TOTAL_CONDITIONS = 56

BOOTSTRAP_SAMPLES = 10_000
BASE_SEED = 20260919

RESULT_RELATIVE_ROOT = Path("secondary_capability") / "gsm8k"

CROSS_MODEL_OUTPUT_ROOT = (
    Path("results") / "cross_model" / "secondary_capability" / "gsm8k" / "postprocessed"
)


# =============================================================================
# Correct strict-marker definition
# =============================================================================

# Numerical only:
#
#   42
#   -3
#   +7
#   3.5
#   .25
#   1e3
#   1/2
#
# Deliberately NOT allowed:
#
#   $42
#   1,000
#   42 dollars
#   ##### 42
#   ### 42
#   Final answer: #### 42
#
# The full stripped final non-empty line must match.
NUMBER_COMPONENT = (
    r"[+-]?"
    r"(?:"
    r"\d+(?:\.\d*)?"
    r"|"
    r"\.\d+"
    r")"
    r"(?:[eE][+-]?\d+)?"
)

STRICT_NUMBER = (
    rf"{NUMBER_COMPONENT}"
    rf"(?:/{NUMBER_COMPONENT})?"
)

EXACT_FINAL_PATTERN = re.compile(rf"^####(?!#)\s*({STRICT_NUMBER})\s*$")


# =============================================================================
# General helpers
# =============================================================================


def stable_seed(
    *parts,
):
    payload = "|".join(str(part) for part in parts).encode("utf-8")

    digest = hashlib.sha256(payload).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    ) % (2**32)


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


def write_json(
    payload,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
        )
        f.write("\n")

    temporary.replace(path)


def as_bool_series(
    series: pd.Series,
):
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    normalized = series.astype(str).str.strip().str.lower()

    unknown = ~normalized.isin(mapping)

    if unknown.any():
        values = sorted(normalized[unknown].unique())

        raise ValueError(f"Could not parse boolean values: {values}")

    return normalized.map(mapping).astype(bool)


# =============================================================================
# Exact numeric parsing
# =============================================================================


def decimal_fraction(
    text: str,
):
    try:
        return Fraction(Decimal(text))
    except (
        InvalidOperation,
        ValueError,
        ZeroDivisionError,
    ):
        return None


def parse_numeric(
    text,
    *,
    allow_gold_formatting: bool = False,
):
    if text is None:
        return None

    if isinstance(
        text,
        float,
    ) and np.isnan(text):
        return None

    value = str(text).strip()

    if not value:
        return None

    # Gold answers were already normalized by the
    # production data preparation. This is only a
    # defensive convenience for CSV representation.
    if allow_gold_formatting:
        value = (
            value.replace(
                ",",
                "",
            )
            .replace(
                "$",
                "",
            )
            .strip()
        )

    if "/" in value:
        pieces = value.split("/")

        if len(pieces) != 2:
            return None

        numerator = decimal_fraction(pieces[0])

        denominator = decimal_fraction(pieces[1])

        if numerator is None or denominator is None or denominator == 0:
            return None

        return numerator / denominator

    return decimal_fraction(value)


def extract_exact_final_prediction(
    response,
):
    """
    Correct strict diagnostic.

    The final non-empty line must consist of:
        exactly four hashes
        optional whitespace
        one numerical answer
        optional whitespace

    The four hashes must not be the beginning of
    a longer hash run such as #####.
    """

    if response is None:
        return (
            None,
            None,
        )

    if isinstance(
        response,
        float,
    ) and np.isnan(response):
        return (
            None,
            None,
        )

    lines = [line.strip() for line in str(response).splitlines() if line.strip()]

    if not lines:
        return (
            None,
            None,
        )

    final_line = lines[-1]

    match = EXACT_FINAL_PATTERN.fullmatch(final_line)

    if match is None:
        return (
            None,
            None,
        )

    prediction_text = match.group(1)

    prediction_value = parse_numeric(prediction_text)

    if prediction_value is None:
        return (
            None,
            None,
        )

    return (
        prediction_text,
        prediction_value,
    )


# =============================================================================
# Item-level correction
# =============================================================================


def validate_item_frame(
    frame: pd.DataFrame,
    *,
    path: Path,
):
    required = {
        "subset_index",
        "source_index",
        "gold_answer",
        "is_correct_flexible",
        "parse_method_flexible",
        "new_token_count",
        "hit_max_tokens",
        "response",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    if len(frame) != EXPECTED_ITEMS_PER_CONDITION:
        raise RuntimeError(
            f"{path}: expected "
            f"{EXPECTED_ITEMS_PER_CONDITION} "
            f"items, found {len(frame)}."
        )

    if frame["subset_index"].duplicated().any():
        raise RuntimeError(f"{path}: duplicate subset_index values.")


def correct_item_frame(
    frame: pd.DataFrame,
):
    result = frame.copy()

    predictions = []
    marker_flags = []
    strict_correct = []

    for row in result.to_dict(orient="records"):
        (
            prediction_text,
            prediction_value,
        ) = extract_exact_final_prediction(row["response"])

        gold_value = parse_numeric(
            row["gold_answer"],
            allow_gold_formatting=True,
        )

        if gold_value is None:
            raise RuntimeError(
                f"Could not parse saved GSM8K gold answer {row['gold_answer']!r}."
            )

        has_marker = prediction_value is not None

        is_correct = bool(has_marker and prediction_value == gold_value)

        predictions.append(prediction_text)

        marker_flags.append(has_marker)

        strict_correct.append(is_correct)

    result["predicted_answer_strict_exact"] = predictions

    result["used_exact_final_marker"] = marker_flags

    result["is_correct_strict_exact"] = strict_correct

    # Normalize the already-frozen primary fields.
    #
    # We DO NOT parse the response again for the
    # flexible metric. This is the production result.
    result["is_correct_flexible"] = as_bool_series(result["is_correct_flexible"])

    result["hit_max_tokens"] = as_bool_series(result["hit_max_tokens"])

    return result


# =============================================================================
# Paired statistics
# =============================================================================


def paired_delta_statistics(
    baseline_values,
    steered_values,
    *,
    seed: int,
):
    baseline_values = np.asarray(
        baseline_values,
        dtype=float,
    )

    steered_values = np.asarray(
        steered_values,
        dtype=float,
    )

    if baseline_values.shape != steered_values.shape:
        raise ValueError("Paired arrays have different shapes.")

    differences = steered_values - baseline_values

    delta = float(differences.mean())

    if np.all(differences == 0):
        return {
            "delta": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
        }

    rng = np.random.default_rng(seed)

    indices = rng.integers(
        0,
        len(differences),
        size=(
            BOOTSTRAP_SAMPLES,
            len(differences),
        ),
    )

    boot = differences[indices].mean(axis=1)

    (
        low,
        high,
    ) = np.quantile(
        boot,
        [
            0.025,
            0.975,
        ],
    )

    return {
        "delta": float(delta),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def align_items(
    baseline: pd.DataFrame,
    steered: pd.DataFrame,
):
    columns = [
        "subset_index",
        "source_index",
    ]

    baseline_keys = baseline[columns].sort_values(columns).reset_index(drop=True)

    steered_keys = steered[columns].sort_values(columns).reset_index(drop=True)

    if not baseline_keys.equals(steered_keys):
        raise RuntimeError("Baseline and steered GSM8K items are not exactly aligned.")

    baseline_aligned = baseline.sort_values(columns).reset_index(drop=True)

    steered_aligned = steered.sort_values(columns).reset_index(drop=True)

    return (
        baseline_aligned,
        steered_aligned,
    )


def condition_statistics(
    *,
    model_slug: str,
    behavior: str,
    direction: str,
    baseline: pd.DataFrame,
    steered: pd.DataFrame,
):
    (
        baseline,
        steered,
    ) = align_items(
        baseline,
        steered,
    )

    baseline_flexible = baseline["is_correct_flexible"].astype(float).to_numpy()

    steered_flexible = steered["is_correct_flexible"].astype(float).to_numpy()

    baseline_strict = baseline["is_correct_strict_exact"].astype(float).to_numpy()

    steered_strict = steered["is_correct_strict_exact"].astype(float).to_numpy()

    flexible_stats = paired_delta_statistics(
        baseline_flexible,
        steered_flexible,
        seed=stable_seed(
            BASE_SEED,
            model_slug,
            behavior,
            direction,
            "flexible",
        ),
    )

    strict_stats = paired_delta_statistics(
        baseline_strict,
        steered_strict,
        seed=stable_seed(
            BASE_SEED,
            model_slug,
            behavior,
            direction,
            "strict_exact",
        ),
    )

    baseline_parse_rate = float(
        (
            ~baseline["parse_method_flexible"].isin(
                [
                    "unparsed",
                    "truncated_unparsed",
                ]
            )
        )
        .astype(float)
        .mean()
    )

    steered_parse_rate = float(
        (
            ~steered["parse_method_flexible"].isin(
                [
                    "unparsed",
                    "truncated_unparsed",
                ]
            )
        )
        .astype(float)
        .mean()
    )

    return {
        # ------------------------------------------------------------
        # Frozen PRIMARY result.
        # ------------------------------------------------------------
        "frozen_baseline_flexible_accuracy": float(baseline_flexible.mean()),
        "frozen_flexible_accuracy": float(steered_flexible.mean()),
        "frozen_flexible_delta_accuracy": float(flexible_stats["delta"]),
        "frozen_flexible_delta_ci_low": float(flexible_stats["ci_low"]),
        "frozen_flexible_delta_ci_high": float(flexible_stats["ci_high"]),
        # ------------------------------------------------------------
        # Corrected strict FORMAT diagnostic.
        # ------------------------------------------------------------
        "corrected_baseline_strict_accuracy": float(baseline_strict.mean()),
        "corrected_strict_accuracy": float(steered_strict.mean()),
        "corrected_strict_delta_accuracy": float(strict_stats["delta"]),
        "corrected_strict_delta_ci_low": float(strict_stats["ci_low"]),
        "corrected_strict_delta_ci_high": float(strict_stats["ci_high"]),
        "corrected_baseline_marker_rate": float(
            baseline["used_exact_final_marker"].astype(float).mean()
        ),
        "corrected_marker_rate": float(
            steered["used_exact_final_marker"].astype(float).mean()
        ),
        # ------------------------------------------------------------
        # Other unchanged diagnostics.
        # ------------------------------------------------------------
        "baseline_flexible_parse_rate": (baseline_parse_rate),
        "flexible_parse_rate": (steered_parse_rate),
        "baseline_truncation_rate": float(
            baseline["hit_max_tokens"].astype(float).mean()
        ),
        "truncation_rate": float(steered["hit_max_tokens"].astype(float).mean()),
        "baseline_mean_generated_tokens": float(
            baseline["new_token_count"].astype(float).mean()
        ),
        "mean_generated_tokens": float(steered["new_token_count"].astype(float).mean()),
        "num_examples": int(len(steered)),
        "bootstrap_samples_postprocess": int(BOOTSTRAP_SAMPLES),
    }


# =============================================================================
# Model processing
# =============================================================================


def process_model(
    model_slug: str,
):
    root = Path("results") / model_slug / RESULT_RELATIVE_ROOT

    baseline_path = root / "baseline_items.csv"

    item_root = root / "items"

    original_summary_path = root / "gsm8k_summary.csv"

    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)

    if not item_root.is_dir():
        raise FileNotFoundError(item_root)

    if not original_summary_path.is_file():
        raise FileNotFoundError(original_summary_path)

    original_summary = pd.read_csv(original_summary_path)

    required_summary = {
        "behavior",
        "direction",
    }

    missing = required_summary - set(original_summary.columns)

    if missing:
        raise RuntimeError(f"{original_summary_path} missing {sorted(missing)}.")

    if len(original_summary) != EXPECTED_CONDITIONS_PER_MODEL:
        raise RuntimeError(
            f"{model_slug}: expected "
            f"{EXPECTED_CONDITIONS_PER_MODEL} "
            "summary rows, found "
            f"{len(original_summary)}."
        )

    # ----------------------------------------------------------------
    # Baseline
    # ----------------------------------------------------------------

    baseline_raw = pd.read_csv(baseline_path)

    validate_item_frame(
        baseline_raw,
        path=baseline_path,
    )

    baseline = correct_item_frame(baseline_raw)

    post_root = root / "postprocessed"

    write_csv(
        baseline,
        post_root / "baseline_items_corrected.csv",
    )

    # ----------------------------------------------------------------
    # Conditions
    # ----------------------------------------------------------------

    item_paths = sorted(item_root.glob("*.csv"))

    if len(item_paths) != EXPECTED_CONDITIONS_PER_MODEL:
        raise RuntimeError(
            f"{model_slug}: expected "
            f"{EXPECTED_CONDITIONS_PER_MODEL} "
            f"item files, found "
            f"{len(item_paths)}."
        )

    rows = []

    for path in item_paths:
        stem = path.stem

        pieces = stem.rsplit(
            "__",
            1,
        )

        if len(pieces) != 2:
            raise RuntimeError(f"Unexpected GSM8K item filename: {path}")

        (
            behavior,
            direction,
        ) = pieces

        if direction not in {
            "negative",
            "positive",
        }:
            raise RuntimeError(f"Unexpected direction in {path}.")

        condition_raw = pd.read_csv(path)

        validate_item_frame(
            condition_raw,
            path=path,
        )

        condition = correct_item_frame(condition_raw)

        corrected_item_path = post_root / "items" / path.name

        write_csv(
            condition,
            corrected_item_path,
        )

        metadata_rows = original_summary[
            (original_summary["behavior"] == behavior)
            & (original_summary["direction"] == direction)
        ]

        if len(metadata_rows) != 1:
            raise RuntimeError(
                f"{model_slug}: expected "
                "one original summary row "
                f"for {behavior}/{direction}; "
                f"found {len(metadata_rows)}."
            )

        # Preserve every original summary field for auditability.
        row = metadata_rows.iloc[0].to_dict()

        corrected = condition_statistics(
            model_slug=model_slug,
            behavior=behavior,
            direction=direction,
            baseline=baseline,
            steered=condition,
        )

        row.update(corrected)

        rows.append(row)

    corrected_summary = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "behavior",
                "direction",
            ]
        )
        .reset_index(drop=True)
    )

    if len(corrected_summary) != EXPECTED_CONDITIONS_PER_MODEL:
        raise RuntimeError(f"{model_slug}: corrected summary row count mismatch.")

    write_csv(
        corrected_summary,
        post_root / "gsm8k_summary_corrected.csv",
    )

    # ----------------------------------------------------------------
    # Baseline diagnostics
    # ----------------------------------------------------------------

    baseline_record = {
        "model_slug": (model_slug),
        "num_examples": int(len(baseline)),
        "flexible_accuracy": float(
            baseline["is_correct_flexible"].astype(float).mean()
        ),
        "corrected_strict_accuracy": float(
            baseline["is_correct_strict_exact"].astype(float).mean()
        ),
        "corrected_marker_rate": float(
            baseline["used_exact_final_marker"].astype(float).mean()
        ),
        "original_strict_accuracy": float(
            baseline["is_correct_strict"].astype(float).mean()
        )
        if ("is_correct_strict" in baseline.columns)
        else float("nan"),
        "original_marker_rate": float(
            baseline["used_required_marker"].astype(float).mean()
        )
        if ("used_required_marker" in baseline.columns)
        else float("nan"),
        "flexible_parse_rate": float(
            (
                ~baseline["parse_method_flexible"].isin(
                    [
                        "unparsed",
                        "truncated_unparsed",
                    ]
                )
            )
            .astype(float)
            .mean()
        ),
        "truncation_rate": float(baseline["hit_max_tokens"].astype(float).mean()),
        "mean_generated_tokens": float(
            baseline["new_token_count"].astype(float).mean()
        ),
    }

    # Recover label if it is available in original summary.
    if "model_label" in corrected_summary.columns:
        baseline_record["model_label"] = str(corrected_summary["model_label"].iloc[0])

    return (
        corrected_summary,
        baseline_record,
    )


# =============================================================================
# Cross-model summaries
# =============================================================================


def build_model_summary(
    combined: pd.DataFrame,
    baselines: pd.DataFrame,
):
    rows = []

    for model_slug in MODEL_ORDER:
        frame = combined[combined["model_slug"] == model_slug].copy()

        if frame.empty:
            raise RuntimeError(f"No GSM8K rows for {model_slug}.")

        if len(frame) != EXPECTED_CONDITIONS_PER_MODEL:
            raise RuntimeError(
                f"{model_slug}: expected "
                f"{EXPECTED_CONDITIONS_PER_MODEL} "
                f"rows, found {len(frame)}."
            )

        baseline_rows = baselines[baselines["model_slug"] == model_slug]

        if len(baseline_rows) != 1:
            raise RuntimeError(f"{model_slug}: baseline diagnostic row mismatch.")

        baseline = baseline_rows.iloc[0]

        flexible_delta = frame["frozen_flexible_delta_accuracy"].astype(float)

        strict_delta = frame["corrected_strict_delta_accuracy"].astype(float)

        supported_flexible_decrease = (
            frame["frozen_flexible_delta_ci_high"].astype(float) < 0.0
        )

        supported_flexible_increase = (
            frame["frozen_flexible_delta_ci_low"].astype(float) > 0.0
        )

        supported_strict_decrease = (
            frame["corrected_strict_delta_ci_high"].astype(float) < 0.0
        )

        supported_strict_increase = (
            frame["corrected_strict_delta_ci_low"].astype(float) > 0.0
        )

        model_label = (
            str(frame["model_label"].iloc[0])
            if ("model_label" in frame.columns)
            else model_slug
        )

        rows.append(
            {
                "model_slug": (model_slug),
                "model_label": (model_label),
                "num_conditions": int(len(frame)),
                # Primary flexible metric
                "baseline_flexible_accuracy": float(baseline["flexible_accuracy"]),
                "mean_flexible_delta_accuracy": float(flexible_delta.mean()),
                "median_flexible_delta_accuracy": float(flexible_delta.median()),
                "min_flexible_delta_accuracy": float(flexible_delta.min()),
                "max_flexible_delta_accuracy": float(flexible_delta.max()),
                "num_ci_supported_flexible_decreases": int(
                    supported_flexible_decrease.sum()
                ),
                "num_ci_supported_flexible_increases": int(
                    supported_flexible_increase.sum()
                ),
                "num_flexible_ci_overlap_zero": int(
                    (~supported_flexible_decrease & ~supported_flexible_increase).sum()
                ),
                # Corrected strict diagnostic
                "corrected_baseline_strict_accuracy": float(
                    baseline["corrected_strict_accuracy"]
                ),
                "corrected_baseline_marker_rate": float(
                    baseline["corrected_marker_rate"]
                ),
                "original_baseline_strict_accuracy": float(
                    baseline["original_strict_accuracy"]
                ),
                "original_baseline_marker_rate": float(
                    baseline["original_marker_rate"]
                ),
                "mean_corrected_strict_delta_accuracy": float(strict_delta.mean()),
                "mean_corrected_marker_rate": float(
                    frame["corrected_marker_rate"].astype(float).mean()
                ),
                "num_ci_supported_strict_decreases": int(
                    supported_strict_decrease.sum()
                ),
                "num_ci_supported_strict_increases": int(
                    supported_strict_increase.sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_overall_summary(
    combined: pd.DataFrame,
    model_summary: pd.DataFrame,
):
    flexible_delta = combined["frozen_flexible_delta_accuracy"].astype(float)

    strict_delta = combined["corrected_strict_delta_accuracy"].astype(float)

    flexible_decrease = combined["frozen_flexible_delta_ci_high"].astype(float) < 0

    flexible_increase = combined["frozen_flexible_delta_ci_low"].astype(float) > 0

    return pd.DataFrame(
        [
            {
                "num_models": int(combined["model_slug"].nunique()),
                "num_conditions": int(len(combined)),
                "mean_model_baseline_flexible_accuracy": float(
                    model_summary["baseline_flexible_accuracy"].mean()
                ),
                "mean_flexible_delta_accuracy": float(flexible_delta.mean()),
                "median_flexible_delta_accuracy": float(flexible_delta.median()),
                "min_flexible_delta_accuracy": float(flexible_delta.min()),
                "max_flexible_delta_accuracy": float(flexible_delta.max()),
                "num_ci_supported_flexible_decreases": int(flexible_decrease.sum()),
                "num_ci_supported_flexible_increases": int(flexible_increase.sum()),
                "num_flexible_ci_overlap_zero": int(
                    (~flexible_decrease & ~flexible_increase).sum()
                ),
                "mean_corrected_strict_delta_accuracy": float(strict_delta.mean()),
                "mean_model_corrected_baseline_strict_accuracy": float(
                    model_summary["corrected_baseline_strict_accuracy"].mean()
                ),
                "mean_model_corrected_baseline_marker_rate": float(
                    model_summary["corrected_baseline_marker_rate"].mean()
                ),
                "mean_condition_corrected_marker_rate": float(
                    combined["corrected_marker_rate"].astype(float).mean()
                ),
            }
        ]
    )


# =============================================================================
# Main
# =============================================================================


def main():
    corrected_frames = []
    baseline_records = []

    print("=" * 100)
    print("GSM8K POSTPROCESSING")
    print("=" * 100)
    print(
        "Primary flexible scores are read from the "
        "saved production item files and are NOT reparsed."
    )
    print("Only the strict final-marker diagnostic is corrected.")

    for model_slug in MODEL_ORDER:
        print()
        print("=" * 100)
        print(model_slug)
        print("=" * 100)

        (
            summary,
            baseline,
        ) = process_model(model_slug)

        corrected_frames.append(summary)

        baseline_records.append(baseline)

        print(f"Baseline flexible accuracy: {baseline['flexible_accuracy']:.4f}")

        print(
            "Baseline strict accuracy:   "
            f"{baseline['original_strict_accuracy']:.4f} "
            "-> "
            f"{baseline['corrected_strict_accuracy']:.4f}"
        )

        print(
            "Baseline marker rate:       "
            f"{baseline['original_marker_rate']:.4f} "
            "-> "
            f"{baseline['corrected_marker_rate']:.4f}"
        )

        print(
            "Mean flexible delta:        "
            f"{summary['frozen_flexible_delta_accuracy'].mean():+.6f}"
        )

    combined = pd.concat(
        corrected_frames,
        ignore_index=True,
    )

    if len(combined) != EXPECTED_TOTAL_CONDITIONS:
        raise RuntimeError(
            "Expected "
            f"{EXPECTED_TOTAL_CONDITIONS} "
            "cross-model conditions, "
            f"found {len(combined)}."
        )

    baselines = pd.DataFrame(baseline_records)

    model_summary = build_model_summary(
        combined,
        baselines,
    )

    overall_summary = build_overall_summary(
        combined,
        model_summary,
    )

    # ----------------------------------------------------------------
    # Persist cross-model outputs
    # ----------------------------------------------------------------

    write_csv(
        combined,
        CROSS_MODEL_OUTPUT_ROOT / "gsm8k_summary_corrected.csv",
    )

    write_csv(
        baselines,
        CROSS_MODEL_OUTPUT_ROOT / "gsm8k_baseline_diagnostics_corrected.csv",
    )

    write_csv(
        model_summary,
        CROSS_MODEL_OUTPUT_ROOT / "gsm8k_model_summary_corrected.csv",
    )

    write_csv(
        overall_summary,
        CROSS_MODEL_OUTPUT_ROOT / "gsm8k_overall_summary_corrected.csv",
    )

    manifest = {
        "protocol": ("gsm8k_secondary_postprocess_exact_marker_v1"),
        "production_protocol": ("gsm8k_secondary_64_8shot_clean_v1"),
        "num_models": int(len(MODEL_ORDER)),
        "num_conditions": int(len(combined)),
        "examples_per_condition": int(EXPECTED_ITEMS_PER_CONDITION),
        "bootstrap_samples": int(BOOTSTRAP_SAMPLES),
        "primary_metric_policy": (
            "Flexible correctness is copied from "
            "saved production is_correct_flexible "
            "fields. Responses are not reparsed for "
            "the primary metric."
        ),
        "strict_diagnostic_policy": (
            "The final non-empty response line must "
            "contain exactly four hash characters "
            "followed only by a numerical answer. "
            "Longer or shorter hash runs, currency "
            "symbols, commas, units, surrounding text, "
            "or non-final marker lines do not qualify."
        ),
        "raw_inference_files_modified": False,
        "outputs": {
            "summary": str(CROSS_MODEL_OUTPUT_ROOT / "gsm8k_summary_corrected.csv"),
            "baseline_diagnostics": str(
                CROSS_MODEL_OUTPUT_ROOT / "gsm8k_baseline_diagnostics_corrected.csv"
            ),
            "model_summary": str(
                CROSS_MODEL_OUTPUT_ROOT / "gsm8k_model_summary_corrected.csv"
            ),
            "overall_summary": str(
                CROSS_MODEL_OUTPUT_ROOT / "gsm8k_overall_summary_corrected.csv"
            ),
        },
    }

    write_json(
        manifest,
        CROSS_MODEL_OUTPUT_ROOT / "manifest.json",
    )

    # ----------------------------------------------------------------
    # Console report
    # ----------------------------------------------------------------

    print()
    print("=" * 100)
    print("CORRECTED BASELINE DIAGNOSTICS")
    print("=" * 100)

    print(
        baselines[
            [
                "model_slug",
                "flexible_accuracy",
                "original_strict_accuracy",
                "corrected_strict_accuracy",
                "original_marker_rate",
                "corrected_marker_rate",
                "flexible_parse_rate",
                "truncation_rate",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 100)
    print("PRIMARY GSM8K CAPABILITY SUMMARY")
    print("=" * 100)

    print(
        model_summary[
            [
                "model_slug",
                "baseline_flexible_accuracy",
                "mean_flexible_delta_accuracy",
                "median_flexible_delta_accuracy",
                "min_flexible_delta_accuracy",
                "max_flexible_delta_accuracy",
                "num_ci_supported_flexible_decreases",
                "num_ci_supported_flexible_increases",
                "num_flexible_ci_overlap_zero",
            ]
        ].to_string(index=False)
    )

    overall = overall_summary.iloc[0]

    print()
    print("=" * 100)
    print("OVERALL")
    print("=" * 100)

    print(f"Conditions:                    {int(overall['num_conditions'])}")

    print(
        f"Mean flexible delta accuracy:  {overall['mean_flexible_delta_accuracy']:+.6f}"
    )

    print(
        f"Median flexible delta:         "
        f"{overall['median_flexible_delta_accuracy']:+.6f}"
    )

    print(
        f"Range:                         "
        f"[{overall['min_flexible_delta_accuracy']:+.6f}, "
        f"{overall['max_flexible_delta_accuracy']:+.6f}]"
    )

    print(
        f"CI-supported decreases:        "
        f"{int(overall['num_ci_supported_flexible_decreases'])}/"
        f"{int(overall['num_conditions'])}"
    )

    print(
        f"CI-supported increases:        "
        f"{int(overall['num_ci_supported_flexible_increases'])}/"
        f"{int(overall['num_conditions'])}"
    )

    print(
        f"CI overlaps zero:              "
        f"{int(overall['num_flexible_ci_overlap_zero'])}/"
        f"{int(overall['num_conditions'])}"
    )

    print()
    print(f"Saved to: {CROSS_MODEL_OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
