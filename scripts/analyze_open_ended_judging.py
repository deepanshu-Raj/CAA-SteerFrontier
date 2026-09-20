"""Analyze blinded open-ended steering judgments after all cases are scored.

The condition key is applied only during analysis, preserving judge blinding;
persistent nonterminations remain a separate generation-stability outcome.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# Frozen analysis protocol
# =============================================================================

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260919

MODEL_ORDER = [
    "llama2-7b-chat",
    "llama3.1-8b-instruct",
    "mistral-7b-instruct-v0.3",
    "qwen2.5-7b-instruct",
]

BEHAVIOR_ORDER = [
    "sycophancy",
    "refusal",
    "hallucination",
]

CONDITIONS = [
    "negative",
    "baseline",
    "positive",
]

EXPECTED_CASES = 276
EXPECTED_RAW_TRIPLETS = 288
EXPECTED_RAW_RESPONSES = 864
EXPECTED_PERSISTENT_FAILURES = 12

ROOT = Path("results") / "cross_model" / "open_ended_generation" / "judging"

JUDGE_RESULTS_PATH = ROOT / "judge_results" / "judge_results.csv"

CONDITION_KEY_PATH = ROOT / "_condition_key.csv"

FAILURES_PATH = ROOT / "generation_stability_failures.csv"

OUTPUT_ROOT = Path("results") / "cross_model" / "open_ended_generation" / "analysis"


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


# =============================================================================
# Cluster bootstrap
# =============================================================================


def cluster_bootstrap_mean_ci(
    frame: pd.DataFrame,
    *,
    value_column: str,
    cluster_columns,
    seed: int,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
):
    """
    Bootstrap complete clusters with replacement.

    For pooled cross-model summaries, clusters correspond to the
    underlying prompt (behavior, prompt_id), preserving dependence
    between different models' responses to the same prompt.

    Within a model/behavior cell, one row corresponds to one prompt,
    so this reduces naturally to the ordinary paired bootstrap.
    """

    if frame.empty:
        raise ValueError("Cannot bootstrap an empty frame.")

    values = frame[value_column].astype(float)

    observed = float(values.mean())

    grouped = [
        group[value_column].astype(float).to_numpy()
        for _, group in frame.groupby(
            cluster_columns,
            sort=False,
            dropna=False,
        )
    ]

    num_clusters = len(grouped)

    if num_clusters < 1:
        raise RuntimeError("No bootstrap clusters found.")

    group_sums = np.asarray(
        [values.sum() for values in grouped],
        dtype=float,
    )

    group_counts = np.asarray(
        [len(values) for values in grouped],
        dtype=float,
    )

    if num_clusters == 1:
        return (
            observed,
            observed,
            observed,
            num_clusters,
        )

    rng = np.random.default_rng(seed)

    bootstrap_means = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for index in range(n_bootstrap):
        sampled = rng.integers(
            0,
            num_clusters,
            size=num_clusters,
        )

        bootstrap_means[index] = group_sums[sampled].sum() / group_counts[sampled].sum()

    low, high = np.quantile(
        bootstrap_means,
        [
            0.025,
            0.975,
        ],
    )

    return (
        observed,
        float(low),
        float(high),
        num_clusters,
    )


# =============================================================================
# Unblinding
# =============================================================================


def load_and_unblind():
    if not JUDGE_RESULTS_PATH.is_file():
        raise FileNotFoundError(JUDGE_RESULTS_PATH)

    if not CONDITION_KEY_PATH.is_file():
        raise FileNotFoundError(CONDITION_KEY_PATH)

    judge = pd.read_csv(JUDGE_RESULTS_PATH)

    key = pd.read_csv(CONDITION_KEY_PATH)

    if len(judge) != EXPECTED_CASES:
        raise RuntimeError(f"Expected {EXPECTED_CASES} judge rows, found {len(judge)}.")

    if len(key) != EXPECTED_CASES:
        raise RuntimeError(
            f"Expected {EXPECTED_CASES} condition-key rows, found {len(key)}."
        )

    if judge["case_id"].duplicated().any():
        raise RuntimeError("Duplicate judge case IDs.")

    if key["case_id"].duplicated().any():
        raise RuntimeError("Duplicate condition-key case IDs.")

    required_judge = {
        "case_id",
        "behavior",
        "A_score",
        "A_rationale",
        "B_score",
        "B_rationale",
        "C_score",
        "C_rationale",
        "judge_model",
        "reasoning_effort",
    }

    required_key = {
        "case_id",
        "model_slug",
        "behavior",
        "prompt_id",
        "domain",
        "A_condition",
        "B_condition",
        "C_condition",
    }

    missing_judge = required_judge - set(judge.columns)

    missing_key = required_key - set(key.columns)

    if missing_judge:
        raise RuntimeError(f"Judge results missing {sorted(missing_judge)}.")

    if missing_key:
        raise RuntimeError(f"Condition key missing {sorted(missing_key)}.")

    merged = judge.merge(
        key,
        on="case_id",
        how="inner",
        validate="one_to_one",
        suffixes=(
            "_judge",
            "_key",
        ),
    )

    if len(merged) != EXPECTED_CASES:
        raise RuntimeError("Judge/key merge lost cases.")

    if not (
        merged["behavior_judge"].astype(str) == merged["behavior_key"].astype(str)
    ).all():
        raise RuntimeError("Behavior differs between judge file and condition key.")

    rows = []

    for row in merged.to_dict(orient="records"):
        condition_scores = {}
        condition_rationales = {}

        for label in [
            "A",
            "B",
            "C",
        ]:
            condition = str(row[f"{label}_condition"])

            if condition not in CONDITIONS:
                raise RuntimeError(f"Unexpected condition {condition!r}.")

            if condition in condition_scores:
                raise RuntimeError(f"{row['case_id']}: condition assigned twice.")

            score = int(row[f"{label}_score"])

            if not (0 <= score <= 4):
                raise RuntimeError(f"{row['case_id']}: invalid score {score}.")

            condition_scores[condition] = score

            condition_rationales[condition] = str(row[f"{label}_rationale"])

        if set(condition_scores) != set(CONDITIONS):
            raise RuntimeError(f"{row['case_id']}: condition mapping incomplete.")

        negative = int(condition_scores["negative"])

        baseline = int(condition_scores["baseline"])

        positive = int(condition_scores["positive"])

        positive_delta = positive - baseline

        negative_delta = baseline - negative

        span_delta = positive - negative

        rows.append(
            {
                "case_id": (row["case_id"]),
                "model_slug": (row["model_slug"]),
                "behavior": (row["behavior_key"]),
                "prompt_id": (row["prompt_id"]),
                "domain": (row["domain"]),
                "negative_score": (negative),
                "baseline_score": (baseline),
                "positive_score": (positive),
                "positive_delta": (positive_delta),
                "negative_delta": (negative_delta),
                "span_delta": (span_delta),
                "positive_success": int(positive_delta > 0),
                "negative_success": int(negative_delta > 0),
                "positive_nonnegative": int(positive_delta >= 0),
                "negative_nonnegative": int(negative_delta >= 0),
                "positive_reverse": int(positive_delta < 0),
                "negative_reverse": int(negative_delta < 0),
                "positive_tie": int(positive_delta == 0),
                "negative_tie": int(negative_delta == 0),
                # Equivalent to both directional
                # comparisons being strictly successful.
                "strict_monotonic": int(negative < baseline < positive),
                # Allows one or both adjacent
                # comparisons to tie.
                "weak_monotonic": int(negative <= baseline <= positive),
                "negative_rationale": (condition_rationales["negative"]),
                "baseline_rationale": (condition_rationales["baseline"]),
                "positive_rationale": (condition_rationales["positive"]),
                "judge_model": (row["judge_model"]),
                "reasoning_effort": (row["reasoning_effort"]),
            }
        )

    result = pd.DataFrame(rows)

    if len(result) != EXPECTED_CASES:
        raise RuntimeError("Unblinded case count mismatch.")

    return result


# =============================================================================
# Summary
# =============================================================================

BOOTSTRAP_METRICS = {
    "positive_delta": ("mean_positive_delta"),
    "negative_delta": ("mean_negative_delta"),
    "span_delta": ("mean_positive_minus_negative"),
    "positive_success": ("positive_direction_success_rate"),
    "negative_success": ("negative_direction_success_rate"),
    "strict_monotonic": ("strict_monotonic_rate"),
    "weak_monotonic": ("weak_monotonic_rate"),
}


def summarize_group(
    frame: pd.DataFrame,
    *,
    label: str,
    cluster_columns,
    metadata=None,
):
    if frame.empty:
        raise ValueError(f"{label}: empty analysis group.")

    row = {
        "group": (label),
        "num_cases": int(len(frame)),
        "num_bootstrap_clusters": int(
            frame.groupby(
                cluster_columns,
                dropna=False,
            ).ngroups
        ),
        "mean_negative_score": float(frame["negative_score"].mean()),
        "mean_baseline_score": float(frame["baseline_score"].mean()),
        "mean_positive_score": float(frame["positive_score"].mean()),
        "median_positive_delta": float(frame["positive_delta"].median()),
        "median_negative_delta": float(frame["negative_delta"].median()),
        "median_positive_minus_negative": float(frame["span_delta"].median()),
        "positive_tie_rate": float(frame["positive_tie"].mean()),
        "negative_tie_rate": float(frame["negative_tie"].mean()),
        "positive_reverse_rate": float(frame["positive_reverse"].mean()),
        "negative_reverse_rate": float(frame["negative_reverse"].mean()),
        "positive_nonnegative_rate": float(frame["positive_nonnegative"].mean()),
        "negative_nonnegative_rate": float(frame["negative_nonnegative"].mean()),
    }

    if metadata:
        row.update(metadata)

    for (
        value_column,
        output_name,
    ) in BOOTSTRAP_METRICS.items():
        (
            observed,
            low,
            high,
            num_clusters,
        ) = cluster_bootstrap_mean_ci(
            frame,
            value_column=(value_column),
            cluster_columns=(cluster_columns),
            seed=stable_seed(
                BOOTSTRAP_SEED,
                label,
                value_column,
            ),
        )

        row[output_name] = observed

        row[f"{output_name}_ci_low"] = low

        row[f"{output_name}_ci_high"] = high

        if num_clusters != row["num_bootstrap_clusters"]:
            raise RuntimeError("Bootstrap cluster-count mismatch.")

    return row


# =============================================================================
# Main summary tables
# =============================================================================


def build_overall_summary(
    cases: pd.DataFrame,
):
    # The same underlying prompt appears under
    # multiple models, so cluster across model
    # replicates of one behavior/prompt.
    return pd.DataFrame(
        [
            summarize_group(
                cases,
                label="overall",
                cluster_columns=[
                    "behavior",
                    "prompt_id",
                ],
            )
        ]
    )


def build_behavior_summary(
    cases: pd.DataFrame,
):
    rows = []

    for behavior in BEHAVIOR_ORDER:
        frame = cases[cases["behavior"] == behavior]

        rows.append(
            summarize_group(
                frame,
                label=(f"behavior:{behavior}"),
                cluster_columns=[
                    "prompt_id",
                ],
                metadata={
                    "behavior": (behavior),
                },
            )
        )

    return pd.DataFrame(rows)


def build_model_summary(
    cases: pd.DataFrame,
):
    rows = []

    for model_slug in MODEL_ORDER:
        frame = cases[cases["model_slug"] == model_slug]

        rows.append(
            summarize_group(
                frame,
                label=(f"model:{model_slug}"),
                cluster_columns=[
                    "behavior",
                    "prompt_id",
                ],
                metadata={
                    "model_slug": (model_slug),
                },
            )
        )

    return pd.DataFrame(rows)


def build_model_behavior_summary(
    cases: pd.DataFrame,
):
    rows = []

    for model_slug in MODEL_ORDER:
        for behavior in BEHAVIOR_ORDER:
            frame = cases[
                (cases["model_slug"] == model_slug) & (cases["behavior"] == behavior)
            ]

            if frame.empty:
                continue

            rows.append(
                summarize_group(
                    frame,
                    label=(f"model_behavior:{model_slug}:{behavior}"),
                    cluster_columns=[
                        "prompt_id",
                    ],
                    metadata={
                        "model_slug": (model_slug),
                        "behavior": (behavior),
                    },
                )
            )

    return pd.DataFrame(rows)


def build_sycophancy_domain_summary(
    cases: pd.DataFrame,
):
    syc = cases[cases["behavior"] == "sycophancy"]

    rows = []

    for domain in [
        "nlp",
        "political",
    ]:
        frame = syc[syc["domain"] == domain]

        rows.append(
            summarize_group(
                frame,
                label=(f"sycophancy_domain:{domain}"),
                cluster_columns=[
                    "prompt_id",
                ],
                metadata={
                    "behavior": ("sycophancy"),
                    "domain": (domain),
                },
            )
        )

    return pd.DataFrame(rows)


# =============================================================================
# Score distribution
# =============================================================================


def build_score_distribution(
    cases: pd.DataFrame,
):
    rows = []

    for behavior in BEHAVIOR_ORDER:
        behavior_frame = cases[cases["behavior"] == behavior]

        for condition in CONDITIONS:
            column = f"{condition}_score"

            counts = (
                behavior_frame[column]
                .value_counts()
                .reindex(
                    [
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    fill_value=0,
                )
            )

            total = int(counts.sum())

            for score in range(5):
                count = int(counts.loc[score])

                rows.append(
                    {
                        "behavior": (behavior),
                        "condition": (condition),
                        "score": int(score),
                        "count": (count),
                        "proportion": float(count / total),
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# Generation stability
# =============================================================================


def build_stability_summaries():
    if not FAILURES_PATH.is_file():
        raise FileNotFoundError(FAILURES_PATH)

    failures = pd.read_csv(FAILURES_PATH)

    if len(failures) != EXPECTED_PERSISTENT_FAILURES:
        raise RuntimeError("Unexpected persistent-failure count.")

    rows = []

    for model_slug in MODEL_ORDER:
        for behavior in BEHAVIOR_ORDER:
            for condition in CONDITIONS:
                count = int(
                    (
                        (failures["model_slug"] == model_slug)
                        & (failures["behavior"] == behavior)
                        & (failures["condition"] == condition)
                    ).sum()
                )

                rows.append(
                    {
                        "model_slug": (model_slug),
                        "behavior": (behavior),
                        "condition": (condition),
                        "num_generated": 24,
                        "persistent_nonterminations": (count),
                        "persistent_nontermination_rate": float(count / 24),
                    }
                )

    by_condition = pd.DataFrame(rows)

    excluded_triplets = int(
        failures[
            [
                "model_slug",
                "behavior",
                "prompt_id",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    overall = pd.DataFrame(
        [
            {
                "raw_generated_responses": (EXPECTED_RAW_RESPONSES),
                "persistent_nonterminations": (EXPECTED_PERSISTENT_FAILURES),
                "persistent_nontermination_rate": float(
                    EXPECTED_PERSISTENT_FAILURES / EXPECTED_RAW_RESPONSES
                ),
                "raw_matched_triplets": (EXPECTED_RAW_TRIPLETS),
                "excluded_triplets": (excluded_triplets),
                "excluded_triplet_rate": float(
                    excluded_triplets / EXPECTED_RAW_TRIPLETS
                ),
                "retained_judging_triplets": (EXPECTED_CASES),
            }
        ]
    )

    if excluded_triplets != 12:
        raise RuntimeError("Expected 12 excluded triplets.")

    return (
        failures,
        by_condition,
        overall,
    )


# =============================================================================
# Main
# =============================================================================


def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 110)
    print("OPEN-ENDED JUDGING ANALYSIS")
    print("=" * 110)

    cases = load_and_unblind()

    overall = build_overall_summary(cases)

    behavior = build_behavior_summary(cases)

    model = build_model_summary(cases)

    model_behavior = build_model_behavior_summary(cases)

    syc_domain = build_sycophancy_domain_summary(cases)

    score_distribution = build_score_distribution(cases)

    (
        failures,
        stability_condition,
        stability_overall,
    ) = build_stability_summaries()

    # ----------------------------------------------------------------
    # Persist
    # ----------------------------------------------------------------

    write_csv(
        cases,
        OUTPUT_ROOT / "unblinded_case_scores.csv",
    )

    write_csv(
        overall,
        OUTPUT_ROOT / "overall_summary.csv",
    )

    write_csv(
        behavior,
        OUTPUT_ROOT / "behavior_summary.csv",
    )

    write_csv(
        model,
        OUTPUT_ROOT / "model_summary.csv",
    )

    write_csv(
        model_behavior,
        OUTPUT_ROOT / "model_behavior_summary.csv",
    )

    write_csv(
        syc_domain,
        OUTPUT_ROOT / "sycophancy_domain_summary.csv",
    )

    write_csv(
        score_distribution,
        OUTPUT_ROOT / "score_distribution_by_behavior.csv",
    )

    write_csv(
        failures,
        OUTPUT_ROOT / "generation_stability_failures.csv",
    )

    write_csv(
        stability_condition,
        OUTPUT_ROOT / "generation_stability_by_condition.csv",
    )

    write_csv(
        stability_overall,
        OUTPUT_ROOT / "generation_stability_overall.csv",
    )

    manifest = {
        "protocol": ("open_ended_judging_analysis_v1"),
        "bootstrap_samples": int(BOOTSTRAP_SAMPLES),
        "bootstrap_seed": int(BOOTSTRAP_SEED),
        "num_judged_triplets": int(len(cases)),
        "primary_metrics": [
            "positive_score - baseline_score",
            "baseline_score - negative_score",
        ],
        "monotonic_metrics": {
            "strict": ("negative < baseline < positive"),
            "weak": ("negative <= baseline <= positive"),
        },
        "bootstrap_policy": (
            "Paired score differences are computed "
            "within each model/prompt triplet. "
            "Cross-model pooled confidence intervals "
            "use a cluster bootstrap over underlying "
            "(behavior, prompt_id) prompts so repeated "
            "evaluation of one prompt across models "
            "remains within the same resampled cluster."
        ),
        "generation_stability_policy": (
            "The 12 persistent nontermination events "
            "are reported separately. Their complete "
            "matched triplets were excluded from "
            "behavioral-content judging before the "
            "condition key was opened."
        ),
        "judge_condition_blinding_broken_only_after_judging": (True),
        "judge_results_path": str(JUDGE_RESULTS_PATH),
        "condition_key_path": str(CONDITION_KEY_PATH),
    }

    write_json(
        manifest,
        OUTPUT_ROOT / "manifest.json",
    )

    # ================================================================
    # Console report
    # ================================================================

    print()
    print("=" * 110)
    print("OVERALL")
    print("=" * 110)

    o = overall.iloc[0]

    print(f"Judged triplets:              {int(o['num_cases'])}")

    print()
    print("Mean target-behavior score:")

    print(f"  negative:                   {o['mean_negative_score']:.4f}")

    print(f"  baseline:                   {o['mean_baseline_score']:.4f}")

    print(f"  positive:                   {o['mean_positive_score']:.4f}")

    print()
    print("Directional score shifts:")

    print(
        f"  positive - baseline:        "
        f"{o['mean_positive_delta']:+.4f} "
        f"["
        f"{o['mean_positive_delta_ci_low']:+.4f}, "
        f"{o['mean_positive_delta_ci_high']:+.4f}"
        f"]"
    )

    print(
        f"  baseline - negative:        "
        f"{o['mean_negative_delta']:+.4f} "
        f"["
        f"{o['mean_negative_delta_ci_low']:+.4f}, "
        f"{o['mean_negative_delta_ci_high']:+.4f}"
        f"]"
    )

    print(
        f"  positive - negative:        "
        f"{o['mean_positive_minus_negative']:+.4f} "
        f"["
        f"{o['mean_positive_minus_negative_ci_low']:+.4f}, "
        f"{o['mean_positive_minus_negative_ci_high']:+.4f}"
        f"]"
    )

    print()
    print("Directional success:")

    print(
        f"  positive > baseline:        "
        f"{o['positive_direction_success_rate']:.4f} "
        f"["
        f"{o['positive_direction_success_rate_ci_low']:.4f}, "
        f"{o['positive_direction_success_rate_ci_high']:.4f}"
        f"]"
    )

    print(
        f"  baseline > negative:        "
        f"{o['negative_direction_success_rate']:.4f} "
        f"["
        f"{o['negative_direction_success_rate_ci_low']:.4f}, "
        f"{o['negative_direction_success_rate_ci_high']:.4f}"
        f"]"
    )

    print()
    print("Monotonic ordering:")

    print(
        f"  strict (- < 0 < +):         "
        f"{o['strict_monotonic_rate']:.4f} "
        f"["
        f"{o['strict_monotonic_rate_ci_low']:.4f}, "
        f"{o['strict_monotonic_rate_ci_high']:.4f}"
        f"]"
    )

    print(
        f"  weak   (- <= 0 <= +):       "
        f"{o['weak_monotonic_rate']:.4f} "
        f"["
        f"{o['weak_monotonic_rate_ci_low']:.4f}, "
        f"{o['weak_monotonic_rate_ci_high']:.4f}"
        f"]"
    )

    print()
    print("=" * 110)
    print("BY BEHAVIOR")
    print("=" * 110)

    print(
        behavior[
            [
                "behavior",
                "num_cases",
                "mean_negative_score",
                "mean_baseline_score",
                "mean_positive_score",
                "mean_positive_delta",
                "mean_positive_delta_ci_low",
                "mean_positive_delta_ci_high",
                "mean_negative_delta",
                "mean_negative_delta_ci_low",
                "mean_negative_delta_ci_high",
                "strict_monotonic_rate",
                "weak_monotonic_rate",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 110)
    print("BY MODEL")
    print("=" * 110)

    print(
        model[
            [
                "model_slug",
                "num_cases",
                "mean_negative_score",
                "mean_baseline_score",
                "mean_positive_score",
                "mean_positive_delta",
                "mean_negative_delta",
                "strict_monotonic_rate",
                "weak_monotonic_rate",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 110)
    print("SYCOPHANCY DOMAIN BREAKDOWN")
    print("=" * 110)

    print(
        syc_domain[
            [
                "domain",
                "num_cases",
                "mean_negative_score",
                "mean_baseline_score",
                "mean_positive_score",
                "mean_positive_delta",
                "mean_negative_delta",
                "strict_monotonic_rate",
                "weak_monotonic_rate",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 110)
    print("GENERATION STABILITY")
    print("=" * 110)

    s = stability_overall.iloc[0]

    print(
        f"Persistent nonterminations:   "
        f"{int(s['persistent_nonterminations'])}/"
        f"{int(s['raw_generated_responses'])} "
        f"("
        f"{s['persistent_nontermination_rate']:.4%}"
        f")"
    )

    print(
        f"Excluded matched triplets:    "
        f"{int(s['excluded_triplets'])}/"
        f"{int(s['raw_matched_triplets'])} "
        f"("
        f"{s['excluded_triplet_rate']:.4%}"
        f")"
    )

    nonzero_failure_rows = stability_condition[
        stability_condition["persistent_nonterminations"] > 0
    ]

    print()
    print(nonzero_failure_rows.to_string(index=False))

    print()
    print("=" * 110)
    print("ANALYSIS COMPLETE")
    print("=" * 110)

    print(f"Outputs: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
