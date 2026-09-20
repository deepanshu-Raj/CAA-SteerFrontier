"""Select high-contrast qualitative examples from the frozen judged set.

Selection is explicitly post hoc and intended for illustration only; candidate
rankings are exported to preserve the audit trail.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

BEHAVIOR_ORDER = [
    "sycophancy",
    "refusal",
    "hallucination",
]

BEHAVIOR_LABELS = {
    "sycophancy": "Sycophancy",
    "refusal": "Refusal",
    "hallucination": "Hallucination",
}

CONDITION_ORDER = [
    "negative",
    "baseline",
    "positive",
]

MODEL_LABELS = {
    "llama2-7b-chat": "Llama 2",
    "llama3.1-8b-instruct": "Llama 3.1",
    "mistral-7b-instruct-v0.3": "Mistral",
    "qwen2.5-7b-instruct": "Qwen 2.5",
}

ANALYSIS_ROOT = Path("results") / "cross_model" / "open_ended_generation" / "analysis"

JUDGING_ROOT = Path("results") / "cross_model" / "open_ended_generation" / "judging"

SCORES_PATH = ANALYSIS_ROOT / "unblinded_case_scores.csv"

RESPONSES_PATH = JUDGING_ROOT / "retained_judging_responses.csv"

OUTPUT_ROOT = (
    Path("results") / "cross_model" / "open_ended_generation" / "qualitative_examples"
)

NUM_CANDIDATES_PER_BEHAVIOR = 10


# =============================================================================
# Loading and validation
# =============================================================================


def load_inputs():
    if not SCORES_PATH.is_file():
        raise FileNotFoundError(f"Missing score file:\n{SCORES_PATH}")

    if not RESPONSES_PATH.is_file():
        raise FileNotFoundError(f"Missing retained responses:\n{RESPONSES_PATH}")

    scores = pd.read_csv(SCORES_PATH)

    responses = pd.read_csv(RESPONSES_PATH)

    required_scores = {
        "case_id",
        "model_slug",
        "behavior",
        "prompt_id",
        "domain",
        "negative_score",
        "baseline_score",
        "positive_score",
        "positive_delta",
        "negative_delta",
        "span_delta",
        "strict_monotonic",
        "weak_monotonic",
    }

    required_responses = {
        "model_slug",
        "behavior",
        "condition",
        "prompt_id",
        "domain",
        "prompt",
        "response",
        "new_token_count",
        "ended_with_eos",
        "hit_max_tokens",
    }

    missing = required_scores - set(scores.columns)

    if missing:
        raise RuntimeError(f"Score file missing columns: {sorted(missing)}")

    missing = required_responses - set(responses.columns)

    if missing:
        raise RuntimeError(f"Response file missing columns: {sorted(missing)}")

    if len(scores) != 276:
        raise RuntimeError(
            f"Expected 276 retained judged triplets, found {len(scores)}."
        )

    if len(responses) != 828:
        raise RuntimeError(f"Expected 828 retained responses, found {len(responses)}.")

    return (
        scores,
        responses,
    )


# =============================================================================
# Attach response-level metadata to each judged triplet
# =============================================================================


def build_triplet_metadata(
    responses: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    group_columns = [
        "model_slug",
        "behavior",
        "prompt_id",
        "domain",
    ]

    for key, frame in responses.groupby(
        group_columns,
        sort=False,
        dropna=False,
    ):
        (
            model_slug,
            behavior,
            prompt_id,
            domain,
        ) = key

        conditions = set(frame["condition"].astype(str))

        if conditions != set(CONDITION_ORDER):
            raise RuntimeError(
                f"Incomplete triplet for "
                f"{model_slug}/{behavior}/{prompt_id}: "
                f"{sorted(conditions)}"
            )

        prompts = frame["prompt"].astype(str).unique()

        if len(prompts) != 1:
            raise RuntimeError(
                f"Prompt mismatch within triplet {model_slug}/{behavior}/{prompt_id}."
            )

        condition_rows = {row.condition: row for row in frame.itertuples(index=False)}

        total_new_tokens = sum(
            int(condition_rows[condition].new_token_count)
            for condition in CONDITION_ORDER
        )

        rows.append(
            {
                "model_slug": (model_slug),
                "behavior": (behavior),
                "prompt_id": (prompt_id),
                "domain": (domain),
                "prompt": (prompts[0]),
                "negative_response": (condition_rows["negative"].response),
                "baseline_response": (condition_rows["baseline"].response),
                "positive_response": (condition_rows["positive"].response),
                "negative_new_tokens": int(condition_rows["negative"].new_token_count),
                "baseline_new_tokens": int(condition_rows["baseline"].new_token_count),
                "positive_new_tokens": int(condition_rows["positive"].new_token_count),
                "total_new_tokens": int(total_new_tokens),
            }
        )

    triplets = pd.DataFrame(rows)

    if len(triplets) != 276:
        raise RuntimeError(
            f"Expected 276 complete response triplets, found {len(triplets)}."
        )

    return triplets


# =============================================================================
# Candidate ranking
# =============================================================================


def rank_candidates(
    scores: pd.DataFrame,
    triplets: pd.DataFrame,
) -> pd.DataFrame:
    merged = scores.merge(
        triplets,
        on=[
            "model_slug",
            "behavior",
            "prompt_id",
            "domain",
        ],
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(scores):
        raise RuntimeError(
            "Failed to match every judged case with its response triplet."
        )

    # Both steering directions move in the intended direction relative to
    # baseline.
    merged["bidirectional_success"] = (merged["positive_delta"] > 0) & (
        merged["negative_delta"] > 0
    )

    # Useful tie-breaker: prefers examples where both sides visibly move,
    # rather than one enormous shift and one negligible shift.
    merged["minimum_directional_shift"] = merged[
        [
            "positive_delta",
            "negative_delta",
        ]
    ].min(axis=1)

    # Secondary magnitude criterion.
    merged["total_directional_shift"] = (
        merged["positive_delta"] + merged["negative_delta"]
    )

    ranked_frames = []

    for behavior in BEHAVIOR_ORDER:
        frame = merged[merged["behavior"] == behavior].copy()

        frame = frame.sort_values(
            by=[
                # Prefer clean monotonic examples first.
                "bidirectional_success",
                "strict_monotonic",
                # Then maximize the full negative-to-positive steering span.
                "span_delta",
                # Prefer examples where both sides contribute.
                "minimum_directional_shift",
                "total_directional_shift",
                # Among otherwise equivalent examples, shorter is easier to
                # present qualitatively.
                "total_new_tokens",
                # Stable deterministic final tie-break.
                "case_id",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                False,
                True,
                True,
            ],
            kind="mergesort",
        )

        frame["behavior_rank"] = range(
            1,
            len(frame) + 1,
        )

        ranked_frames.append(frame)

    ranked = pd.concat(
        ranked_frames,
        ignore_index=True,
    )

    return ranked


# =============================================================================
# Selection
# =============================================================================


def select_examples(
    ranked: pd.DataFrame,
) -> pd.DataFrame:
    selected = []

    for behavior in BEHAVIOR_ORDER:
        frame = ranked[ranked["behavior"] == behavior]

        if frame.empty:
            raise RuntimeError(f"No candidates for {behavior}.")

        selected.append(frame.iloc[0])

    return pd.DataFrame(selected).reset_index(drop=True)


# =============================================================================
# Human-readable Markdown export
# =============================================================================


def clean_text(
    value,
) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def write_markdown(
    selected: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# High-Contrast Open-Ended Steering Examples",
        "",
        (
            "These examples are selected deterministically as high-contrast "
            "qualitative demonstrations. They are not intended to represent "
            "average-case behavior."
        ),
        "",
        (
            "Ranking prioritizes correct bidirectional steering and strict "
            "monotonicity, followed by the largest negative-to-positive score "
            "span."
        ),
        "",
    ]

    for behavior in BEHAVIOR_ORDER:
        row = selected[selected["behavior"] == behavior].iloc[0]

        model = MODEL_LABELS.get(
            row["model_slug"],
            row["model_slug"],
        )

        lines.extend(
            [
                f"## {BEHAVIOR_LABELS[behavior]}",
                "",
                f"- Case: `{row['case_id']}`",
                f"- Model: {model}",
                f"- Domain: {row['domain']}",
                (
                    "- Scores: "
                    f"$S_-={int(row['negative_score'])}$, "
                    f"$S_0={int(row['baseline_score'])}$, "
                    f"$S_+={int(row['positive_score'])}$"
                ),
                (f"- Steering span: $S_+ - S_- = {row['span_delta']:.0f}$"),
                "",
                "### Prompt",
                "",
                clean_text(row["prompt"]),
                "",
                (f"### Negative steering ($S_-={int(row['negative_score'])}$)"),
                "",
                clean_text(row["negative_response"]),
                "",
                (f"### Baseline ($S_0={int(row['baseline_score'])}$)"),
                "",
                clean_text(row["baseline_response"]),
                "",
                (f"### Positive steering ($S_+={int(row['positive_score'])}$)"),
                "",
                clean_text(row["positive_response"]),
                "",
                "---",
                "",
            ]
        )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        scores,
        responses,
    ) = load_inputs()

    triplets = build_triplet_metadata(responses)

    ranked = rank_candidates(
        scores,
        triplets,
    )

    selected = select_examples(ranked)

    # -------------------------------------------------------------------------
    # Save top candidates for selection audit.
    # -------------------------------------------------------------------------

    top_candidates = ranked[
        ranked["behavior_rank"] <= NUM_CANDIDATES_PER_BEHAVIOR
    ].copy()

    candidate_columns = [
        "behavior",
        "behavior_rank",
        "case_id",
        "model_slug",
        "prompt_id",
        "domain",
        "negative_score",
        "baseline_score",
        "positive_score",
        "negative_delta",
        "positive_delta",
        "span_delta",
        "bidirectional_success",
        "strict_monotonic",
        "weak_monotonic",
        "minimum_directional_shift",
        "total_new_tokens",
        "prompt",
    ]

    candidate_path = OUTPUT_ROOT / "top_candidates.csv"

    top_candidates[candidate_columns].to_csv(
        candidate_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save the exact selected triplets including full response text.
    # -------------------------------------------------------------------------

    selected_columns = [
        "behavior",
        "case_id",
        "model_slug",
        "prompt_id",
        "domain",
        "negative_score",
        "baseline_score",
        "positive_score",
        "negative_delta",
        "positive_delta",
        "span_delta",
        "bidirectional_success",
        "strict_monotonic",
        "weak_monotonic",
        "prompt",
        "negative_response",
        "baseline_response",
        "positive_response",
        "negative_new_tokens",
        "baseline_new_tokens",
        "positive_new_tokens",
        "total_new_tokens",
        "negative_rationale",
        "baseline_rationale",
        "positive_rationale",
    ]

    selected_path = OUTPUT_ROOT / "selected_examples.csv"

    selected[selected_columns].to_csv(
        selected_path,
        index=False,
    )

    markdown_path = OUTPUT_ROOT / "selected_examples.md"

    write_markdown(
        selected,
        markdown_path,
    )

    # -------------------------------------------------------------------------
    # Console audit.
    # -------------------------------------------------------------------------

    print("=" * 110)
    print("HIGH-CONTRAST OPEN-ENDED EXAMPLE SELECTION")
    print("=" * 110)

    audit_columns = [
        "behavior",
        "case_id",
        "model_slug",
        "prompt_id",
        "domain",
        "negative_score",
        "baseline_score",
        "positive_score",
        "negative_delta",
        "positive_delta",
        "span_delta",
        "bidirectional_success",
        "strict_monotonic",
        "total_new_tokens",
    ]

    print(selected[audit_columns].to_string(index=False))

    print()
    print("Selection rule:")

    print("  1. Prefer bidirectional steering success")

    print("  2. Prefer strict S_- < S_0 < S_+ ordering")

    print("  3. Maximize S_+ - S_-")

    print("  4. Maximize the weaker directional shift")

    print("  5. Prefer shorter response triplets")

    print()
    print(f"Selected examples: {selected_path}")

    print(f"Readable version:  {markdown_path}")

    print(f"Top candidates:    {candidate_path}")


if __name__ == "__main__":
    main()
