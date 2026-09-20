"""Render quantitative summaries of blinded open-ended steering judgments.

Behavior, model, and domain panels use the frozen retained triplets; persistent
nontermination is reported separately as generation stability.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260919

CONDITION_ORDER = [
    "negative",
    "baseline",
    "positive",
]

CONDITION_LABELS = {
    "negative": "Negative\n($-$)",
    "baseline": "Baseline\n($0$)",
    "positive": "Positive\n($+$)",
}

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

# Open-ended figures use their own palette so the paper does not visually
# repeat the blue/gray and cross-model palettes used in earlier sections.
BEHAVIOR_COLORS = {
    "sycophancy": "#6C5CE7",
    "refusal": "#E17055",
    "hallucination": "#00B894",
}

MODEL_ORDER = [
    "llama2-7b-chat",
    "llama3.1-8b-instruct",
    "mistral-7b-instruct-v0.3",
    "qwen2.5-7b-instruct",
]

MODEL_LABELS = {
    "llama2-7b-chat": "Llama 2",
    "llama3.1-8b-instruct": "Llama 3.1",
    "mistral-7b-instruct-v0.3": "Mistral",
    "qwen2.5-7b-instruct": "Qwen 2.5",
}

MODEL_COLORS = {
    "llama2-7b-chat": "#4C78A8",
    "llama3.1-8b-instruct": "#F58518",
    "mistral-7b-instruct-v0.3": "#54A24B",
    "qwen2.5-7b-instruct": "#B279A2",
}

DOMAIN_ORDER = [
    "nlp",
    "political",
]

DOMAIN_LABELS = {
    "nlp": "NLP",
    "political": "Political",
}

DOMAIN_COLORS = {
    "nlp": "#5E60CE",
    "political": "#E07A5F",
}

# Small offsets prevent the two domain trajectories from occupying exactly the
# same x-coordinate when their scores are similar.
DOMAIN_X_OFFSETS = {
    "nlp": -0.035,
    "political": 0.035,
}

ROOT = Path("results") / "cross_model" / "open_ended_generation" / "analysis"

CASES_PATH = ROOT / "unblinded_case_scores.csv"

STABILITY_PATH = ROOT / "generation_stability_by_condition.csv"

OUTPUT_ROOT = Path("plots") / "open_ended_generation"


# =============================================================================
# Plot style
# =============================================================================


def configure_matplotlib() -> None:
    """
    Compact publication-oriented styling.

    Titles remain human-readable while axes retain mathematical notation for
    quantities defined explicitly in the experimental protocol.
    """

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Avenir",
                "Helvetica",
                "Arial",
                "DejaVu Sans",
            ],
            "mathtext.fontset": "dejavusans",
            "font.size": 8.4,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# =============================================================================
# Loading and validation
# =============================================================================


def load_inputs():
    if not CASES_PATH.is_file():
        raise FileNotFoundError(f"Missing judged cases:\n{CASES_PATH}")

    if not STABILITY_PATH.is_file():
        raise FileNotFoundError(f"Missing stability analysis:\n{STABILITY_PATH}")

    cases = pd.read_csv(CASES_PATH)

    stability = pd.read_csv(STABILITY_PATH)

    required_cases = {
        "case_id",
        "model_slug",
        "behavior",
        "prompt_id",
        "domain",
        "negative_score",
        "baseline_score",
        "positive_score",
    }

    required_stability = {
        "model_slug",
        "behavior",
        "condition",
        "num_generated",
        "persistent_nonterminations",
        "persistent_nontermination_rate",
    }

    missing = required_cases - set(cases.columns)

    if missing:
        raise RuntimeError(f"Unblinded cases missing columns: {sorted(missing)}")

    missing = required_stability - set(stability.columns)

    if missing:
        raise RuntimeError(
            f"Generation-stability results missing columns: {sorted(missing)}"
        )

    if len(cases) != 276:
        raise RuntimeError(
            f"Expected 276 retained judged triplets, found {len(cases)}."
        )

    return (
        cases,
        stability,
    )


# =============================================================================
# Deterministic clustered bootstrap
# =============================================================================


def stable_seed(
    *parts,
) -> int:
    """
    Derive a deterministic 32-bit seed from the analysis seed and group labels.
    """

    payload = "|".join(str(part) for part in parts).encode("utf-8")

    digest = hashlib.sha256(payload).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    ) % (2**32)


def cluster_bootstrap_mean(
    frame: pd.DataFrame,
    *,
    value_column: str,
    cluster_columns,
    seed_parts,
):
    """
    Bootstrap complete prompt clusters.

    When the same underlying prompt is evaluated across several models, those
    evaluations remain together in each resampled cluster. Within model-level
    summaries, behavior/prompt pairs form the corresponding clusters.
    """

    if frame.empty:
        raise ValueError("Cannot bootstrap an empty frame.")

    grouped = [
        group[value_column].astype(float).to_numpy()
        for _, group in frame.groupby(
            cluster_columns,
            sort=False,
            dropna=False,
        )
    ]

    group_sums = np.asarray(
        [values.sum() for values in grouped],
        dtype=float,
    )

    group_counts = np.asarray(
        [len(values) for values in grouped],
        dtype=float,
    )

    observed = float(frame[value_column].astype(float).mean())

    num_clusters = len(grouped)

    if num_clusters == 1:
        return {
            "mean": observed,
            "ci_low": observed,
            "ci_high": observed,
            "num_cases": int(len(frame)),
            "num_clusters": 1,
        }

    rng = np.random.default_rng(
        stable_seed(
            BOOTSTRAP_SEED,
            *seed_parts,
        )
    )

    bootstrap_means = np.empty(
        BOOTSTRAP_SAMPLES,
        dtype=float,
    )

    for index in range(BOOTSTRAP_SAMPLES):
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

    return {
        "mean": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "num_cases": int(len(frame)),
        "num_clusters": int(num_clusters),
    }


# =============================================================================
# Plot-facing summaries
# =============================================================================


def build_behavior_condition_summary(
    cases: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate each behavior across models while clustering by underlying prompt.
    """

    rows = []

    for behavior in BEHAVIOR_ORDER:
        frame = cases[cases["behavior"] == behavior]

        for condition in CONDITION_ORDER:
            result = cluster_bootstrap_mean(
                frame,
                value_column=(f"{condition}_score"),
                cluster_columns=[
                    "prompt_id",
                ],
                seed_parts=(
                    "behavior",
                    behavior,
                    condition,
                ),
            )

            rows.append(
                {
                    "behavior": (behavior),
                    "condition": (condition),
                    **result,
                }
            )

    return pd.DataFrame(rows)


def build_model_condition_summary(
    cases: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate across the three open-ended behaviors within each model.
    """

    rows = []

    for model_slug in MODEL_ORDER:
        frame = cases[cases["model_slug"] == model_slug]

        for condition in CONDITION_ORDER:
            result = cluster_bootstrap_mean(
                frame,
                value_column=(f"{condition}_score"),
                cluster_columns=[
                    "behavior",
                    "prompt_id",
                ],
                seed_parts=(
                    "model",
                    model_slug,
                    condition,
                ),
            )

            rows.append(
                {
                    "model_slug": (model_slug),
                    "condition": (condition),
                    **result,
                }
            )

    return pd.DataFrame(rows)


def build_domain_condition_summary(
    cases: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare open-ended sycophancy responses separately for NLP and political
    prompts.
    """

    rows = []

    sycophancy = cases[cases["behavior"] == "sycophancy"]

    for domain in DOMAIN_ORDER:
        frame = sycophancy[sycophancy["domain"] == domain]

        for condition in CONDITION_ORDER:
            result = cluster_bootstrap_mean(
                frame,
                value_column=(f"{condition}_score"),
                cluster_columns=[
                    "prompt_id",
                ],
                seed_parts=(
                    "sycophancy_domain",
                    domain,
                    condition,
                ),
            )

            rows.append(
                {
                    "domain": (domain),
                    "condition": (condition),
                    **result,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Plotting helpers
# =============================================================================


def ordered_series(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Restore the frozen negative -> baseline -> positive ordering.
    """

    return summary.set_index("condition").loc[CONDITION_ORDER].reset_index()


def add_mean_labels(
    ax,
    *,
    x: np.ndarray,
    mean: np.ndarray,
    ci_high: np.ndarray,
    color: str = "#222222",
    fontsize: float = 6.8,
    offset: float = 0.035,
) -> None:
    """
    Report the plotted mean directly above the upper 95% confidence bound.
    """

    for (
        x_value,
        mean_value,
        upper,
    ) in zip(
        x,
        mean,
        ci_high,
    ):
        ax.text(
            x_value,
            upper + offset,
            f"{mean_value:.2f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=color,
            zorder=5,
        )


def style_score_axis(
    ax,
) -> None:
    """
    Shared 0-4 judge scale. Keeping identical limits preserves visual
    comparability across behaviors, domains, and models.
    """

    ax.set_ylim(
        -0.02,
        4.05,
    )

    ax.set_yticks(
        np.arange(
            0.0,
            4.1,
            0.5,
        )
    )

    ax.grid(
        axis="y",
        alpha=0.14,
        linewidth=0.6,
        zorder=0,
    )

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)


# =============================================================================
# Plot 1: behavior-level panels
# =============================================================================


def plot_behavior_panels(
    summary: pd.DataFrame,
) -> None:
    """
    Main open-ended figure.

    Each behavior receives its own compact panel so behavior-specific response
    patterns remain visible instead of being obscured by a combined trajectory.
    """

    x = np.arange(
        len(CONDITION_ORDER),
        dtype=float,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            7.35,
            2.35,
        ),
        sharex=True,
        sharey=True,
    )

    for ax, behavior in zip(
        axes,
        BEHAVIOR_ORDER,
    ):
        frame = ordered_series(summary[summary["behavior"] == behavior])

        mean = frame["mean"].to_numpy(dtype=float)

        low = frame["ci_low"].to_numpy(dtype=float)

        high = frame["ci_high"].to_numpy(dtype=float)

        yerr = np.vstack(
            [
                mean - low,
                high - mean,
            ]
        )

        color = BEHAVIOR_COLORS[behavior]

        ax.errorbar(
            x,
            mean,
            yerr=yerr,
            marker="o",
            markersize=4.8,
            linewidth=1.35,
            elinewidth=0.9,
            capsize=2.0,
            color=color,
            alpha=0.95,
            zorder=3,
        )

        add_mean_labels(
            ax,
            x=x,
            mean=mean,
            ci_high=high,
            color=color,
            fontsize=6.8,
            offset=0.03,
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            [CONDITION_LABELS[condition] for condition in CONDITION_ORDER]
        )

        ax.set_xlim(
            -0.15,
            2.15,
        )

        ax.set_title(
            BEHAVIOR_LABELS[behavior],
            pad=4.5,
        )

        style_score_axis(ax)

    fig.suptitle(
        "Open-Ended Target-Behavior Scores",
        fontsize=10.4,
        y=0.995,
    )

    fig.supxlabel(
        "Steering condition",
        fontsize=8.8,
        y=0.015,
    )

    fig.supylabel(
        r"Target-behavior score $S$",
        fontsize=8.8,
        x=0.01,
    )

    fig.tight_layout(
        pad=0.25,
        w_pad=0.45,
    )

    pdf_path = OUTPUT_ROOT / "behavior_trajectories.pdf"

    png_path = OUTPUT_ROOT / "behavior_trajectories.png"

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {pdf_path}")

    print(f"Saved: {png_path}")


# =============================================================================
# Plot 2: sycophancy domain comparison
# =============================================================================


def plot_sycophancy_domains(
    summary: pd.DataFrame,
) -> None:
    """
    Compare free-form sycophancy steering on NLP and political prompts.
    """

    x = np.arange(
        len(CONDITION_ORDER),
        dtype=float,
    )

    fig, ax = plt.subplots(
        figsize=(
            4.15,
            2.95,
        )
    )

    for domain in DOMAIN_ORDER:
        frame = ordered_series(summary[summary["domain"] == domain])

        mean = frame["mean"].to_numpy(dtype=float)

        low = frame["ci_low"].to_numpy(dtype=float)

        high = frame["ci_high"].to_numpy(dtype=float)

        yerr = np.vstack(
            [
                mean - low,
                high - mean,
            ]
        )

        plotted_x = x + DOMAIN_X_OFFSETS[domain]

        color = DOMAIN_COLORS[domain]

        ax.errorbar(
            plotted_x,
            mean,
            yerr=yerr,
            marker="o",
            markersize=5.0,
            linewidth=1.35,
            elinewidth=0.9,
            capsize=2.0,
            color=color,
            label=DOMAIN_LABELS[domain],
            alpha=0.95,
            zorder=3,
        )

        add_mean_labels(
            ax,
            x=plotted_x,
            mean=mean,
            ci_high=high,
            color=color,
            fontsize=6.6,
            offset=0.025,
        )

    ax.set_xticks(x)

    ax.set_xticklabels([CONDITION_LABELS[condition] for condition in CONDITION_ORDER])

    ax.set_xlim(
        -0.20,
        2.20,
    )

    ax.set_xlabel(
        "Steering condition",
        labelpad=3,
    )

    ax.set_ylabel(
        r"Target-behavior score $S$",
        labelpad=3,
    )

    ax.set_title(
        "Open-Ended Sycophancy by Domain",
        pad=5,
    )

    style_score_axis(ax)

    ax.legend(
        frameon=False,
        loc="upper left",
        handletextpad=0.35,
        borderaxespad=0.25,
        labelspacing=0.25,
    )

    fig.tight_layout(pad=0.3)

    pdf_path = OUTPUT_ROOT / "sycophancy_domain_trajectories.pdf"

    png_path = OUTPUT_ROOT / "sycophancy_domain_trajectories.png"

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {pdf_path}")

    print(f"Saved: {png_path}")


# =============================================================================
# Plot 3: model-level panels
# =============================================================================


def plot_model_panels(
    summary: pd.DataFrame,
) -> None:
    """
    Supporting 2x2 model view.

    Each model receives its own panel because combining the model trajectories
    hides small but meaningful differences in their aggregate response.
    """

    x = np.arange(
        len(CONDITION_ORDER),
        dtype=float,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            5.85,
            4.15,
        ),
        sharex=True,
        sharey=True,
    )

    axes = axes.flatten()

    for ax, model_slug in zip(
        axes,
        MODEL_ORDER,
    ):
        frame = ordered_series(summary[summary["model_slug"] == model_slug])

        mean = frame["mean"].to_numpy(dtype=float)

        low = frame["ci_low"].to_numpy(dtype=float)

        high = frame["ci_high"].to_numpy(dtype=float)

        yerr = np.vstack(
            [
                mean - low,
                high - mean,
            ]
        )

        color = MODEL_COLORS[model_slug]

        ax.errorbar(
            x,
            mean,
            yerr=yerr,
            marker="o",
            markersize=4.8,
            linewidth=1.3,
            elinewidth=0.9,
            capsize=2.0,
            color=color,
            alpha=0.95,
            zorder=3,
        )

        add_mean_labels(
            ax,
            x=x,
            mean=mean,
            ci_high=high,
            color=color,
            fontsize=6.3,
            offset=0.025,
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            [CONDITION_LABELS[condition] for condition in CONDITION_ORDER]
        )

        ax.set_xlim(
            -0.15,
            2.15,
        )

        ax.set_title(
            MODEL_LABELS[model_slug],
            pad=3.5,
            fontsize=8.5,
        )

        style_score_axis(ax)

        num_cases = int(frame["num_cases"].iloc[0])

        ax.text(
            0.97,
            0.95,
            rf"$n={num_cases}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.3,
            color="#4A4A4A",
        )

    fig.suptitle(
        "Open-Ended Target-Behavior Scores",
        fontsize=10.0,
        y=0.992,
    )

    fig.supxlabel(
        "Steering condition",
        fontsize=8.6,
        y=0.02,
    )

    fig.supylabel(
        r"Target-behavior score $S$",
        fontsize=8.6,
        x=0.01,
    )

    fig.tight_layout(
        pad=0.3,
        h_pad=0.55,
        w_pad=0.4,
    )

    pdf_path = OUTPUT_ROOT / "model_trajectories.pdf"

    png_path = OUTPUT_ROOT / "model_trajectories.png"

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {pdf_path}")

    print(f"Saved: {png_path}")


# =============================================================================
# Appendix generation-stability table
# =============================================================================


def build_stability_appendix_table(
    stability: pd.DataFrame,
) -> pd.DataFrame:
    """
    Only conditions with persistent nontermination are retained because all
    remaining model/behavior/condition cells have zero failures.
    """

    table = stability[stability["persistent_nonterminations"] > 0].copy()

    table["model"] = table["model_slug"].map(MODEL_LABELS)

    table["rate_percent"] = 100.0 * table["persistent_nontermination_rate"]

    return table[
        [
            "model",
            "behavior",
            "condition",
            "num_generated",
            "persistent_nonterminations",
            "persistent_nontermination_rate",
            "rate_percent",
        ]
    ]


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    configure_matplotlib()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        cases,
        stability,
    ) = load_inputs()

    print("=" * 110)
    print("OPEN-ENDED GENERATION PLOTTING")
    print("=" * 110)

    print(f"Retained judged triplets: {len(cases)}")

    # -------------------------------------------------------------------------
    # Build bootstrap summaries for the absolute condition scores.
    # -------------------------------------------------------------------------

    behavior_summary = build_behavior_condition_summary(cases)

    model_summary = build_model_condition_summary(cases)

    domain_summary = build_domain_condition_summary(cases)

    # -------------------------------------------------------------------------
    # Generate paper-facing figures.
    # -------------------------------------------------------------------------

    plot_behavior_panels(behavior_summary)

    plot_sycophancy_domains(domain_summary)

    plot_model_panels(model_summary)

    # -------------------------------------------------------------------------
    # Persist plot-facing summaries.
    # -------------------------------------------------------------------------

    behavior_path = OUTPUT_ROOT / "behavior_condition_summary.csv"

    model_path = OUTPUT_ROOT / "model_condition_summary.csv"

    domain_path = OUTPUT_ROOT / "sycophancy_domain_condition_summary.csv"

    behavior_summary.to_csv(
        behavior_path,
        index=False,
    )

    model_summary.to_csv(
        model_path,
        index=False,
    )

    domain_summary.to_csv(
        domain_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Appendix-ready generation-stability table.
    # -------------------------------------------------------------------------

    stability_appendix = build_stability_appendix_table(stability)

    stability_path = OUTPUT_ROOT / "generation_stability_appendix_table.csv"

    stability_appendix.to_csv(
        stability_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Console audit.
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("BEHAVIOR CONDITION MEANS")
    print("=" * 110)

    print(behavior_summary.to_string(index=False))

    print()
    print("=" * 110)
    print("MODEL CONDITION MEANS")
    print("=" * 110)

    print(model_summary.to_string(index=False))

    print()
    print("=" * 110)
    print("SYCOPHANCY DOMAIN CONDITION MEANS")
    print("=" * 110)

    print(domain_summary.to_string(index=False))

    print()
    print("=" * 110)
    print("PERSISTENT NONTERMINATION CONDITIONS")
    print("=" * 110)

    print(stability_appendix.to_string(index=False))

    print()
    print("=" * 110)
    print("OPEN-ENDED PLOTTING COMPLETE")
    print("=" * 110)

    print(f"Behavior summary: {behavior_path}")

    print(f"Model summary:    {model_path}")

    print(f"Domain summary:   {domain_path}")

    print(f"Stability table:  {stability_path}")

    print(f"Plots:            {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
