"""Render paper-facing out-of-distribution generalization figures.

The script visualizes sycophancy cross-domain transfer and TruthfulQA results
from frozen analysis tables only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# =============================================================================
# Configuration
# =============================================================================

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
    "llama2-7b-chat": "#0072B2",
    "llama3.1-8b-instruct": "#E69F00",
    "mistral-7b-instruct-v0.3": "#009E73",
    "qwen2.5-7b-instruct": "#CC79A7",
}

DIRECTION_MARKERS = {
    "negative": "o",
    "positive": "s",
}

REFERENCE_COLOR = "#777777"

SYCOPHANCY_PATH = (
    Path("results")
    / "cross_model"
    / "ood_generalization"
    / "sycophancy_transfer_pairs.csv"
)

TRUTHFULQA_PATH = (
    Path("results")
    / "cross_model"
    / "ood_generalization"
    / "hallucination_truthfulqa_results.csv"
)

OUTPUT_ROOT = Path("plots") / "ood_generalization"


# =============================================================================
# Plot style
# =============================================================================


def configure_matplotlib() -> None:
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
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# =============================================================================
# Loading and validation
# =============================================================================


def load_inputs():
    if not SYCOPHANCY_PATH.is_file():
        raise FileNotFoundError(f"Missing sycophancy results:\n{SYCOPHANCY_PATH}")

    if not TRUTHFULQA_PATH.is_file():
        raise FileNotFoundError(f"Missing TruthfulQA results:\n{TRUTHFULQA_PATH}")

    sycophancy = pd.read_csv(SYCOPHANCY_PATH)

    truthfulqa = pd.read_csv(TRUTHFULQA_PATH)

    required_sycophancy = {
        "model_slug",
        "source_domain",
        "id_test_domain",
        "ood_test_domain",
        "direction",
        "id_directional_gain",
        "id_ci_low",
        "id_ci_high",
        "id_supported",
        "ood_directional_gain",
        "ood_ci_low",
        "ood_ci_high",
        "ood_supported",
    }

    required_truthfulqa = {
        "model_slug",
        "direction",
        "directional_gain",
        "directional_gain_ci_low",
        "directional_gain_ci_high",
        "direction_supported",
    }

    missing = required_sycophancy - set(sycophancy.columns)

    if missing:
        raise RuntimeError(f"Sycophancy OOD results missing columns: {sorted(missing)}")

    missing = required_truthfulqa - set(truthfulqa.columns)

    if missing:
        raise RuntimeError(f"TruthfulQA results missing columns: {sorted(missing)}")

    if len(sycophancy) != 16:
        raise RuntimeError(
            f"Expected 16 sycophancy transfer pairs, found {len(sycophancy)}."
        )

    if len(truthfulqa) != 8:
        raise RuntimeError(f"Expected 8 TruthfulQA results, found {len(truthfulqa)}.")

    return (
        sycophancy,
        truthfulqa,
    )


# =============================================================================
# Shared helpers
# =============================================================================


def padded_limits(
    values: np.ndarray,
    *,
    include_zero: bool = True,
    padding_fraction: float = 0.10,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[np.isfinite(values)]

    if include_zero:
        values = np.concatenate(
            [
                values,
                np.asarray([0.0]),
            ]
        )

    lower = float(values.min())

    upper = float(values.max())

    span = max(
        upper - lower,
        1e-6,
    )

    return (
        lower - padding_fraction * span,
        upper + padding_fraction * span,
    )


def build_model_legend():
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=MODEL_COLORS[model_slug],
            markeredgecolor="none",
            markersize=6,
            label=MODEL_LABELS[model_slug],
        )
        for model_slug in MODEL_ORDER
    ]


def build_direction_legend():
    return [
        Line2D(
            [0],
            [0],
            marker=DIRECTION_MARKERS["negative"],
            linestyle="none",
            markerfacecolor="#777777",
            markeredgecolor="none",
            markersize=6,
            label="Negative steering",
        ),
        Line2D(
            [0],
            [0],
            marker=DIRECTION_MARKERS["positive"],
            linestyle="none",
            markerfacecolor="#777777",
            markeredgecolor="none",
            markersize=6,
            label="Positive steering",
        ),
    ]


# =============================================================================
# Sycophancy domain transfer
# =============================================================================


def get_shared_sycophancy_limits(
    frame: pd.DataFrame,
):
    """
    Both transfer plots use identical x/y limits so they remain directly
    comparable when placed side-by-side in the manuscript.
    """

    values = np.concatenate(
        [
            frame["id_ci_low"].to_numpy(dtype=float),
            frame["id_ci_high"].to_numpy(dtype=float),
            frame["ood_ci_low"].to_numpy(dtype=float),
            frame["ood_ci_high"].to_numpy(dtype=float),
        ]
    )

    return padded_limits(
        values,
        include_zero=True,
        padding_fraction=0.10,
    )


def plot_sycophancy_domain_transfer(
    frame: pd.DataFrame,
    *,
    source_domain: str,
    target_domain: str,
    title: str,
    output_name: str,
    limits,
) -> pd.DataFrame:
    subset = frame[
        (frame["source_domain"] == source_domain)
        & (frame["ood_test_domain"] == target_domain)
    ].copy()

    if len(subset) != 8:
        raise RuntimeError(
            f"{source_domain}->{target_domain}: expected 8 rows, found {len(subset)}."
        )

    fig, ax = plt.subplots(
        figsize=(
            4.45,
            3.8,
        )
    )

    for row in subset.itertuples(index=False):
        color = MODEL_COLORS[row.model_slug]

        marker = DIRECTION_MARKERS[row.direction]

        xerr = np.asarray(
            [
                [row.id_directional_gain - row.id_ci_low],
                [row.id_ci_high - row.id_directional_gain],
            ]
        )

        yerr = np.asarray(
            [
                [row.ood_directional_gain - row.ood_ci_low],
                [row.ood_ci_high - row.ood_directional_gain],
            ]
        )

        ax.errorbar(
            row.id_directional_gain,
            row.ood_directional_gain,
            xerr=xerr,
            yerr=yerr,
            fmt=marker,
            markersize=6.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            ecolor=color,
            elinewidth=0.9,
            capsize=2.2,
            alpha=0.88,
            zorder=3,
        )

    lower, upper = limits

    ax.plot(
        [
            lower,
            upper,
        ],
        [
            lower,
            upper,
        ],
        linestyle="--",
        linewidth=1.0,
        color="#333333",
        alpha=0.80,
        zorder=1,
    )

    ax.axhline(
        0.0,
        color=REFERENCE_COLOR,
        linewidth=0.8,
        alpha=0.65,
        zorder=1,
    )

    ax.axvline(
        0.0,
        color=REFERENCE_COLOR,
        linewidth=0.8,
        alpha=0.65,
        zorder=1,
    )

    ax.set_xlim(
        lower,
        upper,
    )

    ax.set_ylim(
        lower,
        upper,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel(r"In-domain gain $G_{\mathrm{ID}}$")

    ax.set_ylabel(r"OOD gain $G_{\mathrm{OOD}}$")

    ax.set_title(
        title,
        pad=8,
    )

    positive_count = int((subset["ood_directional_gain"] > 0).sum())

    supported_count = int(subset["ood_supported"].astype(bool).sum())

    annotation = (
        rf"$G_{{\mathrm{{OOD}}}}>0$: "
        f"{positive_count}/{len(subset)}"
        "\n"
        rf"95% CI-supported: "
        f"{supported_count}/{len(subset)}"
    )

    ax.text(
        0.97,
        0.96,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.4,
        linespacing=1.25,
    )

    ax.grid(
        alpha=0.15,
        linewidth=0.7,
        zorder=0,
    )

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    model_legend = ax.legend(
        handles=build_model_legend(),
        frameon=False,
        loc="upper left",
        title="Model",
        title_fontsize=7.4,
        fontsize=6.9,
        handletextpad=0.35,
        labelspacing=0.22,
        borderaxespad=0.4,
    )

    ax.add_artist(model_legend)

    ax.legend(
        handles=build_direction_legend(),
        frameon=False,
        loc="lower right",
        fontsize=6.9,
        handletextpad=0.4,
        labelspacing=0.25,
    )

    fig.tight_layout(pad=0.7)

    pdf_path = OUTPUT_ROOT / f"{output_name}.pdf"

    png_path = OUTPUT_ROOT / f"{output_name}.png"

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

    return pd.DataFrame(
        [
            {
                "source_domain": (source_domain),
                "target_domain": (target_domain),
                "num_pairs": int(len(subset)),
                "mean_id_gain": float(subset["id_directional_gain"].mean()),
                "mean_ood_gain": float(subset["ood_directional_gain"].mean()),
                "mean_ood_minus_id": float(
                    (
                        subset["ood_directional_gain"] - subset["id_directional_gain"]
                    ).mean()
                ),
                "positive_ood_gain_count": (positive_count),
                "supported_ood_gain_count": (supported_count),
            }
        ]
    )


# =============================================================================
# TruthfulQA hallucination transfer
# =============================================================================


def plot_truthfulqa(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for model_index, model_slug in enumerate(MODEL_ORDER):
        for direction in [
            "negative",
            "positive",
        ]:
            matched = frame[
                (frame["model_slug"] == model_slug) & (frame["direction"] == direction)
            ]

            if len(matched) != 1:
                raise RuntimeError(
                    f"{model_slug}/{direction}: expected exactly one TruthfulQA row."
                )

            row = matched.iloc[0]

            rows.append(
                {
                    "model_slug": (model_slug),
                    "model_index": (model_index),
                    "direction": (direction),
                    "gain": float(row["directional_gain"]),
                    "ci_low": float(row["directional_gain_ci_low"]),
                    "ci_high": float(row["directional_gain_ci_high"]),
                    "supported": bool(row["direction_supported"]),
                }
            )

    plot_frame = pd.DataFrame(rows)

    # Compact categorical figure: four model positions do not need the width of
    # a continuous scatter plot.
    fig, ax = plt.subplots(
        figsize=(
            4.25,
            3.35,
        )
    )

    base_x = np.arange(
        len(MODEL_ORDER),
        dtype=float,
    )

    offsets = {
        "negative": -0.10,
        "positive": 0.10,
    }

    for model_index, model_slug in enumerate(MODEL_ORDER):
        color = MODEL_COLORS[model_slug]

        for direction in [
            "negative",
            "positive",
        ]:
            row = plot_frame[
                (plot_frame["model_slug"] == model_slug)
                & (plot_frame["direction"] == direction)
            ].iloc[0]

            x = model_index + offsets[direction]

            yerr = np.asarray(
                [
                    [row["gain"] - row["ci_low"]],
                    [row["ci_high"] - row["gain"]],
                ]
            )

            ax.errorbar(
                x,
                row["gain"],
                yerr=yerr,
                fmt=DIRECTION_MARKERS[direction],
                markersize=6.2,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.5,
                ecolor=color,
                elinewidth=1.0,
                capsize=2.3,
                alpha=0.92,
                zorder=3,
            )

    ax.axhline(
        0.0,
        color=REFERENCE_COLOR,
        linewidth=0.8,
        alpha=0.75,
        zorder=1,
    )

    ax.set_xticks(base_x)

    ax.set_xticklabels([MODEL_LABELS[model_slug] for model_slug in MODEL_ORDER])

    # Tight categorical limits remove unused whitespace beyond the end models.
    ax.set_xlim(
        -0.32,
        len(MODEL_ORDER) - 1 + 0.32,
    )

    ax.set_ylabel(
        r"TruthfulQA directional gain "
        r"$G_{\mathrm{TQA}}$"
    )

    ax.set_title(
        "Hallucination Steering Transfers to TruthfulQA",
        pad=24,
    )

    y_lower = float(plot_frame["ci_low"].min())

    y_upper = float(plot_frame["ci_high"].max())

    y_span = max(
        y_upper - y_lower,
        1e-6,
    )

    ax.set_ylim(
        y_lower - 0.10 * y_span,
        y_upper + 0.25 * y_span,
    )

    positive = int((plot_frame["gain"] > 0).sum())

    supported = int(plot_frame["supported"].sum())

    annotation = (
        rf"$G_{{\mathrm{{TQA}}}}>0$: "
        f"{positive}/{len(plot_frame)}"
        "\n"
        rf"95% CI-supported: "
        f"{supported}/{len(plot_frame)}"
    )

    ax.text(
        0.98,
        0.94,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        linespacing=1.2,
    )

    ax.legend(
        handles=build_direction_legend(),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            1.13,
        ),
        ncols=2,
        fontsize=7.0,
        handletextpad=0.35,
        columnspacing=1.0,
        borderaxespad=0.0,
    )

    ax.grid(
        axis="y",
        alpha=0.16,
        linewidth=0.7,
        zorder=0,
    )

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.7)

    pdf_path = OUTPUT_ROOT / "hallucination_truthfulqa.pdf"

    png_path = OUTPUT_ROOT / "hallucination_truthfulqa.png"

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

    return plot_frame


# =============================================================================
# Appendix-facing tables
# =============================================================================


def build_sycophancy_appendix_table(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    table = frame[
        [
            "model_slug",
            "source_domain",
            "ood_test_domain",
            "direction",
            "id_directional_gain",
            "id_ci_low",
            "id_ci_high",
            "id_supported",
            "ood_directional_gain",
            "ood_ci_low",
            "ood_ci_high",
            "ood_supported",
        ]
    ].copy()

    table["model"] = table["model_slug"].map(MODEL_LABELS)

    table["transfer"] = (
        table["source_domain"].str.upper()
        + " -> "
        + table["ood_test_domain"].str.capitalize()
    )

    return table[
        [
            "model",
            "transfer",
            "direction",
            "id_directional_gain",
            "id_ci_low",
            "id_ci_high",
            "id_supported",
            "ood_directional_gain",
            "ood_ci_low",
            "ood_ci_high",
            "ood_supported",
        ]
    ]


def build_truthfulqa_appendix_table(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    table = frame[
        [
            "model_slug",
            "direction",
            "directional_gain",
            "directional_gain_ci_low",
            "directional_gain_ci_high",
            "direction_supported",
        ]
    ].copy()

    table["model"] = table["model_slug"].map(MODEL_LABELS)

    return table[
        [
            "model",
            "direction",
            "directional_gain",
            "directional_gain_ci_low",
            "directional_gain_ci_high",
            "direction_supported",
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
        sycophancy,
        truthfulqa,
    ) = load_inputs()

    print("=" * 110)
    print("OOD GENERALIZATION PLOTTING")
    print("=" * 110)

    print(f"Sycophancy transfer pairs: {len(sycophancy)}")

    print(f"TruthfulQA directions:      {len(truthfulqa)}")

    shared_limits = get_shared_sycophancy_limits(sycophancy)

    nlp_to_political = plot_sycophancy_domain_transfer(
        sycophancy,
        source_domain="nlp",
        target_domain="political",
        title="Sycophancy: NLP → Political",
        output_name=("sycophancy_nlp_to_political"),
        limits=shared_limits,
    )

    political_to_nlp = plot_sycophancy_domain_transfer(
        sycophancy,
        source_domain="political",
        target_domain="nlp",
        title="Sycophancy: Political → NLP",
        output_name=("sycophancy_political_to_nlp"),
        limits=shared_limits,
    )

    syc_summary = pd.concat(
        [
            nlp_to_political,
            political_to_nlp,
        ],
        ignore_index=True,
    )

    truthfulqa_summary = plot_truthfulqa(truthfulqa)

    # Plot-facing summaries.
    syc_summary_path = OUTPUT_ROOT / "sycophancy_plot_summary.csv"

    truthfulqa_summary_path = OUTPUT_ROOT / "truthfulqa_plot_summary.csv"

    syc_summary.to_csv(
        syc_summary_path,
        index=False,
    )

    truthfulqa_summary.to_csv(
        truthfulqa_summary_path,
        index=False,
    )

    # Exact point-level values retained for appendix tables.
    syc_appendix = build_sycophancy_appendix_table(sycophancy)

    truth_appendix = build_truthfulqa_appendix_table(truthfulqa)

    syc_appendix_path = OUTPUT_ROOT / "sycophancy_appendix_table.csv"

    truth_appendix_path = OUTPUT_ROOT / "truthfulqa_appendix_table.csv"

    syc_appendix.to_csv(
        syc_appendix_path,
        index=False,
    )

    truth_appendix.to_csv(
        truth_appendix_path,
        index=False,
    )

    print()
    print("=" * 110)
    print("SYCOPHANCY SUMMARY")
    print("=" * 110)

    print(syc_summary.to_string(index=False))

    print()
    print("=" * 110)
    print("TRUTHFULQA SUMMARY")
    print("=" * 110)

    print(
        truthfulqa_summary[
            [
                "model_slug",
                "direction",
                "gain",
                "ci_low",
                "ci_high",
                "supported",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 110)
    print("OOD GENERALIZATION PLOTTING COMPLETE")
    print("=" * 110)

    print(f"Sycophancy summary: {syc_summary_path}")

    print(f"TruthfulQA summary:  {truthfulqa_summary_path}")

    print(f"Sycophancy appendix: {syc_appendix_path}")

    print(f"TruthfulQA appendix:  {truth_appendix_path}")

    print(f"Plots:                {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
