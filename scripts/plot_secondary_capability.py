"""Render cross-model secondary-capability degradation figures.

GSM8K capability loss and WikiText-2 NLL increase are related to the frozen
residual-relative perturbation magnitude without re-fitting interventions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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

MODEL_LABELS = {
    "llama2-7b-chat": "Llama 2",
    "llama3.1-8b-instruct": "Llama 3.1",
    "mistral-7b-instruct-v0.3": "Mistral",
    "qwen2.5-7b-instruct": "Qwen 2.5",
}

# Fixed model identity palette used throughout cross-model figures.
MODEL_COLORS = {
    "llama2-7b-chat": "#0072B2",
    "llama3.1-8b-instruct": "#E69F00",
    "mistral-7b-instruct-v0.3": "#009E73",
    "qwen2.5-7b-instruct": "#CC79A7",
}

TREND_COLOR = "#333333"
REFERENCE_COLOR = "#777777"

GSM8K_PATH = (
    Path("results")
    / "cross_model"
    / "secondary_capability"
    / "gsm8k"
    / "postprocessed"
    / "gsm8k_summary_corrected.csv"
)

PERPLEXITY_PATH = (
    Path("results")
    / "cross_model"
    / "secondary_capability"
    / "perplexity"
    / "combined_results.csv"
)

OUTPUT_ROOT = Path("plots") / "secondary_capability"


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
            "legend.fontsize": 7.8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# =============================================================================
# Loading and preparation
# =============================================================================


def load_inputs():
    if not GSM8K_PATH.is_file():
        raise FileNotFoundError(f"Missing GSM8K results:\n{GSM8K_PATH}")

    if not PERPLEXITY_PATH.is_file():
        raise FileNotFoundError(f"Missing perplexity results:\n{PERPLEXITY_PATH}")

    gsm8k = pd.read_csv(GSM8K_PATH)

    perplexity = pd.read_csv(PERPLEXITY_PATH)

    required_gsm8k = {
        "model_slug",
        "behavior",
        "direction",
        "eta_star",
        "frozen_baseline_flexible_accuracy",
        "frozen_flexible_delta_accuracy",
    }

    required_perplexity = {
        "model_slug",
        "behavior",
        "direction",
        "eta_star",
        "abs_eta",
        "delta_nll",
    }

    missing = required_gsm8k - set(gsm8k.columns)

    if missing:
        raise RuntimeError(f"GSM8K results missing columns: {sorted(missing)}")

    missing = required_perplexity - set(perplexity.columns)

    if missing:
        raise RuntimeError(f"Perplexity results missing columns: {sorted(missing)}")

    gsm8k = gsm8k.copy()

    gsm8k["abs_eta"] = gsm8k["eta_star"].astype(float).abs()

    # Positive values always mean degradation.
    gsm8k["capability_loss"] = -gsm8k["frozen_flexible_delta_accuracy"].astype(float)

    perplexity = perplexity.copy()

    perplexity["abs_eta"] = perplexity["abs_eta"].astype(float)

    return (
        gsm8k,
        perplexity,
    )


# =============================================================================
# Association statistics
# =============================================================================


def pearson_correlation(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    if len(x) < 2:
        return float("nan")

    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )


def spearman_correlation(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    if len(x) < 2:
        return float("nan")

    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)

    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)

    return pearson_correlation(
        x_rank,
        y_rank,
    )


def association_summary(
    frame: pd.DataFrame,
    *,
    y_column: str,
) -> dict:
    x = frame["abs_eta"].to_numpy(dtype=float)

    y = frame[y_column].to_numpy(dtype=float)

    return {
        "n": int(len(frame)),
        "pearson_r": (
            pearson_correlation(
                x,
                y,
            )
        ),
        "spearman_rho": (
            spearman_correlation(
                x,
                y,
            )
        ),
    }


def build_summary(
    frame: pd.DataFrame,
    *,
    metric: str,
    y_column: str,
) -> pd.DataFrame:
    rows = []

    overall = association_summary(
        frame,
        y_column=y_column,
    )

    rows.append(
        {
            "metric": metric,
            "scope": "global",
            "model_slug": "ALL",
            **overall,
        }
    )

    for model_slug in MODEL_ORDER:
        model_frame = frame[frame["model_slug"] == model_slug]

        stat = association_summary(
            model_frame,
            y_column=y_column,
        )

        rows.append(
            {
                "metric": metric,
                "scope": "model",
                "model_slug": model_slug,
                **stat,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Plot helpers
# =============================================================================


def padded_limits(
    values: np.ndarray,
    *,
    include_zero: bool = True,
    padding_fraction: float = 0.08,
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


def add_linear_guide(
    ax,
    *,
    x: np.ndarray,
    y: np.ndarray,
) -> None:
    """
    Least-squares trend shown only as a visual guide. Pearson and Spearman
    associations are reported separately from the plotted line.
    """

    finite = np.isfinite(x) & np.isfinite(y)

    x = x[finite]

    y = y[finite]

    if len(x) < 2:
        return

    slope, intercept = np.polyfit(
        x,
        y,
        deg=1,
    )

    x_line = np.linspace(
        float(x.min()),
        float(x.max()),
        200,
    )

    y_line = slope * x_line + intercept

    ax.plot(
        x_line,
        y_line,
        linestyle="--",
        linewidth=1.15,
        color=TREND_COLOR,
        alpha=0.90,
        zorder=3,
    )


def plot_cross_model(
    frame: pd.DataFrame,
    *,
    y_column: str,
    title: str,
    y_label: str,
    output_name: str,
) -> None:
    x = frame["abs_eta"].to_numpy(dtype=float)

    y = frame[y_column].to_numpy(dtype=float)

    stat = association_summary(
        frame,
        y_column=y_column,
    )

    x_limits = padded_limits(
        x,
        include_zero=True,
        padding_fraction=0.05,
    )

    y_limits = padded_limits(
        y,
        include_zero=True,
        padding_fraction=0.08,
    )

    fig, ax = plt.subplots(
        figsize=(
            4.8,
            3.65,
        )
    )

    for model_slug in MODEL_ORDER:
        model_frame = frame[frame["model_slug"] == model_slug]

        ax.scatter(
            model_frame["abs_eta"],
            model_frame[y_column],
            s=32,
            color=MODEL_COLORS[model_slug],
            alpha=0.78,
            edgecolors="white",
            linewidths=0.4,
            label=MODEL_LABELS[model_slug],
            zorder=2,
        )

    add_linear_guide(
        ax,
        x=x,
        y=y,
    )

    ax.axhline(
        0.0,
        color=REFERENCE_COLOR,
        linewidth=0.8,
        alpha=0.70,
        zorder=1,
    )

    ax.set_xlim(*x_limits)

    ax.set_ylim(*y_limits)

    ax.set_xlabel(r"Relative perturbation magnitude $|\eta|$")

    ax.set_ylabel(y_label)

    ax.set_title(
        title,
        pad=8,
    )

    annotation = (
        rf"$r={stat['pearson_r']:.3f}$"
        "\n"
        rf"$\rho_s={stat['spearman_rho']:.3f}$"
        "\n"
        rf"$n={stat['n']}$"
    )

    ax.text(
        0.97,
        0.96,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.9,
        linespacing=1.25,
    )

    ax.grid(
        alpha=0.16,
        linewidth=0.7,
        zorder=0,
    )

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        loc="upper left",
        fontsize=7.4,
        handletextpad=0.45,
        labelspacing=0.35,
        borderaxespad=0.4,
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
        gsm8k,
        perplexity,
    ) = load_inputs()

    print("=" * 110)
    print("SECONDARY CAPABILITY PLOTTING")
    print("=" * 110)

    print(f"GSM8K intervention points: {len(gsm8k)}")

    print(f"Perplexity intervention points: {len(perplexity)}")

    if len(gsm8k) != 56:
        raise RuntimeError(f"Expected 56 GSM8K points, found {len(gsm8k)}.")

    if len(perplexity) != 56:
        raise RuntimeError(f"Expected 56 perplexity points, found {len(perplexity)}.")

    plot_cross_model(
        gsm8k,
        y_column="capability_loss",
        title=("GSM8K Degradation vs. Perturbation Magnitude"),
        y_label=(
            r"GSM8K capability loss "
            r"$(-\Delta A_{\mathrm{GSM8K}})$"
        ),
        output_name=("gsm8k_cross_model"),
    )

    plot_cross_model(
        perplexity,
        y_column="delta_nll",
        title=("Language-Model Degradation vs. Perturbation Magnitude"),
        y_label=(r"NLL increase $(\Delta\mathrm{NLL})$"),
        output_name=("perplexity_cross_model"),
    )

    gsm8k_summary = build_summary(
        gsm8k,
        metric=("gsm8k_capability_loss"),
        y_column=("capability_loss"),
    )

    perplexity_summary = build_summary(
        perplexity,
        metric="delta_nll",
        y_column="delta_nll",
    )

    summary = pd.concat(
        [
            gsm8k_summary,
            perplexity_summary,
        ],
        ignore_index=True,
    )

    summary_path = OUTPUT_ROOT / "association_plot_summary.csv"

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 110)
    print("ASSOCIATION SUMMARY")
    print("=" * 110)

    print(
        summary[
            [
                "metric",
                "scope",
                "model_slug",
                "n",
                "pearson_r",
                "spearman_rho",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 110)
    print("SECONDARY CAPABILITY PLOTTING COMPLETE")
    print("=" * 110)

    print(f"Summary: {summary_path}")

    print(f"Plots:   {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
