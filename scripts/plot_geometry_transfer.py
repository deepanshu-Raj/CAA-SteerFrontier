"""Render geometry-versus-transfer and transfer-by-depth figures.

Plots consume the frozen causal-transfer association tables and expose signed
Spearman relationships without introducing additional statistical selection.
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

MODEL_OUTPUT_NAMES = {
    "llama2-7b-chat": "llama2_signed_transfer",
    "llama3.1-8b-instruct": "llama3.1_signed_transfer",
    "mistral-7b-instruct-v0.3": "mistral_signed_transfer",
    "qwen2.5-7b-instruct": "qwen_signed_transfer",
}

PAIRWISE_PATH = (
    Path("results")
    / "cross_model"
    / "causal_transfer"
    / "association"
    / "pairwise_transfer_summary.csv"
)

ASSOCIATION_PATH = (
    Path("results")
    / "cross_model"
    / "causal_transfer"
    / "association"
    / "association_summary.csv"
)

LAYER_SUMMARY_PATH = (
    Path("results") / "cross_model" / "causal_transfer" / "transfer_layer_summary.csv"
)

OUTPUT_ROOT = Path("plots") / "geometry_transfer"

POINT_COLOR = "#4C72B0"
TREND_COLOR = "#333333"
REFERENCE_COLOR = "#777777"

MODEL_COLORS = {
    "llama2-7b-chat": "#365F91",
    "llama3.1-8b-instruct": "#4C72B0",
    "mistral-7b-instruct-v0.3": "#7A98C5",
    "qwen2.5-7b-instruct": "#A8B9D4",
}

OVERALL_COLOR = "#222222"


# =============================================================================
# Plot style
# =============================================================================


def configure_matplotlib() -> None:
    """
    Keep titles human-readable while using mathematical notation for quantities
    whose definitions are fixed by the experimental protocol.
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
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.0,
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
    for path in [
        PAIRWISE_PATH,
        ASSOCIATION_PATH,
        LAYER_SUMMARY_PATH,
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file:\n{path}")

    pairwise = pd.read_csv(PAIRWISE_PATH)

    association = pd.read_csv(ASSOCIATION_PATH)

    layer_summary = pd.read_csv(LAYER_SUMMARY_PATH)

    required_pairwise = {
        "model_slug",
        "requested_fractional_depth",
        "layer",
        "behavior_i",
        "behavior_j",
        "cosine_similarity",
        "transfer_i_to_j",
        "transfer_j_to_i",
    }

    required_association = {
        "scope",
        "scope_value",
        "variant",
        "metric",
        "n_blocks",
        "relations_per_block",
        "observed_statistic",
        "exact_permutation_p_two_sided",
    }

    required_layer = {
        "model_slug",
        "requested_fractional_depth",
        "fractional_depth",
        "mean_abs_transfer",
    }

    missing = required_pairwise - set(pairwise.columns)

    if missing:
        raise RuntimeError(f"pairwise_transfer_summary.csv missing {sorted(missing)}")

    missing = required_association - set(association.columns)

    if missing:
        raise RuntimeError(f"association_summary.csv missing {sorted(missing)}")

    missing = required_layer - set(layer_summary.columns)

    if missing:
        raise RuntimeError(f"transfer_layer_summary.csv missing {sorted(missing)}")

    return (
        pairwise,
        association,
        layer_summary,
    )


# =============================================================================
# Ordered causal-transfer representation
# =============================================================================


def build_ordered_transfer_table(
    pairwise: pd.DataFrame,
) -> pd.DataFrame:
    """
    Steering-vector cosine similarity is symmetric, but causal transfer is not.

    Each unordered behavior pair therefore contributes two observations:
        i -> j
        j -> i

    Both directions share cos(v_i, v_j), while their transfer values remain
    distinct.
    """

    rows = []

    for row in pairwise.itertuples(index=False):
        common = {
            "model_slug": (row.model_slug),
            "requested_fractional_depth": float(row.requested_fractional_depth),
            "layer": int(row.layer),
            "cosine_similarity": float(row.cosine_similarity),
        }

        rows.append(
            {
                **common,
                "source_behavior": (row.behavior_i),
                "target_behavior": (row.behavior_j),
                "signed_transfer": float(row.transfer_i_to_j),
            }
        )

        rows.append(
            {
                **common,
                "source_behavior": (row.behavior_j),
                "target_behavior": (row.behavior_i),
                "signed_transfer": float(row.transfer_j_to_i),
            }
        )

    ordered = pd.DataFrame(rows)

    expected = 2 * len(pairwise)

    if len(ordered) != expected:
        raise RuntimeError(
            "Ordered transfer expansion produced an unexpected number of rows."
        )

    return ordered


# =============================================================================
# Frozen association statistics
# =============================================================================


def get_signed_spearman(
    association: pd.DataFrame,
    *,
    scope: str,
    scope_value: str,
):
    matched = association[
        (association["scope"] == scope)
        & (association["scope_value"].astype(str) == str(scope_value))
        & (association["variant"] == "ordered")
        & (association["metric"] == "signed_spearman")
    ]

    if len(matched) != 1:
        raise RuntimeError(
            "Expected exactly one signed "
            "Spearman row for "
            f"{scope}={scope_value}, "
            f"found {len(matched)}."
        )

    row = matched.iloc[0]

    return {
        "rho": float(row["observed_statistic"]),
        "p": float(row["exact_permutation_p_two_sided"]),
        "n_blocks": int(row["n_blocks"]),
        "relations_per_block": int(row["relations_per_block"]),
    }


def format_p_value_math(
    value: float,
) -> str:
    """
    Return publication-style math text for permutation p-values.
    """

    if value < 0.001:
        exponent = int(np.floor(np.log10(value)))

        coefficient = value / (10.0**exponent)

        return (
            f"{coefficient:.2f}"
            rf"\times10^{{{exponent}}}"
        )

    return f"{value:.3f}"


# =============================================================================
# Scatter plotting
# =============================================================================


def add_linear_guide(
    ax,
    *,
    x: np.ndarray,
    y: np.ndarray,
) -> None:
    """
    Add a least-squares trend line purely as a visual guide.

    The statistical result reported on the plot remains the frozen signed
    Spearman association with its exact permutation p-value.
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


def plot_signed_transfer(
    frame: pd.DataFrame,
    *,
    title: str,
    statistic: dict,
    output_name: str,
) -> None:
    x = frame["cosine_similarity"].to_numpy(dtype=float)

    y = frame["signed_transfer"].to_numpy(dtype=float)

    fig, ax = plt.subplots(
        figsize=(
            4.65,
            3.65,
        )
    )

    ax.scatter(
        x,
        y,
        s=18,
        color=POINT_COLOR,
        alpha=0.38,
        linewidths=0,
        rasterized=False,
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

    ax.set_xlabel(
        r"Steering-vector similarity "
        r"$\cos(\mathbf{v}_i,\mathbf{v}_j)$"
    )

    ax.set_ylabel(r"Causal transfer $T_{i\rightarrow j}$")

    ax.set_title(
        title,
        pad=8,
    )

    p_text = format_p_value_math(statistic["p"])

    annotation = (
        rf"$\rho_s={statistic['rho']:.3f}$"
        "\n"
        rf"$p_{{\mathrm{{perm}}}}={p_text}$"
    )

    ax.text(
        0.97,
        0.96,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.0,
        linespacing=1.3,
    )

    ax.grid(
        alpha=0.16,
        linewidth=0.7,
        zorder=0,
    )

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

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
# Depth summary
# =============================================================================


def build_depth_summary(
    layer_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        depth,
        frame,
    ) in layer_summary.groupby(
        "requested_fractional_depth",
        sort=True,
    ):
        rows.append(
            {
                "requested_fractional_depth": float(depth),
                "mean_abs_transfer_overall": float(frame["mean_abs_transfer"].mean()),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("requested_fractional_depth")
        .reset_index(drop=True)
    )


def plot_transfer_by_depth(
    layer_summary: pd.DataFrame,
) -> pd.DataFrame:
    overall = build_depth_summary(layer_summary)

    fig, ax = plt.subplots(
        figsize=(
            5.0,
            3.55,
        )
    )

    for model_slug in MODEL_ORDER:
        frame = layer_summary[layer_summary["model_slug"] == model_slug].sort_values(
            "requested_fractional_depth"
        )

        ax.plot(
            frame["requested_fractional_depth"],
            frame["mean_abs_transfer"],
            marker="o",
            markersize=4.0,
            linewidth=1.2,
            color=MODEL_COLORS[model_slug],
            label=MODEL_LABELS[model_slug],
            alpha=0.95,
        )

    ax.plot(
        overall["requested_fractional_depth"],
        overall["mean_abs_transfer_overall"],
        marker="o",
        markersize=4.5,
        linewidth=1.7,
        linestyle="--",
        color=OVERALL_COLOR,
        label="Overall",
        zorder=5,
    )

    ax.set_xlabel(r"Fractional depth $d=\ell/(L-1)$")

    ax.set_ylabel(
        r"Mean transfer magnitude "
        r"$\mathbb{E}[|T_{i\rightarrow j}|]$"
    )

    ax.set_title(
        "Causal Transfer Across Model Depth",
        pad=8,
    )

    ax.set_xticks(overall["requested_fractional_depth"])

    ax.set_xticklabels(
        [f"{value:.2f}" for value in overall["requested_fractional_depth"]]
    )

    ax.grid(
        axis="y",
        alpha=0.18,
        linewidth=0.7,
    )

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        ncols=2,
        loc="upper right",
        columnspacing=1.1,
        handlelength=1.8,
    )

    fig.tight_layout(pad=0.7)

    pdf_path = OUTPUT_ROOT / "transfer_by_depth.pdf"

    png_path = OUTPUT_ROOT / "transfer_by_depth.png"

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

    return overall


# =============================================================================
# Plot-facing summary
# =============================================================================


def build_plot_summary(
    ordered: pd.DataFrame,
    association: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    global_stat = get_signed_spearman(
        association,
        scope="global",
        scope_value="all",
    )

    rows.append(
        {
            "scope": "global",
            "scope_value": "all",
            "num_ordered_points": int(len(ordered)),
            **global_stat,
        }
    )

    for model_slug in MODEL_ORDER:
        frame = ordered[ordered["model_slug"] == model_slug]

        stat = get_signed_spearman(
            association,
            scope="model",
            scope_value=model_slug,
        )

        rows.append(
            {
                "scope": "model",
                "scope_value": (model_slug),
                "num_ordered_points": int(len(frame)),
                **stat,
            }
        )

    return pd.DataFrame(rows)


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
        pairwise,
        association,
        layer_summary,
    ) = load_inputs()

    ordered = build_ordered_transfer_table(pairwise)

    print("=" * 110)
    print("GEOMETRY -> CAUSAL TRANSFER")
    print("=" * 110)

    print(f"Unordered behavior-pair rows: {len(pairwise)}")

    print(f"Ordered causal-transfer rows: {len(ordered)}")

    if len(ordered) != 1176:
        raise RuntimeError(
            f"Expected 1,176 ordered causal transfers, found {len(ordered)}."
        )

    plot_summary = build_plot_summary(
        ordered,
        association,
    )

    print()
    print(
        plot_summary[
            [
                "scope",
                "scope_value",
                "num_ordered_points",
                "rho",
                "p",
            ]
        ].to_string(index=False)
    )

    # -------------------------------------------------------------------------
    # Per-model geometry / transfer relationships
    # -------------------------------------------------------------------------

    for model_slug in MODEL_ORDER:
        frame = ordered[ordered["model_slug"] == model_slug]

        stat = get_signed_spearman(
            association,
            scope="model",
            scope_value=model_slug,
        )

        plot_signed_transfer(
            frame,
            title=(f"{MODEL_LABELS[model_slug]} — Geometry vs. Transfer"),
            statistic=stat,
            output_name=(MODEL_OUTPUT_NAMES[model_slug]),
        )

    # -------------------------------------------------------------------------
    # Cross-model relationship
    # -------------------------------------------------------------------------

    global_stat = get_signed_spearman(
        association,
        scope="global",
        scope_value="all",
    )

    plot_signed_transfer(
        ordered,
        title=("Steering Geometry Predicts Causal Transfer"),
        statistic=global_stat,
        output_name=("cross_model_signed_transfer"),
    )

    # -------------------------------------------------------------------------
    # Transfer magnitude across depth
    # -------------------------------------------------------------------------

    depth_summary = plot_transfer_by_depth(layer_summary)

    # -------------------------------------------------------------------------
    # Persist plot-facing summaries
    # -------------------------------------------------------------------------

    ordered_path = OUTPUT_ROOT / "ordered_transfer_points.csv"

    plot_summary_path = OUTPUT_ROOT / "association_plot_summary.csv"

    depth_summary_path = OUTPUT_ROOT / "transfer_by_depth_summary.csv"

    ordered.to_csv(
        ordered_path,
        index=False,
    )

    plot_summary.to_csv(
        plot_summary_path,
        index=False,
    )

    depth_summary.to_csv(
        depth_summary_path,
        index=False,
    )

    print()
    print("=" * 110)
    print("GEOMETRY / TRANSFER PLOTTING COMPLETE")
    print("=" * 110)

    print(f"Ordered points: {ordered_path}")

    print(f"Association summary: {plot_summary_path}")

    print(f"Depth summary: {depth_summary_path}")

    print(f"Plots: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
