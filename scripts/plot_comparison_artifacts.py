import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

# ================================================================
# Arguments
# ================================================================


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--behaviors",
        nargs="+",
        default=[
            "sycophancy",
            "refusal",
            "hallucination",
        ],
    )

    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[
            16,
            13,
            13,
        ],
        help="Selected intervention layer for each behavior.",
    )

    parser.add_argument(
        "--layer-window",
        type=int,
        default=6,
        help=("Number of layers shown in each layer-alpha heatmap. Default: 6."),
    )

    parser.add_argument(
        "--cross-layer",
        type=int,
        default=13,
        help="Common layer used for cross-behavior steering analysis.",
    )

    parser.add_argument(
        "--model-label",
        default="Llama 2 7B Chat",
        help="Human-readable model name used inside figures.",
    )

    parser.add_argument(
        "--model-slug",
        default="llama2-7b-chat",
        help="Filesystem-safe model name used in output filenames.",
    )

    return parser.parse_args()


# ================================================================
# Style / IO
# ================================================================


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Avenir",
                "Helvetica",
                "Arial",
                "DejaVu Sans",
            ],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def save(fig, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        path.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    fig.savefig(
        path.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:", path.with_suffix(".pdf"))
    print("Saved:", path.with_suffix(".png"))


def display_name(behavior):
    return behavior.replace("-", " ").title()


# ================================================================
# Data helpers
# ================================================================


def load_selected_curves(
    behavior,
    layer,
):
    result_dir = Path(f"results/{behavior}")

    behavior_df = pd.read_csv(result_dir / "coefficient_layer_sweep.csv")

    behavior_df = behavior_df[behavior_df["layer"] == layer].sort_values("alpha").copy()

    if behavior_df.empty:
        raise ValueError(
            f"No coefficient sweep data found for {behavior} at layer {layer}."
        )

    mmlu_path = result_dir / f"mmlu_layer{layer}.csv"

    if not mmlu_path.exists():
        raise FileNotFoundError(f"Missing MMLU file: {mmlu_path}")

    mmlu_df = pd.read_csv(mmlu_path).sort_values("alpha").copy()

    merged = behavior_df[
        [
            "alpha",
            "behavior_score",
            "baseline",
        ]
    ].merge(
        mmlu_df[
            [
                "alpha",
                "capability_score",
                "baseline",
            ]
        ],
        on="alpha",
        suffixes=(
            "_behavior",
            "_capability",
        ),
    )

    merged["delta_behavior"] = merged["behavior_score"] - merged["baseline_behavior"]

    merged["delta_capability"] = (
        merged["capability_score"] - merged["baseline_capability"]
    )

    return merged


def select_layer_window(
    table,
    selected_layer,
    window=6,
):
    """
    Select a fixed-size window around selected_layer.

    If the desired centered window exceeds available layers,
    shift the window so that exactly `window` layers are retained
    whenever possible.
    """

    layers = list(table.index)

    if selected_layer not in layers:
        raise ValueError(
            f"Selected layer {selected_layer} "
            f"is not present in sweep. "
            f"Available layers: {layers}"
        )

    if window >= len(layers):
        return table

    selected_idx = layers.index(selected_layer)

    left = window // 2

    start = selected_idx - left
    end = start + window

    if start < 0:
        start = 0
        end = window

    if end > len(layers):
        end = len(layers)
        start = end - window

    selected_layers = layers[start:end]

    return table.loc[selected_layers]


# ================================================================
# Figure 1:
# Behavior and capability response
# ================================================================


def plot_selected_curves(
    behaviors,
    selected_layers,
    data,
    model_label,
    model_slug,
    output_dir,
):
    fig, axes = plt.subplots(
        1,
        len(behaviors),
        figsize=(13.5, 4.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    if len(behaviors) == 1:
        axes = [axes]

    for ax, behavior, layer in zip(
        axes,
        behaviors,
        selected_layers,
    ):
        df = data[behavior]

        ax.plot(
            df["alpha"],
            df["behavior_score"],
            marker="o",
            label=r"$B_b(\alpha)$",
        )

        ax.plot(
            df["alpha"],
            df["capability_score"],
            marker="o",
            label=r"$C(\alpha)$ (MMLU)",
        )

        ax.axvline(
            0,
            linestyle="--",
            linewidth=1,
            alpha=0.45,
        )

        ax.set_title(
            rf"{display_name(behavior)}, "
            rf"$\ell^*={layer}$"
        )

        ax.set_xlabel(r"Steering coefficient, $\alpha$")

        ax.grid(alpha=0.17)

    axes[0].set_ylabel("Probability")

    axes[0].legend(frameon=False)

    fig.suptitle(
        rf"Behavior control $B_b(\alpha)$ and "
        rf"capability $C(\alpha)$"
        f" — {model_label}"
    )

    save(
        fig,
        output_dir / f"behavior_capability_comparison_{model_slug}",
    )


# ================================================================
# Figure 2:
# Delta behavior / delta capability
# ================================================================


def plot_delta_curves(
    behaviors,
    selected_layers,
    data,
    model_label,
    model_slug,
    output_dir,
):
    # Shared y-limits make the three behavior panels
    # directly comparable.
    max_abs = 0.0

    for behavior in behaviors:
        df = data[behavior]

        max_abs = max(
            max_abs,
            np.abs(df["delta_behavior"]).max(),
            np.abs(df["delta_capability"]).max(),
        )

    margin = 1.08
    ylim = max_abs * margin

    fig, axes = plt.subplots(
        1,
        len(behaviors),
        figsize=(13.5, 4.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    if len(behaviors) == 1:
        axes = [axes]

    for ax, behavior, layer in zip(
        axes,
        behaviors,
        selected_layers,
    ):
        df = data[behavior]

        ax.axhline(
            0,
            linestyle="--",
            linewidth=1,
            alpha=0.55,
        )

        ax.plot(
            df["alpha"],
            df["delta_behavior"],
            marker="o",
            label=r"$\Delta B_b(\alpha)$",
        )

        ax.plot(
            df["alpha"],
            df["delta_capability"],
            marker="o",
            label=r"$\Delta C(\alpha)$",
        )

        ax.set_title(
            rf"{display_name(behavior)}, "
            rf"$\ell^*={layer}$"
        )

        ax.set_xlabel(r"Steering coefficient, $\alpha$")

        ax.set_ylim(
            -ylim,
            ylim,
        )

        ax.grid(alpha=0.17)

    axes[0].set_ylabel("Change from unsteered baseline")

    axes[0].legend(frameon=False)

    fig.suptitle(
        rf"Behavior shift $\Delta B_b(\alpha)$ "
        rf"and capability shift $\Delta C(\alpha)$"
        f" — {model_label}"
    )

    save(
        fig,
        output_dir / f"behavior_capability_delta_comparison_{model_slug}",
    )


# ================================================================
# Figure 3:
# Layer × coefficient response
# ================================================================


def plot_layer_alpha_comparison(
    behaviors,
    selected_layers,
    layer_window,
    model_label,
    model_slug,
    output_dir,
):
    tables = {}
    max_abs = 0.0

    for behavior, selected_layer in zip(
        behaviors,
        selected_layers,
    ):
        path = Path(f"results/{behavior}") / "coefficient_layer_sweep.csv"

        df = pd.read_csv(path)

        df["delta"] = df["behavior_score"] - df["baseline"]

        table = (
            df.pivot(
                index="layer",
                columns="alpha",
                values="delta",
            )
            .sort_index()
            .sort_index(axis=1)
        )

        table = select_layer_window(
            table,
            selected_layer,
            layer_window,
        )

        tables[behavior] = table

        max_abs = max(
            max_abs,
            np.abs(table.to_numpy()).max(),
        )

    norm = TwoSlopeNorm(
        vmin=-max_abs,
        vcenter=0,
        vmax=max_abs,
    )

    # Dedicated final GridSpec column for the colorbar.
    fig = plt.figure(
        figsize=(13.8, 4.7),
        constrained_layout=True,
    )

    gs = fig.add_gridspec(
        1,
        len(behaviors) + 1,
        width_ratios=[
            *([1.0] * len(behaviors)),
            0.06,
        ],
    )

    axes = [fig.add_subplot(gs[0, i]) for i in range(len(behaviors))]

    cax = fig.add_subplot(gs[0, -1])

    im = None

    for ax, behavior, selected_layer in zip(
        axes,
        behaviors,
        selected_layers,
    ):
        table = tables[behavior]

        layers = table.index.to_numpy()

        alphas = table.columns.to_numpy()

        values = table.to_numpy()

        im = ax.imshow(
            values,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="coolwarm",
            norm=norm,
        )

        ax.set_xticks(np.arange(len(alphas)))

        ax.set_xticklabels(
            [f"{alpha:g}" for alpha in alphas],
            rotation=45,
            ha="right",
        )

        ax.set_yticks(np.arange(len(layers)))

        ax.set_yticklabels(layers)

        ax.set_xlabel(r"Steering coefficient, $\alpha$")

        ax.set_ylabel(r"Injection layer, $\ell$")

        ax.set_title(display_name(behavior))

        # --------------------------------------------------------
        # Mark the selected layer.
        # --------------------------------------------------------
        selected_row = list(layers).index(selected_layer)

        rectangle = Rectangle(
            (
                -0.5,
                selected_row - 0.5,
            ),
            width=len(alphas),
            height=1.0,
            fill=False,
            linewidth=1.6,
            edgecolor="black",
        )

        ax.add_patch(rectangle)

    cbar = fig.colorbar(
        im,
        cax=cax,
    )

    cbar.set_label(r"$\Delta B_b(\alpha,\ell)$")

    fig.suptitle(
        rf"Layer–coefficient response "
        rf"$\Delta B_b(\alpha,\ell)$"
        f" — {model_label}"
    )

    save(
        fig,
        output_dir / f"layer_alpha_comparison_{model_slug}",
    )


# ================================================================
# Figure 4:
# Pairwise raw-vector cosine similarity
# ================================================================


def plot_cross_behavior_cosine(
    cross_layer,
    model_label,
    model_slug,
    output_dir,
):
    path = Path("results/comparison") / "cross_behavior_vector_cosine.csv"

    df = pd.read_csv(path)

    fig, ax = plt.subplots(
        figsize=(7.2, 4.4),
        constrained_layout=True,
    )

    for (
        behavior_a,
        behavior_b,
    ), sub in df.groupby(
        [
            "behavior_a",
            "behavior_b",
        ]
    ):
        sub = sub.sort_values("layer")

        ax.plot(
            sub["layer"],
            sub["cosine_similarity"],
            marker="o",
            markersize=3,
            label=(f"{display_name(behavior_a)} ↔ {display_name(behavior_b)}"),
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
        alpha=0.55,
    )

    ax.axvline(
        cross_layer,
        linestyle=":",
        linewidth=1.2,
        alpha=0.65,
        label=rf"Cross-steering layer $\ell={cross_layer}$",
    )

    ax.set_xlabel(r"Layer, $\ell$")

    ax.set_ylabel(r"$\cos(v_{i,\ell},\,v_{j,\ell})$")

    ax.set_title(f"Cross-behavior CAA direction similarity — {model_label}")

    ax.grid(alpha=0.17)

    ax.legend(
        frameon=False,
        fontsize=8,
    )

    save(
        fig,
        output_dir / f"cross_behavior_vector_cosine_{model_slug}",
    )


# ================================================================
# Figure 5:
# Cross-behavior causal steering
# ================================================================


def plot_cross_behavior_steering(
    behaviors,
    layer,
    model_label,
    model_slug,
    output_dir,
):
    path = Path("results/comparison") / f"cross_behavior_steering_layer{layer}.csv"

    df = pd.read_csv(path)

    matrices = {}
    max_abs = 0.0

    for alpha in [
        -1.0,
        1.0,
    ]:
        sub = df[df["alpha"] == alpha]

        matrix = sub.pivot(
            index="source_behavior",
            columns="target_behavior",
            values="delta_from_baseline",
        )

        matrix = matrix.reindex(
            index=behaviors,
            columns=behaviors,
        )

        matrices[alpha] = matrix

        max_abs = max(
            max_abs,
            np.abs(matrix.to_numpy()).max(),
        )

    norm = TwoSlopeNorm(
        vmin=-max_abs,
        vcenter=0,
        vmax=max_abs,
    )

    # ------------------------------------------------------------
    # Dedicated colorbar column fixes overlap.
    # ------------------------------------------------------------
    fig = plt.figure(
        figsize=(9.8, 4.7),
        constrained_layout=True,
    )

    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[
            1.0,
            1.0,
            0.055,
        ],
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
    ]

    cax = fig.add_subplot(gs[0, 2])

    im = None

    for ax, alpha in zip(
        axes,
        [
            -1.0,
            1.0,
        ],
    ):
        matrix = matrices[alpha]

        values = matrix.to_numpy()

        im = ax.imshow(
            values,
            cmap="coolwarm",
            norm=norm,
            aspect="equal",
        )

        labels = [display_name(behavior) for behavior in behaviors]

        ax.set_xticks(range(len(behaviors)))

        ax.set_xticklabels(
            labels,
            rotation=30,
            ha="right",
        )

        ax.set_yticks(range(len(behaviors)))

        ax.set_yticklabels(labels)

        ax.set_xlabel(r"Evaluated behavior, $j$")

        ax.set_ylabel(r"Steering direction, $v_i$")

        ax.set_title(rf"$\alpha={alpha:+g}$")

        # --------------------------------------------------------
        # Annotate actual behavior changes.
        # --------------------------------------------------------
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[
                    i,
                    j,
                ]

                ax.text(
                    j,
                    i,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

    cbar = fig.colorbar(
        im,
        cax=cax,
    )

    cbar.set_label(
        rf"$\Delta B_j^{{(i)}}"
        rf"(\alpha,\ell={layer})$"
    )

    fig.suptitle(
        rf"Cross-behavior CAA transfer at "
        rf"$\ell={layer}$"
        f" — {model_label}"
    )

    save(
        fig,
        output_dir / (f"cross_behavior_steering_layer{layer}_{model_slug}"),
    )


# ================================================================
# Main
# ================================================================


def main():
    configure_style()

    args = parse_args()

    if len(args.behaviors) != len(args.layers):
        raise ValueError("--behaviors and --layers must have the same length.")

    if args.layer_window < 1:
        raise ValueError("--layer-window must be >= 1.")

    output_dir = Path("plots/comparison")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Model:", args.model_label)
    print("Behaviors:", args.behaviors)
    print("Selected layers:", args.layers)
    print("Layer window:", args.layer_window)
    print("Cross-steering layer:", args.cross_layer)
    print()

    # ------------------------------------------------------------
    # Selected-layer behavior + capability data
    # ------------------------------------------------------------
    selected_data = {
        behavior: load_selected_curves(
            behavior,
            layer,
        )
        for behavior, layer in zip(
            args.behaviors,
            args.layers,
        )
    }

    # ------------------------------------------------------------
    # Main report-facing artifacts
    # ------------------------------------------------------------
    plot_selected_curves(
        args.behaviors,
        args.layers,
        selected_data,
        args.model_label,
        args.model_slug,
        output_dir,
    )

    plot_delta_curves(
        args.behaviors,
        args.layers,
        selected_data,
        args.model_label,
        args.model_slug,
        output_dir,
    )

    plot_layer_alpha_comparison(
        args.behaviors,
        args.layers,
        args.layer_window,
        args.model_label,
        args.model_slug,
        output_dir,
    )

    plot_cross_behavior_cosine(
        args.cross_layer,
        args.model_label,
        args.model_slug,
        output_dir,
    )

    plot_cross_behavior_steering(
        args.behaviors,
        args.cross_layer,
        args.model_label,
        args.model_slug,
        output_dir,
    )

    print()
    print("Finished generating comparison artifacts.")


if __name__ == "__main__":
    main()
