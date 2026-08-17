import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from steerfrontier.config import load_experiment_config
from steerfrontier.workflow import parse_selected_layers


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate report-facing comparison figures for one model run."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--selected-layer-spec",
        nargs="+",
        required=True,
        metavar="BEHAVIOR=LAYER",
    )
    parser.add_argument("--layer-window", type=int, default=6)
    parser.add_argument(
        "--cross-layer",
        type=int,
        default=None,
        help="Optional common layer for the cross-behavior steering heatmap.",
    )
    return parser.parse_args()


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Avenir", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def display_name(behavior):
    return behavior.replace("-", " ").title()


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path.with_suffix(".pdf"))
    print("Saved:", path.with_suffix(".png"))


def select_layer_window(table, selected_layer, window):
    layers = list(table.index)
    if selected_layer not in layers:
        raise ValueError(
            f"Selected layer {selected_layer} not present; available: {layers}"
        )
    if window >= len(layers):
        return table
    idx = layers.index(selected_layer)
    start = idx - window // 2
    end = start + window
    if start < 0:
        start, end = 0, window
    if end > len(layers):
        end = len(layers)
        start = end - window
    return table.loc[layers[start:end]]


def load_curves(config, behavior, layer):
    root = config.layout.results_dir(behavior)
    behavior_df = pd.read_csv(root / "coefficient_layer_sweep.csv")
    behavior_df = behavior_df[behavior_df["layer"] == layer].sort_values("alpha")
    capability_df = pd.read_csv(root / f"mmlu_layer{layer}.csv").sort_values("alpha")

    df = behavior_df[["alpha", "behavior_score", "baseline"]].merge(
        capability_df[["alpha", "capability_score", "baseline"]],
        on="alpha",
        suffixes=("_behavior", "_capability"),
    )
    df["delta_behavior"] = df["behavior_score"] - df["baseline_behavior"]
    df["delta_capability"] = df["capability_score"] - df["baseline_capability"]
    return df


def plot_selected_curves(config, selected, data, output_dir):
    behaviors = config.behaviors
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

    for ax, behavior in zip(axes, behaviors):
        layer = selected[behavior]
        df = data[behavior]
        ax.plot(df["alpha"], df["behavior_score"], marker="o", label=r"$B_b(\alpha)$")
        ax.plot(
            df["alpha"],
            df["capability_score"],
            marker="o",
            label=r"$C_b(\alpha)$ (MMLU)",
        )
        ax.axvline(0, linestyle="--", linewidth=1, alpha=0.45)
        ax.set_title(rf"{display_name(behavior)}, $\ell_b^*={layer}$")
        ax.set_xlabel(r"Steering coefficient, $\alpha$")
        ax.grid(alpha=0.17)

    axes[0].set_ylabel("Probability")
    axes[0].legend(frameon=False)
    fig.suptitle(
        rf"Behavior control $B_b(\alpha)$ and capability $C_b(\alpha)$"
        f" — {config.model_label}"
    )
    save(
        fig,
        output_dir / f"behavior_capability_comparison_{config.model_slug}",
    )


def plot_delta_curves(config, selected, data, output_dir):
    behaviors = config.behaviors
    max_abs = max(
        max(
            float(np.abs(data[b]["delta_behavior"]).max()),
            float(np.abs(data[b]["delta_capability"]).max()),
        )
        for b in behaviors
    )
    if max_abs == 0.0:
        max_abs = 1e-12

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

    for ax, behavior in zip(axes, behaviors):
        layer = selected[behavior]
        df = data[behavior]
        ax.axhline(0, linestyle="--", linewidth=1, alpha=0.55)
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
            label=r"$\Delta C_b(\alpha)$",
        )
        ax.set_title(rf"{display_name(behavior)}, $\ell_b^*={layer}$")
        ax.set_xlabel(r"Steering coefficient, $\alpha$")
        ax.set_ylim(-1.08 * max_abs, 1.08 * max_abs)
        ax.grid(alpha=0.17)

    axes[0].set_ylabel("Change from unsteered baseline")
    axes[0].legend(frameon=False)
    fig.suptitle(
        rf"Behavior shift $\Delta B_b(\alpha)$ and capability shift "
        rf"$\Delta C_b(\alpha)$ — {config.model_label}"
    )
    save(
        fig,
        output_dir / f"behavior_capability_delta_comparison_{config.model_slug}",
    )


def plot_layer_alpha_comparison(config, selected, layer_window, output_dir):
    tables = {}
    max_abs = 0.0

    for behavior in config.behaviors:
        df = pd.read_csv(
            config.layout.results_dir(behavior) / "coefficient_layer_sweep.csv"
        )
        df["delta"] = df["behavior_score"] - df["baseline"]
        table = (
            df.pivot(index="layer", columns="alpha", values="delta")
            .sort_index()
            .sort_index(axis=1)
        )
        table = select_layer_window(table, selected[behavior], layer_window)
        tables[behavior] = table
        max_abs = max(max_abs, float(np.abs(table.to_numpy()).max()))

    if max_abs == 0.0:
        max_abs = 1e-12
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

    fig = plt.figure(figsize=(13.8, 4.7), constrained_layout=True)
    gs = fig.add_gridspec(
        1,
        len(config.behaviors) + 1,
        width_ratios=[*([1.0] * len(config.behaviors)), 0.06],
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(len(config.behaviors))]
    cax = fig.add_subplot(gs[0, -1])
    image = None

    for ax, behavior in zip(axes, config.behaviors):
        table = tables[behavior]
        layers = table.index.to_numpy()
        alphas = table.columns.to_numpy()
        values = table.to_numpy()
        image = ax.imshow(
            values,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="coolwarm",
            norm=norm,
        )
        ax.set_xticks(np.arange(len(alphas)))
        ax.set_xticklabels([f"{a:g}" for a in alphas], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(layers)))
        ax.set_yticklabels(layers)
        ax.set_xlabel(r"Steering coefficient, $\alpha$")
        ax.set_ylabel(r"Injection layer, $\ell$")
        ax.set_title(display_name(behavior))

        selected_row = list(layers).index(selected[behavior])
        ax.add_patch(
            Rectangle(
                (-0.5, selected_row - 0.5),
                width=len(alphas),
                height=1.0,
                fill=False,
                linewidth=1.6,
                edgecolor="black",
            )
        )

    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label(r"$\Delta B_b(\alpha,\ell)$")
    fig.suptitle(
        rf"Layer–coefficient response $\Delta B_b(\alpha,\ell)$"
        f" — {config.model_label}"
    )
    save(fig, output_dir / f"layer_alpha_comparison_{config.model_slug}")


def plot_cross_behavior_cosine(config, output_dir):
    path = config.layout.comparison_results_dir() / "cross_behavior_vector_cosine.csv"
    if not path.exists():
        print("Skipping cross-behavior cosine; missing:", path)
        return

    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for (a, b), sub in df.groupby(["behavior_a", "behavior_b"]):
        sub = sub.sort_values("layer")
        ax.plot(
            sub["layer"],
            sub["cosine_similarity"],
            marker="o",
            markersize=3,
            label=f"{display_name(a)} ↔ {display_name(b)}",
        )
    ax.axhline(0, linestyle="--", linewidth=1, alpha=0.55)
    ax.set_xlabel(r"Layer, $\ell$")
    ax.set_ylabel(r"$\cos(v_{i,\ell},v_{j,\ell})$")
    ax.set_title(f"Cross-behavior CAA direction similarity — {config.model_label}")
    ax.grid(alpha=0.17)
    ax.legend(frameon=False, fontsize=8)
    save(fig, output_dir / f"cross_behavior_vector_cosine_{config.model_slug}")


def plot_cross_behavior_steering(config, layer, output_dir):
    if layer is None:
        return
    path = (
        config.layout.comparison_results_dir()
        / f"cross_behavior_steering_layer{layer}.csv"
    )
    if not path.exists():
        print("Skipping cross-behavior steering; missing:", path)
        return

    df = pd.read_csv(path)
    plot_alphas = [-1.0, 1.0]
    matrices = {}
    max_abs = 0.0
    for alpha in plot_alphas:
        sub = df[df["alpha"] == alpha]
        matrix = sub.pivot(
            index="source_behavior",
            columns="target_behavior",
            values="delta_from_baseline",
        ).reindex(index=config.behaviors, columns=config.behaviors)
        matrices[alpha] = matrix
        max_abs = max(max_abs, float(np.abs(matrix.to_numpy()).max()))

    if max_abs == 0.0:
        max_abs = 1e-12
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

    fig = plt.figure(figsize=(9.8, 4.7), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.055])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2])
    image = None

    labels = [display_name(b) for b in config.behaviors]
    for ax, alpha in zip(axes, plot_alphas):
        matrix = matrices[alpha]
        values = matrix.to_numpy()
        image = ax.imshow(values, cmap="coolwarm", norm=norm, aspect="equal")
        ax.set_xticks(range(len(config.behaviors)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticks(range(len(config.behaviors)))
        ax.set_yticklabels(labels)
        ax.set_xlabel(r"Evaluated behavior, $j$")
        ax.set_ylabel(r"Steering direction, $v_i$")
        ax.set_title(rf"$\alpha={alpha:+g}$")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(
                    j, i, f"{values[i, j]:+.2f}", ha="center", va="center", fontsize=9
                )

    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label(rf"$\Delta B_j^{{(i)}}(\alpha,\ell={layer})$")
    fig.suptitle(
        rf"Cross-behavior CAA transfer at $\ell={layer}$ — {config.model_label}"
    )
    save(
        fig,
        output_dir / f"cross_behavior_steering_layer{layer}_{config.model_slug}",
    )


def main():
    configure_style()
    args = parse_args()
    config = load_experiment_config(args.config)
    selected = parse_selected_layers(args.selected_layer_spec)

    missing = [b for b in config.behaviors if b not in selected]
    if missing:
        raise ValueError(f"Missing selected layers for: {missing}")

    data = {
        behavior: load_curves(config, behavior, selected[behavior])
        for behavior in config.behaviors
    }
    output_dir = config.layout.comparison_plots_dir()

    plot_selected_curves(config, selected, data, output_dir)
    plot_delta_curves(config, selected, data, output_dir)
    plot_layer_alpha_comparison(config, selected, args.layer_window, output_dir)
    plot_cross_behavior_cosine(config, output_dir)
    plot_cross_behavior_steering(config, args.cross_layer, output_dir)


if __name__ == "__main__":
    main()
