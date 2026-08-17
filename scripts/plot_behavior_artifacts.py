import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from steerfrontier.config import load_experiment_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate scientific artifacts for one behavior/model run."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--behavior", required=True)
    parser.add_argument("--selected-layer", type=int, required=True)
    parser.add_argument("--layer-window", type=int, default=6)
    parser.add_argument("--include-diagnostics", action="store_true")
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
            "figure.titlesize": 12,
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


def blue_green_cmap():
    # Explicitly requested blue/green palette while retaining a white zero point.
    return LinearSegmentedColormap.from_list(
        "blue_white_green",
        ["#2166ac", "#f7f7f7", "#1b7837"],
    )


def select_layer_window(table, selected_layer, window):
    layers = list(table.index)
    if selected_layer not in layers:
        raise ValueError(
            f"Selected layer {selected_layer} is absent; available: {layers}"
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


def load_curves(result_dir, layer):
    behavior_df = pd.read_csv(result_dir / "coefficient_layer_sweep.csv")
    behavior_df = behavior_df[behavior_df["layer"] == layer].sort_values("alpha")
    mmlu_path = result_dir / f"mmlu_layer{layer}.csv"
    if not mmlu_path.exists():
        raise FileNotFoundError(
            f"Missing {mmlu_path}. Run the finalize phase before plotting capability."
        )
    mmlu_df = pd.read_csv(mmlu_path).sort_values("alpha")

    merged = behavior_df[["alpha", "behavior_score", "baseline"]].merge(
        mmlu_df[["alpha", "capability_score", "baseline"]],
        on="alpha",
        suffixes=("_behavior", "_capability"),
    )
    merged["delta_behavior"] = merged["behavior_score"] - merged["baseline_behavior"]
    merged["delta_capability"] = (
        merged["capability_score"] - merged["baseline_capability"]
    )
    return merged


def plot_layer_sweep(config, behavior, selected_layer, result_dir, output_dir):
    df = pd.read_csv(result_dir / "layer_sweep.csv")
    baseline = float(df["baseline"].iloc[0])

    fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    for alpha in sorted(df["alpha"].unique()):
        sub = df[df["alpha"] == alpha].sort_values("layer")
        ax.plot(
            sub["layer"],
            sub["behavior_score"],
            marker="o",
            markersize=3,
            linewidth=1.6,
            label=rf"$\alpha={alpha:+g}$",
        )

    ax.axhline(
        baseline,
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label=r"$B_b(0)$",
    )
    ax.axvline(
        selected_layer,
        linestyle=":",
        linewidth=1.3,
        alpha=0.8,
        label=rf"$\ell_b^*={selected_layer}$",
    )
    ax.set_xlabel(r"Injection layer, $\ell$")
    ax.set_ylabel(r"$B_b(\alpha,\ell)$")
    ax.set_title(
        rf"Layer sensitivity $B_b(\alpha,\ell)$ — "
        f"{display_name(behavior)}, {config.model_label}"
    )
    layers = sorted(df["layer"].unique())
    ax.set_xlim(min(layers), max(layers))
    ax.set_xticks(range(min(layers), max(layers) + 1, 2))
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, ncol=2)
    save(fig, output_dir / f"layer_sweep_{config.model_slug}")


def plot_layer_alpha_heatmap(
    config,
    behavior,
    selected_layer,
    layer_window,
    result_dir,
    output_dir,
):
    df = pd.read_csv(result_dir / "coefficient_layer_sweep.csv")
    df["delta_behavior"] = df["behavior_score"] - df["baseline"]
    table = (
        df.pivot(index="layer", columns="alpha", values="delta_behavior")
        .sort_index()
        .sort_index(axis=1)
    )
    table = select_layer_window(table, selected_layer, layer_window)

    layers = table.index.to_numpy()
    alphas = table.columns.to_numpy()
    values = table.to_numpy()
    max_abs = float(np.abs(values).max())
    if max_abs == 0.0:
        max_abs = 1e-12
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    fig = plt.figure(figsize=(7.4, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.055])
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    image = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=blue_green_cmap(),
        norm=norm,
    )
    ax.set_xticks(np.arange(len(alphas)))
    ax.set_xticklabels([f"{a:g}" for a in alphas], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(layers)))
    ax.set_yticklabels(layers)
    ax.set_xlabel(r"Steering coefficient, $\alpha$")
    ax.set_ylabel(r"Injection layer, $\ell$")
    ax.set_title(
        rf"Layer–coefficient response $\Delta B_b(\alpha,\ell)$ — "
        f"{display_name(behavior)}, {config.model_label}"
    )
    selected_row = list(layers).index(selected_layer)
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
    save(fig, output_dir / f"layer_alpha_heatmap_{config.model_slug}")


def plot_behavior_capability(config, behavior, selected_layer, curves, output_dir):
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.plot(
        curves["alpha"],
        curves["behavior_score"],
        marker="o",
        linewidth=1.8,
        label=r"$B_b(\alpha)$",
    )
    ax.plot(
        curves["alpha"],
        curves["capability_score"],
        marker="o",
        linewidth=1.8,
        label=r"$C_b(\alpha)$ (MMLU)",
    )
    ax.axvline(0.0, linestyle="--", linewidth=1.0, alpha=0.5)
    ax.set_xticks(curves["alpha"])
    ax.set_xlabel(r"Steering coefficient, $\alpha$")
    ax.set_ylabel("Probability")
    ax.set_title(
        rf"$B_b(\alpha)$ and $C_b(\alpha)$ at $\ell_b^*={selected_layer}$ — "
        f"{display_name(behavior)}, {config.model_label}"
    )
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    save(
        fig,
        output_dir / f"behavior_capability_layer{selected_layer}_{config.model_slug}",
    )


def plot_behavior_capability_delta(
    config, behavior, selected_layer, curves, output_dir
):
    max_abs = max(
        float(np.abs(curves["delta_behavior"]).max()),
        float(np.abs(curves["delta_capability"]).max()),
    )
    if max_abs == 0.0:
        max_abs = 1e-12

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.plot(
        curves["alpha"],
        curves["delta_behavior"],
        marker="o",
        linewidth=1.8,
        label=r"$\Delta B_b(\alpha)$",
    )
    ax.plot(
        curves["alpha"],
        curves["delta_capability"],
        marker="o",
        linewidth=1.8,
        label=r"$\Delta C_b(\alpha)$",
    )
    ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.55)
    ax.axvline(0.0, linestyle=":", linewidth=1.0, alpha=0.45)
    ax.set_xticks(curves["alpha"])
    ax.set_ylim(-1.08 * max_abs, 1.08 * max_abs)
    ax.set_xlabel(r"Steering coefficient, $\alpha$")
    ax.set_ylabel("Change from unsteered baseline")
    ax.set_title(
        rf"$\Delta B_b(\alpha)$ and $\Delta C_b(\alpha)$ at "
        rf"$\ell_b^*={selected_layer}$ — "
        f"{display_name(behavior)}, {config.model_label}"
    )
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    save(
        fig,
        output_dir
        / f"behavior_capability_delta_layer{selected_layer}_{config.model_slug}",
    )


def plot_coefficient_curves(config, behavior, selected_layer, result_dir, output_dir):
    df = pd.read_csv(result_dir / "coefficient_layer_sweep.csv")
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)

    other_color = "#9AA0A6"
    selected_color = "#6A3D9A"
    baseline_color = "#1f77b4"

    for layer in sorted(df["layer"].unique()):
        sub = df[df["layer"] == layer].sort_values("alpha")
        is_selected = layer == selected_layer
        ax.plot(
            sub["alpha"],
            sub["behavior_score"],
            marker="o" if is_selected else None,
            markersize=4.5 if is_selected else 0,
            linewidth=2.4 if is_selected else 1.1,
            alpha=1.0 if is_selected else 0.70,
            color=selected_color if is_selected else other_color,
            zorder=3 if is_selected else 1,
        )

    baseline = float(df["baseline"].iloc[0])
    ax.axhline(
        baseline,
        linestyle="--",
        linewidth=1.2,
        alpha=0.85,
        color=baseline_color,
        zorder=2,
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=selected_color,
            linewidth=2.4,
            marker="o",
            markersize=4.5,
            label=rf"Selected layer, $\ell_b^*={selected_layer}$",
        ),
        Line2D(
            [0],
            [0],
            color=other_color,
            linewidth=1.1,
            label=r"Other layers, $\ell \neq \ell_b^*$",
        ),
        Line2D(
            [0],
            [0],
            color=baseline_color,
            linestyle="--",
            linewidth=1.2,
            label=r"$B_b(0)$",
        ),
    ]
    ax.set_xticks(sorted(df["alpha"].unique()))
    ax.set_xlabel(r"Steering coefficient, $\alpha$")
    ax.set_ylabel(r"$B_b(\alpha,\ell)$")
    ax.set_title(
        f"Coefficient response across injection layers — "
        f"{display_name(behavior)}, {config.model_label}"
    )
    ax.grid(alpha=0.18)
    ax.legend(handles=handles, frameon=False)
    save(fig, output_dir / f"coefficient_curves_{config.model_slug}")


def plot_layer_gains(config, behavior, selected_layer, result_dir, output_dir):
    df = pd.read_csv(result_dir / "layer_selection_metrics.csv")
    layers = df["layer"].to_numpy()
    x = np.arange(len(layers))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    ax.bar(
        x - width / 2,
        df["negative_gain"],
        width,
        label=r"Counter-steering gain, $G_b^-$",
    )
    ax.bar(
        x + width / 2,
        df["positive_gain"],
        width,
        label=r"In-direction gain, $G_b^+$",
    )
    if selected_layer in set(layers):
        idx = int(np.where(layers == selected_layer)[0][0])
        ax.axvspan(
            idx - 0.5,
            idx + 0.5,
            fill=False,
            linewidth=1.5,
            edgecolor="black",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_xlabel(r"Injection layer, $\ell$")
    ax.set_ylabel(r"Behavior-score gain, $\Delta B_b$")
    ax.set_title(
        f"Directional steering gains — {display_name(behavior)}, {config.model_label}"
    )
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False)
    save(fig, output_dir / f"layer_steering_gains_{config.model_slug}")


def plot_vector_similarity(config, behavior, vector_path, output_dir):
    vectors = torch.load(vector_path, map_location="cpu").float()
    normalized = F.normalize(vectors, dim=1)
    similarity = normalized @ normalized.T

    fig = plt.figure(figsize=(6.8, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.055])
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    image = ax.imshow(
        similarity.numpy(),
        origin="lower",
        aspect="equal",
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
    )
    ticks = range(0, vectors.shape[0], 4)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel(r"Layer, $\ell_2$")
    ax.set_ylabel(r"Layer, $\ell_1$")
    ax.set_title(
        f"Intra-behavior CAA direction similarity — "
        f"{display_name(behavior)}, {config.model_label}"
    )
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label(r"$\cos(v_{b,\ell_1},v_{b,\ell_2})$")
    save(fig, output_dir / f"vector_cosine_similarity_{config.model_slug}")


def plot_adjacent_similarity(config, behavior, selected_layer, vector_path, output_dir):
    vectors = torch.load(vector_path, map_location="cpu").float()
    vectors = F.normalize(vectors, dim=1)
    similarities = torch.sum(vectors[:-1] * vectors[1:], dim=1).numpy()
    layers = np.arange(len(similarities))

    fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    ax.plot(layers, similarities, marker="o", markersize=4, linewidth=1.6)
    ax.axvline(
        selected_layer,
        linestyle=":",
        linewidth=1.2,
        alpha=0.7,
        label=rf"$\ell_b^*={selected_layer}$",
    )
    ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.5)
    ax.set_xlabel(r"Adjacent layer pair, $(\ell,\ell+1)$")
    ax.set_ylabel(r"$\cos(v_{b,\ell},v_{b,\ell+1})$")
    ax.set_title(
        f"Adjacent-layer CAA direction similarity — "
        f"{display_name(behavior)}, {config.model_label}"
    )
    ax.set_xlim(0, len(similarities) - 1)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks(range(0, len(similarities), 2))
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    save(fig, output_dir / f"adjacent_layer_similarity_{config.model_slug}")


def main():
    configure_style()
    args = parse_args()
    config = load_experiment_config(args.config)

    if args.behavior not in config.behaviors:
        raise ValueError(
            f"Behavior {args.behavior!r} is not configured: {config.behaviors}"
        )
    if args.layer_window < 1:
        raise ValueError("--layer-window must be >= 1.")

    result_dir = config.layout.results_dir(args.behavior)
    output_dir = config.layout.plots_dir(args.behavior)
    vector_path = config.layout.raw_vectors_path(args.behavior)
    curves = load_curves(result_dir, args.selected_layer)

    plot_layer_sweep(config, args.behavior, args.selected_layer, result_dir, output_dir)
    plot_layer_alpha_heatmap(
        config,
        args.behavior,
        args.selected_layer,
        args.layer_window,
        result_dir,
        output_dir,
    )
    plot_behavior_capability(
        config, args.behavior, args.selected_layer, curves, output_dir
    )
    plot_behavior_capability_delta(
        config, args.behavior, args.selected_layer, curves, output_dir
    )

    if args.include_diagnostics:
        plot_coefficient_curves(
            config, args.behavior, args.selected_layer, result_dir, output_dir
        )
        plot_layer_gains(
            config, args.behavior, args.selected_layer, result_dir, output_dir
        )
        if vector_path.exists():
            plot_vector_similarity(config, args.behavior, vector_path, output_dir)
            plot_adjacent_similarity(
                config,
                args.behavior,
                args.selected_layer,
                vector_path,
                output_dir,
            )


if __name__ == "__main__":
    main()
