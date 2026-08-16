import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate report-ready artifacts for one CAA behavior. "
            "Core plots are always generated; exploratory/appendix plots "
            "can be enabled with --include-diagnostics."
        )
    )

    parser.add_argument("--behavior", required=True)
    parser.add_argument("--selected-layer", required=True, type=int)
    parser.add_argument(
        "--layer-window",
        type=int,
        default=6,
        help="Number of layers shown in the layer-alpha heatmap (default: 6).",
    )
    parser.add_argument(
        "--model-label",
        default="Llama 2 7B Chat",
        help="Human-readable model name used in figure titles.",
    )
    parser.add_argument(
        "--model-slug",
        default="llama2-7b-chat",
        help="Filesystem-safe model name used in output filenames.",
    )
    parser.add_argument(
        "--include-diagnostics",
        action="store_true",
        help=(
            "Also generate coefficient curves, layer-selection gains, "
            "and intra-behavior vector geometry diagnostics."
        ),
    )
    parser.add_argument(
        "--heatmap-cmap",
        default="BrBG",
        help=(
            "Matplotlib colormap used for the individual behavior heatmap "
            "(default: BrBG)."
        ),
    )

    return parser.parse_args()


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
            "figure.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def display_name(behavior):
    return behavior.replace("-", " ").title()


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    pdf = path.with_suffix(".pdf")
    png = path.with_suffix(".png")

    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", pdf)
    print("Saved:", png)


def select_layer_window(table, selected_layer, window=6):
    layers = list(table.index)

    if selected_layer not in layers:
        raise ValueError(
            f"Selected layer {selected_layer} is not present in the coefficient sweep. "
            f"Available layers: {layers}"
        )

    if window >= len(layers):
        return table

    selected_idx = layers.index(selected_layer)
    start = selected_idx - window // 2
    end = start + window

    if start < 0:
        start = 0
        end = window

    if end > len(layers):
        end = len(layers)
        start = end - window

    return table.loc[layers[start:end]]


def load_selected_curves(result_dir, selected_layer):
    behavior_df = pd.read_csv(result_dir / "coefficient_layer_sweep.csv")
    behavior_df = (
        behavior_df[behavior_df["layer"] == selected_layer].sort_values("alpha").copy()
    )

    if behavior_df.empty:
        raise ValueError(f"No coefficient sweep data found at layer {selected_layer}.")

    mmlu_path = result_dir / f"mmlu_layer{selected_layer}.csv"
    if not mmlu_path.exists():
        raise FileNotFoundError(f"Missing MMLU result: {mmlu_path}")

    capability_df = pd.read_csv(mmlu_path).sort_values("alpha").copy()

    merged = behavior_df[["alpha", "behavior_score", "baseline"]].merge(
        capability_df[["alpha", "capability_score", "baseline", "accuracy"]],
        on="alpha",
        suffixes=("_behavior", "_capability"),
    )

    merged["delta_behavior"] = merged["behavior_score"] - merged["baseline_behavior"]
    merged["delta_capability"] = (
        merged["capability_score"] - merged["baseline_capability"]
    )

    return merged


def plot_layer_sweep(
    behavior,
    selected_layer,
    model_label,
    model_slug,
    result_dir,
    output_dir,
):
    df = pd.read_csv(result_dir / "layer_sweep.csv")
    baseline = df["baseline"].iloc[0]

    neg = df[df["alpha"] == -1.0].sort_values("layer")
    pos = df[df["alpha"] == 1.0].sort_values("layer")

    fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)

    ax.plot(
        neg["layer"],
        neg["behavior_score"],
        marker="o",
        markersize=3,
        linewidth=1.6,
        label=r"$\alpha=-1$",
    )
    ax.plot(
        pos["layer"],
        pos["behavior_score"],
        marker="o",
        markersize=3,
        linewidth=1.6,
        label=r"$\alpha=+1$",
    )
    ax.axhline(
        baseline,
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label=r"$\alpha=0$",
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
        f"{display_name(behavior)}, {model_label}"
    )

    layers = sorted(df["layer"].unique())
    ax.set_xlim(min(layers), max(layers))
    ax.set_xticks(range(min(layers), max(layers) + 1, 2))
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, ncol=2)

    save(
        fig,
        output_dir / f"layer_sweep_{model_slug}",
    )


def plot_layer_alpha_heatmap(
    behavior,
    selected_layer,
    layer_window,
    model_label,
    model_slug,
    heatmap_cmap,
    result_dir,
    output_dir,
):
    df = pd.read_csv(result_dir / "coefficient_layer_sweep.csv")
    df["delta_behavior"] = df["behavior_score"] - df["baseline"]

    table = (
        df.pivot(
            index="layer",
            columns="alpha",
            values="delta_behavior",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    table = select_layer_window(
        table,
        selected_layer,
        layer_window,
    )

    layers = table.index.to_numpy()
    alphas = table.columns.to_numpy()
    values = table.to_numpy()

    max_abs = np.abs(values).max()
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
        cmap=heatmap_cmap,
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
    ax.set_title(
        rf"Layer–coefficient response $\Delta B_b(\alpha,\ell)$ — "
        f"{display_name(behavior)}, {model_label}"
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

    save(
        fig,
        output_dir / f"layer_alpha_heatmap_{model_slug}",
    )


def plot_behavior_capability(
    behavior,
    selected_layer,
    curves,
    model_label,
    model_slug,
    output_dir,
):
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
        f"{display_name(behavior)}, {model_label}"
    )

    ax.grid(alpha=0.18)
    ax.legend(frameon=False)

    save(
        fig,
        output_dir / f"behavior_capability_layer{selected_layer}_{model_slug}",
    )


def plot_behavior_capability_delta(
    behavior,
    selected_layer,
    curves,
    model_label,
    model_slug,
    output_dir,
):
    max_abs = max(
        np.abs(curves["delta_behavior"]).max(),
        np.abs(curves["delta_capability"]).max(),
    )

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
        f"{display_name(behavior)}, {model_label}"
    )

    ax.grid(alpha=0.18)
    ax.legend(frameon=False)

    save(
        fig,
        output_dir / f"behavior_capability_delta_layer{selected_layer}_{model_slug}",
    )


def plot_coefficient_curves(
    behavior,
    selected_layer,
    model_label,
    model_slug,
    result_dir,
    output_dir,
):
    df = pd.read_csv(result_dir / "coefficient_layer_sweep.csv")

    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)

    other_color = "#9AA0A6"
    selected_color = "#6A3D9A"

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

    baseline = df["baseline"].iloc[0]
    ax.axhline(
        baseline,
        linestyle="--",
        linewidth=1.2,
        alpha=0.85,
        color="#1f77b4",
        zorder=2,
    )

    legend_handles = [
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
            alpha=0.9,
            label=r"Other layers, $\ell \neq \ell_b^*$",
        ),
        Line2D(
            [0], [0], color="#1f77b4", linestyle="--", linewidth=1.2, label=r"$B_b(0)$"
        ),
    ]

    ax.set_xticks(sorted(df["alpha"].unique()))
    ax.set_xlabel(r"Steering coefficient, $\alpha$")
    ax.set_ylabel(r"$B_b(\alpha,\ell)$")
    ax.set_title(
        rf"Coefficient response across injection layers — "
        f"{display_name(behavior)}, {model_label}"
    )
    ax.grid(alpha=0.18)
    ax.legend(handles=legend_handles, frameon=False, loc="best")

    save(
        fig,
        output_dir / f"coefficient_curves_{model_slug}",
    )


def plot_layer_selection_gains(
    behavior,
    selected_layer,
    model_label,
    model_slug,
    result_dir,
    output_dir,
):
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
        selected_idx = int(np.where(layers == selected_layer)[0][0])
        ax.axvspan(
            selected_idx - 0.5,
            selected_idx + 0.5,
            fill=False,
            linewidth=1.5,
            edgecolor="black",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_xlabel(r"Injection layer, $\ell$")
    ax.set_ylabel(r"Behavior-score gain, $\Delta B_b$")
    ax.set_title(
        rf"Directional steering gains — "
        f"{display_name(behavior)}, {model_label}"
    )
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False)

    save(
        fig,
        output_dir / f"layer_steering_gains_{model_slug}",
    )


def plot_vector_similarity(
    behavior,
    model_label,
    model_slug,
    vector_path,
    output_dir,
):
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
        cmap="viridis",
    )

    major_ticks = range(0, vectors.shape[0], 4)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.set_xlabel(r"Layer, $\ell_2$")
    ax.set_ylabel(r"Layer, $\ell_1$")
    ax.set_title(
        rf"Intra-behavior CAA direction similarity — "
        f"{display_name(behavior)}, {model_label}"
    )

    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label(rf"$\cos(v_{{b,\ell_1}},\,v_{{b,\ell_2}})$")

    save(
        fig,
        output_dir / f"vector_cosine_similarity_{model_slug}",
    )


def plot_adjacent_similarity(
    behavior,
    selected_layer,
    model_label,
    model_slug,
    vector_path,
    output_dir,
):
    vectors = torch.load(vector_path, map_location="cpu").float()
    vectors = F.normalize(vectors, dim=1)

    similarities = torch.sum(vectors[:-1] * vectors[1:], dim=1).numpy()
    layers = np.arange(len(similarities))

    fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)

    ax.plot(
        layers,
        similarities,
        marker="o",
        markersize=4,
        linewidth=1.6,
    )
    ax.axvline(
        selected_layer,
        linestyle=":",
        linewidth=1.2,
        alpha=0.7,
        label=rf"$\ell_b^*={selected_layer}$",
    )

    ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.5)
    ax.set_xlabel(r"Adjacent layer pair, $(\ell,\ell+1)$")
    ax.set_ylabel(r"$\cos(v_{b,\ell},\,v_{b,\ell+1})$")
    ax.set_title(
        rf"Adjacent-layer CAA direction similarity — "
        f"{display_name(behavior)}, {model_label}"
    )
    ax.set_xlim(0, len(similarities) - 1)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks(range(0, len(similarities), 2))
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)

    save(
        fig,
        output_dir / f"adjacent_layer_similarity_{model_slug}",
    )


def plot_frontier(
    behavior,
    selected_layer,
    curves,
    model_label,
    model_slug,
    output_dir,
):
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)

    ax.plot(
        curves["behavior_score"],
        curves["capability_score"],
        marker="o",
        linewidth=1.5,
    )

    for _, row in curves.iterrows():
        ax.annotate(
            rf"$\alpha={row['alpha']:g}$",
            (row["behavior_score"], row["capability_score"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    baseline = curves[curves["alpha"] == 0.0].iloc[0]
    ax.scatter(
        [baseline["behavior_score"]],
        [baseline["capability_score"]],
        marker="*",
        s=120,
        zorder=5,
        label=r"$\alpha=0$",
    )

    ax.set_xlabel(r"Behavior score, $B_b(\alpha)$")
    ax.set_ylabel(r"Capability, $C_b(\alpha)$")
    ax.set_title(
        rf"Behavior–capability frontier at $\ell_b^*={selected_layer}$ — "
        f"{display_name(behavior)}, {model_label}"
    )
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)

    save(
        fig,
        output_dir / f"behavior_capability_frontier_layer{selected_layer}_{model_slug}",
    )


def main():
    configure_style()
    args = parse_args()

    if args.layer_window < 1:
        raise ValueError("--layer-window must be >= 1")

    result_dir = Path(f"results/{args.behavior}")
    output_dir = Path(f"plots/{args.behavior}")
    vector_path = Path(f"vectors/{args.behavior}/raw_vectors.pt")

    print("Behavior:", args.behavior)
    print("Selected layer:", args.selected_layer)
    print("Layer window:", args.layer_window)
    print("Model:", args.model_label)
    print("Diagnostics:", args.include_diagnostics)
    print("Heatmap cmap:", args.heatmap_cmap)
    print()

    curves = load_selected_curves(result_dir, args.selected_layer)

    # Core report-ready artifacts.
    plot_layer_sweep(
        args.behavior,
        args.selected_layer,
        args.model_label,
        args.model_slug,
        result_dir,
        output_dir,
    )
    plot_layer_alpha_heatmap(
        args.behavior,
        args.selected_layer,
        args.layer_window,
        args.model_label,
        args.model_slug,
        args.heatmap_cmap,
        result_dir,
        output_dir,
    )
    plot_behavior_capability(
        args.behavior,
        args.selected_layer,
        curves,
        args.model_label,
        args.model_slug,
        output_dir,
    )
    plot_behavior_capability_delta(
        args.behavior,
        args.selected_layer,
        curves,
        args.model_label,
        args.model_slug,
        output_dir,
    )

    # Optional appendix / diagnostic artifacts.
    if args.include_diagnostics:
        plot_coefficient_curves(
            args.behavior,
            args.selected_layer,
            args.model_label,
            args.model_slug,
            result_dir,
            output_dir,
        )
        plot_layer_selection_gains(
            args.behavior,
            args.selected_layer,
            args.model_label,
            args.model_slug,
            result_dir,
            output_dir,
        )
        if vector_path.exists():
            plot_vector_similarity(
                args.behavior,
                args.model_label,
                args.model_slug,
                vector_path,
                output_dir,
            )
            plot_adjacent_similarity(
                args.behavior,
                args.selected_layer,
                args.model_label,
                args.model_slug,
                vector_path,
                output_dir,
            )
        else:
            print("Skipping vector-geometry diagnostics; missing:", vector_path)

    print()
    print("Finished generating individual-behavior artifacts.")


if __name__ == "__main__":
    main()
