from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle

from steerfrontier.config import load_experiment_config
from steerfrontier.plot_style import (
    BEHAVIOR_BLUE,
    CAPABILITY_ORANGE,
    CAPABILITY_STYLES,
    OTHER_GRAY,
    SELECTED_PURPLE,
    ZERO_GRAY,
    configure_matplotlib,
    get_heatmap_cmap,
    pretty_behavior,
    save_figure,
)

# ============================================================================
# CLI
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-behavior P0-P2 plotting artifacts for CAA-SteerFrontier."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--behavior",
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("plots/per_model"),
    )

    parser.add_argument(
        "--heatmap-cmap",
        choices=[
            "diverging",
            "sequential",
        ],
        default="diverging",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    return parser.parse_args()


# ============================================================================
# Loading / validation
# ============================================================================


def require_columns(
    frame: pd.DataFrame,
    required,
    *,
    source: Path,
):
    missing = set(required) - set(frame.columns)

    if missing:
        raise ValueError(f"{source} missing columns: {sorted(missing)}")


def load_behavior_artifacts(
    *,
    results_root: Path,
    behavior: str,
):
    behavior_root = results_root / behavior

    layer_sweep_path = behavior_root / "layer_sweep.csv"

    coefficient_path = behavior_root / "coefficient_layer_sweep.csv"

    selection_path = behavior_root / "layer_selection_metrics.csv"

    capability_path = results_root / "capability_selection" / "summary.csv"

    frontier_path = results_root / "steering_frontiers.csv"

    paths = [
        layer_sweep_path,
        coefficient_path,
        selection_path,
        capability_path,
        frontier_path,
    ]

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    layer_sweep = pd.read_csv(layer_sweep_path)

    coefficient = pd.read_csv(coefficient_path)

    selection = pd.read_csv(selection_path)

    capability = pd.read_csv(capability_path)

    frontier = pd.read_csv(frontier_path)

    require_columns(
        layer_sweep,
        [
            "behavior",
            "layer",
            "alpha",
            "behavior_score",
            "baseline",
            "delta_from_baseline",
        ],
        source=layer_sweep_path,
    )

    require_columns(
        coefficient,
        [
            "behavior",
            "layer",
            "alpha",
            "behavior_score",
            "baseline",
            "delta_from_baseline",
        ],
        source=coefficient_path,
    )

    require_columns(
        selection,
        [
            "behavior",
            "layer",
            "negative_gain",
            "positive_gain",
            "bidirectional_gain",
        ],
        source=selection_path,
    )

    require_columns(
        capability,
        [
            "behavior",
            "layer",
            "benchmark",
            "alpha",
            "baseline_accuracy",
            "accuracy",
            "delta_accuracy",
            "delta_ci_low",
            "delta_ci_high",
        ],
        source=capability_path,
    )

    require_columns(
        frontier,
        [
            "behavior",
            "layer",
            "epsilon",
            "direction",
            "alpha_star",
        ],
        source=frontier_path,
    )

    layer_sweep = layer_sweep.loc[layer_sweep["behavior"] == behavior].copy()

    coefficient = coefficient.loc[coefficient["behavior"] == behavior].copy()

    selection = selection.loc[selection["behavior"] == behavior].copy()

    capability = capability.loc[capability["behavior"] == behavior].copy()

    frontier = frontier.loc[frontier["behavior"] == behavior].copy()

    if frontier.empty:
        raise ValueError(f"No steering-frontier rows for {behavior}.")

    selected_layers = sorted(set(frontier["layer"].astype(int)))

    if len(selected_layers) != 1:
        raise ValueError(
            f"{behavior}: expected one selected layer, found {selected_layers}"
        )

    selected_layer = selected_layers[0]

    if selected_layer not in set(coefficient["layer"].astype(int)):
        raise ValueError(
            f"Selected layer {selected_layer} missing from coefficient sweep."
        )

    return {
        "layer_sweep": (layer_sweep),
        "coefficient": (coefficient),
        "selection": (selection),
        "capability": (capability),
        "frontier": (frontier),
        "selected_layer": (selected_layer),
    }


# ============================================================================
# Plot 1: layer × coefficient heatmap
# ============================================================================


def plot_layer_alpha_heatmap(
    *,
    frame: pd.DataFrame,
    selected_layer: int,
    behavior_label: str,
    model_label: str,
    output_stem: Path,
    cmap_name: str,
    dpi: int,
):
    pivot = (
        frame.pivot(
            index="layer",
            columns="alpha",
            values="delta_from_baseline",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    layers = pivot.index.astype(int).to_list()

    alphas = pivot.columns.astype(float).to_list()

    values = pivot.to_numpy(dtype=float)

    cmap = get_heatmap_cmap(cmap_name)

    if cmap_name == "diverging":
        maximum = float(np.nanmax(np.abs(values)))

        maximum = max(
            maximum,
            1e-12,
        )

        norm = TwoSlopeNorm(
            vmin=-maximum,
            vcenter=0.0,
            vmax=maximum,
        )

    else:
        norm = Normalize(
            vmin=float(np.nanmin(values)),
            vmax=float(np.nanmax(values)),
        )

    fig, ax = plt.subplots(
        figsize=(
            10.8,
            6.7,
        ),
        constrained_layout=True,
    )

    image = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    ax.grid(False)

    ax.set_xticks(np.arange(len(alphas)))

    ax.set_xticklabels(
        [f"{alpha:g}" for alpha in (alphas)],
        rotation=45,
        ha="right",
    )

    ax.set_yticks(np.arange(len(layers)))

    ax.set_yticklabels(layers)

    ax.set_xlabel(r"Steering coefficient, $\alpha$")

    ax.set_ylabel(r"Injection layer, $\ell$")

    ax.set_title(
        r"Layer–coefficient response "
        r"$\Delta B_b(\alpha,\ell)$"
        f" — {behavior_label}, {model_label}"
    )

    if selected_layer in layers:
        row = layers.index(selected_layer)

        rectangle = Rectangle(
            (
                -0.5,
                row - 0.5,
            ),
            width=len(alphas),
            height=1.0,
            fill=False,
            edgecolor="black",
            linewidth=2.0,
        )

        ax.add_patch(rectangle)

    colorbar = fig.colorbar(
        image,
        ax=ax,
        pad=0.015,
    )

    colorbar.set_label(r"$\Delta B_b(\alpha,\ell)$")

    return save_figure(
        fig,
        output_stem,
        dpi=dpi,
    )


# ============================================================================
# Plot 2: coefficient curves across candidate layers
# ============================================================================


def plot_coefficient_curves(
    *,
    frame: pd.DataFrame,
    selected_layer: int,
    behavior_label: str,
    model_label: str,
    output_stem: Path,
    dpi: int,
):
    fig, ax = plt.subplots(
        figsize=(
            10.8,
            6.7,
        ),
        constrained_layout=True,
    )

    layers = sorted(frame["layer"].astype(int).unique())

    for layer in layers:
        if layer == selected_layer:
            continue

        current = frame.loc[frame["layer"] == layer].sort_values("alpha")

        ax.plot(
            current["alpha"],
            current["behavior_score"],
            color=OTHER_GRAY,
            linewidth=1.35,
            alpha=0.72,
            zorder=1,
        )

    selected = frame.loc[frame["layer"] == selected_layer].sort_values("alpha")

    ax.plot(
        selected["alpha"],
        selected["behavior_score"],
        color=SELECTED_PURPLE,
        linewidth=3.0,
        marker="o",
        markersize=6.5,
        label=(
            r"Selected layer, "
            rf"$\ell_b^*={selected_layer}$"
        ),
        zorder=3,
    )

    baseline = float(selected["baseline"].iloc[0])

    ax.axhline(
        baseline,
        color=BEHAVIOR_BLUE,
        linestyle="--",
        linewidth=1.6,
        alpha=0.82,
        label=r"$B_b(0)$",
        zorder=0,
    )

    # Dummy line for concise legend entry for all gray layers.
    ax.plot(
        [],
        [],
        color=OTHER_GRAY,
        linewidth=1.5,
        label=(
            r"Other layers, "
            r"$\ell\neq\ell_b^*$"
        ),
    )

    ax.set_xlabel(r"Steering coefficient, $\alpha$")

    ax.set_ylabel(r"$B_b(\alpha,\ell)$")

    ax.set_title(
        "Coefficient response across injection layers"
        f" — {behavior_label}, {model_label}"
    )

    ax.legend(
        loc="best",
        frameon=False,
    )

    return save_figure(
        fig,
        output_stem,
        dpi=dpi,
    )


# ============================================================================
# Plot 3: full layer sensitivity
# ============================================================================


def plot_layer_sweep(
    *,
    frame: pd.DataFrame,
    selected_layer: int,
    behavior_label: str,
    model_label: str,
    output_stem: Path,
    dpi: int,
):
    fig, ax = plt.subplots(
        figsize=(
            10.8,
            6.7,
        ),
        constrained_layout=True,
    )

    alphas = sorted(frame["alpha"].astype(float).unique())

    negative = [alpha for alpha in alphas if alpha < 0.0]

    positive = [alpha for alpha in alphas if alpha > 0.0]

    colors = [
        BEHAVIOR_BLUE,
        CAPABILITY_ORANGE,
    ]

    plotted = []

    for alpha, color in zip(
        [
            negative[0] if negative else None,
            positive[0] if positive else None,
        ],
        colors,
    ):
        if alpha is None:
            continue

        current = frame.loc[
            np.isclose(
                frame["alpha"].astype(float),
                alpha,
            )
        ].sort_values("layer")

        ax.plot(
            current["layer"],
            current["behavior_score"],
            color=color,
            marker="o",
            markersize=4.8,
            linewidth=2.4,
            label=(rf"$\alpha={alpha:+g}$"),
        )

        plotted.append(current)

    if not plotted:
        raise ValueError("No nonzero alpha values found in layer sweep.")

    baseline = float(frame["baseline"].iloc[0])

    ax.axhline(
        baseline,
        color=BEHAVIOR_BLUE,
        linestyle="--",
        linewidth=1.5,
        alpha=0.75,
        label=r"$B_b(0)$",
    )

    ax.axvline(
        selected_layer,
        color=SELECTED_PURPLE,
        linestyle=":",
        linewidth=2.1,
        alpha=0.9,
        label=(rf"$\ell_b^*={selected_layer}$"),
    )

    ax.set_xlabel(r"Injection layer, $\ell$")

    ax.set_ylabel(r"$B_b(\alpha,\ell)$")

    ax.set_title(
        r"Layer sensitivity $B_b(\alpha,\ell)$"
        f" — {behavior_label}, {model_label}"
    )

    ax.legend(
        loc="best",
        frameon=False,
    )

    return save_figure(
        fig,
        output_stem,
        dpi=dpi,
    )


# ============================================================================
# Plot 4: directional steering gains by candidate layer
# ============================================================================


def plot_layer_steering_gains(
    *,
    frame: pd.DataFrame,
    selected_layer: int,
    behavior_label: str,
    model_label: str,
    output_stem: Path,
    dpi: int,
):
    frame = frame.sort_values("layer").reset_index(drop=True)

    layers = frame["layer"].astype(int).to_numpy()

    x = np.arange(len(layers))

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(
            10.8,
            6.7,
        ),
        constrained_layout=True,
    )

    ax.bar(
        x - width / 2,
        frame["negative_gain"],
        width=width,
        color=BEHAVIOR_BLUE,
        label=(
            r"Counter-steering gain, "
            r"$G_b^-$"
        ),
    )

    ax.bar(
        x + width / 2,
        frame["positive_gain"],
        width=width,
        color=CAPABILITY_ORANGE,
        label=(
            r"In-direction gain, "
            r"$G_b^+$"
        ),
    )

    ax.set_xticks(x)

    ax.set_xticklabels(layers)

    if selected_layer in layers:
        selected_index = int(np.where(layers == selected_layer)[0][0])

        ymax = float(
            max(
                frame["negative_gain"].max(),
                frame["positive_gain"].max(),
            )
        )

        rectangle = Rectangle(
            (
                selected_index - 0.5,
                0.0,
            ),
            width=1.0,
            height=(ymax * 1.08 if ymax > 0.0 else 1.0),
            fill=False,
            edgecolor="black",
            linewidth=2.0,
        )

        ax.add_patch(rectangle)

    ax.set_xlabel(r"Injection layer, $\ell$")

    ax.set_ylabel(r"Directional gain, $G_b^{\pm}(\ell)$")

    ax.set_title(
        r"Layerwise directional gains $G_b^{\pm}(\ell)$"
        f" — {behavior_label}, {model_label}"
    )

    ax.legend(
        loc="best",
        frameon=False,
    )

    return save_figure(
        fig,
        output_stem,
        dpi=dpi,
    )


# ============================================================================
# Plot 5: behavior + capability response at selected layer
# ============================================================================


def plot_behavior_capability_delta(
    *,
    coefficient_frame: pd.DataFrame,
    capability_frame: pd.DataFrame,
    selected_layer: int,
    behavior_label: str,
    model_label: str,
    output_stem: Path,
    dpi: int,
):
    behavior = coefficient_frame.loc[
        coefficient_frame["layer"] == selected_layer
    ].sort_values("alpha")

    capability = capability_frame.loc[
        capability_frame["layer"] == selected_layer
    ].copy()

    fig, ax = plt.subplots(
        figsize=(10.8, 6.7),
        constrained_layout=True,
    )

    ax.plot(
        behavior["alpha"],
        behavior["delta_from_baseline"],
        color=BEHAVIOR_BLUE,
        linewidth=2.8,
        marker="o",
        markersize=6.3,
        label=r"$\Delta B_b(\alpha)$",
        zorder=4,
    )

    benchmark_order = [
        "mmlu",
        "arc_challenge",
        "hellaswag",
    ]

    for benchmark in benchmark_order:
        current = capability.loc[capability["benchmark"] == benchmark].sort_values(
            "alpha"
        )

        if current.empty:
            continue

        style = CAPABILITY_STYLES[benchmark]

        ax.plot(
            current["alpha"],
            current["delta_accuracy"],
            color=CAPABILITY_ORANGE,
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=5.2,
            linewidth=2.0,
            alpha=style["alpha"],
            label=(
                rf"$\Delta C_"
                rf"{{\mathrm{{{style['label']}}}}}"
                rf"(\alpha)$"
            ),
            zorder=2,
        )

    ax.axhline(
        0.0,
        color=ZERO_GRAY,
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
    )

    ax.axvline(
        0.0,
        color=SELECTED_PURPLE,
        linestyle=":",
        linewidth=1.5,
        alpha=0.65,
    )

    ax.set_xlabel(r"Steering coefficient, $\alpha$")

    ax.set_ylabel("Change from unsteered baseline")

    ax.set_title(
        rf"$\Delta B_b(\alpha)$ and $\Delta C_k(\alpha)$ "
        rf"at $\ell_b^*={selected_layer}$"
        f" — {behavior_label}, {model_label}"
    )

    ax.legend(
        loc="best",
        frameon=False,
    )

    return save_figure(
        fig,
        output_stem,
        dpi=dpi,
    )


# ============================================================================
# Main
# ============================================================================


def main():
    args = parse_args()

    configure_matplotlib()

    config = load_experiment_config(args.config)

    if args.behavior not in (config.behaviors):
        raise ValueError(
            f"Unknown behavior: {args.behavior}\nAvailable: {list(config.behaviors)}"
        )

    artifacts = load_behavior_artifacts(
        results_root=(config.results_root),
        behavior=(args.behavior),
    )

    behavior_label = pretty_behavior(args.behavior)

    model_label = config.model_label

    selected_layer = int(artifacts["selected_layer"])

    output_dir = args.output_root / config.model_slug / args.behavior

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 76)

    print("PER-BEHAVIOR P0-P2 ARTIFACTS")

    print("=" * 76)

    print(f"Model:          {model_label}")

    print(f"Model slug:     {config.model_slug}")

    print(f"Behavior:       {args.behavior}")

    print(f"Selected layer: {selected_layer}")

    print(f"Heatmap cmap:   {args.heatmap_cmap}")

    print(f"Output:         {output_dir}")

    outputs = []

    outputs.extend(
        plot_layer_alpha_heatmap(
            frame=(artifacts["coefficient"]),
            selected_layer=(selected_layer),
            behavior_label=(behavior_label),
            model_label=(model_label),
            output_stem=(output_dir / (f"layer_alpha_heatmap_{config.model_slug}")),
            cmap_name=(args.heatmap_cmap),
            dpi=args.dpi,
        )
    )

    outputs.extend(
        plot_coefficient_curves(
            frame=(artifacts["coefficient"]),
            selected_layer=(selected_layer),
            behavior_label=(behavior_label),
            model_label=(model_label),
            output_stem=(output_dir / (f"coefficient_curves_{config.model_slug}")),
            dpi=args.dpi,
        )
    )

    outputs.extend(
        plot_layer_sweep(
            frame=(artifacts["layer_sweep"]),
            selected_layer=(selected_layer),
            behavior_label=(behavior_label),
            model_label=(model_label),
            output_stem=(output_dir / (f"layer_sweep_{config.model_slug}")),
            dpi=args.dpi,
        )
    )

    outputs.extend(
        plot_layer_steering_gains(
            frame=(artifacts["selection"]),
            selected_layer=(selected_layer),
            behavior_label=(behavior_label),
            model_label=(model_label),
            output_stem=(output_dir / (f"layer_steering_gains_{config.model_slug}")),
            dpi=args.dpi,
        )
    )

    outputs.extend(
        plot_behavior_capability_delta(
            coefficient_frame=(artifacts["coefficient"]),
            capability_frame=(artifacts["capability"]),
            selected_layer=(selected_layer),
            behavior_label=(behavior_label),
            model_label=(model_label),
            output_stem=(
                output_dir
                / (
                    "behavior_capability_delta_"
                    f"layer{selected_layer}_"
                    f"{config.model_slug}"
                )
            ),
            dpi=args.dpi,
        )
    )

    print()
    print("Generated:")

    for path in outputs:
        print(f"  {path}")

    print()
    print(f"Artifacts: {len(outputs)} files (5 figures × PNG/PDF)")


if __name__ == "__main__":
    main()
