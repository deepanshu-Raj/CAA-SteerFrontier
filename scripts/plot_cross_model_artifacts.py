import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from steerfrontier.config import load_experiment_config


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate cross-model CAA comparison artifacts from two or more "
            "canonical model-specific experiment trees."
        )
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Experiment YAML configs to compare.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots/cross_model",
        help="Directory for cross-model PDF/PNG artifacts.",
    )
    parser.add_argument(
        "--cross-layer",
        type=int,
        default=None,
        help=(
            "Optional common layer at which to compare cross-behavior causal "
            "steering matrices. Requires cross_behavior_steering_layer<L>.csv "
            "for every model."
        ),
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


def display_name(text):
    return text.replace("-", " ").title()


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path.with_suffix(".pdf"))
    print("Saved:", path.with_suffix(".png"))


def load_summary(config, behavior):
    path = config.layout.results_dir(behavior) / "summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the finalize phase for {behavior} first."
        )
    with path.open("r") as f:
        return json.load(f)


def curve_frame(summary):
    behavior = pd.DataFrame(summary["behavior_curve"])
    capability = pd.DataFrame(summary["capability_curve"])

    df = behavior.merge(capability, on="alpha", how="inner")
    if df.empty:
        raise ValueError(
            f"No shared alpha values for {summary['model_slug']} / "
            f"{summary['behavior']}."
        )

    baseline_capability = float(summary["baseline_capability"])
    df["delta_capability"] = df["capability_score"] - baseline_capability
    df["delta_behavior"] = df["delta_from_baseline"]
    return df.sort_values("alpha").reset_index(drop=True)


def validate_configs(configs):
    if len(configs) < 2:
        raise ValueError("Cross-model plotting requires at least two configs.")

    behavior_lists = [tuple(config.behaviors) for config in configs]
    if len(set(behavior_lists)) != 1:
        raise ValueError(
            "All compared configs must contain the same behaviors in the same "
            f"order. Found: {behavior_lists}"
        )

    slugs = [config.model_slug for config in configs]
    if len(set(slugs)) != len(slugs):
        raise ValueError(f"Duplicate model_slug values: {slugs}")


def load_all(configs):
    data = {}
    for config in configs:
        data[config.model_slug] = {
            "config": config,
            "summaries": {},
            "curves": {},
        }
        for behavior in config.behaviors:
            summary = load_summary(config, behavior)
            data[config.model_slug]["summaries"][behavior] = summary
            data[config.model_slug]["curves"][behavior] = curve_frame(summary)
    return data


def write_cross_model_summary(configs, data, output_dir):
    rows = []

    for config in configs:
        for behavior in config.behaviors:
            summary = data[config.model_slug]["summaries"][behavior]
            curve = data[config.model_slug]["curves"][behavior]

            selected_layer = int(summary["selected_layer"])
            num_layers = None
            cosine_path = (
                config.layout.comparison_results_dir()
                / "cross_behavior_vector_cosine.csv"
            )
            if cosine_path.exists():
                cosine_df = pd.read_csv(cosine_path)
                if not cosine_df.empty:
                    num_layers = int(cosine_df["layer"].max()) + 1

            selected_depth = (
                selected_layer / (num_layers - 1)
                if num_layers is not None and num_layers > 1
                else np.nan
            )

            def value_at(alpha, column):
                sub = curve[np.isclose(curve["alpha"], float(alpha))]
                if sub.empty:
                    return np.nan
                return float(sub.iloc[0][column])

            selection = summary["layer_selection"]

            rows.append(
                {
                    "model_slug": config.model_slug,
                    "model_label": config.model_label,
                    "behavior": behavior,
                    "selected_layer": selected_layer,
                    "selected_relative_depth": selected_depth,
                    "baseline_behavior": float(summary["baseline_behavior"]),
                    "baseline_capability": float(summary["baseline_capability"]),
                    "spearman_rho": float(selection["spearman_rho"]),
                    "monotonic_violations": int(selection["monotonic_violations"]),
                    "negative_gain": float(selection["negative_gain"]),
                    "positive_gain": float(selection["positive_gain"]),
                    "bidirectional_gain": float(selection["bidirectional_gain"]),
                    "total_range": float(selection["total_range"]),
                    "delta_behavior_alpha_-2": value_at(-2.0, "delta_behavior"),
                    "delta_behavior_alpha_-1": value_at(-1.0, "delta_behavior"),
                    "delta_behavior_alpha_+1": value_at(1.0, "delta_behavior"),
                    "delta_behavior_alpha_+2": value_at(2.0, "delta_behavior"),
                    "delta_capability_alpha_-2": value_at(-2.0, "delta_capability"),
                    "delta_capability_alpha_-1": value_at(-1.0, "delta_capability"),
                    "delta_capability_alpha_+1": value_at(1.0, "delta_capability"),
                    "delta_capability_alpha_+2": value_at(2.0, "delta_capability"),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "cross_model_selected_layer_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print("Saved:", path)


def model_styles(configs):
    # Use Matplotlib's default color cycle so the script remains style-agnostic.
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    linestyles = ["-", "--", "-.", ":"]

    styles = {}
    for index, config in enumerate(configs):
        styles[config.model_slug] = {
            "color": colors[index % len(colors)] if colors else None,
            "linestyle": linestyles[index % len(linestyles)],
            "marker": "o" if index % 2 == 0 else "s",
        }
    return styles


def plot_behavior_response(configs, data, output_dir):
    behaviors = configs[0].behaviors
    styles = model_styles(configs)

    fig, axes = plt.subplots(
        1,
        len(behaviors),
        figsize=(13.7, 4.1),
        sharex=True,
        constrained_layout=True,
    )
    if len(behaviors) == 1:
        axes = [axes]

    for ax, behavior in zip(axes, behaviors):
        for config in configs:
            summary = data[config.model_slug]["summaries"][behavior]
            curve = data[config.model_slug]["curves"][behavior]
            style = styles[config.model_slug]

            ax.plot(
                curve["alpha"],
                curve["behavior_score"],
                label=config.model_label,
                linewidth=1.8,
                markersize=4,
                **style,
            )

        ax.axvline(0.0, linestyle=":", linewidth=1.0, alpha=0.5)
        selected_layers = {
            int(data[config.model_slug]["summaries"][behavior]["selected_layer"])
            for config in configs
        }
        if len(selected_layers) != 1:
            raise ValueError(
                f"Compared models use different selected layers for {behavior}: "
                f"{sorted(selected_layers)}"
            )
        selected_layer = next(iter(selected_layers))
        ax.set_title(rf"{display_name(behavior)} ($\ell_b^*={selected_layer}$)")
        ax.set_xlabel(r"Steering coefficient, $\alpha$")
        ax.set_ylabel(r"$B_b(\alpha)$")
        ax.grid(alpha=0.17)

    axes[0].legend(frameon=False)
    fig.suptitle("Cross-model behavioral response at selected intervention layers")
    save(fig, output_dir / "cross_model_behavior_response")


def plot_behavior_shift(configs, data, output_dir):
    behaviors = configs[0].behaviors
    styles = model_styles(configs)

    max_abs = 0.0
    for config in configs:
        for behavior in behaviors:
            curve = data[config.model_slug]["curves"][behavior]
            max_abs = max(
                max_abs,
                float(np.abs(curve["delta_behavior"]).max()),
            )
    if max_abs == 0:
        max_abs = 1e-12

    fig, axes = plt.subplots(
        1,
        len(behaviors),
        figsize=(13.7, 4.1),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(behaviors) == 1:
        axes = [axes]

    for ax, behavior in zip(axes, behaviors):
        for config in configs:
            summary = data[config.model_slug]["summaries"][behavior]
            curve = data[config.model_slug]["curves"][behavior]
            style = styles[config.model_slug]

            ax.plot(
                curve["alpha"],
                curve["delta_behavior"],
                label=config.model_label,
                linewidth=1.8,
                markersize=4,
                **style,
            )

        ax.axhline(0.0, linestyle=":", linewidth=1.0, alpha=0.55)
        selected_layers = {
            int(data[config.model_slug]["summaries"][behavior]["selected_layer"])
            for config in configs
        }
        if len(selected_layers) != 1:
            raise ValueError(
                f"Compared models use different selected layers for {behavior}: "
                f"{sorted(selected_layers)}"
            )
        selected_layer = next(iter(selected_layers))
        ax.set_title(rf"{display_name(behavior)} ($\ell_b^*={selected_layer}$)")
        ax.set_xlabel(r"Steering coefficient, $\alpha$")
        ax.set_ylim(-1.08 * max_abs, 1.08 * max_abs)
        ax.grid(alpha=0.17)

    axes[0].set_ylabel(r"Behavior shift, $\Delta B_b(\alpha)$")
    axes[0].legend(frameon=False)
    fig.suptitle("Cross-model behavioral controllability")
    save(fig, output_dir / "cross_model_behavior_shift")


def plot_capability_shift(configs, data, output_dir):
    behaviors = configs[0].behaviors
    styles = model_styles(configs)

    max_abs = 0.0
    for config in configs:
        for behavior in behaviors:
            curve = data[config.model_slug]["curves"][behavior]
            max_abs = max(
                max_abs,
                float(np.abs(curve["delta_capability"]).max()),
            )
    if max_abs == 0:
        max_abs = 1e-12

    fig, axes = plt.subplots(
        1,
        len(behaviors),
        figsize=(13.7, 4.1),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(behaviors) == 1:
        axes = [axes]

    for ax, behavior in zip(axes, behaviors):
        for config in configs:
            summary = data[config.model_slug]["summaries"][behavior]
            curve = data[config.model_slug]["curves"][behavior]
            style = styles[config.model_slug]

            ax.plot(
                curve["alpha"],
                curve["delta_capability"],
                label=config.model_label,
                linewidth=1.8,
                markersize=4,
                **style,
            )

        ax.axhline(0.0, linestyle=":", linewidth=1.0, alpha=0.55)
        selected_layers = {
            int(data[config.model_slug]["summaries"][behavior]["selected_layer"])
            for config in configs
        }
        if len(selected_layers) != 1:
            raise ValueError(
                f"Compared models use different selected layers for {behavior}: "
                f"{sorted(selected_layers)}"
            )
        selected_layer = next(iter(selected_layers))
        ax.set_title(rf"{display_name(behavior)} ($\ell_b^*={selected_layer}$)")
        ax.set_xlabel(r"Steering coefficient, $\alpha$")
        ax.set_ylim(-1.08 * max_abs, 1.08 * max_abs)
        ax.grid(alpha=0.17)

    axes[0].set_ylabel(r"Capability shift, $\Delta C_b(\alpha)$")
    axes[0].legend(frameon=False)
    fig.suptitle("Cross-model capability tradeoff at selected intervention layers")
    save(fig, output_dir / "cross_model_capability_shift")


def load_cosine(config):
    path = config.layout.comparison_results_dir() / "cross_behavior_vector_cosine.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run run_cross_behavior.py for {config.model_label}."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty cosine table: {path}")

    num_layers = int(df["layer"].max()) + 1
    if num_layers <= 1:
        raise ValueError(f"Invalid number of layers inferred from {path}.")
    df["relative_depth"] = df["layer"] / float(num_layers - 1)
    return df


def plot_vector_geometry(configs, output_dir):
    behaviors = configs[0].behaviors
    pairs = [
        (behaviors[i], behaviors[j])
        for i in range(len(behaviors))
        for j in range(i + 1, len(behaviors))
    ]

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    pair_colors = {
        pair: colors[i % len(colors)] if colors else None
        for i, pair in enumerate(pairs)
    }
    model_line_styles = ["-", "--", "-.", ":"]
    model_markers = ["o", "s", "^", "D"]

    cosine_tables = {config.model_slug: load_cosine(config) for config in configs}

    fig, axes = plt.subplots(
        1,
        len(pairs),
        figsize=(13.7, 4.15),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(pairs) == 1:
        axes = [axes]

    for ax, pair in zip(axes, pairs):
        a, b = pair
        for model_index, config in enumerate(configs):
            df = cosine_tables[config.model_slug]
            sub = df[(df["behavior_a"] == a) & (df["behavior_b"] == b)].sort_values(
                "layer"
            )

            if sub.empty:
                sub = df[(df["behavior_a"] == b) & (df["behavior_b"] == a)].sort_values(
                    "layer"
                )

            if sub.empty:
                raise ValueError(
                    f"Missing pair {a}/{b} in cosine table for {config.model_label}."
                )

            ax.plot(
                sub["relative_depth"],
                sub["cosine_similarity"],
                color=pair_colors[pair],
                linestyle=model_line_styles[model_index % len(model_line_styles)],
                marker=model_markers[model_index % len(model_markers)],
                markevery=max(1, len(sub) // 10),
                markersize=3,
                linewidth=1.7,
                label=config.model_label,
            )

        ax.axhline(0.0, linestyle=":", linewidth=1.0, alpha=0.55)
        ax.set_title(f"{display_name(a)} ↔ {display_name(b)}")
        ax.set_xlabel(r"Relative depth, $d=\ell/(L-1)$")
        ax.grid(alpha=0.17)

    axes[0].set_ylabel(r"$\cos(v_{i,\ell},v_{j,\ell})$")
    axes[0].legend(frameon=False)
    fig.suptitle("Cross-model geometry of CAA behavior directions")
    save(fig, output_dir / "cross_model_vector_cosine_geometry")


def load_cross_steering(config, layer):
    path = (
        config.layout.comparison_results_dir()
        / f"cross_behavior_steering_layer{layer}.csv"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run run_cross_behavior.py --layers {layer} "
            f"for {config.model_label}."
        )
    return pd.read_csv(path)


def plot_cross_steering(configs, layer, output_dir):
    if layer is None:
        return

    behaviors = configs[0].behaviors
    plot_alphas = [-1.0, 1.0]
    tables = {
        config.model_slug: load_cross_steering(config, layer) for config in configs
    }

    matrices = {}
    max_abs = 0.0

    for config in configs:
        df = tables[config.model_slug]
        for alpha in plot_alphas:
            sub = df[np.isclose(df["alpha"], alpha)]
            matrix = sub.pivot(
                index="source_behavior",
                columns="target_behavior",
                values="delta_from_baseline",
            ).reindex(index=behaviors, columns=behaviors)
            if matrix.isnull().any().any():
                raise ValueError(
                    f"Incomplete cross-steering matrix for "
                    f"{config.model_label}, alpha={alpha:+g}, layer={layer}."
                )
            matrices[(config.model_slug, alpha)] = matrix
            max_abs = max(
                max_abs,
                float(np.abs(matrix.to_numpy()).max()),
            )

    if max_abs == 0:
        max_abs = 1e-12
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    n_models = len(configs)
    fig = plt.figure(
        figsize=(9.8, 4.1 * n_models),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(
        n_models,
        len(plot_alphas) + 1,
        width_ratios=[1.0, 1.0, 0.055],
    )

    labels = [display_name(b) for b in behaviors]
    image = None

    for row, config in enumerate(configs):
        for col, alpha in enumerate(plot_alphas):
            ax = fig.add_subplot(gs[row, col])
            matrix = matrices[(config.model_slug, alpha)]
            values = matrix.to_numpy()

            image = ax.imshow(
                values,
                cmap="coolwarm",
                norm=norm,
                aspect="equal",
            )
            ax.set_xticks(range(len(behaviors)))
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_yticks(range(len(behaviors)))
            ax.set_yticklabels(labels)
            ax.set_xlabel(r"Evaluated behavior, $j$")
            ax.set_ylabel(r"Steering direction, $v_i$")
            ax.set_title(
                f"{config.model_label}, "
                rf"$\alpha={alpha:+g}$"
            )

            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    ax.text(
                        j,
                        i,
                        f"{values[i, j]:+.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                    )

    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label(rf"$\Delta B_j^{{(i)}}(\alpha,\ell={layer})$")

    fig.suptitle(rf"Cross-model causal transfer at common layer $\ell={layer}$")
    save(
        fig,
        output_dir / f"cross_model_cross_behavior_steering_layer{layer}",
    )


def main():
    print(f"[cross-model plotter] file={Path(__file__).resolve()}")
    configure_style()
    args = parse_args()

    configs = [load_experiment_config(path) for path in args.configs]
    validate_configs(configs)
    data = load_all(configs)

    output_dir = Path(args.output_dir)
    write_cross_model_summary(configs, data, output_dir)

    plot_behavior_response(configs, data, output_dir)
    plot_behavior_shift(configs, data, output_dir)
    plot_capability_shift(configs, data, output_dir)
    plot_vector_geometry(configs, output_dir)
    plot_cross_steering(
        configs=configs,
        layer=args.cross_layer,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
