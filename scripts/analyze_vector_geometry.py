"""Compute cross-behavior steering-vector geometry for one model.

Pairwise cosine similarities are evaluated across all layers and at fixed
fractional-depth anchors used by the causal-transfer experiment.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from steerfrontier.config import load_experiment_config
from steerfrontier.workflow import load_normalized_vectors

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DEPTH_ANCHORS = [
    0.25,
    0.33,
    0.42,
    0.50,
    0.58,
    0.67,
    0.75,
]


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute pairwise CAA steering-vector geometry across behaviors and layers."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Experiment YAML for one model.",
    )

    parser.add_argument(
        "--depth-anchors",
        type=float,
        nargs="*",
        default=DEFAULT_DEPTH_ANCHORS,
        help=("Fractional-depth anchors used by the later causal-transfer experiment."),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=("Default: <results_root>/geometry"),
    )

    return parser.parse_args()


# =============================================================================
# Helpers
# =============================================================================


def select_anchor_layers(
    num_layers: int,
    depth_anchors,
):
    """
    Map requested fractional depths d to nearest actual layer:

        d_l = l / (L - 1)

    Preserve anchor order and verify that the requested anchors map
    to unique layers.
    """

    if num_layers < 2:
        raise ValueError(f"Need at least two layers; got {num_layers}.")

    rows = []

    used_layers = set()

    for requested_depth in depth_anchors:
        requested_depth = float(requested_depth)

        if not (0.0 <= requested_depth <= 1.0):
            raise ValueError(
                f"Depth anchors must lie in [0, 1]. Got {requested_depth}."
            )

        layer = int(np.rint(requested_depth * (num_layers - 1)))

        if layer in used_layers:
            raise ValueError(
                "Two requested fractional depths map to "
                f"the same layer {layer}. "
                "Choose more widely separated anchors."
            )

        used_layers.add(layer)

        actual_depth = layer / (num_layers - 1)

        rows.append(
            {
                "requested_fractional_depth": (requested_depth),
                "layer": (layer),
                "fractional_depth": (actual_depth),
                "depth_error": (actual_depth - requested_depth),
            }
        )

    return pd.DataFrame(rows)


def validate_vectors(
    vectors,
    behaviors,
):
    """
    Verify all behavior tensors have compatible [L, H] shape.
    """

    shapes = {}

    for behavior in behaviors:
        if behavior not in vectors:
            raise KeyError(f"Missing normalized vectors for {behavior}.")

        tensor = vectors[behavior]

        if tensor.ndim != 2:
            raise ValueError(
                f"{behavior}: expected [layers, hidden] tensor; "
                f"got shape {tuple(tensor.shape)}."
            )

        shapes[behavior] = tuple(tensor.shape)

    unique_shapes = set(shapes.values())

    if len(unique_shapes) != 1:
        raise ValueError(
            f"Behavior steering-vector tensors have inconsistent shapes: {shapes}"
        )

    num_layers, hidden_size = next(iter(unique_shapes))

    return (
        int(num_layers),
        int(hidden_size),
    )


# =============================================================================
# Geometry
# =============================================================================


def compute_pairwise_geometry(
    *,
    normalized_vectors,
    behaviors,
    model_slug: str,
    model_label: str,
):
    num_layers = int(normalized_vectors[behaviors[0]].shape[0])

    rows = []

    for layer in range(num_layers):
        fractional_depth = layer / (num_layers - 1)

        for (
            behavior_i,
            behavior_j,
        ) in itertools.combinations(
            behaviors,
            2,
        ):
            vector_i = (
                normalized_vectors[behavior_i][layer].detach().cpu().float().reshape(-1)
            )

            vector_j = (
                normalized_vectors[behavior_j][layer].detach().cpu().float().reshape(-1)
            )

            norm_i = float(torch.linalg.vector_norm(vector_i).item())

            norm_j = float(torch.linalg.vector_norm(vector_j).item())

            if (
                not np.isfinite(norm_i)
                or not np.isfinite(norm_j)
                or norm_i <= 0.0
                or norm_j <= 0.0
            ):
                raise ValueError(
                    "Invalid steering-vector norm at "
                    f"layer={layer}, "
                    f"pair=({behavior_i}, {behavior_j}): "
                    f"{norm_i}, {norm_j}"
                )

            cosine = float(
                F.cosine_similarity(
                    vector_i.unsqueeze(0),
                    vector_j.unsqueeze(0),
                    dim=1,
                ).item()
            )

            rows.append(
                {
                    "model_slug": (model_slug),
                    "model_label": (model_label),
                    "layer": (layer),
                    "fractional_depth": (fractional_depth),
                    "behavior_i": (behavior_i),
                    "behavior_j": (behavior_j),
                    "cosine_similarity": (cosine),
                    "abs_cosine_similarity": (abs(cosine)),
                    "vector_i_norm": (norm_i),
                    "vector_j_norm": (norm_j),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    config = load_experiment_config(args.config)

    behaviors = list(config.behaviors)

    if len(behaviors) != 7:
        print(f"WARNING: expected 7 configured behaviors, found {len(behaviors)}.")

    output_root = (
        args.output_root
        if args.output_root is not None
        else (config.results_root / "geometry")
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 76)
    print("CAA VECTOR GEOMETRY")
    print("=" * 76)
    print(f"Model:       {config.model_label}")
    print(f"Model slug:  {config.model_slug}")
    print(f"Behaviors:   {len(behaviors)}")
    print(f"Output:      {output_root}")

    print()
    print("Loading normalized vectors...")

    vectors = load_normalized_vectors(config)

    (
        num_layers,
        hidden_size,
    ) = validate_vectors(
        vectors,
        behaviors,
    )

    print(f"Layers:      {num_layers}")
    print(f"Hidden size: {hidden_size}")
    print(f"Pairs/layer: {len(list(itertools.combinations(behaviors, 2)))}")

    anchor_layers = select_anchor_layers(
        num_layers,
        args.depth_anchors,
    )

    print()
    print("Causal-transfer depth anchors:")
    print(anchor_layers.to_string(index=False))

    print()
    print("Computing pairwise geometry...")

    geometry = compute_pairwise_geometry(
        normalized_vectors=vectors,
        behaviors=behaviors,
        model_slug=(config.model_slug),
        model_label=(config.model_label),
    )

    expected_rows = num_layers * (len(behaviors) * (len(behaviors) - 1) // 2)

    if len(geometry) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} geometry rows, found {len(geometry)}."
        )

    anchor_geometry = geometry.loc[
        geometry["layer"].isin(anchor_layers["layer"].astype(int))
    ].copy()

    anchor_geometry = anchor_geometry.merge(
        anchor_layers[
            [
                "requested_fractional_depth",
                "layer",
            ]
        ],
        on="layer",
        how="left",
        validate="many_to_one",
    )

    geometry_path = output_root / "pairwise_cosine_all_layers.csv"

    anchor_path = output_root / "anchor_layers.csv"

    anchor_geometry_path = output_root / "pairwise_cosine_anchor_layers.csv"

    geometry.to_csv(
        geometry_path,
        index=False,
    )

    anchor_layers.to_csv(
        anchor_path,
        index=False,
    )

    anchor_geometry.to_csv(
        anchor_geometry_path,
        index=False,
    )

    # Compact layer-level statistics, useful later for plotting.
    layer_summary = geometry.groupby(
        [
            "model_slug",
            "model_label",
            "layer",
            "fractional_depth",
        ],
        as_index=False,
    ).agg(
        cosine_mean=(
            "cosine_similarity",
            "mean",
        ),
        cosine_std=(
            "cosine_similarity",
            "std",
        ),
        cosine_min=(
            "cosine_similarity",
            "min",
        ),
        cosine_max=(
            "cosine_similarity",
            "max",
        ),
        mean_abs_cosine=(
            "abs_cosine_similarity",
            "mean",
        ),
        max_abs_cosine=(
            "abs_cosine_similarity",
            "max",
        ),
    )

    layer_summary_path = output_root / "geometry_layer_summary.csv"

    layer_summary.to_csv(
        layer_summary_path,
        index=False,
    )

    manifest = {
        "analysis": ("pairwise_steering_vector_geometry"),
        "model_name": (config.model_name),
        "model_label": (config.model_label),
        "model_slug": (config.model_slug),
        "behaviors": (behaviors),
        "num_behaviors": (len(behaviors)),
        "num_layers": (num_layers),
        "hidden_size": (hidden_size),
        "pairwise_relations_per_layer": (
            len(
                list(
                    itertools.combinations(
                        behaviors,
                        2,
                    )
                )
            )
        ),
        "num_geometry_rows": int(len(geometry)),
        "requested_depth_anchors": [float(value) for value in args.depth_anchors],
        "actual_anchor_layers": (anchor_layers.to_dict(orient="records")),
        "geometry_definition": (
            "Signed cosine similarity between normalized CAA "
            "steering vectors for each unordered behavior pair "
            "at the same transformer layer."
        ),
        "transfer_eta_frozen": (0.20),
        "transfer_eta_note": (
            "Frozen before causal-transfer outcomes were observed; "
            "chosen conservatively from the existing positive "
            "epsilon=0.03 frontier eta distribution."
        ),
    }

    manifest_path = output_root / "manifest.json"

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print()
    print("=" * 76)
    print("VECTOR GEOMETRY COMPLETE")
    print("=" * 76)
    print(f"All-layer rows:    {len(geometry)}")
    print(f"Anchor-layer rows: {len(anchor_geometry)}")
    print(f"Geometry:          {geometry_path}")
    print(f"Anchors:           {anchor_path}")
    print(f"Anchor geometry:   {anchor_geometry_path}")
    print(f"Layer summary:     {layer_summary_path}")
    print(f"Manifest:          {manifest_path}")


if __name__ == "__main__":
    main()
