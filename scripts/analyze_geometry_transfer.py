"""Quantify the association between steering-vector geometry and causal transfer.

Association statistics operate on frozen geometry and transfer artifacts, with
exact permutation tests used for the primary signed analysis.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# Constants
# =============================================================================

EXPECTED_BEHAVIORS = 7
EXPECTED_BLOCK_ROWS = 42
EXPECTED_PAIR_ROWS = 21


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the association between CAA vector geometry "
            "and ordered cross-behavior causal transfer."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/cross_model/causal_transfer/transfer_results.csv"),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/cross_model/causal_transfer/association"),
    )

    return parser.parse_args()


# =============================================================================
# Statistics
# =============================================================================


def average_rank(values):
    return (
        pd.Series(
            np.asarray(
                values,
                dtype=float,
            )
        )
        .rank(method="average")
        .to_numpy(dtype=float)
    )


def rowwise_correlation(
    x,
    y,
):
    """
    Correlation between every row of x and fixed vector y.

    x: [P, N]
    y: [N]
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    xc = x - x.mean(
        axis=1,
        keepdims=True,
    )

    yc = y - y.mean()

    numerator = xc @ yc

    denominator = np.linalg.norm(
        xc,
        axis=1,
    ) * np.linalg.norm(yc)

    result = np.full(
        x.shape[0],
        np.nan,
        dtype=float,
    )

    valid = denominator > 0.0

    result[valid] = numerator[valid] / denominator[valid]

    return result


def exact_two_sided_p(
    null_statistics,
):
    null_statistics = np.asarray(
        null_statistics,
        dtype=float,
    )

    observed = float(null_statistics[0])

    valid = np.isfinite(null_statistics)

    if not valid.any():
        return np.nan

    return float(np.mean(np.abs(null_statistics[valid]) >= (abs(observed) - 1e-12)))


# =============================================================================
# Block construction
# =============================================================================


def build_block_matrices(
    block,
    behaviors,
):
    n = len(behaviors)

    index = {behavior: i for i, behavior in enumerate(behaviors)}

    transfer = np.full(
        (
            n,
            n,
        ),
        np.nan,
        dtype=float,
    )

    geometry = np.full(
        (
            n,
            n,
        ),
        np.nan,
        dtype=float,
    )

    for row in block.to_dict(orient="records"):
        source = str(row["source_behavior"])

        target = str(row["target_behavior"])

        i = index[source]

        j = index[target]

        if i == j:
            raise ValueError(f"Diagonal transfer condition found for {source}.")

        if np.isfinite(
            transfer[
                i,
                j,
            ]
        ):
            raise ValueError(f"Duplicate ordered transfer {source}->{target}.")

        transfer[
            i,
            j,
        ] = float(row["transfer_effect"])

        cosine = float(row["cosine_similarity"])

        if np.isfinite(
            geometry[
                i,
                j,
            ]
        ):
            if not np.isclose(
                geometry[
                    i,
                    j,
                ],
                cosine,
                atol=1e-10,
                rtol=1e-8,
            ):
                raise ValueError("Inconsistent geometry value.")

        geometry[
            i,
            j,
        ] = cosine

    offdiag = ~np.eye(
        n,
        dtype=bool,
    )

    if not np.isfinite(transfer[offdiag]).all():
        raise ValueError("Transfer matrix is incomplete.")

    if not np.isfinite(geometry[offdiag]).all():
        raise ValueError("Geometry matrix is incomplete.")

    # Geometry must be symmetric.
    for i in range(n):
        for j in range(
            i + 1,
            n,
        ):
            if not np.isclose(
                geometry[
                    i,
                    j,
                ],
                geometry[
                    j,
                    i,
                ],
                atol=1e-10,
                rtol=1e-8,
            ):
                raise ValueError(
                    f"Geometry is not symmetric for {behaviors[i]}, {behaviors[j]}."
                )

    return (
        transfer,
        geometry,
    )


# =============================================================================
# Permutation statistics
# =============================================================================


def ranked_matrix(
    matrix,
    mask,
):
    result = np.zeros_like(
        matrix,
        dtype=float,
    )

    result[mask] = average_rank(matrix[mask])

    return result


def permuted_matrix_values(
    matrix,
    permutations,
    mask,
):
    """
    Apply the same behavior-label permutation to rows and columns,
    then extract the requested matrix cells.
    """

    permuted = matrix[
        permutations[
            :,
            :,
            None,
        ],
        permutations[
            :,
            None,
            :,
        ],
    ]

    return permuted[
        :,
        mask,
    ]


def ordered_statistics(
    transfer,
    geometry,
    permutations,
):
    n = transfer.shape[0]

    mask = ~np.eye(
        n,
        dtype=bool,
    )

    target_signed = transfer[mask]

    target_signed_rank = average_rank(target_signed)

    target_magnitude = np.abs(target_signed)

    target_magnitude_rank = average_rank(target_magnitude)

    geometry_rank = ranked_matrix(
        geometry,
        mask,
    )

    geometry_abs = np.abs(geometry)

    geometry_abs_rank = ranked_matrix(
        geometry_abs,
        mask,
    )

    perm_geometry = permuted_matrix_values(
        geometry,
        permutations,
        mask,
    )

    perm_geometry_rank = permuted_matrix_values(
        geometry_rank,
        permutations,
        mask,
    )

    perm_geometry_abs_rank = permuted_matrix_values(
        geometry_abs_rank,
        permutations,
        mask,
    )

    return {
        "signed_spearman": (
            rowwise_correlation(
                perm_geometry_rank,
                target_signed_rank,
            )
        ),
        "signed_pearson": (
            rowwise_correlation(
                perm_geometry,
                target_signed,
            )
        ),
        "magnitude_spearman": (
            rowwise_correlation(
                perm_geometry_abs_rank,
                target_magnitude_rank,
            )
        ),
    }


def pair_mean_statistics(
    transfer,
    geometry,
    permutations,
):
    n = transfer.shape[0]

    upper = np.triu(
        np.ones(
            (
                n,
                n,
            ),
            dtype=bool,
        ),
        k=1,
    )

    pair_transfer = np.full(
        (
            n,
            n,
        ),
        np.nan,
        dtype=float,
    )

    for i in range(n):
        for j in range(
            i + 1,
            n,
        ):
            value = (
                transfer[
                    i,
                    j,
                ]
                + transfer[
                    j,
                    i,
                ]
            ) / 2.0

            pair_transfer[
                i,
                j,
            ] = value

            pair_transfer[
                j,
                i,
            ] = value

    target_signed = pair_transfer[upper]

    target_signed_rank = average_rank(target_signed)

    target_magnitude = np.abs(target_signed)

    target_magnitude_rank = average_rank(target_magnitude)

    geometry_rank = ranked_matrix(
        geometry,
        upper,
    )

    geometry_abs_rank = ranked_matrix(
        np.abs(geometry),
        upper,
    )

    # The ranked matrices above only populate the upper triangle.
    # Mirror them because simultaneous behavior permutations can map
    # an upper-triangle pair into a lower-triangle location.
    geometry_rank = geometry_rank + geometry_rank.T

    geometry_abs_rank = geometry_abs_rank + geometry_abs_rank.T

    perm_geometry = permuted_matrix_values(
        geometry,
        permutations,
        upper,
    )

    perm_geometry_rank = permuted_matrix_values(
        geometry_rank,
        permutations,
        upper,
    )

    perm_geometry_abs_rank = permuted_matrix_values(
        geometry_abs_rank,
        permutations,
        upper,
    )

    return {
        "signed_spearman": (
            rowwise_correlation(
                perm_geometry_rank,
                target_signed_rank,
            )
        ),
        "signed_pearson": (
            rowwise_correlation(
                perm_geometry,
                target_signed,
            )
        ),
        "magnitude_spearman": (
            rowwise_correlation(
                perm_geometry_abs_rank,
                target_magnitude_rank,
            )
        ),
    }


# =============================================================================
# Aggregation
# =============================================================================


def aggregate_nulls(
    records,
    *,
    variant,
    metric,
):
    selected = [record["nulls"][variant][metric] for record in records]

    matrix = np.stack(
        selected,
        axis=0,
    )

    return np.nanmean(
        matrix,
        axis=0,
    )


def make_aggregate_row(
    *,
    scope,
    scope_value,
    records,
    variant,
    metric,
):
    nulls = aggregate_nulls(
        records,
        variant=variant,
        metric=metric,
    )

    return {
        "scope": (scope),
        "scope_value": (scope_value),
        "variant": (variant),
        "metric": (metric),
        "n_blocks": (len(records)),
        "relations_per_block": (
            EXPECTED_BLOCK_ROWS if variant == "ordered" else EXPECTED_PAIR_ROWS
        ),
        "observed_statistic": float(nulls[0]),
        "exact_permutation_p_two_sided": (exact_two_sided_p(nulls)),
        "num_exact_permutations": int(len(nulls)),
    }


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    df = pd.read_csv(args.input)

    required = {
        "model_slug",
        "model_label",
        "requested_fractional_depth",
        "layer",
        "source_behavior",
        "target_behavior",
        "cosine_similarity",
        "transfer_effect",
        "eta_transfer",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    behaviors = sorted(set(df["source_behavior"]) | set(df["target_behavior"]))

    if len(behaviors) != EXPECTED_BEHAVIORS:
        raise ValueError(
            f"Expected {EXPECTED_BEHAVIORS} behaviors, "
            f"found {len(behaviors)}: {behaviors}"
        )

    permutations = np.asarray(
        list(itertools.permutations(range(len(behaviors)))),
        dtype=int,
    )

    if len(permutations) != 5040:
        raise RuntimeError("Expected 7! = 5040 permutations.")

    identity = np.arange(len(behaviors))

    if not np.array_equal(
        permutations[0],
        identity,
    ):
        raise RuntimeError("First permutation is not identity.")

    print("=" * 80)
    print("GEOMETRY <-> CAUSAL TRANSFER")
    print("=" * 80)
    print(
        "Rows:          ",
        len(df),
    )
    print(
        "Behaviors:     ",
        behaviors,
    )
    print(
        "Permutations:  ",
        len(permutations),
    )
    print(
        "eta values:    ",
        sorted(df["eta_transfer"].unique()),
    )

    block_records = []
    block_rows = []
    pair_rows = []

    group_columns = [
        "model_slug",
        "model_label",
        "requested_fractional_depth",
        "layer",
    ]

    for keys, block in df.groupby(
        group_columns,
        sort=True,
    ):
        (
            model_slug,
            model_label,
            depth,
            layer,
        ) = keys

        if len(block) != EXPECTED_BLOCK_ROWS:
            raise ValueError(
                f"{model_slug}, depth={depth}: "
                f"expected {EXPECTED_BLOCK_ROWS} rows, "
                f"found {len(block)}."
            )

        (
            transfer,
            geometry,
        ) = build_block_matrices(
            block,
            behaviors,
        )

        ordered = ordered_statistics(
            transfer,
            geometry,
            permutations,
        )

        pair_mean = pair_mean_statistics(
            transfer,
            geometry,
            permutations,
        )

        nulls = {
            "ordered": (ordered),
            "pair_mean": (pair_mean),
        }

        record = {
            "model_slug": (model_slug),
            "model_label": (model_label),
            "requested_fractional_depth": float(depth),
            "layer": int(layer),
            "nulls": (nulls),
        }

        block_records.append(record)

        for variant in [
            "ordered",
            "pair_mean",
        ]:
            for metric in [
                "signed_spearman",
                "signed_pearson",
                "magnitude_spearman",
            ]:
                values = nulls[variant][metric]

                block_rows.append(
                    {
                        "model_slug": (model_slug),
                        "model_label": (model_label),
                        "requested_fractional_depth": float(depth),
                        "layer": int(layer),
                        "variant": (variant),
                        "metric": (metric),
                        "n_relations": (
                            EXPECTED_BLOCK_ROWS
                            if variant == "ordered"
                            else EXPECTED_PAIR_ROWS
                        ),
                        "observed_statistic": float(values[0]),
                        "exact_permutation_p_two_sided": (exact_two_sided_p(values)),
                        "num_exact_permutations": (len(values)),
                    }
                )

        # Explicit 21-pair table for interpretation.
        index = {behavior: i for i, behavior in enumerate(behaviors)}

        for behavior_i, behavior_j in itertools.combinations(
            behaviors,
            2,
        ):
            i = index[behavior_i]

            j = index[behavior_j]

            t_ij = float(
                transfer[
                    i,
                    j,
                ]
            )

            t_ji = float(
                transfer[
                    j,
                    i,
                ]
            )

            pair_rows.append(
                {
                    "model_slug": (model_slug),
                    "model_label": (model_label),
                    "requested_fractional_depth": float(depth),
                    "layer": int(layer),
                    "behavior_i": (behavior_i),
                    "behavior_j": (behavior_j),
                    "cosine_similarity": float(
                        geometry[
                            i,
                            j,
                        ]
                    ),
                    "transfer_i_to_j": (t_ij),
                    "transfer_j_to_i": (t_ji),
                    "mean_symmetric_transfer": ((t_ij + t_ji) / 2.0),
                    "directional_asymmetry": (t_ij - t_ji),
                    "mean_abs_directional_transfer": ((abs(t_ij) + abs(t_ji)) / 2.0),
                }
            )

    if len(block_records) != 28:
        raise RuntimeError(
            f"Expected 28 model-depth blocks, found {len(block_records)}."
        )

    # ----------------------------------------------------------------
    # Aggregate exact permutation tests.
    #
    # The same semantic behavior-label permutation is applied
    # synchronously across all selected blocks.
    # ----------------------------------------------------------------

    aggregate_rows = []

    metrics = [
        "signed_spearman",
        "signed_pearson",
        "magnitude_spearman",
    ]

    variants = [
        "ordered",
        "pair_mean",
    ]

    for variant in variants:
        for metric in metrics:
            aggregate_rows.append(
                make_aggregate_row(
                    scope="global",
                    scope_value="all",
                    records=(block_records),
                    variant=(variant),
                    metric=(metric),
                )
            )

    model_slugs = sorted({record["model_slug"] for record in block_records})

    for model_slug in model_slugs:
        selected = [
            record for record in block_records if record["model_slug"] == model_slug
        ]

        for variant in variants:
            for metric in metrics:
                aggregate_rows.append(
                    make_aggregate_row(
                        scope="model",
                        scope_value=(model_slug),
                        records=(selected),
                        variant=(variant),
                        metric=(metric),
                    )
                )

    depths = sorted(
        {float(record["requested_fractional_depth"]) for record in block_records}
    )

    for depth in depths:
        selected = [
            record
            for record in block_records
            if np.isclose(
                float(record["requested_fractional_depth"]),
                depth,
            )
        ]

        for variant in variants:
            for metric in metrics:
                aggregate_rows.append(
                    make_aggregate_row(
                        scope="depth",
                        scope_value=(f"{depth:.2f}"),
                        records=(selected),
                        variant=(variant),
                        metric=(metric),
                    )
                )

    block_df = pd.DataFrame(block_rows)

    aggregate_df = pd.DataFrame(aggregate_rows)

    pair_df = pd.DataFrame(pair_rows)

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    block_path = args.output_root / "block_association.csv"

    aggregate_path = args.output_root / "association_summary.csv"

    pair_path = args.output_root / "pairwise_transfer_summary.csv"

    block_df.to_csv(
        block_path,
        index=False,
    )

    aggregate_df.to_csv(
        aggregate_path,
        index=False,
    )

    pair_df.to_csv(
        pair_path,
        index=False,
    )

    manifest = {
        "analysis": ("geometry_causal_transfer_association"),
        "input": str(args.input),
        "num_transfer_rows": int(len(df)),
        "num_model_depth_blocks": int(len(block_records)),
        "behaviors": (behaviors),
        "num_behaviors": (len(behaviors)),
        "num_exact_behavior_label_permutations": int(len(permutations)),
        "primary_analysis": (
            "Mean within-model-depth ordered signed Spearman "
            "correlation between cosine geometry and signed "
            "causal transfer."
        ),
        "primary_permutation_test": (
            "Exact 7! behavior-label permutation. The same semantic "
            "label permutation is applied synchronously across blocks."
        ),
        "secondary_analyses": [
            ("Ordered signed Pearson correlation."),
            ("Ordered |cosine| versus |transfer| Spearman."),
            ("21-pair symmetrized transfer analysis using (T_i_to_j + T_j_to_i)/2."),
        ],
        "p_value_definition": (
            "Two-sided exact permutation p-value including the identity permutation."
        ),
        "iid_note": (
            "Standard IID correlation p-values are not used because "
            "geometry is symmetric while transfer is directional and "
            "the same behaviors recur across layers/models."
        ),
    }

    with (args.output_root / "manifest.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("PRIMARY RESULTS")
    print("=" * 80)

    primary = aggregate_df[
        (aggregate_df["variant"] == "ordered")
        & (aggregate_df["metric"] == "signed_spearman")
    ].copy()

    print(
        primary[
            [
                "scope",
                "scope_value",
                "n_blocks",
                "observed_statistic",
                "exact_permutation_p_two_sided",
            ]
        ].to_string(index=False)
    )

    print()
    print("Global secondary checks:")

    global_rows = aggregate_df[aggregate_df["scope"] == "global"]

    print(
        global_rows[
            [
                "variant",
                "metric",
                "observed_statistic",
                "exact_permutation_p_two_sided",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Block results: {block_path}")
    print(f"Summary:       {aggregate_path}")
    print(f"Pair table:    {pair_path}")


if __name__ == "__main__":
    main()
