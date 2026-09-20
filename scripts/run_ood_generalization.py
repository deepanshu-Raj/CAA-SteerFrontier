"""Evaluate frozen steering interventions on out-of-distribution behavior data.

Sycophancy tests cross-domain transfer and hallucination tests TruthfulQA without
re-selecting layers or coefficients.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Reuse exact held-out behavior evaluation/statistics path.
from run_final_evaluation import (
    behavior_statistics,
    evaluate_behavior_condition,
)

from steerfrontier.config import load_experiment_config
from steerfrontier.data import load_test_data
from steerfrontier.models import load_model
from steerfrontier.prompts import get_prompt_adapter
from steerfrontier.workflow import load_normalized_vectors

# =============================================================================
# Frozen protocol
# =============================================================================

EPSILON = 0.03
BOOTSTRAP_SAMPLES = 10_000
BASE_SEED = 20260919

SYCOPHANCY_DOMAINS = (
    "nlp",
    "political",
)

EXPECTED_RESULTS_PER_MODEL = 10

OOD_DATA_ROOT = Path("data/ood")


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen CAA interventions under OOD / cross-domain generalization."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=BOOTSTRAP_SAMPLES,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help=(
            "Optional prefix truncation for smoke testing. "
            "Requires a separate --output-root."
        ),
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
    )

    return parser.parse_args()


# =============================================================================
# Helpers
# =============================================================================


def stable_seed(
    *parts,
):
    payload = "|".join(str(part) for part in parts).encode("utf-8")

    digest = hashlib.sha256(payload).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    ) % (2**32)


def sha256_file(
    path: Path,
):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def write_csv(
    frame: pd.DataFrame,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(path.suffix + ".tmp")

    frame.to_csv(
        temporary,
        index=False,
    )

    temporary.replace(path)


def write_json(
    value,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            value,
            f,
            indent=2,
        )
        f.write("\n")

    temporary.replace(path)


def as_dataframe(
    rows,
):
    return pd.DataFrame(rows)


def load_dataset(
    path: Path,
    *,
    max_examples=None,
):
    data = load_test_data(path)

    full_size = len(data)

    if max_examples is not None:
        if max_examples < 1:
            raise ValueError("--max-examples must be >= 1.")

        data = data[:max_examples]

    return (
        data,
        full_size,
    )


def valid_condition_file(
    path: Path,
    expected_examples: int,
):
    if not path.is_file():
        return False

    try:
        frame = pd.read_csv(path)
    except Exception:
        return False

    required = {
        "item_index",
        "matching_prob",
    }

    if not required.issubset(frame.columns):
        return False

    if len(frame) != expected_examples:
        return False

    if frame["item_index"].duplicated().any():
        return False

    return True


# =============================================================================
# Frozen operating points
# =============================================================================


def load_frozen_points(
    config,
):
    path = config.results_root / "final_evaluation" / "frontier_test_results.csv"

    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)

    required = {
        "behavior",
        "direction",
        "epsilon",
        "layer",
        "alpha_star",
        "eta_star",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    frame = frame[
        frame["behavior"].isin(
            [
                "sycophancy",
                "hallucination",
            ]
        )
        & np.isclose(
            frame["epsilon"].astype(float),
            EPSILON,
            atol=1e-12,
            rtol=0.0,
        )
    ].copy()

    for behavior in [
        "sycophancy",
        "hallucination",
    ]:
        rows = frame[frame["behavior"] == behavior]

        if len(rows) != 2:
            raise RuntimeError(
                f"{behavior}: expected two epsilon={EPSILON} rows, found {len(rows)}."
            )

        if set(rows["direction"]) != {
            "negative",
            "positive",
        }:
            raise RuntimeError(f"{behavior}: missing positive/negative frozen point.")

        if rows["layer"].astype(int).nunique() != 1:
            raise RuntimeError(f"{behavior}: frozen directions use different layers.")

    return (
        frame.sort_values(
            [
                "behavior",
                "direction",
            ]
        ).reset_index(drop=True),
        path,
    )


def point_for(
    points,
    behavior,
    direction,
):
    rows = points[(points["behavior"] == behavior) & (points["direction"] == direction)]

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one frozen point for {behavior}/{direction}, found {len(rows)}."
        )

    return rows.iloc[0].to_dict()


# =============================================================================
# Vectors
# =============================================================================


def load_sycophancy_domain_vector(
    model_slug: str,
    domain: str,
):
    path = (
        Path("vectors")
        / model_slug
        / "ood"
        / "sycophancy"
        / domain
        / "normalized_vectors.pt"
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    vector = torch.load(
        path,
        map_location="cpu",
    ).float()

    if vector.ndim != 2:
        raise ValueError(
            f"{path}: expected [num_layers, hidden_size], found {tuple(vector.shape)}."
        )

    return (
        vector,
        path,
    )


# =============================================================================
# Statistics
# =============================================================================


def directional_statistics(
    stats,
    *,
    direction: str,
):
    delta = float(stats["delta_behavior"])

    low = float(stats["delta_ci_low"])

    high = float(stats["delta_ci_high"])

    if direction == "positive":
        directional_gain = delta
        directional_low = low
        directional_high = high

    elif direction == "negative":
        directional_gain = -delta
        directional_low = -high
        directional_high = -low

    else:
        raise ValueError(f"Unknown direction: {direction}")

    supported = bool(directional_low > 0.0)

    return {
        "baseline_behavior_score": float(stats["baseline_behavior_score"]),
        "behavior_score": float(stats["behavior_score"]),
        "behavior_delta": (delta),
        "behavior_delta_ci_low": (low),
        "behavior_delta_ci_high": (high),
        "directional_gain": float(directional_gain),
        "directional_gain_ci_low": float(directional_low),
        "directional_gain_ci_high": float(directional_high),
        "direction_supported": (supported),
        "num_examples": int(stats["num_examples"]),
    }


# =============================================================================
# Evaluation helpers
# =============================================================================


def get_or_run_condition(
    *,
    path,
    model,
    tokenizer,
    adapter,
    dataset,
    layer,
    vector,
    alpha,
    device,
    resume,
    description,
):
    if resume and valid_condition_file(
        path,
        len(dataset),
    ):
        print(f"Reusing: {description}")

        return pd.read_csv(path)

    print(f"Running: {description}")

    (
        _,
        items,
    ) = evaluate_behavior_condition(
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        dataset=dataset,
        layer=layer,
        vector=vector,
        alpha=alpha,
        device=device,
    )

    frame = as_dataframe(items)

    if len(frame) != len(dataset):
        raise RuntimeError(
            f"{description}: expected {len(dataset)} rows, found {len(frame)}."
        )

    write_csv(
        frame,
        path,
    )

    return frame


def completed_key(
    row,
):
    return (
        str(row["experiment"]),
        str(row["source_domain"]),
        str(row["test_domain"]),
        str(row["direction"]),
    )


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    if args.max_examples is not None and args.output_root is None:
        raise ValueError(
            "--max-examples is only for smoke testing and requires --output-root."
        )

    config = load_experiment_config(args.config)

    output_root = (
        args.output_root
        if args.output_root is not None
        else (config.results_root / "ood_generalization")
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    resume = not (args.no_resume)

    results_path = output_root / "ood_results.csv"

    manifest_path = output_root / "manifest.json"

    (
        frozen_points,
        frozen_points_path,
    ) = load_frozen_points(config)

    syc_points = frozen_points[frozen_points["behavior"] == "sycophancy"]

    hall_points = frozen_points[frozen_points["behavior"] == "hallucination"]

    syc_layer = int(syc_points["layer"].iloc[0])

    hall_layer = int(hall_points["layer"].iloc[0])

    # ----------------------------------------------------------------
    # Load datasets before model
    # ----------------------------------------------------------------

    datasets = {}
    dataset_full_sizes = {}
    dataset_paths = {}

    for domain in SYCOPHANCY_DOMAINS:
        path = OOD_DATA_ROOT / "sycophancy" / f"{domain}_test.json"

        (
            dataset,
            full_size,
        ) = load_dataset(
            path,
            max_examples=(args.max_examples),
        )

        key = (
            "sycophancy",
            domain,
        )

        datasets[key] = dataset

        dataset_full_sizes[key] = full_size

        dataset_paths[key] = path

    truthfulqa_path = OOD_DATA_ROOT / "hallucination" / "truthfulqa_test.json"

    (
        truthfulqa,
        truthfulqa_full_size,
    ) = load_dataset(
        truthfulqa_path,
        max_examples=(args.max_examples),
    )

    datasets[
        (
            "hallucination",
            "truthfulqa",
        )
    ] = truthfulqa

    dataset_full_sizes[
        (
            "hallucination",
            "truthfulqa",
        )
    ] = truthfulqa_full_size

    dataset_paths[
        (
            "hallucination",
            "truthfulqa",
        )
    ] = truthfulqa_path

    # ----------------------------------------------------------------
    # Load vectors
    # ----------------------------------------------------------------

    syc_vectors = {}
    syc_vector_paths = {}

    for domain in SYCOPHANCY_DOMAINS:
        (
            tensor,
            path,
        ) = load_sycophancy_domain_vector(
            config.model_slug,
            domain,
        )

        if not (0 <= syc_layer < tensor.shape[0]):
            raise RuntimeError(
                f"{domain}: selected layer {syc_layer} outside {tuple(tensor.shape)}."
            )

        syc_vectors[domain] = tensor[syc_layer]

        syc_vector_paths[domain] = path

    canonical_vectors = load_normalized_vectors(config)

    hall_tensor = canonical_vectors["hallucination"]

    if not (0 <= hall_layer < hall_tensor.shape[0]):
        raise RuntimeError(
            "Hallucination selected layer outside canonical vector tensor."
        )

    hall_vector = hall_tensor[hall_layer]

    # ----------------------------------------------------------------
    # Plan
    # ----------------------------------------------------------------

    print("=" * 92)
    print("OOD / GENERALIZATION EVALUATION")
    print("=" * 92)
    print(f"Model:             {config.model_label}")
    print(f"Model slug:        {config.model_slug}")
    print(f"Epsilon:           {EPSILON}")
    print(f"Sycophancy layer:  {syc_layer}")
    print(f"Hallucination l:   {hall_layer}")
    print(f"Steered conditions: {EXPECTED_RESULTS_PER_MODEL}")

    if args.max_examples is not None:
        print(f"SMOKE prefix:      {args.max_examples}/dataset")

    for behavior in [
        "sycophancy",
        "hallucination",
    ]:
        print()

        for direction in [
            "negative",
            "positive",
        ]:
            point = point_for(
                frozen_points,
                behavior,
                direction,
            )

            print(
                f"{behavior:14s} "
                f"{direction:8s} "
                f"L={int(point['layer']):2d} "
                f"alpha="
                f"{float(point['alpha_star']):+.3f} "
                f"eta="
                f"{float(point['eta_star']):+.6f}"
            )

    # ----------------------------------------------------------------
    # Resume summary
    # ----------------------------------------------------------------

    if resume and results_path.is_file():
        existing_results = pd.read_csv(results_path)

        result_rows = existing_results.to_dict(orient="records")

        done = {completed_key(row) for row in result_rows}

    else:
        result_rows = []
        done = set()

    # ----------------------------------------------------------------
    # Load model
    # ----------------------------------------------------------------

    print()
    print("Loading model...")

    (
        model,
        tokenizer,
    ) = load_model(
        model_name=(config.model_name),
        device=args.device,
        dtype=(config.torch_dtype),
    )

    model.eval()

    adapter = get_prompt_adapter(
        model=model,
        tokenizer=tokenizer,
        model_name=(config.model_name),
        prompt_family=(config.prompt_family),
    )

    print(f"Prompt adapter: {adapter.family}")

    # =================================================================
    # Baselines
    # =================================================================

    baselines = {}

    print()
    print("=" * 92)
    print("OOD BASELINES")
    print("=" * 92)

    # Sycophancy baselines: one per test domain.
    for test_domain in SYCOPHANCY_DOMAINS:
        dataset = datasets[
            (
                "sycophancy",
                test_domain,
            )
        ]

        baseline_path = output_root / "sycophancy" / "baselines" / f"{test_domain}.csv"

        baseline = get_or_run_condition(
            path=baseline_path,
            model=model,
            tokenizer=tokenizer,
            adapter=adapter,
            dataset=dataset,
            layer=syc_layer,
            vector=(syc_vectors[test_domain]),
            alpha=0.0,
            device=args.device,
            resume=resume,
            description=(f"sycophancy baseline [{test_domain}] N={len(dataset)}"),
        )

        baselines[
            (
                "sycophancy",
                test_domain,
            )
        ] = baseline

    # Hallucination baseline.
    hall_baseline_path = output_root / "hallucination" / "baseline_truthfulqa.csv"

    hall_baseline = get_or_run_condition(
        path=hall_baseline_path,
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        dataset=truthfulqa,
        layer=hall_layer,
        vector=hall_vector,
        alpha=0.0,
        device=args.device,
        resume=resume,
        description=(f"hallucination baseline [TruthfulQA] N={len(truthfulqa)}"),
    )

    baselines[
        (
            "hallucination",
            "truthfulqa",
        )
    ] = hall_baseline

    # =================================================================
    # Sycophancy cross-domain evaluation
    # =================================================================

    print()
    print("=" * 92)
    print("SYCOPHANCY CROSS-DOMAIN TRANSFER")
    print("=" * 92)

    for source_domain in SYCOPHANCY_DOMAINS:
        for test_domain in SYCOPHANCY_DOMAINS:
            transfer_type = "id" if source_domain == test_domain else "ood"

            dataset = datasets[
                (
                    "sycophancy",
                    test_domain,
                )
            ]

            baseline_df = baselines[
                (
                    "sycophancy",
                    test_domain,
                )
            ]

            vector = syc_vectors[source_domain]

            for direction in [
                "negative",
                "positive",
            ]:
                point = point_for(
                    frozen_points,
                    "sycophancy",
                    direction,
                )

                alpha = float(point["alpha_star"])

                eta = float(point["eta_star"])

                key = (
                    "sycophancy_cross_domain",
                    source_domain,
                    test_domain,
                    direction,
                )

                if key in done:
                    print(
                        "Skipping completed: "
                        f"{source_domain} -> "
                        f"{test_domain} / "
                        f"{direction}"
                    )
                    continue

                condition_path = (
                    output_root
                    / "sycophancy"
                    / "conditions"
                    / (f"source_{source_domain}__test_{test_domain}__{direction}.csv")
                )

                steered_df = get_or_run_condition(
                    path=condition_path,
                    model=model,
                    tokenizer=tokenizer,
                    adapter=adapter,
                    dataset=dataset,
                    layer=syc_layer,
                    vector=vector,
                    alpha=alpha,
                    device=args.device,
                    resume=resume,
                    description=(
                        f"sycophancy "
                        f"{source_domain} -> "
                        f"{test_domain} "
                        f"[{transfer_type}] "
                        f"{direction} "
                        f"N={len(dataset)}"
                    ),
                )

                bootstrap_seed = stable_seed(
                    BASE_SEED,
                    config.model_slug,
                    "sycophancy",
                    source_domain,
                    test_domain,
                    direction,
                )

                stats = behavior_statistics(
                    baseline_df=(baseline_df),
                    steered_df=(steered_df),
                    behavior=("sycophancy"),
                    alpha=alpha,
                    bootstrap_samples=(args.bootstrap_samples),
                    bootstrap_seed=(bootstrap_seed),
                )

                canonical = directional_statistics(
                    stats,
                    direction=(direction),
                )

                result_rows.append(
                    {
                        "experiment": ("sycophancy_cross_domain"),
                        "model_name": (config.model_name),
                        "model_label": (config.model_label),
                        "model_slug": (config.model_slug),
                        "behavior": ("sycophancy"),
                        "source_domain": (source_domain),
                        "test_domain": (test_domain),
                        "transfer_type": (transfer_type),
                        "layer": int(syc_layer),
                        "direction": (direction),
                        "alpha_star": (alpha),
                        "eta_star": (eta),
                        "epsilon": (EPSILON),
                        "vector_path": str(syc_vector_paths[source_domain]),
                        "dataset_path": str(
                            dataset_paths[
                                (
                                    "sycophancy",
                                    test_domain,
                                )
                            ]
                        ),
                        "bootstrap_seed": int(bootstrap_seed),
                        **canonical,
                    }
                )

                done.add(key)

                result_frame = (
                    pd.DataFrame(result_rows)
                    .sort_values(
                        [
                            "experiment",
                            "source_domain",
                            "test_domain",
                            "direction",
                        ]
                    )
                    .reset_index(drop=True)
                )

                write_csv(
                    result_frame,
                    results_path,
                )

                print(
                    f"  directional gain="
                    f"{canonical['directional_gain']:+.6f} "
                    f"["
                    f"{canonical['directional_gain_ci_low']:+.6f}, "
                    f"{canonical['directional_gain_ci_high']:+.6f}"
                    f"]"
                )

    # =================================================================
    # Hallucination -> TruthfulQA
    # =================================================================

    print()
    print("=" * 92)
    print("HALLUCINATION -> TRUTHFULQA")
    print("=" * 92)

    for direction in [
        "negative",
        "positive",
    ]:
        point = point_for(
            frozen_points,
            "hallucination",
            direction,
        )

        alpha = float(point["alpha_star"])

        eta = float(point["eta_star"])

        key = (
            "hallucination_truthfulqa",
            "canonical",
            "truthfulqa",
            direction,
        )

        if key in done:
            print(f"Skipping completed: TruthfulQA / {direction}")
            continue

        condition_path = (
            output_root
            / "hallucination"
            / "conditions"
            / f"truthfulqa__{direction}.csv"
        )

        steered_df = get_or_run_condition(
            path=condition_path,
            model=model,
            tokenizer=tokenizer,
            adapter=adapter,
            dataset=truthfulqa,
            layer=hall_layer,
            vector=hall_vector,
            alpha=alpha,
            device=args.device,
            resume=resume,
            description=(
                f"hallucination canonical -> TruthfulQA {direction} N={len(truthfulqa)}"
            ),
        )

        bootstrap_seed = stable_seed(
            BASE_SEED,
            config.model_slug,
            "hallucination",
            "canonical",
            "truthfulqa",
            direction,
        )

        stats = behavior_statistics(
            baseline_df=(hall_baseline),
            steered_df=(steered_df),
            behavior=("hallucination"),
            alpha=alpha,
            bootstrap_samples=(args.bootstrap_samples),
            bootstrap_seed=(bootstrap_seed),
        )

        canonical = directional_statistics(
            stats,
            direction=direction,
        )

        result_rows.append(
            {
                "experiment": ("hallucination_truthfulqa"),
                "model_name": (config.model_name),
                "model_label": (config.model_label),
                "model_slug": (config.model_slug),
                "behavior": ("hallucination"),
                "source_domain": ("canonical"),
                "test_domain": ("truthfulqa"),
                "transfer_type": ("external_ood"),
                "layer": int(hall_layer),
                "direction": (direction),
                "alpha_star": (alpha),
                "eta_star": (eta),
                "epsilon": (EPSILON),
                "vector_path": ("canonical normalized hallucination vector"),
                "dataset_path": str(truthfulqa_path),
                "bootstrap_seed": int(bootstrap_seed),
                **canonical,
            }
        )

        done.add(key)

        result_frame = (
            pd.DataFrame(result_rows)
            .sort_values(
                [
                    "experiment",
                    "source_domain",
                    "test_domain",
                    "direction",
                ]
            )
            .reset_index(drop=True)
        )

        write_csv(
            result_frame,
            results_path,
        )

        print(
            f"  directional gain="
            f"{canonical['directional_gain']:+.6f} "
            f"["
            f"{canonical['directional_gain_ci_low']:+.6f}, "
            f"{canonical['directional_gain_ci_high']:+.6f}"
            f"]"
        )

    # =================================================================
    # Final validation
    # =================================================================

    final = pd.DataFrame(result_rows)

    if len(final) != EXPECTED_RESULTS_PER_MODEL:
        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_RESULTS_PER_MODEL} "
            f"completed OOD conditions, "
            f"found {len(final)}."
        )

    if (
        final[
            [
                "experiment",
                "source_domain",
                "test_domain",
                "direction",
            ]
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError("Duplicate OOD result keys.")

    # ----------------------------------------------------------------
    # Manifest
    # ----------------------------------------------------------------

    manifest = {
        "protocol": ("ood_generalization_v1"),
        "model_name": (config.model_name),
        "model_label": (config.model_label),
        "model_slug": (config.model_slug),
        "epsilon": (EPSILON),
        "bootstrap_samples": int(args.bootstrap_samples),
        "smoke_prefix": bool(args.max_examples is not None),
        "max_examples": (args.max_examples),
        "frozen_frontier_path": str(frozen_points_path),
        "frozen_frontier_sha256": (sha256_file(frozen_points_path)),
        "num_steered_conditions": int(len(final)),
        "sycophancy": {
            "selected_layer": int(syc_layer),
            "design": (
                "Each source-domain vector "
                "evaluated both in-domain and "
                "cross-domain using the frozen "
                "epsilon=0.03 positive and "
                "negative coefficients."
            ),
            "nlp_vector_path": str(syc_vector_paths["nlp"]),
            "political_vector_path": str(syc_vector_paths["political"]),
            "nlp_test_path": str(
                dataset_paths[
                    (
                        "sycophancy",
                        "nlp",
                    )
                ]
            ),
            "political_test_path": str(
                dataset_paths[
                    (
                        "sycophancy",
                        "political",
                    )
                ]
            ),
            "full_test_sizes": {
                "nlp": int(
                    dataset_full_sizes[
                        (
                            "sycophancy",
                            "nlp",
                        )
                    ]
                ),
                "political": int(
                    dataset_full_sizes[
                        (
                            "sycophancy",
                            "political",
                        )
                    ]
                ),
            },
        },
        "hallucination": {
            "selected_layer": int(hall_layer),
            "design": (
                "Frozen canonical hallucination "
                "vector evaluated on untouched "
                "TruthfulQA binary OOD set."
            ),
            "truthfulqa_test_path": str(truthfulqa_path),
            "full_test_size": int(truthfulqa_full_size),
        },
        "no_ood_tuning": True,
        "target_data_used_for_vector_learning": False,
        "target_data_used_for_layer_selection": False,
        "target_data_used_for_coefficient_selection": False,
    }

    write_json(
        manifest,
        manifest_path,
    )

    print()
    print("=" * 92)
    print("OOD / GENERALIZATION COMPLETE")
    print("=" * 92)
    print(f"Completed conditions: {len(final)}/{EXPECTED_RESULTS_PER_MODEL}")
    print(f"Results:  {results_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
