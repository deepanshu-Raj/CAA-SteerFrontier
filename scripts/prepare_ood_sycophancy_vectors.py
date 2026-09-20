"""Extract and normalize domain-specific sycophancy steering vectors.

Domain vectors reuse the main experiment's layer-wise normalization reference so
OOD transfer remains directly comparable to the canonical intervention scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from steerfrontier.config import load_experiment_config
from steerfrontier.models import load_model
from steerfrontier.normalization import normalize_to_layer_norms
from steerfrontier.prompts import get_prompt_adapter
from steerfrontier.vectors import extract_all_layer_vectors

# =============================================================================
# Frozen protocol
# =============================================================================

DOMAINS = [
    "nlp",
    "political",
]

DEFAULT_DATA_ROOT = Path("data") / "ood" / "sycophancy"

DEFAULT_EPSILON = 0.03


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Construct source-domain sycophancy CAA vectors for "
            "OOD cross-domain transfer."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Model experiment YAML.",
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--domain",
        choices=DOMAINS,
        default=None,
        help=(
            "Optional single domain. "
            "By default both NLP and political vectors are built."
        ),
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional custom output root. "
            "Production default is "
            "vectors/<model_slug>/ood/sycophancy."
        ),
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help=(
            "Optional prefix truncation for infrastructure smoke tests. "
            "Requires --output-root."
        ),
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute vectors even if completed outputs already exist.",
    )

    return parser.parse_args()


# =============================================================================
# Helpers
# =============================================================================


def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def sha256_file(
    path: Path,
):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def write_json(
    path: Path,
    value,
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


def save_tensor(
    tensor: torch.Tensor,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(path.suffix + ".tmp")

    torch.save(
        tensor.detach().cpu().float(),
        temporary,
    )

    temporary.replace(path)


def tensor_cosine_by_layer(
    a: torch.Tensor,
    b: torch.Tensor,
):
    if a.shape != b.shape:
        raise ValueError(
            f"Tensor shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}."
        )

    numerator = (a * b).sum(dim=1)

    denominator = a.norm(dim=1) * b.norm(dim=1)

    if torch.any(denominator == 0):
        raise ValueError("Cannot compute cosine with zero-norm layer vector.")

    return numerator / denominator


# =============================================================================
# Frozen main-experiment artifacts
# =============================================================================


def load_reference_norms(
    model_slug: str,
):
    path = Path("vectors") / model_slug / "normalization" / "reference_layer_norms.pt"

    if not path.is_file():
        raise FileNotFoundError(path)

    tensor = torch.load(
        path,
        map_location="cpu",
    ).float()

    if tensor.ndim != 1:
        raise ValueError(
            "Expected reference_layer_norms "
            f"with shape [num_layers], found "
            f"{tuple(tensor.shape)}."
        )

    if torch.any(tensor <= 0):
        raise ValueError("Reference layer norms must all be positive.")

    return (
        tensor,
        path,
    )


def load_canonical_sycophancy_vector(
    model_slug: str,
):
    path = Path("vectors") / model_slug / "sycophancy" / "normalized_vectors.pt"

    if not path.is_file():
        raise FileNotFoundError(path)

    tensor = torch.load(
        path,
        map_location="cpu",
    ).float()

    if tensor.ndim != 2:
        raise ValueError(
            "Expected canonical sycophancy vectors "
            f"with shape [num_layers, hidden_size], "
            f"found {tuple(tensor.shape)}."
        )

    return (
        tensor,
        path,
    )


def load_frozen_sycophancy_layer(
    config,
    epsilon: float = DEFAULT_EPSILON,
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

    rows = frame[
        (frame["behavior"] == "sycophancy")
        & np.isclose(
            frame["epsilon"].astype(float),
            float(epsilon),
            atol=1e-12,
            rtol=0.0,
        )
    ].copy()

    if len(rows) != 2:
        raise RuntimeError(
            "Expected exactly two frozen "
            "sycophancy epsilon=0.03 rows "
            f"(positive/negative), found {len(rows)}."
        )

    if set(rows["direction"]) != {
        "positive",
        "negative",
    }:
        raise RuntimeError(
            "Frozen sycophancy rows do not contain "
            "exactly positive and negative directions."
        )

    layers = rows["layer"].astype(int).unique()

    if len(layers) != 1:
        raise RuntimeError(
            "Positive and negative frozen sycophancy points use different layers."
        )

    layer = int(layers[0])

    operating_points = {}

    for row in rows.to_dict(orient="records"):
        direction = str(row["direction"])

        operating_points[direction] = {
            "alpha_star": float(row["alpha_star"]),
            "eta_star": float(row["eta_star"]),
        }

    return (
        layer,
        operating_points,
        path,
    )


# =============================================================================
# Domain data
# =============================================================================


def load_domain_vector_data(
    *,
    data_root: Path,
    domain: str,
    max_examples=None,
):
    path = data_root / f"{domain}_vector.json"

    if not path.is_file():
        raise FileNotFoundError(path)

    data = load_json(path)

    if not data:
        raise ValueError(f"{path} is empty.")

    for index, item in enumerate(data):
        required = {
            "question",
            "answer_matching_behavior",
            "answer_not_matching_behavior",
        }

        missing = required - set(item)

        if missing:
            raise ValueError(f"{path}: item {index} missing {sorted(missing)}.")

        if item.get("ood_domain") != domain:
            raise ValueError(
                f"{path}: item {index} has "
                f"ood_domain="
                f"{item.get('ood_domain')!r}; "
                f"expected {domain!r}."
            )

    full_count = int(len(data))

    if max_examples is not None:
        if max_examples < 1:
            raise ValueError("--max-examples must be >= 1.")

        data = data[:max_examples]

    return (
        data,
        full_count,
        path,
    )


# =============================================================================
# Domain vector construction
# =============================================================================


def build_domain_vector(
    *,
    model,
    tokenizer,
    adapter,
    device: str,
    dataset,
    reference_norms: torch.Tensor,
):
    raw = (
        extract_all_layer_vectors(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            device=device,
            prompt_adapter=adapter,
        )
        .detach()
        .cpu()
        .float()
    )

    if raw.ndim != 2:
        raise RuntimeError(
            f"Raw vector extractor returned unexpected shape {tuple(raw.shape)}."
        )

    if raw.shape[0] != reference_norms.shape[0]:
        raise RuntimeError(
            "Raw vector layer count does not "
            "match frozen reference norms: "
            f"{raw.shape[0]} vs "
            f"{reference_norms.shape[0]}."
        )

    raw_norms = raw.norm(dim=1)

    if torch.any(raw_norms == 0):
        raise RuntimeError("At least one raw domain vector has zero norm.")

    normalized = (
        normalize_to_layer_norms(
            raw,
            reference_norms,
        )
        .detach()
        .cpu()
        .float()
    )

    normalized_norms = normalized.norm(dim=1)

    relative_error = (normalized_norms - reference_norms).abs() / reference_norms

    max_relative_error = float(relative_error.max().item())

    if max_relative_error > 1e-5:
        raise RuntimeError(
            "Normalized vector norms do not "
            "match frozen reference norms. "
            f"Max relative error="
            f"{max_relative_error:.3e}."
        )

    return (
        raw,
        normalized,
        max_relative_error,
    )


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    if args.max_examples is not None and args.output_root is None:
        raise ValueError(
            "--max-examples is for smoke testing and requires a separate --output-root."
        )

    config = load_experiment_config(args.config)

    domains = [args.domain] if args.domain is not None else DOMAINS

    output_root = (
        args.output_root
        if args.output_root is not None
        else (Path("vectors") / config.model_slug / "ood" / "sycophancy")
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 92)
    print("OOD SYCOPHANCY DOMAIN-VECTOR PREPARATION")
    print("=" * 92)
    print(f"Model:          {config.model_label}")
    print(f"Model slug:     {config.model_slug}")
    print(f"Domains:        {domains}")
    print(f"Device:         {args.device}")
    print(f"Output:         {output_root}")

    if args.max_examples is not None:
        print(f"SMOKE PREFIX:   {args.max_examples} examples/domain")

    # ----------------------------------------------------------------
    # Frozen normalization and operating point metadata
    # ----------------------------------------------------------------

    (
        reference_norms,
        reference_norms_path,
    ) = load_reference_norms(config.model_slug)

    (
        canonical_vector,
        canonical_vector_path,
    ) = load_canonical_sycophancy_vector(config.model_slug)

    (
        frozen_layer,
        frozen_operating_points,
        frontier_path,
    ) = load_frozen_sycophancy_layer(config)

    if canonical_vector.shape[0] != reference_norms.shape[0]:
        raise RuntimeError(
            "Canonical sycophancy vector and reference norm layer counts differ."
        )

    print()
    print("Frozen main-experiment artifacts")
    print(f"  layers:                 {reference_norms.shape[0]}")
    print(f"  selected sycophancy l:  {frozen_layer}")

    for direction in [
        "negative",
        "positive",
    ]:
        point = frozen_operating_points[direction]

        print(
            f"  {direction:8s}: "
            f"alpha={point['alpha_star']:+.6f}  "
            f"eta={point['eta_star']:+.6f}"
        )

    # ----------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------

    print()
    print("Loading model...")

    (
        model,
        tokenizer,
    ) = load_model(
        model_name=(config.model_name),
        device=(args.device),
        dtype=(config.torch_dtype),
    )

    model.eval()

    adapter = get_prompt_adapter(
        model=model,
        tokenizer=tokenizer,
        model_name=(config.model_name),
        prompt_family=(config.prompt_family),
    )

    print(f"Prompt family: {adapter.family}")

    # ----------------------------------------------------------------
    # Build / reuse domain vectors
    # ----------------------------------------------------------------

    records = {}

    normalized_by_domain = {}

    for domain in domains:
        domain_root = output_root / domain

        raw_path = domain_root / "raw_vectors.pt"

        normalized_path = domain_root / "normalized_vectors.pt"

        metadata_path = domain_root / "metadata.json"

        (
            dataset,
            full_count,
            data_path,
        ) = load_domain_vector_data(
            data_root=(args.data_root),
            domain=domain,
            max_examples=(args.max_examples),
        )

        print()
        print("=" * 92)
        print(domain.upper())
        print("=" * 92)
        print(f"Data:            {data_path}")
        print(f"Full examples:   {full_count}")
        print(f"Used examples:   {len(dataset)}")

        can_resume = (
            raw_path.is_file()
            and normalized_path.is_file()
            and metadata_path.is_file()
            and not args.no_resume
        )

        if can_resume:
            print("Reusing completed domain vector.")

            raw = torch.load(
                raw_path,
                map_location="cpu",
            ).float()

            normalized = torch.load(
                normalized_path,
                map_location="cpu",
            ).float()

            metadata = load_json(metadata_path)

            if int(metadata["num_examples_used"]) != len(dataset):
                raise RuntimeError(
                    f"{domain}: cached vector was "
                    "built from a different number "
                    "of examples."
                )

        else:
            (
                raw,
                normalized,
                max_norm_error,
            ) = build_domain_vector(
                model=model,
                tokenizer=tokenizer,
                adapter=adapter,
                device=args.device,
                dataset=dataset,
                reference_norms=(reference_norms),
            )

            domain_root.mkdir(
                parents=True,
                exist_ok=True,
            )

            save_tensor(
                raw,
                raw_path,
            )

            save_tensor(
                normalized,
                normalized_path,
            )

            canonical_cosine = tensor_cosine_by_layer(
                normalized,
                canonical_vector,
            )

            metadata = {
                "model_name": (config.model_name),
                "model_label": (config.model_label),
                "model_slug": (config.model_slug),
                "behavior": ("sycophancy"),
                "domain": domain,
                "vector_definition": (
                    "mean(h_matching_behavior - h_not_matching_behavior)"
                ),
                "normalization": (
                    "Each layer rescaled to the "
                    "frozen main-experiment "
                    "reference_layer_norms."
                ),
                "data_path": str(data_path),
                "data_sha256": (sha256_file(data_path)),
                "num_examples_available": int(full_count),
                "num_examples_used": int(len(dataset)),
                "smoke_prefix": bool(args.max_examples is not None),
                "raw_vectors_path": str(raw_path),
                "normalized_vectors_path": str(normalized_path),
                "reference_layer_norms_path": str(reference_norms_path),
                "canonical_sycophancy_vector_path": str(canonical_vector_path),
                "frozen_frontier_path": str(frontier_path),
                "frozen_epsilon": float(DEFAULT_EPSILON),
                "frozen_layer": int(frozen_layer),
                "frozen_operating_points": (frozen_operating_points),
                "max_reference_norm_relative_error": float(max_norm_error),
                "canonical_cosine_at_frozen_layer": float(
                    canonical_cosine[frozen_layer].item()
                ),
                "canonical_cosine_all_layers": {
                    "min": float(canonical_cosine.min().item()),
                    "median": float(canonical_cosine.median().item()),
                    "max": float(canonical_cosine.max().item()),
                },
            }

            write_json(
                metadata_path,
                metadata,
            )

        # Revalidate cached or freshly generated output.
        if raw.shape != canonical_vector.shape:
            raise RuntimeError(
                f"{domain}: raw vector shape "
                f"{tuple(raw.shape)} != canonical "
                f"{tuple(canonical_vector.shape)}."
            )

        if normalized.shape != canonical_vector.shape:
            raise RuntimeError(
                f"{domain}: normalized vector shape "
                f"{tuple(normalized.shape)} != canonical "
                f"{tuple(canonical_vector.shape)}."
            )

        normalized_norms = normalized.norm(dim=1)

        relative_error = (normalized_norms - reference_norms).abs() / reference_norms

        max_relative_error = float(relative_error.max().item())

        if max_relative_error > 1e-5:
            raise RuntimeError(
                f"{domain}: normalization relative error {max_relative_error:.3e}."
            )

        cosine_to_canonical = tensor_cosine_by_layer(
            normalized,
            canonical_vector,
        )

        print(f"Raw shape:       {tuple(raw.shape)}")
        print(f"Normalized:      {tuple(normalized.shape)}")
        print(f"Max norm error:  {max_relative_error:.3e}")
        print(
            f"cos(domain, canonical) "
            f"@ frozen layer {frozen_layer}: "
            f"{cosine_to_canonical[frozen_layer].item():+.6f}"
        )

        normalized_by_domain[domain] = normalized

        records[domain] = {
            "raw_vectors_path": str(raw_path),
            "normalized_vectors_path": str(normalized_path),
            "metadata_path": str(metadata_path),
            "raw_sha256": (sha256_file(raw_path)),
            "normalized_sha256": (sha256_file(normalized_path)),
            "num_examples": int(len(dataset)),
            "max_reference_norm_relative_error": float(max_relative_error),
            "canonical_cosine_at_frozen_layer": float(
                cosine_to_canonical[frozen_layer].item()
            ),
        }

    # ----------------------------------------------------------------
    # Cross-domain vector geometry
    # ----------------------------------------------------------------

    if all(domain in normalized_by_domain for domain in DOMAINS):
        cross_cosine = tensor_cosine_by_layer(
            normalized_by_domain["nlp"],
            normalized_by_domain["political"],
        )

        print()
        print("=" * 92)
        print("NLP vs POLITICAL VECTOR GEOMETRY")
        print("=" * 92)
        print(f"Frozen layer {frozen_layer}: {cross_cosine[frozen_layer].item():+.6f}")
        print(f"All-layer min:             {cross_cosine.min().item():+.6f}")
        print(f"All-layer median:          {cross_cosine.median().item():+.6f}")
        print(f"All-layer max:             {cross_cosine.max().item():+.6f}")

        cross_domain_geometry = {
            "cosine_at_frozen_layer": float(cross_cosine[frozen_layer].item()),
            "all_layers_min": float(cross_cosine.min().item()),
            "all_layers_median": float(cross_cosine.median().item()),
            "all_layers_max": float(cross_cosine.max().item()),
        }

    else:
        cross_domain_geometry = None

    # ----------------------------------------------------------------
    # Model-level manifest
    # ----------------------------------------------------------------

    manifest = {
        "protocol": ("sycophancy_cross_domain_vectors_v1"),
        "model_name": (config.model_name),
        "model_label": (config.model_label),
        "model_slug": (config.model_slug),
        "behavior": ("sycophancy"),
        "domains_requested": domains,
        "frozen_epsilon": float(DEFAULT_EPSILON),
        "frozen_layer": int(frozen_layer),
        "frozen_operating_points": (frozen_operating_points),
        "normalization_policy": (
            "Source-domain raw vectors are normalized "
            "to the existing model-level reference norms "
            "from the original seven-behavior experiment. "
            "No OOD-specific normalization scale is learned."
        ),
        "reference_layer_norms_path": str(reference_norms_path),
        "canonical_sycophancy_vector_path": str(canonical_vector_path),
        "smoke_prefix": bool(args.max_examples is not None),
        "domains": records,
        "nlp_vs_political_geometry": (cross_domain_geometry),
    }

    manifest_path = output_root / "manifest.json"

    write_json(
        manifest_path,
        manifest,
    )

    print()
    print("=" * 92)
    print("OOD SYCOPHANCY VECTOR PREPARATION COMPLETE")
    print("=" * 92)

    for domain in domains:
        record = records[domain]

        print(
            f"{domain:10s} "
            f"N={record['num_examples']:3d}  "
            f"cos@layer={record['canonical_cosine_at_frozen_layer']:+.6f}"
        )

    print()
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
