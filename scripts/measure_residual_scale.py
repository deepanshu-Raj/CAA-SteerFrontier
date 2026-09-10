import argparse
import json

import torch

from steerfrontier.calibration import measure_residual_scale
from steerfrontier.config import load_experiment_config
from steerfrontier.data import load_vector_data
from steerfrontier.models import load_model
from steerfrontier.prompts import get_prompt_adapter


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure unsteered residual-stream scale for steering calibration."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--examples-per-behavior",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_experiment_config(args.config)

    print("=" * 76)
    print("RESIDUAL-SCALE CALIBRATION")
    print("=" * 76)
    print("Model:", config.model_label)
    print(
        "Examples per behavior:",
        args.examples_per_behavior,
    )
    print()

    model, tokenizer = load_model(
        model_name=config.model_name,
        device=args.device,
        dtype=config.torch_dtype,
    )

    adapter = get_prompt_adapter(
        model=model,
        tokenizer=tokenizer,
        model_name=config.model_name,
        prompt_family=config.prompt_family,
    )

    datasets = {}

    for behavior in config.behaviors:
        path = config.layout.vector_data_path(behavior)

        datasets[behavior] = load_vector_data(path)

        print(f"{behavior:18s} available={len(datasets[behavior])}")

    print()

    residual_scale, counts = measure_residual_scale(
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        behavior_datasets=datasets,
        max_examples_per_behavior=args.examples_per_behavior,
        device=args.device,
    )

    normalization_dir = config.layout.normalization_dir()

    normalization_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    residual_path = normalization_dir / "residual_scale.pt"

    torch.save(
        residual_scale,
        residual_path,
    )

    reference_path = config.layout.reference_norms_path()

    if not reference_path.exists():
        raise FileNotFoundError(
            f"Missing normalized-vector reference norms: {reference_path}"
        )

    reference_norms = torch.load(
        reference_path,
        map_location="cpu",
    ).float()

    if reference_norms.shape != residual_scale.shape:
        raise ValueError(
            "Reference vector norms and residual scale have different shapes."
        )

    perturbation_ratio = reference_norms / residual_scale

    ratio_path = normalization_dir / "perturbation_ratio.pt"

    torch.save(
        perturbation_ratio,
        ratio_path,
    )

    metadata = {
        "model_name": config.model_name,
        "model_label": config.model_label,
        "model_slug": config.model_slug,
        "prompt_family": adapter.family,
        "source_split": "vector",
        "measurement_position": ("steering_start_position"),
        "examples_per_behavior_requested": (args.examples_per_behavior),
        "examples_used": counts,
        "total_examples": sum(counts.values()),
        "definition": ("r_l = mean_x ||x_{l,t_start}||_2"),
        "eta_definition": (
            "eta(alpha,l) = |alpha| * reference_vector_norm_l / residual_scale_l"
        ),
    }

    metadata_path = normalization_dir / "residual_scale_metadata.json"

    with metadata_path.open("w") as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print("=" * 76)
    print("CALIBRATION SUMMARY")
    print("=" * 76)
    print("layer   depth      m_l        r_l       m_l/r_l")

    num_layers = residual_scale.shape[0]

    for layer in range(num_layers):
        depth = layer / (num_layers - 1) if num_layers > 1 else 0.0

        print(
            f"{layer:5d}  "
            f"{depth:6.3f}  "
            f"{reference_norms[layer]:9.4f}  "
            f"{residual_scale[layer]:9.4f}  "
            f"{perturbation_ratio[layer]:9.6f}"
        )

    print()
    print("Saved:")
    print(" ", residual_path)
    print(" ", ratio_path)
    print(" ", metadata_path)


if __name__ == "__main__":
    main()
