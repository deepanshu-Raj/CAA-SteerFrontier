import argparse
from itertools import combinations

import pandas as pd
import torch
import torch.nn.functional as F

from steerfrontier.config import load_experiment_config
from steerfrontier.data import load_test_data
from steerfrontier.evaluation import evaluate_behavior
from steerfrontier.models import load_model
from steerfrontier.prompts import get_prompt_adapter
from steerfrontier.workflow import load_normalized_vectors


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generic cross-behavior analysis: raw-vector cosine geometry and "
            "causal cross-steering at one or more common layers."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--layers",
        nargs="*",
        type=int,
        default=[],
        help="Common layers for causal cross-steering. Geometry always runs.",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[-1.0, 0.0, 1.0],
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def run_geometry(config):
    layout = config.layout
    raw = {
        behavior: torch.load(
            layout.raw_vectors_path(behavior),
            map_location="cpu",
        ).float()
        for behavior in config.behaviors
    }

    shapes = {tuple(v.shape) for v in raw.values()}
    if len(shapes) != 1:
        raise ValueError(f"Raw-vector shapes differ: {shapes}")

    rows = []
    for behavior_a, behavior_b in combinations(config.behaviors, 2):
        a = F.normalize(raw[behavior_a], dim=1)
        b = F.normalize(raw[behavior_b], dim=1)
        cosine = torch.sum(a * b, dim=1)

        for layer, value in enumerate(cosine.tolist()):
            rows.append(
                {
                    "model_slug": config.model_slug,
                    "behavior_a": behavior_a,
                    "behavior_b": behavior_b,
                    "layer": layer,
                    "cosine_similarity": float(value),
                }
            )

    output_dir = layout.comparison_results_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cross_behavior_vector_cosine.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print("Saved:", output_path)


def run_cross_steering(config, layers, alphas, device):
    if not layers:
        return

    model, tokenizer = load_model(
        model_name=config.model_name,
        device=device,
        dtype=config.torch_dtype,
    )
    adapter = get_prompt_adapter(
        model=model,
        tokenizer=tokenizer,
        model_name=config.model_name,
        prompt_family=config.prompt_family,
    )

    vectors = load_normalized_vectors(config)
    datasets = {
        behavior: load_test_data(config.layout.test_data_path(behavior))
        for behavior in config.behaviors
    }

    for layer in layers:
        if layer < 0 or layer >= next(iter(vectors.values())).shape[0]:
            raise ValueError(f"Invalid cross-steering layer {layer}.")

        baselines = {}
        for target in config.behaviors:
            reference_behavior = config.behaviors[0]
            baselines[target] = float(
                evaluate_behavior(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=datasets[target],
                    layer_idx=layer,
                    vector=vectors[reference_behavior][layer],
                    coefficient=0.0,
                    device=device,
                    prompt_adapter=adapter,
                )["behavior_score"]
            )

        rows = []
        for source in config.behaviors:
            vector = vectors[source][layer]
            for target in config.behaviors:
                baseline = baselines[target]
                for alpha in alphas:
                    if float(alpha) == 0.0:
                        score = baseline
                    else:
                        score = float(
                            evaluate_behavior(
                                model=model,
                                tokenizer=tokenizer,
                                dataset=datasets[target],
                                layer_idx=layer,
                                vector=vector,
                                coefficient=float(alpha),
                                device=device,
                                prompt_adapter=adapter,
                            )["behavior_score"]
                        )

                    rows.append(
                        {
                            "model_slug": config.model_slug,
                            "layer": int(layer),
                            "source_behavior": source,
                            "target_behavior": target,
                            "alpha": float(alpha),
                            "behavior_score": score,
                            "baseline": baseline,
                            "delta_from_baseline": score - baseline,
                        }
                    )

        output_dir = config.layout.comparison_results_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"cross_behavior_steering_layer{layer}.csv"
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print("Saved:", output_path)


def main():
    args = parse_args()
    config = load_experiment_config(args.config)

    run_geometry(config)
    run_cross_steering(
        config=config,
        layers=args.layers,
        alphas=args.alphas,
        device=args.device,
    )


if __name__ == "__main__":
    main()
