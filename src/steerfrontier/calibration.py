from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from steerfrontier.activations import LayerActivationCapture
from steerfrontier.models import get_decoder_layers
from steerfrontier.prompts import PromptAdapter


@torch.no_grad()
def measure_residual_scale(
    model,
    tokenizer,
    adapter: PromptAdapter,
    behavior_datasets: Dict[str, List[dict]],
    max_examples_per_behavior: int = 100,
    device: str = "cuda",
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Measure the mean unsteered residual-stream norm at each decoder layer.

    For each behavior, use at most `max_examples_per_behavior` prompts.
    Behaviors therefore contribute approximately equally to the estimate.

    Residual activations are measured at the same token position from which
    the steering intervention begins.

    Returns:
        residual_scale:
            Tensor [num_layers], where element l is

                E_x ||x_{l,t_start}||_2

        counts:
            Number of examples used from each behavior.
    """

    if not behavior_datasets:
        raise ValueError("At least one behavior dataset is required.")

    num_layers = len(get_decoder_layers(model))

    norm_sum = torch.zeros(
        num_layers,
        dtype=torch.float64,
    )

    total_examples = 0
    counts = {}

    with LayerActivationCapture(model) as capture:
        for behavior, dataset in behavior_datasets.items():
            if not dataset:
                raise ValueError(f"Residual calibration dataset is empty: {behavior}")

            selected = dataset[:max_examples_per_behavior]
            counts[behavior] = len(selected)

            for item in tqdm(
                selected,
                desc=f"Residual scale — {behavior}",
            ):
                scoring = adapter.build_scoring_prompt(
                    tokenizer,
                    item["question"],
                )

                capture.clear()

                model(
                    input_ids=scoring.input_ids.to(device),
                    use_cache=False,
                )

                activations = capture.at_position(scoring.steering_start_position)

                if activations.shape[0] != num_layers:
                    raise RuntimeError(
                        "Unexpected number of captured layers: "
                        f"{activations.shape[0]} != {num_layers}"
                    )

                norms = activations.norm(
                    p=2,
                    dim=1,
                )

                norm_sum += norms.double()
                total_examples += 1

    if total_examples == 0:
        raise RuntimeError("No examples were used for residual-scale calibration.")

    residual_scale = (norm_sum / total_examples).float()

    if torch.any(residual_scale <= 0):
        raise RuntimeError("Residual scale must be positive at every layer.")

    return residual_scale, counts


def steering_calibration(
    alpha: float,
    layer: int,
    vectors: torch.Tensor,
    residual_scale: torch.Tensor,
):
    """
    Return calibrated steering quantities for one layer/coefficient.

    eta = |alpha| * ||v_tilde_l|| / r_l
    signed_eta = alpha * ||v_tilde_l|| / r_l
    """

    if vectors.ndim != 2:
        raise ValueError(
            f"Expected vectors [layers, hidden], got {tuple(vectors.shape)}"
        )

    if residual_scale.ndim != 1:
        raise ValueError("Expected residual_scale to be rank-1.")

    if vectors.shape[0] != residual_scale.shape[0]:
        raise ValueError("Vector layers and residual-scale layers do not match.")

    vector_norm = float(vectors[layer].float().norm(p=2).item())

    residual_norm = float(residual_scale[layer].item())

    if residual_norm <= 0.0:
        raise ValueError(f"Residual scale must be positive at layer {layer}.")

    ratio = vector_norm / residual_norm

    eta = abs(float(alpha)) * ratio
    signed_eta = float(alpha) * ratio

    num_layers = vectors.shape[0]

    fractional_depth = layer / (num_layers - 1) if num_layers > 1 else 0.0

    return {
        "vector_norm": vector_norm,
        "residual_scale": residual_norm,
        "perturbation_ratio": ratio,
        "eta": eta,
        "signed_eta": signed_eta,
        "fractional_depth": fractional_depth,
    }
