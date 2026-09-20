"""Shared layer-wise normalization for behavior steering vectors.

All behavior vectors at a layer are rescaled to the same reference norm, preserving
direction while avoiding behavior-specific magnitude confounds.
"""

from typing import Dict, Tuple

import torch


# =============================================================================
# Vector normalization
# =============================================================================

def _validate_behavior_vectors(behavior_vectors: Dict[str, torch.Tensor]):
    if not behavior_vectors:
        raise ValueError("At least one behavior vector tensor is required.")

    shapes = {tuple(v.shape) for v in behavior_vectors.values()}
    if len(shapes) != 1:
        raise ValueError(f"Behavior vector tensors must share one shape; found {shapes}.")

    shape = next(iter(shapes))
    if len(shape) != 2:
        raise ValueError(
            "Expected behavior vectors with shape [num_layers, hidden_size], "
            f"found {shape}."
        )


def compute_mean_layer_norms(
    behavior_vectors: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute the mean steering-vector norm across behaviors at each layer."""

    _validate_behavior_vectors(behavior_vectors)
    norms = [vectors.norm(dim=1) for vectors in behavior_vectors.values()]
    return torch.stack(norms, dim=0).mean(dim=0)


def normalize_to_layer_norms(
    vectors: torch.Tensor,
    target_norms: torch.Tensor,
) -> torch.Tensor:
    """Rescale each layer's vector to a supplied target norm."""

    if vectors.ndim != 2:
        raise ValueError("vectors must have shape [num_layers, hidden_size].")
    if target_norms.ndim != 1 or target_norms.shape[0] != vectors.shape[0]:
        raise ValueError(
            "target_norms must have shape [num_layers] matching vectors."
        )

    current_norms = vectors.norm(dim=1)
    if torch.any(current_norms == 0):
        raise ValueError("Cannot normalize a zero steering vector.")

    scale = target_norms.to(current_norms) / current_norms
    return vectors * scale.unsqueeze(1)


def normalize_behavior_vectors(
    behavior_vectors: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Normalize all behaviors to their shared mean per-layer norm.

    Returns ``(normalized_vectors, reference_layer_norms)``.
    """

    target_norms = compute_mean_layer_norms(behavior_vectors)
    normalized = {
        behavior: normalize_to_layer_norms(vectors, target_norms)
        for behavior, vectors in behavior_vectors.items()
    }
    return normalized, target_norms
