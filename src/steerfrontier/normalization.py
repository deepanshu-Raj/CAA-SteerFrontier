import torch


def compute_mean_layer_norms(
    behavior_vectors: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Compute the mean steering-vector norm across behaviors for each layer.

    Each input tensor must have shape:
        [num_layers, hidden_size]

    Returns:
        [num_layers]
    """

    norms = []

    for vectors in behavior_vectors.values():
        norms.append(vectors.norm(dim=1))

    return torch.stack(norms).mean(dim=0)


def normalize_to_layer_norms(
    vectors: torch.Tensor,
    target_norms: torch.Tensor,
) -> torch.Tensor:
    """
    Rescale each layer's vector to the provided target norm.
    """

    current_norms = vectors.norm(dim=1)

    if torch.any(current_norms == 0):
        raise ValueError("Cannot normalize a zero steering vector.")

    scale = target_norms / current_norms

    return vectors * scale.unsqueeze(1)
