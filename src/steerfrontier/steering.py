from typing import List

import torch

from steerfrontier.models import get_decoder_layers


class SteeringIntervention:
    """Add one CAA vector to the residual output of a decoder layer."""

    def __init__(
        self,
        model,
        layer_idx: int,
        vector: torch.Tensor,
        coefficient: float,
        from_position: int,
    ):
        self.model = model
        self.layer_idx = int(layer_idx)
        self.vector = vector
        self.coefficient = float(coefficient)
        self.from_position = int(from_position)

        self.decoder_layers = get_decoder_layers(model)
        if self.layer_idx < 0 or self.layer_idx >= len(self.decoder_layers):
            raise ValueError(
                f"Invalid layer {self.layer_idx}; model has "
                f"{len(self.decoder_layers)} layers."
            )

        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def _resolve_positions(self, hidden_states, kwargs):
        position_ids = kwargs.get("position_ids")
        if position_ids is not None:
            return position_ids

        # Newer HF decoder implementations may expose cache_position instead.
        cache_position = kwargs.get("cache_position")
        if cache_position is not None:
            if cache_position.ndim == 1:
                cache_position = cache_position.unsqueeze(0)
            return cache_position

        # Fallback for evaluation/extraction path, which always performs a
        # full forward pass with use_cache=False.
        seq_len = hidden_states.shape[1]
        return torch.arange(
            seq_len,
            device=hidden_states.device,
            dtype=torch.long,
        ).unsqueeze(0)

    def _hook(self, module, args, kwargs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        positions = self._resolve_positions(hidden_states, kwargs)

        vector = self.coefficient * self.vector.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        mask = (positions >= self.from_position).unsqueeze(-1)
        steered_hidden = hidden_states + mask.to(hidden_states.dtype) * vector

        if isinstance(output, tuple):
            return (steered_hidden,) + output[1:]
        return steered_hidden

    def __enter__(self):
        layer = self.decoder_layers[self.layer_idx]
        handle = layer.register_forward_hook(
            self._hook,
            with_kwargs=True,
        )
        self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
