"""Activation-capture utilities for decoder-layer residual streams.

Hooks expose the token representations required for contrastive vector extraction
without modifying the underlying model computation.
"""

from typing import Dict, List, Optional

import torch

from steerfrontier.models import get_decoder_layers


# =============================================================================
# Activation capture
# =============================================================================

class LayerActivationCapture:
    """Capture residual-stream outputs from transformer decoder blocks."""

    def __init__(self, model, layers: Optional[List[int]] = None):
        self.model = model
        self.decoder_layers = get_decoder_layers(model)
        self.layer_indices = (
            list(range(len(self.decoder_layers))) if layers is None else list(layers)
        )

        for layer_idx in self.layer_indices:
            if layer_idx < 0 or layer_idx >= len(self.decoder_layers):
                raise ValueError(
                    f"Invalid layer {layer_idx}; model has {len(self.decoder_layers)} layers."
                )

        self.activations: Dict[int, torch.Tensor] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            self.activations[layer_idx] = hidden_states.detach()

        return hook

    def __enter__(self):
        for layer_idx in self.layer_indices:
            layer = self.decoder_layers[layer_idx]
            self.handles.append(
                layer.register_forward_hook(self._make_hook(layer_idx))
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.activations.clear()

    def clear(self):
        self.activations.clear()

    def at_position(self, position: int) -> torch.Tensor:
        """Return captured activations at one token position.

        Shape is ``[num_captured_layers, hidden_size]`` in ``layer_indices`` order.
        The returned tensor is CPU float32 so vector accumulation is stable and
        independent of model execution dtype.
        """

        missing = [idx for idx in self.layer_indices if idx not in self.activations]
        if missing:
            raise RuntimeError(f"Missing captured activations for layers: {missing}")

        layer_activations = []

        for layer_idx in self.layer_indices:
            hidden_states = self.activations[layer_idx]
            seq_len = hidden_states.shape[1]

            resolved_position = position if position >= 0 else seq_len + position
            if resolved_position < 0 or resolved_position >= seq_len:
                raise IndexError(
                    f"Position {position} is out of range for sequence length {seq_len}."
                )

            activation = hidden_states[0, resolved_position, :]
            layer_activations.append(activation.detach().float().cpu())

        return torch.stack(layer_activations)
