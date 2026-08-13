import torch


class LayerActivationCapture:
    """
    Capture the residual-stream output of every transformer decoder block.

    For Llama-2, this corresponds to the output of model.model.layers[i],
    matching the activation location used by the original CAA implementation.
    """

    def __init__(self, model):
        self.model = model
        self.activations: dict[int, torch.Tensor] = {}  # Pure built-in dict
        self.handles: list[torch.utils.hooks.RemovableHandle] = []  # Pure built-in list

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            self.activations[layer_idx] = hidden_states.detach()

        return hook

    def __enter__(self):
        for layer_idx, layer in enumerate(self.model.model.layers):
            handle = layer.register_forward_hook(self._make_hook(layer_idx))
            self.handles.append(handle)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()

        self.handles.clear()
        self.activations.clear()

    def clear(self):
        self.activations.clear()

    def at_position(self, position: int) -> torch.Tensor:
        """
        Return activations at one token position from all layers.

        Shape:
            [num_layers, hidden_size]
        """

        expected_layers = self.model.config.num_hidden_layers

        if len(self.activations) != expected_layers:
            raise RuntimeError(
                f"Expected {expected_layers} captured layers, "
                f"found {len(self.activations)}."
            )

        layer_activations = []

        for layer_idx in range(expected_layers):
            hidden_states = self.activations[layer_idx]

            # [batch, sequence, hidden] -> [hidden]
            activation = hidden_states[0, position, :]

            layer_activations.append(activation.detach().float().cpu())

        return torch.stack(layer_activations)
