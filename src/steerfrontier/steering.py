import torch


class SteeringIntervention:
    """
    Add a steering vector to the residual-stream output of one decoder layer.
    """

    def __init__(
        self,
        model,
        layer_idx: int,
        vector: torch.Tensor,
        coefficient: float,
        from_position: int,
    ):
        self.model = model
        self.layer_idx = layer_idx
        self.vector = vector
        self.coefficient = coefficient
        self.from_position = from_position

        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, module, args, kwargs, output):
        hidden_states = output[0]

        position_ids = kwargs.get("position_ids")

        if position_ids is None:
            raise RuntimeError("Decoder layer did not receive position_ids.")

        vector = self.coefficient * self.vector.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        mask = (position_ids >= self.from_position).unsqueeze(-1)

        steered_hidden = hidden_states + mask.to(hidden_states.dtype) * vector

        return (steered_hidden,) + output[1:]

    def __enter__(self):
        layer = self.model.model.layers[self.layer_idx]

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
