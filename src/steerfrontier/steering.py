from typing import List, Sequence, Union

import torch

from steerfrontier.models import get_decoder_layers

PositionSpec = Union[
    int,
    Sequence[int],
    torch.Tensor,
]


class SteeringIntervention:
    """
    Add one CAA vector to the residual output of a decoder layer.

    `from_position` may be either:

        int
            One steering-start position shared by the entire batch.

        Sequence[int] / Tensor
            One steering-start position per batch element.

    This allows padded batches whose examples have different prompt
    or continuation boundaries.
    """

    def __init__(
        self,
        model,
        layer_idx: int,
        vector: torch.Tensor,
        coefficient: float,
        from_position: PositionSpec,
    ):
        self.model = model
        self.layer_idx = int(layer_idx)

        self.vector = vector

        self.coefficient = float(coefficient)

        if isinstance(
            from_position,
            torch.Tensor,
        ):
            positions = (
                from_position.detach()
                .clone()
                .to(
                    dtype=torch.long,
                    device="cpu",
                )
                .reshape(-1)
            )

            if positions.numel() == 0:
                raise ValueError("from_position tensor cannot be empty.")

            self.from_position = positions

        elif isinstance(
            from_position,
            int,
        ):
            if from_position < 0:
                raise ValueError("from_position must be non-negative.")

            self.from_position = int(from_position)

        else:
            positions = torch.tensor(
                list(from_position),
                dtype=torch.long,
            ).reshape(-1)

            if positions.numel() == 0:
                raise ValueError("from_position sequence cannot be empty.")

            if (positions < 0).any():
                raise ValueError("from_position values must be non-negative.")

            self.from_position = positions

        self.decoder_layers = get_decoder_layers(model)

        if self.layer_idx < 0 or self.layer_idx >= len(self.decoder_layers):
            raise ValueError(
                f"Invalid layer "
                f"{self.layer_idx}; "
                f"model has "
                f"{len(self.decoder_layers)} "
                "layers."
            )

        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def _resolve_positions(
        self,
        hidden_states,
        kwargs,
    ):
        position_ids = kwargs.get("position_ids")

        if position_ids is not None:
            return position_ids

        # Newer HF decoder implementations may
        # expose cache_position instead.
        cache_position = kwargs.get("cache_position")

        if cache_position is not None:
            if cache_position.ndim == 1:
                cache_position = cache_position.unsqueeze(0)

            return cache_position

        # Full-forward fallback.
        seq_len = hidden_states.shape[1]

        return torch.arange(
            seq_len,
            device=(hidden_states.device),
            dtype=torch.long,
        ).unsqueeze(0)

    def _resolve_start_positions(
        self,
        hidden_states,
    ):
        """
        Return either a scalar start position or a
        [batch_size, 1] tensor suitable for broadcasting
        against sequence positions.
        """

        if isinstance(
            self.from_position,
            int,
        ):
            return self.from_position

        batch_size = int(hidden_states.shape[0])

        starts = self.from_position.to(
            device=(hidden_states.device),
            dtype=torch.long,
        ).reshape(-1)

        if starts.numel() == 1:
            return int(starts.item())

        if starts.numel() != batch_size:
            raise ValueError(
                "Per-example "
                "from_position has "
                f"{starts.numel()} values, "
                f"but hidden-state batch "
                f"size is {batch_size}."
            )

        return starts.unsqueeze(1)

    def _hook(
        self,
        module,
        args,
        kwargs,
        output,
    ):
        hidden_states = (
            output[0]
            if isinstance(
                output,
                tuple,
            )
            else output
        )

        positions = self._resolve_positions(
            hidden_states,
            kwargs,
        )

        starts = self._resolve_start_positions(hidden_states)

        vector = self.coefficient * self.vector.to(
            device=(hidden_states.device),
            dtype=(hidden_states.dtype),
        )

        mask = (positions >= starts).unsqueeze(-1)

        steered_hidden = hidden_states + mask.to(hidden_states.dtype) * vector

        if isinstance(
            output,
            tuple,
        ):
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

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        for handle in self.handles:
            handle.remove()

        self.handles.clear()
