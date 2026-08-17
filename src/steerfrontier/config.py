from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

from steerfrontier.experiment import (
    DEFAULT_ALPHAS,
    DEFAULT_LAYER_SWEEP_ALPHAS,
    ExperimentLayout,
)


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    model_label: str
    model_slug: str
    behaviors: List[str]
    data_root: Path
    vector_root: Path
    results_root: Path
    plots_root: Path
    normalization_tag: str = "3beh"
    prompt_family: str = "auto"
    dtype: str = "float32"
    alphas: List[float] = None
    layer_sweep_alphas: List[float] = None
    candidate_width: int = 9

    def __post_init__(self):
        if not self.behaviors:
            raise ValueError("Config must contain at least one behavior.")

        alphas = DEFAULT_ALPHAS if self.alphas is None else self.alphas
        layer_alphas = (
            DEFAULT_LAYER_SWEEP_ALPHAS
            if self.layer_sweep_alphas is None
            else self.layer_sweep_alphas
        )

        object.__setattr__(self, "alphas", [float(x) for x in alphas])
        object.__setattr__(
            self,
            "layer_sweep_alphas",
            [float(x) for x in layer_alphas],
        )

        if 0.0 not in self.alphas:
            raise ValueError("Config alphas must include 0.0.")
        if len(self.layer_sweep_alphas) != 2:
            raise ValueError("layer_sweep_alphas must contain exactly two values.")
        if self.candidate_width < 1:
            raise ValueError("candidate_width must be >= 1.")

    @property
    def layout(self) -> ExperimentLayout:
        return ExperimentLayout(
            data_root=self.data_root,
            vector_root=self.vector_root,
            results_root=self.results_root,
            plots_root=self.plots_root,
        )

    @property
    def torch_dtype(self) -> torch.dtype:
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        key = self.dtype.lower()
        if key not in mapping:
            raise ValueError(
                f"Unsupported dtype {self.dtype!r}. Choose one of {sorted(mapping)}."
            )
        return mapping[key]


def load_experiment_config(path: str) -> ExperimentConfig:
    path_obj = Path(path)
    with path_obj.open("r") as f:
        payload: Dict[str, Any] = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path_obj}.")

    required = {
        "model_name",
        "model_label",
        "model_slug",
        "behaviors",
        "data_root",
        "vector_root",
        "results_root",
        "plots_root",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Config {path_obj} is missing fields: {sorted(missing)}")

    return ExperimentConfig(
        model_name=str(payload["model_name"]),
        model_label=str(payload["model_label"]),
        model_slug=str(payload["model_slug"]),
        behaviors=[str(x) for x in payload["behaviors"]],
        data_root=Path(payload["data_root"]),
        vector_root=Path(payload["vector_root"]),
        results_root=Path(payload["results_root"]),
        plots_root=Path(payload["plots_root"]),
        normalization_tag=str(payload.get("normalization_tag", "3beh")),
        prompt_family=str(payload.get("prompt_family", "auto")),
        dtype=str(payload.get("dtype", "float32")),
        alphas=payload.get("alphas"),
        layer_sweep_alphas=payload.get("layer_sweep_alphas"),
        candidate_width=int(payload.get("candidate_width", 9)),
    )
