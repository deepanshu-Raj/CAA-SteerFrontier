from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]

DEFAULT_ALPHAS = [
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
]

DEFAULT_LAYER_SWEEP_ALPHAS = [-1.0, 1.0]


@dataclass(frozen=True)
class ExperimentLayout:
    """Filesystem layout for one model-specific experiment.

    ``vector_root``, ``results_root`` and ``plots_root`` should normally already
    include the model slug, for example ``vectors/llama3.1-8b-instruct``.
    """

    data_root: Path = Path("data")
    vector_root: Path = Path("vectors")
    results_root: Path = Path("results")
    plots_root: Path = Path("plots")

    def generation_path(self, behavior: str) -> Path:
        return self.data_root / behavior / "generate.json"

    def test_ab_path(self, behavior: str) -> Path:
        return self.data_root / behavior / "test_ab.json"

    def mmlu_path(self) -> Path:
        return self.data_root / "mmlu" / "test.json"

    def vector_dir(self, behavior: str) -> Path:
        return self.vector_root / behavior

    def raw_vectors_path(self, behavior: str) -> Path:
        return self.vector_dir(behavior) / "raw_vectors.pt"

    def normalized_vectors_path(
        self,
        behavior: str,
        tag: Optional[str] = None,
    ) -> Path:
        filename = (
            "normalized_vectors.pt"
            if tag is None
            else f"normalized_vectors_{tag}.pt"
        )
        return self.vector_dir(behavior) / filename

    def normalization_dir(self) -> Path:
        return self.vector_root / "normalization"

    def reference_norms_path(self, tag: str) -> Path:
        return self.normalization_dir() / f"{tag}_reference_layer_norms.pt"

    def results_dir(self, behavior: str) -> Path:
        return self.results_root / behavior

    def comparison_results_dir(self) -> Path:
        return self.results_root / "comparison"

    def plots_dir(self, behavior: str) -> Path:
        return self.plots_root / behavior

    def comparison_plots_dir(self) -> Path:
        return self.plots_root / "comparison"


# Backward-compatible helpers.
def generation_path(behavior: str, data_root: PathLike = "data") -> Path:
    return Path(data_root) / behavior / "generate.json"


def test_ab_path(behavior: str, data_root: PathLike = "data") -> Path:
    return Path(data_root) / behavior / "test_ab.json"


def raw_vectors_path(behavior: str, vector_root: PathLike = "vectors") -> Path:
    return Path(vector_root) / behavior / "raw_vectors.pt"


def normalized_vectors_path(
    behavior: str,
    vector_root: PathLike = "vectors",
    tag: Optional[str] = None,
) -> Path:
    filename = (
        "normalized_vectors.pt"
        if tag is None
        else f"normalized_vectors_{tag}.pt"
    )
    return Path(vector_root) / behavior / filename


def results_dir(behavior: str, results_root: PathLike = "results") -> Path:
    return Path(results_root) / behavior


def plots_dir(behavior: str, plots_root: PathLike = "plots") -> Path:
    return Path(plots_root) / behavior
