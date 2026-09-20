"""Filesystem layout and default sweep settings for one model experiment.

The layout centralizes paths for vectors, normalization artifacts, results, and
plots so scripts share a consistent artifact contract.
"""

from dataclasses import dataclass
from pathlib import Path

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


# =============================================================================
# Experiment layout
# =============================================================================

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

    def vector_data_path(self, behavior: str) -> Path:
        return self.data_root / behavior / "vector.json"

    def validation_data_path(self, behavior: str) -> Path:
        return self.data_root / behavior / "validation.json"

    def test_data_path(self, behavior: str) -> Path:
        return self.data_root / behavior / "test.json"

    def mmlu_path(self) -> Path:
        return self.data_root / "mmlu" / "test.json"

    def vector_dir(self, behavior: str) -> Path:
        return self.vector_root / behavior

    def raw_vectors_path(self, behavior: str) -> Path:
        return self.vector_dir(behavior) / "raw_vectors.pt"

    def normalized_vectors_path(self, behavior: str) -> Path:
        return self.vector_dir(behavior) / "normalized_vectors.pt"

    def normalization_dir(self) -> Path:
        return self.vector_root / "normalization"

    def reference_norms_path(self) -> Path:
        return self.normalization_dir() / "reference_layer_norms.pt"

    def results_dir(self, behavior: str) -> Path:
        return self.results_root / behavior

    def comparison_results_dir(self) -> Path:
        return self.results_root / "comparison"

    def plots_dir(self, behavior: str) -> Path:
        return self.plots_root / behavior

    def comparison_plots_dir(self) -> Path:
        return self.plots_root / "comparison"
