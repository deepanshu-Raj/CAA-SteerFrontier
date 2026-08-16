from pathlib import Path

MODEL = "meta-llama/Llama-2-7b-chat-hf"

BEHAVIORS = [
    "coordinate-other-ais",
    "corrigible-neutral-HHH",
    "hallucination",
    "myopic-reward",
    "refusal",
    "survival-instinct",
    "sycophancy",
]

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


def generation_path(behavior):
    return Path("data") / behavior / "generate.json"


def test_ab_path(behavior):
    return Path("data") / behavior / "test_ab.json"


def raw_vectors_path(behavior):
    return Path("vectors") / behavior / "raw_vectors.pt"


def normalized_vectors_path(behavior):
    return Path("vectors") / behavior / "normalized_vectors.pt"


def results_dir(behavior):
    return Path("results") / behavior


def plots_dir(behavior):
    return Path("plots") / behavior
