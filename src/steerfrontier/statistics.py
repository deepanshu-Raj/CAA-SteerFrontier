"""Deterministic bootstrap utilities used by steering analyses.

Confidence intervals are computed from explicit seeds so uncertainty estimates
remain reproducible across reruns.
"""

from typing import Iterable, Tuple

import numpy as np

DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_CONFIDENCE_LEVEL = 0.95


# =============================================================================
# Bootstrap statistics
# =============================================================================

def bootstrap_mean_ci(
    values: Iterable[float],
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    batch_size: int = 1000,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for the mean.

    For our behavior experiments, `values` should be the paired
    per-example differences:

        steered_matching_prob - baseline_matching_prob

    Because the subtraction is performed before resampling, this is a
    paired bootstrap.

    Returns:
        mean, ci_low, ci_high
    """

    values = np.asarray(
        list(values),
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError("Bootstrap input must be one-dimensional.")

    if len(values) == 0:
        raise ValueError("Cannot bootstrap an empty sample.")

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie between 0 and 1.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    rng = np.random.default_rng(seed)

    n = len(values)

    bootstrap_means = np.empty(
        n_bootstrap,
        dtype=np.float64,
    )

    offset = 0

    while offset < n_bootstrap:
        current_batch = min(
            batch_size,
            n_bootstrap - offset,
        )

        indices = rng.integers(
            0,
            n,
            size=(current_batch, n),
        )

        bootstrap_means[offset : offset + current_batch] = values[indices].mean(axis=1)

        offset += current_batch

    alpha = 1.0 - confidence_level

    ci_low = float(
        np.quantile(
            bootstrap_means,
            alpha / 2.0,
        )
    )

    ci_high = float(
        np.quantile(
            bootstrap_means,
            1.0 - alpha / 2.0,
        )
    )

    mean = float(values.mean())

    return mean, ci_low, ci_high
