import numpy as np
import pytest

from steerfrontier.statistics import (
    bootstrap_mean_ci,
)


def test_bootstrap_mean_ci_is_deterministic():
    values = [
        -0.2,
        -0.1,
        0.0,
        0.1,
        0.2,
    ]

    first = bootstrap_mean_ci(
        values,
        n_bootstrap=1000,
        seed=42,
    )

    second = bootstrap_mean_ci(
        values,
        n_bootstrap=1000,
        seed=42,
    )

    assert first == second


def test_bootstrap_mean_matches_sample_mean():
    values = [
        0.1,
        0.2,
        0.3,
    ]

    mean, _, _ = bootstrap_mean_ci(
        values,
        n_bootstrap=1000,
        seed=42,
    )

    assert mean == pytest.approx(np.mean(values))


def test_bootstrap_constant_sample_has_zero_width_ci():
    mean, low, high = bootstrap_mean_ci(
        [0.25] * 10,
        n_bootstrap=1000,
        seed=42,
    )

    assert mean == pytest.approx(0.25)

    assert low == pytest.approx(0.25)

    assert high == pytest.approx(0.25)


def test_bootstrap_rejects_empty_sample():
    with pytest.raises(
        ValueError,
        match="empty",
    ):
        bootstrap_mean_ci([])
