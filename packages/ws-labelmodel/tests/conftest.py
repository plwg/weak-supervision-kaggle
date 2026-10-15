"""Shared fixtures: synthetic label matrices."""

from collections.abc import Callable

import numpy as np
import pytest

from ws_labelmodel import ABSTAIN


def synthetic_L(
    n: int, prior: list[float], accs: list[float], covs: list[float], seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Conditionally independent LFs: each votes with prob cov, correctly with prob acc."""
    rng = np.random.default_rng(seed)
    k = len(prior)
    Y = rng.choice(k, size=n, p=prior)
    L = np.full((n, len(accs)), ABSTAIN)
    for j, (acc, cov) in enumerate(zip(accs, covs, strict=True)):
        votes = rng.random(n) < cov
        wrong = rng.random(n) >= acc
        noise = (Y + rng.integers(1, k, size=n)) % k
        L[votes, j] = np.where(wrong[votes], noise[votes], Y[votes])
    return L, Y


SyntheticL = Callable[..., tuple[np.ndarray, np.ndarray]]


@pytest.fixture(scope="session")
def make_L() -> SyntheticL:
    """``make_L(n, prior, accs, covs, seed=0) -> (L, Y)``; see ``synthetic_L``."""
    return synthetic_L
