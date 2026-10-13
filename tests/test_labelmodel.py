"""Parity tests: fraud_ws.labelmodel must reproduce Snorkel 0.10.

Snorkel is only a dev dependency; these tests are skipped without it.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from fraud_ws import labelmodel as lm

snorkel_labeling = pytest.importorskip("snorkel.labeling")
snorkel_model = pytest.importorskip("snorkel.labeling.model")


def synthetic_L(
    n: int, prior: list[float], accs: list[float], covs: list[float], seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Conditionally independent LFs: each votes with prob cov, correctly with prob acc."""
    rng = np.random.default_rng(seed)
    k = len(prior)
    Y = rng.choice(k, size=n, p=prior)
    L = np.full((n, len(accs)), lm.ABSTAIN)
    for j, (acc, cov) in enumerate(zip(accs, covs, strict=True)):
        votes = rng.random(n) < cov
        wrong = rng.random(n) >= acc
        noise = (Y + rng.integers(1, k, size=n)) % k
        L[votes, j] = np.where(wrong[votes], noise[votes], Y[votes])
    return L, Y


@dataclass(frozen=True)
class Case:
    prior: list[float]
    accs: list[float]
    covs: list[float]
    fit: dict[str, Any] = field(default_factory=dict)


CASES = {
    # Imbalanced binary, like the fraud setup.
    "imbalanced": Case(
        prior=[0.97, 0.03],
        accs=[0.9, 0.8, 0.75, 0.95, 0.7, 0.85],
        covs=[0.3, 0.5, 0.2, 0.6, 0.4, 0.1],
        fit={"class_balance": [0.97, 0.03], "n_epochs": 500, "seed": 42},
    ),
    # Uniform prior: exercises the column-permutation tie-break.
    "uniform": Case(
        prior=[0.5, 0.5],
        accs=[0.8, 0.7, 0.9, 0.65],
        covs=[0.6, 0.5, 0.4, 0.7],
        fit={"n_epochs": 200, "seed": 1},
    ),
    "multiclass": Case(
        prior=[0.5, 0.3, 0.2],
        accs=[0.8, 0.7, 0.9, 0.75, 0.6],
        covs=[0.5, 0.6, 0.4, 0.5, 0.7],
        fit={"class_balance": [0.5, 0.3, 0.2], "n_epochs": 300, "l2": 0.1, "seed": 7},
    ),
}


@pytest.mark.parametrize("name", CASES)
def test_label_model_matches_snorkel(name: str) -> None:
    case = CASES[name]
    k = len(case.prior)
    L, _ = synthetic_L(20_000, case.prior, case.accs, case.covs)

    ours = lm.LabelModel(cardinality=k).fit(L, **case.fit)
    ref = snorkel_model.LabelModel(cardinality=k, verbose=False)
    ref.fit(L, progress_bar=False, **case.fit)

    # Snorkel optimises in float32, so allow a little drift.
    np.testing.assert_allclose(ours.mu, ref.mu.detach().numpy(), atol=1e-4)
    np.testing.assert_allclose(ours.get_weights(), ref.get_weights(), atol=1e-3)
    np.testing.assert_allclose(
        ours.get_conditional_probs(), ref.get_conditional_probs(), atol=1e-4
    )
    np.testing.assert_allclose(ours.predict_proba(L), ref.predict_proba(L), atol=1e-3)


def test_label_model_recovers_accuracies() -> None:
    accs = [0.9, 0.8, 0.75, 0.95, 0.7]
    L, Y = synthetic_L(50_000, [0.5, 0.5], accs, [0.5] * 5)
    model = lm.LabelModel().fit(L, n_epochs=500, seed=0)
    np.testing.assert_allclose(model.get_weights(), accs, atol=0.03)
    assert (model.predict_proba(L).argmax(axis=1) == Y).mean() > 0.85


def test_majority_vote_matches_snorkel() -> None:
    L, _ = synthetic_L(5_000, [0.5, 0.3, 0.2], [0.7] * 4, [0.4] * 4)
    ref = snorkel_model.MajorityLabelVoter(cardinality=3).predict_proba(L)
    np.testing.assert_array_equal(lm.majority_vote_proba(L, cardinality=3), ref)


@pytest.mark.parametrize("with_y", [False, True])
def test_lf_summary_matches_snorkel(with_y: bool) -> None:
    L, Y = synthetic_L(5_000, [0.6, 0.4], [0.8, 0.7, 0.9, 0.6], [0.3, 0.5, 0.2, 0.6])
    # Include a LEGIT-only LF, as in the fraud LFs.
    L[:, 2] = np.where(L[:, 2] == 1, lm.ABSTAIN, L[:, 2])
    y = Y if with_y else None
    ours = lm.lf_summary(L, y)
    ref = snorkel_labeling.LFAnalysis(L).lf_summary(Y=y)
    assert list(ours) == list(ref.columns)
    assert ours["Polarity"] == ref["Polarity"].tolist()
    for col in list(ours)[1:]:
        np.testing.assert_allclose(ours[col], ref[col].to_numpy(), err_msg=col)


def test_labeled_mask() -> None:
    L = np.array([[-1, -1, -1], [0, -1, 1], [-1, -1, -1], [-1, 1, -1]])
    np.testing.assert_array_equal(lm.labeled_mask(L), [False, True, False, True])


def test_rejects_too_few_lfs() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        lm.LabelModel().fit(np.zeros((10, 2), dtype=int))
