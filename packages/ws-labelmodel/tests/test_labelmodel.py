"""Behaviour and input-validation tests. These need only numpy and pytest."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import ws_labelmodel as lm


@pytest.fixture(scope="module")
def L(make_L: Callable) -> np.ndarray:
    return make_L(2_000, [0.7, 0.3], [0.9, 0.8, 0.75, 0.85], [0.5] * 4)[0]


@pytest.fixture(scope="module")
def fitted(L: np.ndarray) -> lm.LabelModel:
    return lm.LabelModel().fit(L, class_balance=[0.7, 0.3], seed=0)


# --- behaviour --------------------------------------------------------------


def test_label_model_recovers_accuracies(make_L: Callable) -> None:
    accs = [0.9, 0.8, 0.75, 0.95, 0.7]
    L, Y = make_L(50_000, [0.5, 0.5], accs, [0.5] * 5)
    model = lm.LabelModel().fit(L, n_epochs=500, seed=0)
    np.testing.assert_allclose(model.get_weights(), accs, atol=0.03)
    assert (model.predict_proba(L).argmax(axis=1) == Y).mean() > 0.85


def test_multiclass_recovers_labels(make_L: Callable) -> None:
    L, Y = make_L(20_000, [0.5, 0.3, 0.2], [0.85, 0.8, 0.9, 0.75], [0.6] * 4)
    model = lm.LabelModel(cardinality=3).fit(
        L, class_balance=[0.5, 0.3, 0.2], n_epochs=300, seed=0
    )
    probs = model.predict_proba(L)
    assert probs.shape == (len(L), 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1)
    assert (probs.argmax(axis=1) == Y).mean() > 0.8


def test_fit_is_deterministic_with_seed(L: np.ndarray) -> None:
    a = lm.LabelModel().fit(L, seed=3).mu
    b = lm.LabelModel().fit(L, seed=3).mu
    np.testing.assert_array_equal(a, b)


def test_all_abstain_row_gets_prior(fitted: lm.LabelModel) -> None:
    probs = fitted.predict_proba(np.full((1, 4), lm.ABSTAIN))
    np.testing.assert_allclose(probs[0], [0.7, 0.3])


def test_accepts_lists_and_whole_floats(L: np.ndarray, fitted: lm.LabelModel) -> None:
    expected = fitted.predict_proba(L[:10])
    np.testing.assert_allclose(fitted.predict_proba(L[:10].tolist()), expected)
    np.testing.assert_allclose(fitted.predict_proba(L[:10].astype(float)), expected)


def test_conditional_probs_shape(fitted: lm.LabelModel) -> None:
    cp = fitted.get_conditional_probs()
    assert cp.shape == (4, 3, 2)
    np.testing.assert_allclose(cp.sum(axis=1), 1)


def test_silent_lf_warns_and_reports_nan(L: np.ndarray) -> None:
    L2 = L.copy()
    L2[:, 1] = lm.ABSTAIN
    with pytest.warns(UserWarning, match=r"columns \[1\] never vote"):
        model = lm.LabelModel().fit(L2, seed=0)
    weights = model.get_weights()
    assert np.isnan(weights[1])
    assert np.all(np.isfinite(np.delete(weights, 1)))


def test_majority_vote_ties() -> None:
    L = np.array([[0, 0, 1], [0, 1, -1], [-1, -1, -1]])
    np.testing.assert_array_equal(
        lm.majority_vote_proba(L), [[1, 0], [0.5, 0.5], [0.5, 0.5]]
    )


def test_labeled_mask() -> None:
    L = np.array([[-1, -1, -1], [0, -1, 1], [-1, -1, -1], [-1, 1, -1]])
    np.testing.assert_array_equal(lm.labeled_mask(L), [False, True, False, True])


def test_lf_summary_with_labels() -> None:
    L = np.array([[0, 1, -1], [0, 0, -1], [1, -1, -1]])
    s = lm.lf_summary(L, np.array([0, 0, 0]))
    assert s["Polarity"] == [[0, 1], [0, 1], []]
    assert s["Correct"] == [2, 1, 0]
    assert s["Emp. Acc."] == pytest.approx([2 / 3, 0.5, 0.0])


# --- validation -------------------------------------------------------------


@pytest.mark.parametrize(
    ("bad", "error", "match"),
    [
        (np.zeros(5, dtype=int), ValueError, "must be a 2-D array"),
        (np.zeros((0, 3), dtype=int), ValueError, "at least one row"),
        (np.full((4, 3), "0"), TypeError, "integer labels"),
        (np.ones((4, 3), dtype=bool), TypeError, "integer labels"),
        (np.array([[0.0, np.nan, 1.0]] * 4), ValueError, "NaN or inf"),
        (np.array([[0.5, 1.0, 0.0]] * 4), ValueError, "non-integer"),
        (np.array([[0, -2, 1]] * 4), ValueError, "contains -2"),
        (np.array([[0, 2, 1]] * 4), ValueError, "label 2, but cardinality=2"),
    ],
)
def test_rejects_bad_label_matrix(
    bad: np.ndarray, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        lm.LabelModel().fit(bad)


def test_rejects_too_few_lfs() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        lm.LabelModel().fit(np.zeros((10, 2), dtype=int))


@pytest.mark.parametrize("cardinality", [1, 0, -3])
def test_rejects_small_cardinality(cardinality: int) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        lm.LabelModel(cardinality=cardinality)


@pytest.mark.parametrize("cardinality", [2.0, "2", True])
def test_rejects_non_int_cardinality(cardinality: Any) -> None:
    with pytest.raises(TypeError, match="cardinality must be an int"):
        lm.LabelModel(cardinality=cardinality)


@pytest.mark.parametrize(
    ("class_balance", "match"),
    [
        ([0.5, 0.9], "must sum to 1"),
        ([1.0, 0.0], "must all be positive"),
        ([0.2, 0.3, 0.5], "one entry per class"),
        ([np.nan, 0.5], "must all be positive"),
    ],
)
def test_rejects_bad_class_balance(
    L: np.ndarray, class_balance: list[float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        lm.LabelModel().fit(L, class_balance=class_balance)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"n_epochs": -1}, ValueError, "n_epochs must be >= 0"),
        ({"n_epochs": 10.5}, TypeError, "n_epochs must be an int"),
        ({"lr": 0}, ValueError, r"lr must be in \(0"),
        ({"lr": float("inf")}, ValueError, "lr must be in"),
        ({"l2": -0.1}, ValueError, r"l2 must be in \[0"),
        ({"momentum": 1.0}, ValueError, r"momentum must be in \[0, 1\)"),
        ({"prec_init": 0.0}, ValueError, r"prec_init values must be in \(0, 1\]"),
        ({"prec_init": [0.7, 0.7]}, ValueError, "one entry per LF"),
        ({"mu_eps": 0.5}, ValueError, "mu_eps must be in"),
        ({"seed": -1}, ValueError, "seed must be in"),
        ({"seed": "1"}, TypeError, "seed must be an int"),
    ],
)
def test_rejects_bad_hyperparameters(
    L: np.ndarray, kwargs: dict, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        lm.LabelModel().fit(L, **kwargs)


def test_diverging_fit_raises(L: np.ndarray) -> None:
    with pytest.raises(FloatingPointError, match="try a smaller lr"):
        lm.LabelModel().fit(L, lr=1e6, seed=0)


@pytest.mark.parametrize(
    "method", ["predict_proba", "get_weights", "get_conditional_probs"]
)
def test_unfitted_model_raises(L: np.ndarray, method: str) -> None:
    model = lm.LabelModel()
    args = (L,) if method == "predict_proba" else ()
    with pytest.raises(lm.NotFittedError, match="call fit"):
        getattr(model, method)(*args)


def test_predict_rejects_wrong_lf_count(L: np.ndarray, fitted: lm.LabelModel) -> None:
    with pytest.raises(ValueError, match="has 3 LFs, but the model was fitted on 4"):
        fitted.predict_proba(L[:, :3])


def test_lf_summary_rejects_bad_y(L: np.ndarray) -> None:
    with pytest.raises(ValueError, match="Y has 3 labels but L has"):
        lm.lf_summary(L, np.array([0, 1, 0]))
    with pytest.raises(ValueError, match="not abstains"):
        lm.lf_summary(L, np.full(len(L), -1))
