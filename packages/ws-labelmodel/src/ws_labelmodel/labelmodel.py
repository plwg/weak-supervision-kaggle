"""Numpy-only reimplementation of Snorkel 0.10's label model and LF tools.

Reimplements Snorkel 0.10's ``LabelModel`` (conditionally independent LFs),
``MajorityLabelVoter``, ``LFAnalysis.lf_summary`` and
``filter_unlabeled_dataframe`` (as ``labeled_mask``). The LabelModel follows
Snorkel's algorithm step for step - same loss, initialisation,
SGD-with-momentum optimiser, clamping and permutation tie-break - so results
match Snorkel to within float32/float64 rounding (see ``tests/test_parity.py``).

Label matrices use Snorkel's convention: ``L[i, j]`` is the vote of labeling
function ``j`` on row ``i``, an integer in ``{-1, 0, ..., k-1}`` where ``-1``
means the LF abstained and ``k`` is the number of classes.

Invalid input raises ``ValueError`` (or ``TypeError`` for a non-numeric
array) with a message naming the problem; using a model before ``fit`` raises
``NotFittedError``.
"""

import warnings
from itertools import permutations
from typing import Any

import numpy as np

ABSTAIN = -1


class NotFittedError(RuntimeError):
    """Raised when a ``LabelModel`` is used before ``fit`` has been called."""


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def check_label_matrix(
    L: Any, cardinality: int | None = None, name: str = "L"
) -> np.ndarray:
    """Validate a label matrix and return it as a 2-D int64 array.

    Accepts anything ``np.asarray`` turns into a 2-D numeric array: a numpy
    array, a list of lists, a pandas/polars frame's ``.to_numpy()``. Floats
    are accepted only if every value is a whole number (e.g. ``1.0``).

    Raises:
        TypeError: the array is not numeric (strings, objects, booleans).
        ValueError: wrong number of dimensions, no rows or columns, NaN/inf,
            non-integer values, values below -1, or values >= ``cardinality``.
    """
    arr = np.asarray(L)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a 2-D array of shape (n_rows, n_lfs), "
            f"got {arr.ndim}-D array with shape {arr.shape}"
        )
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(
            f"{name} must have at least one row and one LF, got shape {arr.shape}"
        )
    if arr.dtype == bool or not np.issubdtype(arr.dtype, np.number):
        raise TypeError(
            f"{name} must contain integer labels (-1 = abstain, 0..k-1 = class), "
            f"got dtype {arr.dtype}"
        )
    if not np.issubdtype(arr.dtype, np.integer):
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                f"{name} contains NaN or inf; encode abstains as {ABSTAIN}, not NaN"
            )
        if np.any(arr != np.round(arr)):
            raise ValueError(
                f"{name} contains non-integer values; labels must be integers"
            )
    arr = arr.astype(np.int64)
    low = int(arr.min())
    if low < ABSTAIN:
        raise ValueError(
            f"{name} contains {low}; labels must be {ABSTAIN} (abstain) or 0..k-1"
        )
    if cardinality is not None and int(arr.max()) >= cardinality:
        raise ValueError(
            f"{name} contains label {int(arr.max())}, but cardinality={cardinality} "
            f"(valid labels are {ABSTAIN}..{cardinality - 1})"
        )
    return arr


def _check_cardinality(cardinality: Any) -> int:
    if isinstance(cardinality, bool) or not isinstance(cardinality, int | np.integer):
        raise TypeError(f"cardinality must be an int, got {type(cardinality).__name__}")
    if cardinality < 2:
        raise ValueError(f"cardinality must be at least 2, got {cardinality}")
    return int(cardinality)


def _check_class_balance(class_balance: Any, k: int) -> np.ndarray:
    if class_balance is None:
        return np.full(k, 1 / k)
    p = np.asarray(class_balance, dtype=float)
    if p.shape != (k,):
        raise ValueError(
            f"class_balance must have one entry per class ({k}), got shape {p.shape}"
        )
    if not np.all(np.isfinite(p)) or np.any(p <= 0):
        raise ValueError(
            f"class_balance entries must all be positive, got {p.tolist()}"
        )
    if abs(p.sum() - 1) > 1e-6:
        raise ValueError(
            f"class_balance must sum to 1, got {p.tolist()} (sum {p.sum():.6g}); "
            "normalise it first, e.g. p / p.sum()"
        )
    return p


def _check_number(
    name: str,
    value: Any,
    low: float,
    high: float = np.inf,
    low_inclusive: bool = True,
    high_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.number):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    ok_low = value >= low if low_inclusive else value > low
    ok_high = value <= high if high_inclusive else value < high
    if not (np.isfinite(value) and ok_low and ok_high):
        lo = "[" if low_inclusive else "("
        hi = "]" if high_inclusive else ")"
        raise ValueError(f"{name} must be in {lo}{low}, {high}{hi}, got {value}")
    return float(value)


def _check_int(name: str, value: Any, low: int, high: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < low or (high is not None and value > high):
        bounds = f">= {low}" if high is None else f"in [{low}, {high}]"
        raise ValueError(f"{name} must be {bounds}, got {value}")
    return int(value)


# --------------------------------------------------------------------------
# Label model
# --------------------------------------------------------------------------


def _one_hot(L: np.ndarray, cardinality: int) -> np.ndarray:
    """[n, m] label matrix -> [n, m*k] indicators; column j*k + y is ``L[:, j] == y``.

    Abstains become all-zero blocks.
    """
    n, m = L.shape
    L_ind = np.zeros((n, m * cardinality))
    for y in range(cardinality):
        L_ind[:, y::cardinality] = L == y
    return L_ind


class LabelModel:
    """Snorkel's matrix-completion label model, in numpy.

    Estimates how accurate each labeling function (LF) is from the pattern of
    agreements and disagreements between LFs, with no ground-truth labels, and
    combines their votes into a probabilistic label per row.

    It assumes LFs are conditionally independent given the true class, and
    learns ``mu[j*k + l, y] = P(LF_j = l | Y = y)`` by fitting the LF overlap
    matrix ``O = E[L_ind^T L_ind]`` to ``mu P mu^T`` (off the per-LF diagonal
    blocks), where ``P = diag(class_balance)``.

    Example::

        model = LabelModel(cardinality=2).fit(L_train, class_balance=[0.9, 0.1], seed=0)
        probs = model.predict_proba(L_test)   # (n_rows, 2), rows sum to 1
        model.get_weights()                   # estimated accuracy per LF

    Attributes (set by ``fit``):
        mu: ``(n_lfs * k, k)`` array, ``mu[j*k + l, y] = P(LF_j = l | Y = y)``.
        p: ``(k,)`` class prior used for fitting and prediction.
        coverage: ``(n_lfs,)`` fraction of training rows each LF voted on.
        n_lfs: number of LFs (columns of ``L``) the model was fitted on.
    """

    def __init__(self, cardinality: int = 2) -> None:
        """Create an unfitted model for ``cardinality`` classes (labels 0..k-1)."""
        self.cardinality = _check_cardinality(cardinality)

    def fit(
        self,
        L: Any,
        class_balance: list[float] | np.ndarray | None = None,
        n_epochs: int = 100,
        lr: float = 0.01,
        l2: float = 0.0,
        momentum: float = 0.9,
        prec_init: float | list[float] | np.ndarray = 0.7,
        mu_eps: float | None = None,
        seed: int | None = None,
    ) -> "LabelModel":
        """Learn LF accuracies from an unlabeled label matrix.

        Args:
            L: ``(n_rows, n_lfs)`` integer label matrix, ``-1`` = abstain.
                Needs at least 3 LFs. Rows where every LF abstains are allowed
                but contribute nothing.
            class_balance: prior ``P(Y = y)`` for each class; positive and
                summing to 1. Defaults to uniform. **Set this for imbalanced
                problems** - the model cannot estimate it, and a uniform prior
                on, say, a 1%-positive task gives badly calibrated output.
            n_epochs: number of full-batch gradient steps. 100 is Snorkel's
                default; a few hundred is often better and still fast.
            lr: SGD learning rate. Reduce it if ``fit`` raises
                ``FloatingPointError``.
            l2: L2 regularisation strength. As in Snorkel it is applied twice:
                as a penalty ``l2**2 * ||mu - mu_init||^2`` pulling towards
                the initial estimate, and as SGD weight decay ``l2 * mu``.
            momentum: SGD momentum, in ``[0, 1)``.
            prec_init: initial guess at each LF's precision
                ``P(Y = y | LF_j = y)``, in ``(0, 1]``. A scalar applies to
                all LFs; a sequence gives one value per LF.
            mu_eps: ``mu`` is clipped to ``[mu_eps, 1 - mu_eps]`` after
                fitting, which keeps ``log(mu)`` finite. Defaults to
                ``min(0.01, 10 ** -ceil(log10(n_rows)))``; must be in
                ``(0, 0.5)``.
            seed: seed for the random scaling of the initial ``mu`` (as in
                Snorkel). ``None`` gives a non-deterministic fit. Does not
                touch numpy's global random state.

        Returns:
            ``self``, so calls can be chained.

        Raises:
            ValueError, TypeError: invalid ``L`` or hyperparameters.
            FloatingPointError: the optimisation diverged (try a smaller ``lr``).
        """
        k = self.cardinality
        L = check_label_matrix(L, cardinality=k)
        n, m = L.shape
        if m < 3:
            raise ValueError(
                f"L should have at least 3 labeling functions, got {m}; with fewer "
                "the LF accuracies are not identifiable (use majority_vote_proba)"
            )
        p = _check_class_balance(class_balance, k)
        n_epochs = _check_int("n_epochs", n_epochs, 0)
        lr = _check_number("lr", lr, 0, low_inclusive=False)
        l2 = _check_number("l2", l2, 0)
        momentum = _check_number("momentum", momentum, 0, 1, high_inclusive=False)
        prec_arr = np.asarray(prec_init, dtype=float)
        if prec_arr.ndim > 1 or (prec_arr.ndim == 1 and len(prec_arr) != m):
            raise ValueError(
                f"prec_init must be a scalar or have one entry per LF ({m}), "
                f"got shape {prec_arr.shape}"
            )
        if not np.all((prec_arr > 0) & (prec_arr <= 1)):
            raise ValueError(
                f"prec_init values must be in (0, 1], got {prec_arr.tolist()}"
            )
        if mu_eps is not None:
            mu_eps = _check_number(
                "mu_eps", mu_eps, 0, 0.5, low_inclusive=False, high_inclusive=False
            )
        if seed is not None:
            seed = _check_int("seed", seed, 0, 2**32 - 1)

        self.p = p
        P = np.diag(p)
        self.n_lfs = m
        self.coverage = (L != ABSTAIN).mean(axis=0)
        silent = np.flatnonzero(self.coverage == 0)
        if len(silent):
            warnings.warn(
                f"LFs at columns {silent.tolist()} never vote in L; their "
                "accuracy cannot be estimated and get_weights() reports NaN",
                stacklevel=2,
            )

        L_ind = _one_hot(L, k)
        O = L_ind.T @ L_ind / n
        d = m * k

        # Each LF's own k x k block of O is excluded from the fit.
        mask = np.ones((d, d), dtype=bool)
        for j in range(m):
            mask[j * k : (j + 1) * k, j * k : (j + 1) * k] = False

        # mu_init = P(LF=y | Y=y) = P(LF=y) * prec / P(Y=y), on the diagonal only.
        prec = np.broadcast_to(prec_arr, (m,))
        lps = np.diag(O)
        mu_init = np.zeros((d, k))
        for j in range(m):
            for y in range(k):
                idx = j * k + y
                mu_init[idx, y] = np.clip(lps[idx] * prec[j] / p[y], 0, 1)

        # Snorkel scales the init by one draw from np.random after seeding;
        # RandomState(seed) reproduces that draw without touching global state.
        mu = mu_init * np.random.RandomState(seed).random_sample()

        velocity = np.zeros_like(mu)
        diag_O = np.diag(O)
        for epoch in range(n_epochs):
            # Overflow means divergence, which the isfinite check reports.
            with np.errstate(over="ignore", invalid="ignore"):
                R = np.where(mask, O - mu @ P @ mu.T, 0.0)
                e = (mu @ P).sum(axis=1) - diag_O
                # Gradient of ||R||^2 + ||e||^2 + l2*||mu - mu_init||^2 (R is symmetric).
                grad = -4 * R @ mu @ P + 2 * e[:, None] * p[None, :]
                grad += 2 * l2**2 * (mu - mu_init)
                # torch.optim.SGD: weight_decay=l2, momentum without dampening.
                grad += l2 * mu
                velocity = grad if epoch == 0 else momentum * velocity + grad
                mu = mu - lr * velocity
            if not np.all(np.isfinite(mu)):
                raise FloatingPointError(
                    f"LabelModel diverged at epoch {epoch} (lr={lr}); try a smaller lr"
                )

        if mu_eps is None:
            mu_eps = min(0.01, 1 / 10 ** np.ceil(np.log10(n)))
        mu = np.clip(mu, mu_eps, 1 - mu_eps)
        self.mu = self._break_col_permutation_symmetry(mu, m)
        return self

    def _break_col_permutation_symmetry(self, mu: np.ndarray, m: int) -> np.ndarray:
        """Among classes with equal prior, pick the column order that trusts LFs most.

        Swapping columns of mu for classes with the same prior leaves the loss
        unchanged; choose the permutation maximising sum_j sum_y P(LF_j=y, Y=y).
        Cost grows factorially with the number of classes sharing a prior.
        """
        k = self.cardinality
        probs_sum = sum(mu[j * k : (j + 1) * k] for j in range(m)) @ np.diag(self.p)
        groups: dict[float, list[int]] = {}
        for y, prior in enumerate(self.p):
            groups.setdefault(round(float(prior), 3), []).append(y)

        order = np.arange(k)
        for group in groups.values():
            best = max(
                permutations(group),
                key=lambda perm: sum(
                    probs_sum[g, c] for g, c in zip(group, perm, strict=True)
                ),
            )
            order[group] = best
        return mu[:, order]

    def _check_fitted(self) -> None:
        if not hasattr(self, "mu"):
            raise NotFittedError("This LabelModel is not fitted yet; call fit(L) first")

    def predict_proba(self, L: Any) -> np.ndarray:
        """Probabilistic labels ``P(Y = y | LF votes)``, shape ``(n_rows, k)``.

        Combines votes naive-Bayes style using the learned ``mu`` and the
        class prior; abstains are ignored, so a row where every LF abstains
        gets the prior. ``L`` must have the same LF columns, in the same
        order, as the matrix passed to ``fit``.

        Raises:
            NotFittedError: called before ``fit``.
            ValueError, TypeError: invalid ``L`` or wrong number of LFs.
        """
        self._check_fitted()
        L = check_label_matrix(L, cardinality=self.cardinality)
        if L.shape[1] != self.n_lfs:
            raise ValueError(
                f"L has {L.shape[1]} LFs, but the model was fitted on {self.n_lfs}"
            )
        z = _one_hot(L, self.cardinality) @ np.log(self.mu) + np.log(self.p)
        z -= z.max(axis=1, keepdims=True)
        probs = np.exp(z)
        return probs / probs.sum(axis=1, keepdims=True)

    def get_conditional_probs(self) -> np.ndarray:
        """``(n_lfs, k+1, k)`` table of ``P(LF_j = l - 1 | Y = y)``.

        Index ``[j, 0, y]`` is the probability LF ``j`` abstains given class
        ``y``; ``[j, l + 1, y]`` is the probability it votes ``l``.
        """
        self._check_fitted()
        k = self.cardinality
        mu = self.mu.reshape(self.n_lfs, k, k)
        return np.concatenate([1 - mu.sum(axis=1, keepdims=True), mu], axis=1)

    def get_weights(self) -> np.ndarray:
        """Estimated accuracy of each LF when it votes, ``P(Y = LF_j | LF_j votes)``.

        Shape ``(n_lfs,)``, clipped to ``[1e-6, 1]``. Uses coverage from the
        training matrix. NaN for an LF that never voted during ``fit``.
        """
        self._check_fitted()
        k = self.cardinality
        joint = self.mu.reshape(self.n_lfs, k, k) * self.p  # P(LF_j = l, Y = y)
        accs = np.trace(joint, axis1=1, axis2=2)
        ratio = np.divide(
            accs, self.coverage, out=np.full_like(accs, np.nan), where=self.coverage > 0
        )
        return np.clip(ratio, 1e-6, 1.0)


# --------------------------------------------------------------------------
# Baseline and analysis helpers
# --------------------------------------------------------------------------


def majority_vote_proba(L: Any, cardinality: int = 2) -> np.ndarray:
    """Majority vote as probabilities, shape ``(n_rows, cardinality)``.

    Each row puts equal probability on the classes with the most votes, so
    ties (including rows where every LF abstains) are split evenly. Matches
    Snorkel's ``MajorityLabelVoter.predict_proba``.
    """
    cardinality = _check_cardinality(cardinality)
    L = check_label_matrix(L, cardinality=cardinality)
    counts = np.stack([(L == y).sum(axis=1) for y in range(cardinality)], axis=1)
    winners = (counts == counts.max(axis=1, keepdims=True)).astype(float)
    return winners / winners.sum(axis=1, keepdims=True)


def lf_summary(L: Any, Y: Any = None) -> dict[str, list]:
    """Per-LF polarity, coverage, overlaps and conflicts (plus accuracy if Y given).

    Returns column name -> per-LF values, in Snorkel's ``lf_summary`` column
    order, ready for e.g. ``pl.DataFrame(lf_summary(L))`` or
    ``pd.DataFrame(lf_summary(L))``:

    - ``Polarity``: sorted list of labels the LF ever emits.
    - ``Coverage``: fraction of rows where the LF votes.
    - ``Overlaps``: fraction of all rows where the LF votes and another LF
      also votes.
    - ``Conflicts``: fraction of all rows where the LF votes and another LF
      votes differently.
    - With ``Y`` (``(n_rows,)`` true labels, ints ``>= 0``): ``Correct``,
      ``Incorrect`` and ``Emp. Acc.`` (0 for an LF that never votes).
    """
    L = check_label_matrix(L)
    voted = L != ABSTAIN
    overlapped = voted.sum(axis=1) > 1
    # A row conflicts if its non-abstain votes are not all equal.
    vote_max = np.where(voted, L, -np.inf).max(axis=1)
    vote_min = np.where(voted, L, np.inf).min(axis=1)
    conflicted = overlapped & (vote_max != vote_min)

    n = len(L)
    summary = {
        "Polarity": [sorted(set(col[col != ABSTAIN].tolist())) for col in L.T],
        "Coverage": voted.mean(axis=0).tolist(),
        "Overlaps": ((voted & overlapped[:, None]).sum(axis=0) / n).tolist(),
        "Conflicts": ((voted & conflicted[:, None]).sum(axis=0) / n).tolist(),
    }
    if Y is not None:
        Y = check_label_matrix(np.asarray(Y).reshape(-1, 1), name="Y").ravel()
        if len(Y) != n:
            raise ValueError(f"Y has {len(Y)} labels but L has {n} rows")
        if Y.min() < 0:
            raise ValueError("Y must contain true class labels (>= 0), not abstains")
        correct = (voted & (L == Y[:, None])).sum(axis=0)
        incorrect = voted.sum(axis=0) - correct
        summary["Correct"] = correct.tolist()
        summary["Incorrect"] = incorrect.tolist()
        with np.errstate(divide="ignore", invalid="ignore"):
            summary["Emp. Acc."] = np.nan_to_num(correct / voted.sum(axis=0)).tolist()
    return summary


def labeled_mask(L: Any) -> np.ndarray:
    """Boolean row mask, shape ``(n_rows,)``: True where at least one LF voted.

    Use it to drop rows that carry no signal before training an end model,
    like Snorkel's ``filter_unlabeled_dataframe``.
    """
    return (check_label_matrix(L) != ABSTAIN).any(axis=1)
