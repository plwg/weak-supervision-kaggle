"""Numpy-only replacement for the parts of Snorkel this project uses.

Reimplements Snorkel 0.10's ``LabelModel`` (conditionally independent LFs),
``MajorityLabelVoter``, ``LFAnalysis.lf_summary`` and
``filter_unlabeled_dataframe`` (as ``labeled_mask``). The LabelModel follows Snorkel's algorithm
step for step - same loss, initialisation, SGD-with-momentum optimiser,
clamping and permutation tie-break - so results match Snorkel to within
float32/float64 rounding (see ``tests/test_labelmodel.py``).

Label matrices use Snorkel's convention: ``L[i, j]`` in ``{-1, 0, ..., k-1}``
with ``-1`` meaning abstain.
"""

from itertools import permutations

import numpy as np

ABSTAIN = -1


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

    Learns ``mu[j*k + l, y] = P(LF_j = l | Y = y)`` by fitting the LF overlap
    matrix ``O = E[L_ind^T L_ind]`` to ``mu P mu^T`` (off the per-LF diagonal
    blocks), with no ground-truth labels.
    """

    def __init__(self, cardinality: int = 2) -> None:
        self.cardinality = cardinality

    def fit(
        self,
        L: np.ndarray,
        class_balance: list[float] | np.ndarray | None = None,
        n_epochs: int = 100,
        lr: float = 0.01,
        l2: float = 0.0,
        momentum: float = 0.9,
        prec_init: float | np.ndarray = 0.7,
        mu_eps: float | None = None,
        seed: int | None = None,
    ) -> "LabelModel":
        k = self.cardinality
        n, m = L.shape
        if m < 3:
            raise ValueError("L should have at least 3 labeling functions")
        if L.max() >= k:
            raise ValueError(f"L has label {L.max()}, but cardinality={k}")

        p = (
            np.full(k, 1 / k)
            if class_balance is None
            else np.asarray(class_balance, dtype=float)
        )
        if len(p) != k or np.any(p <= 0):
            raise ValueError(f"class_balance must have {k} positive entries, got {p}")
        self.p = p
        P = np.diag(p)
        self.coverage = (L != ABSTAIN).mean(axis=0)

        L_ind = _one_hot(L, k)
        O = L_ind.T @ L_ind / n
        d = m * k

        # Each LF's own k x k block of O is excluded from the fit.
        mask = np.ones((d, d), dtype=bool)
        for j in range(m):
            mask[j * k : (j + 1) * k, j * k : (j + 1) * k] = False

        # mu_init = P(LF=y | Y=y) = P(LF=y) * prec / P(Y=y), on the diagonal only.
        prec = np.broadcast_to(np.asarray(prec_init, dtype=float), (m,))
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
                raise FloatingPointError("Loss is NaN. Consider reducing lr.")

        if mu_eps is None:
            mu_eps = min(0.01, 1 / 10 ** np.ceil(np.log10(n)))
        mu = np.clip(mu, mu_eps, 1 - mu_eps)
        self.mu = self._break_col_permutation_symmetry(mu, m)
        return self

    def _break_col_permutation_symmetry(self, mu: np.ndarray, m: int) -> np.ndarray:
        """Among classes with equal prior, pick the column order that trusts LFs most.

        Swapping columns of mu for classes with the same prior leaves the loss
        unchanged; choose the permutation maximising sum_j sum_y P(LF_j=y, Y=y).
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

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        """P(Y | LF votes) under the naive-Bayes model; abstains are ignored."""
        z = _one_hot(L, self.cardinality) @ np.log(self.mu) + np.log(self.p)
        z -= z.max(axis=1, keepdims=True)
        probs = np.exp(z)
        return probs / probs.sum(axis=1, keepdims=True)

    def get_conditional_probs(self) -> np.ndarray:
        """[m, k+1, k] table of P(LF_j = l - 1 | Y = y); row 0 is abstain."""
        k = self.cardinality
        m = self.mu.shape[0] // k
        mu = self.mu.reshape(m, k, k)
        return np.concatenate([1 - mu.sum(axis=1, keepdims=True), mu], axis=1)

    def get_weights(self) -> np.ndarray:
        """Estimated accuracy of each LF when it votes, P(Y = LF_j | LF_j votes)."""
        k = self.cardinality
        m = self.mu.shape[0] // k
        joint = self.mu.reshape(m, k, k) * self.p  # P(LF_j = l, Y = y)
        accs = np.trace(joint, axis1=1, axis2=2)
        return np.clip(accs / self.coverage, 1e-6, 1.0)


def majority_vote_proba(L: np.ndarray, cardinality: int = 2) -> np.ndarray:
    """Majority vote; ties (including all-abstain rows) split probability evenly."""
    counts = np.stack([(L == y).sum(axis=1) for y in range(cardinality)], axis=1)
    winners = (counts == counts.max(axis=1, keepdims=True)).astype(float)
    return winners / winners.sum(axis=1, keepdims=True)


def lf_summary(L: np.ndarray, Y: np.ndarray | None = None) -> dict[str, list]:
    """Per-LF polarity, coverage, overlaps and conflicts (plus accuracy if Y given).

    Returns column name -> per-LF values, in Snorkel's ``lf_summary`` column
    order, ready for e.g. ``pl.DataFrame(lf_summary(L))``.
    Overlaps/Conflicts are the fraction of all rows where the LF votes and some
    other LF also votes / votes differently, as in Snorkel's ``LFAnalysis``.
    """
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
        correct = (voted & (L == Y[:, None])).sum(axis=0)
        incorrect = voted.sum(axis=0) - correct
        summary["Correct"] = correct.tolist()
        summary["Incorrect"] = incorrect.tolist()
        with np.errstate(divide="ignore", invalid="ignore"):
            summary["Emp. Acc."] = np.nan_to_num(correct / voted.sum(axis=0)).tolist()
    return summary


def labeled_mask(L: np.ndarray) -> np.ndarray:
    """Boolean row mask: True where at least one LF voted (didn't abstain)."""
    return (L != ABSTAIN).any(axis=1)
