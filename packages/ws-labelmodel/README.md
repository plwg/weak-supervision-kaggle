# ws-labelmodel

A numpy-only port of the parts of [Snorkel](https://github.com/snorkel-team/snorkel)
0.10 needed for weak supervision: `LabelModel`, `MajorityLabelVoter`,
`LFAnalysis.lf_summary` and `filter_unlabeled_dataframe`. The only runtime
dependency is numpy (Python >= 3.10). It matches Snorkel's output to within
float32 rounding and fits about 10x faster.

**Weak supervision in one paragraph.** You have unlabeled rows and a handful of
*labeling functions* (LFs): rules that vote a class or abstain, e.g. "message
contains a URL -> SPAM". Stack their votes into a label matrix `L`. The
`LabelModel` estimates how accurate each LF is from how often the LFs agree and
disagree, using **no ground-truth labels**, then turns each row's votes into a
probability per class. Those probabilistic labels usually train a normal
classifier (the "end model").

## Install

From this workspace (already wired up for `fraud_ws` / `spam_ws` at the repo root):

```toml
# consumer pyproject.toml, inside the same uv workspace
dependencies = ["ws-labelmodel"]

[tool.uv.sources]
ws-labelmodel = { workspace = true }
```

From another project on the same machine, or from git:

```bash
uv add /path/to/snorkel/packages/ws-labelmodel           # local path
uv add "ws-labelmodel @ git+https://github.com/<owner>/<repo>#subdirectory=packages/ws-labelmodel"
```

Or copy `src/ws_labelmodel/labelmodel.py` into your project. It is a single
self-contained file that depends only on numpy.

## Quickstart

```python
import numpy as np
from ws_labelmodel import (
    ABSTAIN,
    LabelModel,
    labeled_mask,
    lf_summary,
    majority_vote_proba,
)

# L[i, j] = vote of LF j on row i: -1 = abstain, 0..k-1 = class. Need >= 3 LFs.
L_train = np.array(
    [
        [1, -1, 1, 0],
        [-1, 0, 0, 0],
        [1, 1, -1, -1],
        # ...
    ]
)

print(lf_summary(L_train))  # coverage / overlaps / conflicts per LF

model = LabelModel(cardinality=2).fit(
    L_train,
    class_balance=[0.9, 0.1],  # P(Y=0), P(Y=1): set this for imbalanced data
    n_epochs=500,
    seed=0,
)
probs = model.predict_proba(L_train)  # (n_rows, 2), rows sum to 1
model.get_weights()  # estimated accuracy of each LF

# Typical end-model step: drop rows no LF voted on, train on soft labels.
keep = labeled_mask(L_train)
X_keep, probs_keep = X[keep], probs[keep]
```

Turning DataFrame LFs into `L`: build one integer column per LF (`-1` where it
abstains) and call `.to_numpy()`. See `src/fraud_ws/lfs.py` and
`src/spam_ws/lfs.py` at the repo root for Polars examples.

## API

Everything below is importable from `ws_labelmodel`.

| Name | What it does |
|---|---|
| `ABSTAIN` | `-1`, the abstain value in label matrices. |
| `LabelModel(cardinality=2)` | Unfitted model for `cardinality` classes (labels `0..k-1`). |
| `.fit(L, class_balance=None, n_epochs=100, lr=0.01, l2=0.0, momentum=0.9, prec_init=0.7, mu_eps=None, seed=None)` | Learns LF accuracies from `L` and returns `self`. Full parameter docs are in the docstring. |
| `.predict_proba(L)` | `(n_rows, k)` probabilistic labels. `L` must have the same LF columns as in `fit`. |
| `.get_weights()` | `(n_lfs,)` estimated `P(Y = vote \| LF votes)`. NaN for an LF that never voted. |
| `.get_conditional_probs()` | `(n_lfs, k+1, k)` table of `P(LF_j = l-1 \| Y = y)`; index 0 is abstain. |
| `.mu`, `.p`, `.coverage`, `.n_lfs` | Fitted parameters, class prior, per-LF training coverage, number of LFs. |
| `majority_vote_proba(L, cardinality=2)` | Majority-vote baseline as probabilities; ties split evenly. |
| `lf_summary(L, Y=None)` | `dict` of per-LF `Polarity`, `Coverage`, `Overlaps`, `Conflicts`, plus `Correct`, `Incorrect`, `Emp. Acc.` when true labels `Y` are given. Pass it to `pl.DataFrame(...)` or `pd.DataFrame(...)`. |
| `labeled_mask(L)` | Boolean row mask, True where at least one LF voted. |
| `check_label_matrix(L, cardinality=None)` | The validator the functions above use; returns `L` as int64 or raises. |
| `NotFittedError` | Raised by `LabelModel` methods called before `fit`. Subclass of `RuntimeError`. |

### Input rules and errors

`L` can be anything `np.asarray` turns into a 2-D numeric array: an ndarray,
a list of lists, or `df.to_numpy()`. Float arrays are accepted when every value
is a whole number. Bad input fails up front with a message naming the problem:

| Problem | Error |
|---|---|
| `L` not 2-D, or with no rows or no columns | `ValueError` |
| `L` of strings, objects or booleans | `TypeError` |
| `L` with NaN/inf (abstain must be `-1`, not NaN), non-integers, values `< -1`, or values `>= cardinality` | `ValueError` |
| Fewer than 3 LFs passed to `fit` | `ValueError` (use `majority_vote_proba` instead) |
| `class_balance` with the wrong length, a non-positive entry, or not summing to 1 | `ValueError` |
| Hyperparameter out of range (`n_epochs < 0`, `lr <= 0`, `l2 < 0`, `momentum` outside `[0, 1)`, `prec_init` outside `(0, 1]`, `mu_eps` outside `(0, 0.5)`, `seed` outside `[0, 2**32)`) | `ValueError`, or `TypeError` for the wrong type |
| Optimisation diverged | `FloatingPointError`: lower `lr` |
| `predict_proba`/`get_weights`/`get_conditional_probs` before `fit` | `NotFittedError` |
| `predict_proba` with a different number of LFs than `fit` saw | `ValueError` |
| An LF that never votes in the training `L` | `UserWarning`; its weight is NaN |

## Things to know

- **Always pass `class_balance` when the classes are imbalanced.** The model
  cannot estimate the prior from `L`, and the uniform default badly
  miscalibrates, say, a 1%-positive problem. Use domain knowledge or a small
  labelled dev set.
- **Assumes conditionally independent LFs.** Near-duplicate LFs get counted
  twice and become overconfident. Merge them, or drop one.
- **LFs that only ever vote one class are fine.** That's the normal case.
- **Rows where every LF abstains get the prior from `predict_proba`.** Filter
  them out with `labeled_mask` before training an end model.
- **`seed` controls the only randomness** (a scaling of the initial estimate)
  and doesn't touch numpy's global random state. `seed=None` is not
  reproducible.
- **`l2` is applied twice**, as in Snorkel: as a pull towards the initial
  estimate (`l2**2`) and as SGD weight decay (`l2`).
- **Memory:** `fit` and `predict_proba` build a dense float64 matrix of
  `n_rows x (n_lfs * k)`. For 1.3M rows x 12 LFs x 2 classes that's about
  250 MB. For much bigger data, call `predict_proba` in chunks.
- **Not included from Snorkel:** `predict()` with tie-break policies,
  `save`/`load` (pickling the model works), fitting with a labelled dev set
  (`Y_dev`), and LF dependency structures. Hard labels:
  `probs.argmax(axis=1)`.

## Development

Run the tests from `packages/ws-labelmodel/`. uv installs the package plus its
`dev` and `parity` groups, the latter being Snorkel and CPU torch.

```bash
uv run pytest                                # all 52 tests incl. Snorkel parity (~8 s)
uv run --isolated --no-group parity pytest   # clean env without Snorkel/torch: parity skips
```

Run lint and type checks from the **repo root**. That's where the workspace's
`.venv` lives, and `ty`/`pyrefly` need it to resolve numpy and pytest. They
cover this package along with `fraud_ws`/`spam_ws`:

```bash
uvx ruff check && uvx ruff format && uvx ty check && uvx pyrefly check
```

If you copy the package out of this workspace, run `uv sync` in it first to
create a local `.venv`; then the same commands work from the package directory.

| File | Contents |
|---|---|
| `src/ws_labelmodel/labelmodel.py` | All the code. |
| `src/ws_labelmodel/__init__.py` | Public re-exports. |
| `tests/test_labelmodel.py` | Behaviour and validation tests (numpy + pytest only). |
| `tests/test_parity.py` | Output compared against Snorkel 0.10 on synthetic data. |
| `tests/conftest.py` | `make_L` fixture: synthetic conditionally independent LFs with known accuracies. |

Changes to the numerical code in `fit`/`predict_proba` must keep
`tests/test_parity.py` passing, since the Snorkel match is the main
correctness guarantee.

## License

Apache License 2.0 (see `LICENSE`). The code is ported from
[Snorkel](https://github.com/snorkel-team/snorkel) 0.10.0, which is also
Apache-2.0 and copyright the Snorkel team. `labelmodel.py` is a modified
reimplementation of it.
