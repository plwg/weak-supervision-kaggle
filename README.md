# fraud-ws: weakly supervised fraud detection with Snorkel

Builds a fraud classifier **without using any training labels**, using
[Snorkel](https://github.com/snorkel-team/snorkel) weak supervision on the
Kaggle dataset [`kartik2112/fraud-detection`](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
(simulated card transactions, 1.3M train / 556k test rows, ~0.5% fraud).

## Pipeline

1. **Download & hide the label** (`data.py`): fetches the CSVs via `kagglehub`, drops
   `is_fraud`, throws away the train labels, and writes the test labels to
   `data/vault/` for final evaluation only.
2. **Features**: per-card causal (past-only) behaviour: amount vs. the card's
   running median (overall and per category), 1h/24h velocity, time since the
   last transaction, hour, distance, age.
3. **Labeling functions** (`lfs.py`): 12 analyst heuristics (8 FRAUD, 4 LEGIT)
   written as **Polars expressions** that build Snorkel's label matrix `L`
   directly. This is much faster than the row-wise `PandasLFApplier`.
4. **Snorkel LabelModel** fits LF accuracies from their agreements/disagreements
   (no labels), using an assumed 1% fraud prior.
5. **End model**: `HistGradientBoostingClassifier` trained on the LabelModel's
   probabilistic labels. Each row is used twice, once per class, weighted by
   that class's probability.
6. **Evaluation**: opens the vault and scores everything on the test set, plus a
   fully supervised model as an upper bound for reference.

## Run

```bash
uv sync
uv run fraud-ws            # add --rebuild to re-download, --no-reference to skip upper bound
```

The full run takes about 80 seconds on CPU.

## Results (test set, fraud rate 0.386%)

| Model                    | ROC-AUC | PR-AUC | Precision @ top 0.5% | Recall @ top 0.5% |
|--------------------------|--------:|-------:|---------------------:|------------------:|
| MajorityVote             |  0.883  | 0.053  | 0.094 | 0.122 |
| Snorkel LabelModel       |  0.934  | 0.340  | 0.371 | 0.480 |
| End model (weak labels)  |  0.943  | 0.277  | 0.295 | 0.381 |
| Supervised (true labels) |  0.993  | 0.785  | 0.678 | 0.878 |

The LabelModel is a big improvement over majority vote. With no labels at all,
it catches about half of the fraud in a 0.5% review queue. Most of the remaining
gap to the supervised model comes from noisy FRAUD LFs. Post-hoc test accuracy
shows `lf_night_card_testing` and `lf_velocity_burst` are only around 1–2%
precise, but the LabelModel rates them around 26%. The LEGIT LFs are about 99.9%
precise.

The LFs were *not* tuned against the test labels, because that would leak the
label. To improve them, label a small dev set (a few hundred rows) and iterate
on that.
