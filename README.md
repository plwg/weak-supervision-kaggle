# Weak supervision with a Snorkel-style label model: card fraud and SMS spam

Builds a fraud classifier **without using any training labels**, using
[Snorkel](https://github.com/snorkel-team/snorkel)-style weak supervision on the
Kaggle dataset [`kartik2112/fraud-detection`](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
(simulated card transactions, 1.3M train / 556k test rows, ~0.5% fraud).

## Pipeline

1. **Download & hide the label** (`data.py`): fetches the CSVs via `kagglehub`, drops
   `is_fraud`, throws away the train labels, and writes the test labels to
   `data/vault/` for final evaluation only.
2. **Features** (Polars): per-card causal (past-only) behaviour: amount vs. the card's
   running median (overall and per category), 1h/24h velocity, time since the
   last transaction, hour, distance, age.
3. **Labeling functions** (`lfs.py`): 12 analyst heuristics (8 FRAUD, 4 LEGIT)
   written as **Polars expressions** that build the Snorkel-style label matrix `L`
   directly. This is much faster than Snorkel's row-wise `PandasLFApplier`.
4. **LabelModel** (`ws_labelmodel`, see below) fits LF accuracies from their
   agreements/disagreements (no labels), using an assumed 1% fraud prior. It is a
   numpy-only port of Snorkel 0.10's `LabelModel`, `MajorityLabelVoter` and
   `LFAnalysis`.
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

The full run takes about 30 seconds on CPU.

## Results (test set, fraud rate 0.386%)

| Model                    | ROC-AUC | PR-AUC | Precision @ top 0.5% | Recall @ top 0.5% |
|--------------------------|--------:|-------:|---------------------:|------------------:|
| MajorityVote             |  0.883  | 0.053  | 0.094 | 0.122 |
| LabelModel               |  0.934  | 0.340  | 0.371 | 0.480 |
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

## Second problem: SMS spam (`spam-ws`)

The same approach on text: the Kaggle dataset
[`uciml/sms-spam-collection-dataset`](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
(UCI SMS Spam Collection). There are 5,169 unique messages after exact
duplicates are dropped, split 70/30 into 3,575 train and 1,594 test messages,
and about 12% are spam. It lives in `src/spam_ws/` and uses the
same `ws_labelmodel` package.

- **Data** (`spam_ws/data.py`): deduplicates before splitting so the same text
  can't be in both train and test. Labels go into `data/sms/vault/`.
- **LFs** (`spam_ws/lfs.py`): 14 regex/keyword heuristics as Polars
  expressions. The 9 SPAM LFs look for prices, premium-rate numbers, short
  codes, URLs, prize language, opt-out boilerplate, ringtone/dating marketing
  and shouting. The 5 HAM LFs look for texting slang, emoticons, short messages
  with no digits, first-person chat and everyday plans. Together they cover
  87% of training messages.
- **LabelModel** with an assumed 15% spam prior.
- **End model**: word 1-2 gram plus char 2-5 gram TF-IDF into logistic
  regression, trained on the soft labels.

```bash
uv run spam-ws             # ~10 seconds
```

| Model                    | ROC-AUC | PR-AUC | Precision | Recall |   F1  |
|--------------------------|--------:|-------:|----------:|-------:|------:|
| MajorityVote             |  0.973  | 0.764  | 0.794 | 0.933 | 0.858 |
| LabelModel               |  0.988  | 0.952  | 0.834 | 0.933 | 0.881 |
| End model (weak labels)  |  0.996  | 0.982  | 0.929 | 0.949 | 0.939 |
| Supervised (true labels) |  0.997  | 0.989  | 0.995 | 0.943 | 0.968 |

Precision, recall and F1 are at a 0.5 threshold. Unlike the fraud task, here the
end model clearly beats the LabelModel. TF-IDF picks up spam vocabulary that no
LF names, so it generalises past the heuristics and nearly matches the
supervised model. The one weak LF is `lf_urgent` (`URGENT`, `WINNER!` and the
like), which is only 40% precise on test because people shout in ordinary texts
too. The LabelModel finds that without labels and rates it at 46%, while every
other LF gets 85-100%.

## The `ws-labelmodel` package

Open-source Snorkel has had no release since 0.10.0 (Feb 2024). It also pulls in
torch, tensorboard, protobuf, networkx and munkres, and this project only needs
a few hundred lines of linear algebra from it. Those parts are reimplemented in
numpy in their own uv workspace package,
[`packages/ws-labelmodel/`](packages/ws-labelmodel/README.md), which depends
only on numpy and can be reused in other projects. It uses the same loss,
initialisation, SGD with momentum, clamping and permutation tie-break as
Snorkel. On the fraud data it matches Snorkel's probabilities to within 3e-7
and fits about 10x faster. Its README covers the API, input validation and
error messages, installation in other projects, and how to run its tests.

Snorkel and torch are only needed for the package's parity tests:

```bash
cd packages/ws-labelmodel && uv run pytest
```

