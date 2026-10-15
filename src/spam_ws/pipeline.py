"""Weak supervision pipeline: LFs -> LabelModel -> TF-IDF end classifier."""

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline, make_pipeline, make_union
from ws_labelmodel import (
    LabelModel,
    labeled_mask,
    lf_summary,
    majority_vote_proba,
)

from spam_ws import data
from spam_ws.lfs import LF_NAMES, apply_lfs

# Assumed prior: roughly one SMS in seven is spam. This is a round-number
# guess, not something measured from the hidden labels.
CLASS_BALANCE = [0.85, 0.15]

SEED = 42


def fit_label_model(L_train: np.ndarray) -> LabelModel:
    return LabelModel(cardinality=2).fit(
        L_train, class_balance=CLASS_BALANCE, n_epochs=500, lr=0.01, seed=SEED
    )


def text_classifier() -> Pipeline:
    """Word 1-2 gram + character 2-5 gram TF-IDF into logistic regression."""
    vectorizer = make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True
        ),
    )
    return make_pipeline(vectorizer, LogisticRegression(C=10, max_iter=2000))


def fit_end_model(texts: list[str], probs: np.ndarray) -> Pipeline:
    """Train on soft labels: each row appears once per class, weighted by P(class)."""
    texts2 = texts + texts
    y2 = np.r_[np.zeros(len(texts), dtype=int), np.ones(len(texts), dtype=int)]
    w2 = np.r_[probs[:, 0], probs[:, 1]]
    clf = text_classifier()
    clf.fit(texts2, y2, logisticregression__sample_weight=w2)
    return clf


def metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    pred = (score > 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }


def table(columns: dict[str, list], index: list[str], name: str = "") -> pl.DataFrame:
    """Columns plus a leading label column, for printing."""
    return pl.DataFrame({name: index, **columns}, strict=False)


def supervised_reference(train: pl.DataFrame, test: pl.DataFrame) -> np.ndarray:
    """Upper bound for context only: the same model trained on the true labels.

    This is the *only* place training labels are read, and it plays no part in
    the weakly supervised model.
    """
    clf = text_classifier()
    clf.fit(train["text"].to_list(), data.reveal_train_labels(train))
    return clf.predict_proba(test["text"].to_list())[:, 1]


def run(with_reference: bool = True) -> None:
    with pl.Config(
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_width_chars=200,
        float_precision=4,
        tbl_formatting="ASCII_MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
    ):
        _run(with_reference)


def _run(with_reference: bool) -> None:
    train, test = data.load_unlabeled()
    print(
        f"train: {len(train):,} unlabeled messages | "
        f"test: {len(test):,} messages (labels in vault)"
    )

    L_train = apply_lfs(train)
    L_test = apply_lfs(test)

    print("\nLF summary on the unlabeled training set (no labels used):")
    print(table(lf_summary(L_train), LF_NAMES))

    label_model = fit_label_model(L_train)
    probs_train = label_model.predict_proba(L_train)
    print("\nLearned LF accuracies (LabelModel, unsupervised):")
    for name, acc in zip(LF_NAMES, label_model.get_weights(), strict=True):
        print(f"  {name:32s} {acc:.3f}")

    # Drop rows where every LF abstained - they carry no signal.
    labeled = labeled_mask(L_train)
    texts = train.filter(pl.Series(labeled))["text"].to_list()
    probs_filtered = probs_train[labeled]
    print(
        f"\nTraining end model on {len(texts):,} weakly labeled messages "
        f"(expected spam rate {probs_filtered[:, 1].mean():.1%})"
    )
    end_model = fit_end_model(texts, probs_filtered)

    # ---------------- Evaluation: open the vault ----------------
    y_test = data.reveal_test_labels(test)
    print(f"\n=== Evaluation on held-out test set (spam rate {y_test.mean():.1%}) ===")

    print("\nLF empirical accuracy on test (post hoc, for diagnosis only):")
    print(table(lf_summary(L_test, y_test), LF_NAMES))

    scores = {
        "MajorityVote": majority_vote_proba(L_test)[:, 1],
        "LabelModel": label_model.predict_proba(L_test)[:, 1],
        "End model (weak labels)": end_model.predict_proba(test["text"].to_list())[
            :, 1
        ],
    }
    if with_reference:
        scores["Supervised (true labels)"] = supervised_reference(train, test)

    results = [metrics(y_test, s) for s in scores.values()]
    columns = {col: [r[col] for r in results] for col in results[0]}
    print("\n" + str(table(columns, list(scores), name="model")))
