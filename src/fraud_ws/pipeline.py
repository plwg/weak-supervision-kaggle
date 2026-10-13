"""Weak supervision pipeline: LFs -> LabelModel -> end classifier."""

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fraud_ws import data
from fraud_ws.labelmodel import (
    LabelModel,
    labeled_mask,
    lf_summary,
    majority_vote_proba,
)
from fraud_ws.lfs import LF_NAMES, apply_lfs

# Assumed prior: card fraud is typically well under 1-2% of transactions.
# This is domain knowledge, not something measured from the hidden labels.
CLASS_BALANCE = [0.99, 0.01]

# Operating point: the fraud team can review the top 0.5% of transactions.
ALERT_RATE = 0.005

SEED = 42


def features(df: pl.DataFrame) -> np.ndarray:
    return df.select(data.FEATURES).to_numpy()


def fit_label_model(L_train: np.ndarray) -> LabelModel:
    return LabelModel(cardinality=2).fit(
        L_train, class_balance=CLASS_BALANCE, n_epochs=500, lr=0.01, seed=SEED
    )


def fit_end_model(X: np.ndarray, probs: np.ndarray) -> HistGradientBoostingClassifier:
    """Train on soft labels: each row appears once per class, weighted by P(class)."""
    X2 = np.vstack([X, X])
    y2 = np.r_[np.zeros(len(X), dtype=int), np.ones(len(X), dtype=int)]
    w2 = np.r_[probs[:, 0], probs[:, 1]]
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=63, random_state=SEED
    )
    clf.fit(X2, y2, sample_weight=w2)
    return clf


def metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    k = int(len(score) * ALERT_RATE)
    alert = np.zeros(len(score), dtype=int)
    alert[np.argsort(-score, kind="stable")[:k]] = 1
    return {
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        f"precision@{ALERT_RATE:.1%}": precision_score(y, alert, zero_division=0),
        f"recall@{ALERT_RATE:.1%}": recall_score(y, alert, zero_division=0),
    }


def table(columns: dict[str, list], index: list[str], name: str = "") -> pl.DataFrame:
    """Columns plus a leading label column, for printing."""
    return pl.DataFrame({name: index, **columns}, strict=False)


def supervised_reference(test: pl.DataFrame) -> np.ndarray:
    """Upper bound for context only: the same model trained on the true labels.

    This is the *only* place training labels are read, and it plays no part in
    the weakly supervised model.
    """
    train, _ = data.load_unlabeled()
    labels = pl.read_csv(
        data.RAW_DIR / "fraudTrain.csv", columns=["trans_num", data.LABEL]
    )
    y = data.labels_for(train["trans_num"], labels)
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=63, random_state=SEED
    )
    clf.fit(features(train), y)
    return clf.predict_proba(features(test))[:, 1]


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
        f"train: {len(train):,} unlabeled rows | test: {len(test):,} rows (labels in vault)"
    )

    print("\nApplying labeling functions (Polars)...")
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
    X_train, probs_filtered = features(train)[labeled], probs_train[labeled]
    print(
        f"\nTraining end model on {len(X_train):,} weakly labeled rows "
        f"(expected fraud rate {probs_filtered[:, 1].mean():.3%})"
    )
    end_model = fit_end_model(X_train, probs_filtered)

    # ---------------- Evaluation: open the vault ----------------
    y_test = data.reveal_test_labels(test)
    print(f"\n=== Evaluation on held-out test set (fraud rate {y_test.mean():.3%}) ===")

    print("\nLF empirical accuracy on test (post hoc, for diagnosis only):")
    print(table(lf_summary(L_test, y_test), LF_NAMES))

    mv = majority_vote_proba(L_test)[:, 1]
    scores = {
        "MajorityVote": mv,
        "LabelModel": label_model.predict_proba(L_test)[:, 1],
        "End model (weak labels)": end_model.predict_proba(features(test))[:, 1],
    }
    if with_reference:
        print("\nTraining fully supervised reference (upper bound)...")
        scores["Supervised (true labels)"] = supervised_reference(test)

    results = [metrics(y_test, s) for s in scores.values()]
    columns = {col: [r[col] for r in results] for col in results[0]}
    print("\n" + str(table(columns, list(scores), name="model")))
