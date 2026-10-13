"""Weak supervision pipeline: LFs -> Snorkel LabelModel -> end classifier."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from snorkel.labeling import LFAnalysis, filter_unlabeled_dataframe
from snorkel.labeling.model import LabelModel, MajorityLabelVoter

from fraud_ws import data
from fraud_ws.lfs import LF_NAMES, apply_lfs

# Assumed prior: card fraud is typically well under 1-2% of transactions.
# This is domain knowledge, not something measured from the hidden labels.
CLASS_BALANCE = [0.99, 0.01]

# Operating point: the fraud team can review the top 0.5% of transactions.
ALERT_RATE = 0.005

SEED = 42


def lf_summary(L: np.ndarray, Y: np.ndarray | None = None) -> pd.DataFrame:
    summary = LFAnalysis(L=L).lf_summary(Y=Y)
    summary.index = pd.Index(LF_NAMES)
    return summary


def fit_label_model(L_train: np.ndarray) -> LabelModel:
    model = LabelModel(cardinality=2, verbose=False)
    model.fit(L_train, class_balance=CLASS_BALANCE, n_epochs=500, lr=0.01, seed=SEED)
    return model


def fit_end_model(X: pd.DataFrame, probs: np.ndarray) -> HistGradientBoostingClassifier:
    """Train on soft labels: each row appears once per class, weighted by P(class)."""
    X2 = pd.concat([X, X], ignore_index=True)
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


def supervised_reference(test: pd.DataFrame) -> np.ndarray:
    """Upper bound for context only: the same model trained on the true labels.

    This is the *only* place training labels are read, and it plays no part in
    the weakly supervised model.
    """
    train, _ = data.load_unlabeled()
    raw = pd.read_csv(
        data.RAW_DIR / "fraudTrain.csv", usecols=["trans_num", data.LABEL]
    ).set_index("trans_num")
    y = raw.loc[train["trans_num"], data.LABEL].to_numpy()
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=63, random_state=SEED
    )
    clf.fit(train[data.FEATURES], y)
    return clf.predict_proba(test[data.FEATURES])[:, 1]


def run(with_reference: bool = True) -> None:
    train, test = data.load_unlabeled()
    print(
        f"train: {len(train):,} unlabeled rows | test: {len(test):,} rows (labels in vault)"
    )

    print("\nApplying labeling functions (Polars)...")
    L_train = apply_lfs(train)
    L_test = apply_lfs(test)

    print("\nLF summary on the unlabeled training set (no labels used):")
    print(lf_summary(L_train).round(4).to_string())

    label_model = fit_label_model(L_train)
    probs_train = label_model.predict_proba(L_train)
    print("\nLearned LF accuracies (LabelModel, unsupervised):")
    for name, acc in zip(LF_NAMES, label_model.get_weights(), strict=True):
        print(f"  {name:32s} {acc:.3f}")

    # Drop rows where every LF abstained - they carry no signal.
    X_train, probs_filtered = filter_unlabeled_dataframe(
        X=train, y=probs_train, L=L_train
    )
    print(
        f"\nTraining end model on {len(X_train):,} weakly labeled rows "
        f"(expected fraud rate {probs_filtered[:, 1].mean():.3%})"
    )
    end_model = fit_end_model(X_train[data.FEATURES], probs_filtered)

    # ---------------- Evaluation: open the vault ----------------
    y_test = data.reveal_test_labels(test)
    print(f"\n=== Evaluation on held-out test set (fraud rate {y_test.mean():.3%}) ===")

    print("\nLF empirical accuracy on test (post hoc, for diagnosis only):")
    print(lf_summary(L_test, y_test).round(4).to_string())

    mv = MajorityLabelVoter(cardinality=2).predict_proba(L_test)[:, 1]
    scores = {
        "MajorityVote": mv,
        "Snorkel LabelModel": label_model.predict_proba(L_test)[:, 1],
        "End model (weak labels)": end_model.predict_proba(test[data.FEATURES])[:, 1],
    }
    if with_reference:
        print("\nTraining fully supervised reference (upper bound)...")
        scores["Supervised (true labels)"] = supervised_reference(test)

    results = pd.DataFrame({name: metrics(y_test, s) for name, s in scores.items()}).T
    print("\n" + results.round(4).to_string())
