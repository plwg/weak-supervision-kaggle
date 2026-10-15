"""Snorkel-style weak supervision label model, in numpy. See README.md."""

from ws_labelmodel.labelmodel import (
    ABSTAIN,
    LabelModel,
    NotFittedError,
    check_label_matrix,
    labeled_mask,
    lf_summary,
    majority_vote_proba,
)

__all__ = [
    "ABSTAIN",
    "LabelModel",
    "NotFittedError",
    "check_label_matrix",
    "labeled_mask",
    "lf_summary",
    "majority_vote_proba",
]
