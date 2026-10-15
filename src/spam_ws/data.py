"""Download the Kaggle SMS spam dataset, split it, and hide the label.

Dataset: uciml/sms-spam-collection-dataset (UCI SMS Spam Collection, 5,572
English SMS messages labelled ``ham`` or ``spam``).

Exact-duplicate messages are dropped before the split so the same text cannot
land in both train and test. The label is removed before anything downstream
sees the data. Training labels are only kept for the supervised reference.
Test labels go into a separate "vault" file that only the final evaluation
step reads.
"""

import shutil
from pathlib import Path

import numpy as np
import polars as pl

DATASET = "uciml/sms-spam-collection-dataset"
DATA_DIR = Path("data") / "sms"
RAW_PATH = DATA_DIR / "raw" / "spam.csv"
TRAIN_PATH = DATA_DIR / "train_unlabeled.parquet"
TEST_PATH = DATA_DIR / "test_unlabeled.parquet"
VAULT_DIR = DATA_DIR / "vault"
TEST_LABELS_PATH = VAULT_DIR / "test_labels.parquet"
TRAIN_LABELS_PATH = VAULT_DIR / "train_labels.parquet"

LABEL = "is_spam"
TEST_FRACTION = 0.3
SEED = 42


def download() -> None:
    """Fetch the raw CSV from Kaggle into ``data/sms/raw`` (anonymous download)."""
    if RAW_PATH.exists():
        return
    import kagglehub

    src = Path(kagglehub.dataset_download(DATASET))
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "spam.csv", RAW_PATH)


def read_raw() -> pl.DataFrame:
    """Raw messages as (text, is_spam).

    The CSV is Latin-1 encoded, and messages containing commas spill into
    three unnamed trailing columns, so those are glued back onto the text.
    """
    raw = pl.read_csv(RAW_PATH, encoding="latin1", truncate_ragged_lines=True)
    label, text, *spill = raw.columns
    return raw.select(
        text=pl.concat_str([text, *spill], separator=",", ignore_nulls=True),
        **{LABEL: (pl.col(label) == "spam").cast(pl.Int64)},
    )


def prepare(force: bool = False) -> None:
    """Build the unlabeled train/test sets and the label vault."""
    paths = (TRAIN_PATH, TEST_PATH, TEST_LABELS_PATH, TRAIN_LABELS_PATH)
    if not force and all(p.exists() for p in paths):
        return
    download()
    df = (
        read_raw()
        .unique("text", keep="first", maintain_order=True)
        .with_row_index("msg_id")
    )
    rng = np.random.default_rng(SEED)
    is_test = rng.random(len(df)) < TEST_FRACTION
    df = df.with_columns(split=pl.Series(np.where(is_test, "test", "train")))

    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    for split, data_path, label_path in (
        ("train", TRAIN_PATH, TRAIN_LABELS_PATH),
        ("test", TEST_PATH, TEST_LABELS_PATH),
    ):
        part = df.filter(pl.col("split") == split)
        part.select("msg_id", LABEL).write_parquet(label_path)
        part.select("msg_id", "text").write_parquet(data_path)


def load_unlabeled() -> tuple[pl.DataFrame, pl.DataFrame]:
    prepare()
    return pl.read_parquet(TRAIN_PATH), pl.read_parquet(TEST_PATH)


def _labels(msg_id: pl.Series, path: Path) -> np.ndarray:
    joined = msg_id.to_frame().join(
        pl.read_parquet(path), on="msg_id", how="left", maintain_order="left"
    )
    return joined[LABEL].to_numpy().astype(int)


def reveal_test_labels(test: pl.DataFrame) -> np.ndarray:
    """Open the vault. Only call this for final evaluation."""
    return _labels(test["msg_id"], TEST_LABELS_PATH)


def reveal_train_labels(train: pl.DataFrame) -> np.ndarray:
    """Training labels, for the supervised upper bound only."""
    return _labels(train["msg_id"], TRAIN_LABELS_PATH)
