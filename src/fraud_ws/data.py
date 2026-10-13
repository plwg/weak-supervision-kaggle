"""Download the Kaggle dataset, engineer features, and hide the fraud label.

Dataset: kartik2112/fraud-detection (simulated credit card transactions,
Jan 2019 - Dec 2020, ~1.85M rows, ~0.5% fraud).

The label column ``is_fraud`` is removed before anything downstream sees the
data. Training-set labels are discarded entirely. Test-set labels are written
to a separate "vault" file that is only read by the final evaluation step.
"""

import shutil
from pathlib import Path

import numpy as np
import polars as pl

DATASET = "kartik2112/fraud-detection"
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
TRAIN_PATH = DATA_DIR / "train_unlabeled.parquet"
TEST_PATH = DATA_DIR / "test_unlabeled.parquet"
VAULT_PATH = DATA_DIR / "vault" / "test_labels.parquet"

LABEL = "is_fraud"

FEATURES = [
    "amt",
    "log_amt",
    "hour",
    "is_night",
    "dow",
    "age",
    "city_pop_log",
    "dist_km",
    "amt_ratio_card_median",
    "amt_ratio_card_cat_median",
    "n_txn_24h",
    "amt_sum_24h",
    "n_txn_1h",
    "secs_since_last",
    "card_txn_index",
    "category_code",
    "gender_code",
]


def download() -> None:
    """Fetch the raw CSVs from Kaggle into ``data/raw`` (anonymous download)."""
    if (RAW_DIR / "fraudTrain.csv").exists() and (RAW_DIR / "fraudTest.csv").exists():
        return
    import kagglehub

    src = Path(kagglehub.dataset_download(DATASET))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("fraudTrain.csv", "fraudTest.csv"):
        shutil.copy(src / name, RAW_DIR / name)


def _haversine_km(
    lat1: pl.Expr, lon1: pl.Expr, lat2: pl.Expr, lon2: pl.Expr
) -> pl.Expr:
    lat1, lon1, lat2, lon2 = (c.radians() for c in (lat1, lon1, lat2, lon2))
    a = ((lat2 - lat1) / 2).sin() ** 2 + lat1.cos() * lat2.cos() * (
        (lon2 - lon1) / 2
    ).sin() ** 2
    return 6371.0 * 2 * a.sqrt().arcsin()


def _prev_median(over: str | list[str]) -> pl.Expr:
    """Expanding median of the group's *previous* amounts (null for the first)."""
    return (
        pl.col("amt")
        .rolling_median(window_size=1 << 30, min_samples=1)
        .shift(1)
        .over(over)
    )


def _causal_card_features(df: pl.DataFrame) -> pl.DataFrame:
    """Per-card behavioural features using only each card's *prior* history.

    No labels are involved; these are the kind of signals a fraud team would
    have in production at authorisation time.
    """
    card = "cc_num"
    has_amt = pl.col("amt").is_not_null().cast(pl.Int64)  # rolling count
    # Time windows include every row at the current timestamp; drop the ones
    # *after* this row so same-second transactions only see earlier ones.
    same_ts = [card, "ts"]
    later_same_ts = pl.len().over(same_ts) - 1 - pl.int_range(pl.len()).over(same_ts)
    later_same_ts_amt = pl.col("amt").cum_sum(reverse=True).over(same_ts) - pl.col(
        "amt"
    )
    return df.sort(card, "ts", maintain_order=True).with_columns(
        card_txn_index=pl.int_range(pl.len()).over(card),
        # (The raw ``unix_time`` column is offset differently in the two CSVs,
        # so derive gaps from the parsed timestamp instead.)
        secs_since_last=pl.col("ts")
        .diff()
        .dt.total_seconds(fractional=True)
        .fill_null(1e7)
        .over(card),
        amt_ratio_card_median=(pl.col("amt") / _prev_median(card)).fill_null(1.0),
        amt_ratio_card_cat_median=(
            pl.col("amt") / _prev_median([card, "category"])
        ).fill_null(1.0),
        # Velocity: count / sum of the card's transactions in trailing windows
        # (window includes the current transaction).
        n_txn_24h=has_amt.rolling_sum_by("ts", window_size="24h").over(card)
        - later_same_ts,
        amt_sum_24h=pl.col("amt").rolling_sum_by("ts", window_size="24h").over(card)
        - later_same_ts_amt,
        n_txn_1h=has_amt.rolling_sum_by("ts", window_size="1h").over(card)
        - later_same_ts,
    )


def build_features(raw: pl.DataFrame) -> pl.DataFrame:
    ts = pl.col("trans_date_trans_time").str.to_datetime("%Y-%m-%d %H:%M:%S")
    hour = pl.col("ts").dt.hour()
    df = raw.with_columns(ts=ts).with_columns(
        hour=hour,
        dow=pl.col("ts").dt.weekday() - 1,  # Monday = 0
        is_night=((hour >= 22) | (hour <= 3)).cast(pl.Int64),
        log_amt=pl.col("amt").log1p(),
        age=(
            pl.col("ts") - pl.col("dob").str.to_date().cast(pl.Datetime("us"))
        ).dt.total_days()
        / 365.25,
        city_pop_log=pl.col("city_pop").log10(),
        dist_km=_haversine_km(
            pl.col("lat"), pl.col("long"), pl.col("merch_lat"), pl.col("merch_long")
        ),
        category_code=pl.col("category").rank("dense") - 1,
        gender_code=(pl.col("gender") == "F").cast(pl.Int64),
    )
    return _causal_card_features(df)


def _read_raw(name: str) -> pl.DataFrame:
    # The first, unnamed CSV column is a row index.
    return pl.read_csv(RAW_DIR / name).drop("")


def prepare(force: bool = False) -> None:
    """Build the unlabeled train/test sets and the test-label vault."""
    if not force and TRAIN_PATH.exists() and TEST_PATH.exists() and VAULT_PATH.exists():
        return
    download()
    full = pl.concat(
        [
            _read_raw("fraudTrain.csv").with_columns(split=pl.lit("train")),
            _read_raw("fraudTest.csv").with_columns(split=pl.lit("test")),
        ]
    )

    # --- Hide the label. --------------------------------------------------
    # Training labels are thrown away; test labels go into the vault keyed by
    # transaction id, and are only used for the final evaluation.
    vault = full.filter(pl.col("split") == "test").select("trans_num", LABEL)
    full = full.drop(LABEL)

    # Card history spans both files, so compute causal features on the union.
    feats = build_features(full)
    keep = ["trans_num", "ts", "cc_num", "category", "split", *FEATURES]
    feats = feats.select(keep)

    DATA_DIR.mkdir(exist_ok=True)
    VAULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for split, path in (("train", TRAIN_PATH), ("test", TEST_PATH)):
        feats.filter(pl.col("split") == split).drop("split").write_parquet(path)
    vault.write_parquet(VAULT_PATH)


def load_unlabeled() -> tuple[pl.DataFrame, pl.DataFrame]:
    prepare()
    return pl.read_parquet(TRAIN_PATH), pl.read_parquet(TEST_PATH)


def labels_for(trans_num: pl.Series, labels: pl.DataFrame) -> np.ndarray:
    """Look up ``LABEL`` for each transaction id, in the given order."""
    joined = trans_num.to_frame().join(
        labels, on="trans_num", how="left", maintain_order="left"
    )
    return joined[LABEL].to_numpy().astype(int)


def reveal_test_labels(test: pl.DataFrame) -> np.ndarray:
    """Open the vault. Only call this for final evaluation."""
    return labels_for(test["trans_num"], pl.read_parquet(VAULT_PATH))
