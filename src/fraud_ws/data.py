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
import pandas as pd

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


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _causal_card_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-card behavioural features using only each card's *prior* history.

    No labels are involved; these are the kind of signals a fraud team would
    have in production at authorisation time.
    """
    df = df.sort_values(["cc_num", "ts"]).reset_index(drop=True)
    g = df.groupby("cc_num", sort=False)

    df["card_txn_index"] = g.cumcount()
    # (The raw ``unix_time`` column is offset differently in the two CSVs, so
    # derive gaps from the parsed timestamp instead.)
    df["secs_since_last"] = g["ts"].diff().dt.total_seconds().fillna(1e7)

    # Expanding median of the card's previous amounts (shifted -> past only).
    prev_median = g["amt"].transform(lambda s: s.expanding().median().shift(1))
    df["amt_ratio_card_median"] = (df["amt"] / prev_median).fillna(1.0)

    prev_cat_median = df.groupby(["cc_num", "category"], sort=False)["amt"].transform(
        lambda s: s.expanding().median().shift(1)
    )
    df["amt_ratio_card_cat_median"] = (df["amt"] / prev_cat_median).fillna(1.0)

    # Velocity: count / sum of the card's transactions in trailing windows
    # (window includes the current transaction).
    indexed = df.set_index("ts")
    gi = indexed.groupby("cc_num", sort=False)["amt"]
    df["n_txn_24h"] = gi.rolling("24h").count().to_numpy()
    df["amt_sum_24h"] = gi.rolling("24h").sum().to_numpy()
    df["n_txn_1h"] = gi.rolling("1h").count().to_numpy()
    return df


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["ts"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = df["ts"].dt.hour
    df["dow"] = df["ts"].dt.dayofweek
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 3)).astype(int)
    df["log_amt"] = np.log1p(df["amt"])
    df["age"] = (df["ts"] - pd.to_datetime(df["dob"])).dt.days / 365.25
    df["city_pop_log"] = np.log10(df["city_pop"])
    df["dist_km"] = _haversine_km(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )
    df["category_code"] = df["category"].astype("category").cat.codes
    df["gender_code"] = (df["gender"] == "F").astype(int)
    return _causal_card_features(df)


def prepare(force: bool = False) -> None:
    """Build the unlabeled train/test sets and the test-label vault."""
    if not force and TRAIN_PATH.exists() and TEST_PATH.exists() and VAULT_PATH.exists():
        return
    download()
    train = pd.read_csv(RAW_DIR / "fraudTrain.csv", index_col=0)
    test = pd.read_csv(RAW_DIR / "fraudTest.csv", index_col=0)
    train["split"] = "train"
    test["split"] = "test"
    full = pd.concat([train, test], ignore_index=True)

    # --- Hide the label. --------------------------------------------------
    # Training labels are thrown away; test labels go into the vault keyed by
    # transaction id, and are only used for the final evaluation.
    vault = full.loc[full["split"] == "test", ["trans_num", LABEL]]
    full = full.drop(columns=[LABEL])

    # Card history spans both files, so compute causal features on the union.
    feats = build_features(full)
    keep = ["trans_num", "ts", "cc_num", "category", "split", *FEATURES]
    feats = feats[keep]

    DATA_DIR.mkdir(exist_ok=True)
    VAULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats[feats["split"] == "train"].drop(columns="split").to_parquet(TRAIN_PATH)
    feats[feats["split"] == "test"].drop(columns="split").to_parquet(TEST_PATH)
    vault.to_parquet(VAULT_PATH)


def load_unlabeled() -> tuple[pd.DataFrame, pd.DataFrame]:
    prepare()
    return pd.read_parquet(TRAIN_PATH), pd.read_parquet(TEST_PATH)


def reveal_test_labels(test: pd.DataFrame) -> np.ndarray:
    """Open the vault. Only call this for final evaluation."""
    vault = pd.read_parquet(VAULT_PATH).set_index("trans_num")[LABEL]
    return vault.loc[test["trans_num"]].to_numpy().astype(int)
