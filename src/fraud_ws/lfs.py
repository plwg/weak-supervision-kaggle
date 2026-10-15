"""Labeling functions encoding fraud-analyst heuristics, as Polars expressions.

Each LF votes FRAUD, LEGIT, or ABSTAINs. They are evaluated column-wise in
Polars to build the Snorkel-style label matrix ``L`` (n_rows x n_lfs) directly, which
is orders of magnitude faster than Snorkel's row-by-row ``PandasLFApplier``.
``ws_labelmodel`` is then used on ``L`` for analysis and the LabelModel.

Thresholds come from domain priors and from the *unlabeled* feature
distribution only — no ground-truth labels were used to write or tune them.
"""

import numpy as np
import polars as pl

ABSTAIN = -1
LEGIT = 0
FRAUD = 1

ONLINE = ["shopping_net", "misc_net"]
CARD_PRESENT_HIGH_RISK = ["grocery_pos", "shopping_pos"]
ROUTINE = [
    "gas_transport",
    "home",
    "kids_pets",
    "food_dining",
    "personal_care",
    "health_fitness",
    "entertainment",
]

amt = pl.col("amt")
night = pl.col("is_night") == 1
category = pl.col("category")

# name -> (vote, condition). The LF votes ``vote`` where the condition holds,
# and abstains elsewhere.
LFS: dict[str, tuple[int, pl.Expr]] = {
    # --- FRAUD heuristics ---------------------------------------------------
    # Far above what this card usually spends.
    "lf_amt_spike_vs_card": (
        FRAUD,
        (pl.col("amt_ratio_card_median") > 8) & (amt > 200),
    ),
    # Far above what this card usually spends in this category.
    "lf_amt_spike_vs_card_category": (
        FRAUD,
        (pl.col("amt_ratio_card_cat_median") > 10) & (amt > 150),
    ),
    # Large purchase in the small hours.
    "lf_night_large": (FRAUD, night & (amt > 250)),
    # Card-not-present purchase well above typical basket size.
    "lf_online_large": (FRAUD, category.is_in(ONLINE) & (amt > 600)),
    # Unusually large point-of-sale grocery/shopping ticket.
    "lf_pos_large": (FRAUD, category.is_in(CARD_PRESENT_HIGH_RISK) & (amt > 300)),
    # Small 'card testing' charges at night, in an active burst.
    "lf_night_card_testing": (FRAUD, night & (amt < 20) & (pl.col("n_txn_24h") >= 8)),
    # Many transactions in 24h with a high total spend.
    "lf_velocity_burst": (
        FRAUD,
        (pl.col("n_txn_24h") >= 10) & (pl.col("amt_sum_24h") > 1500),
    ),
    # Multiple sizeable charges within an hour.
    "lf_rapid_fire": (FRAUD, (pl.col("n_txn_1h") >= 3) & (amt > 100)),
    # --- LEGIT heuristics ---------------------------------------------------
    # Low-value spend in an everyday category.
    "lf_routine_small": (LEGIT, category.is_in(ROUTINE) & (amt < 100)),
    # Amount in line with the card's history, at a normal hour.
    "lf_in_card_pattern": (
        LEGIT,
        ~night
        & pl.col("amt_ratio_card_cat_median").is_between(0.3, 2.0, closed="none")
        & (pl.col("card_txn_index") > 20),
    ),
    # Daytime spend on a card with normal activity levels.
    "lf_daytime_quiet": (LEGIT, ~night & (pl.col("n_txn_24h") <= 6) & (amt < 250)),
    # Tiny daytime purchases.
    "lf_tiny_amount_daytime": (LEGIT, (amt < 10) & pl.col("hour").is_between(6, 20)),
}

LF_NAMES = list(LFS)


def apply_lfs(df: pl.DataFrame) -> np.ndarray:
    """Evaluate all LFs and return the label matrix L (int, -1 = abstain)."""
    L = df.select(
        pl.when(cond)
        .then(pl.lit(vote))
        .otherwise(pl.lit(ABSTAIN))
        .cast(pl.Int8)
        .alias(name)
        for name, (vote, cond) in LFS.items()
    )
    return L.to_numpy().astype(int)
