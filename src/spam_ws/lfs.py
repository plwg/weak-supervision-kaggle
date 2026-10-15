"""Labeling functions for SMS spam, as Polars expressions over the raw text.

Each LF votes SPAM, HAM, or ABSTAINs. They encode the usual tells of UK
premium-SMS marketing (prices, premium-rate and short-code numbers, prize
language, opt-out boilerplate) against the tells of person-to-person chat
(texting slang, emoticons, short replies, no numbers at all).

They were written from domain knowledge and the *unlabeled* training texts
only. No ground-truth labels were used to write or tune them.
"""

import numpy as np
import polars as pl

ABSTAIN = -1
HAM = 0
SPAM = 1

text = pl.col("text")
lower = text.str.to_lowercase()
n_chars = text.str.len_chars()


def has(pattern: str) -> pl.Expr:
    """Case-insensitive regex match on the message."""
    return text.str.contains(f"(?i){pattern}")


def words(*ws: str) -> pl.Expr:
    return has(rf"\b({'|'.join(ws)})\b")


# The file is decoded as Latin-1, so a pound sign may show up as "£" or "å£".
money = has(r"£|\$\d|€|\b\d+\s?p\s?(/|per\s)|\b\d+p\b|\bpence\b")
# UK premium-rate (09...) and non-geographic (08...) numbers.
premium_number = has(r"\b0[89]\d{8,9}\b")
short_code = has(r"\b[6-8]\d{4}\b")
call_to_action = words("txt", "text", "reply", "send", "call", "ring", "dial")
any_digit = text.str.contains(r"\d")
long_number = text.str.contains(r"\d{4,}")
url = has(r"https?://|www\.|\.com\b|\.co\.uk\b|\.net\b")

# name -> (vote, condition). The LF votes ``vote`` where the condition holds,
# and abstains elsewhere.
LFS: dict[str, tuple[int, pl.Expr]] = {
    # --- SPAM heuristics ----------------------------------------------------
    # Quotes a price or per-message charge.
    "lf_money": (SPAM, money),
    # Asks the reader to ring a premium-rate / non-geographic number.
    "lf_premium_number": (SPAM, premium_number),
    # "txt WIN to 87121"-style short-code instructions.
    "lf_short_code": (SPAM, short_code & call_to_action),
    # Contains a web address.
    "lf_url": (SPAM, url),
    # Prize / competition language.
    "lf_prize": (
        SPAM,
        words(
            "prize",
            "winner",
            "won",
            "claim",
            "award(ed)?",
            "guaranteed",
            "congratulations",
            "cash",
            "voucher(s)?",
        ),
    ),
    # "Free" paired with a call to action.
    "lf_free_offer": (SPAM, words("free") & call_to_action),
    # Opt-out / subscription boilerplate.
    "lf_opt_out": (
        SPAM,
        words("stop", "opt ?out", "unsubscribe", "unsub", "subscription", "t&cs?")
        | has(r"\bt ?& ?cs?\b|\bstd\s+(txt|rate)|\bpobox|\bpo box"),
    ),
    # Ringtones, chat lines and other mobile-content marketing.
    "lf_mobile_marketing": (
        SPAM,
        words(
            "ringtone(s)?",
            "polyphonic",
            "camcorder",
            "mobile update",
            "latest (nokia|motorola|colour)",
            "sexy",
            "chat ?line",
            "dating",
            "network operator",
            "customer service(s)?",
        ),
    ),
    # Shouting: "URGENT!", "WINNER!!" and the like.
    "lf_urgent": (SPAM, words("urgent", "important") | has(r"\b[A-Z]{3,}!")),
    # --- HAM heuristics -----------------------------------------------------
    # Texting slang that marketers rarely use.
    "lf_slang": (
        HAM,
        words(
            "lol",
            "lor",
            "lar",
            "haha+",
            "hehe+",
            "gonna",
            "wanna",
            "dunno",
            "wat",
            "dun",
            "nite",
            "luv",
            "da",
            "ya",
            "ok+",
            "k",
            "yup",
            "hmm+",
        ),
    ),
    # Emoticons, plus the "&lt;#&gt;" placeholder the corpus uses where
    # personal numbers in private messages were anonymised.
    "lf_emoticon_or_placeholder": (HAM, has(r"[:;]-?[)(pd]|&lt;#&gt;|<#>")),
    # Short message with no numbers: "ok see you there".
    "lf_short_no_digits": (HAM, (n_chars < 40) & ~any_digit),
    # First-person chat with no prices, links or long numbers.
    "lf_first_person_chat": (
        HAM,
        words("i", "i'm", "im", "i'll", "me", "my")
        & ~money
        & ~url
        & ~long_number
        & ~words("free", "txt", "claim"),
    ),
    # Talks about everyday arrangements, with no call to action.
    "lf_plans": (
        HAM,
        words(
            "home",
            "tonight",
            "tomorrow",
            "later",
            "dinner",
            "lunch",
            "class",
            "sleep",
            "work",
            "pick (you|u) up",
            "on (my|the) way",
        )
        & ~call_to_action
        & ~long_number,
    ),
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
