"""Train/test boundary — defined once so every stage agrees.

The held-out test window is the most recent ``TEST_DAYS`` of calendar time. Any
parameter learned from data — the price/supply trend fits in stage 2 and the
feature normalization in stage 3 — must be fit **only on rows before this
boundary**, otherwise information from the test period leaks into the model and
inflates reported accuracy.

A *date-based* boundary (rather than "the last N rows") is used deliberately:
``dropna`` warmup trimming changes the row count between stages, but the most
recent date is stable, so a date cutoff is identical in stages 2, 3 and 4.
"""

import pandas as pd

from ssm.config import TEST_DAYS


def holdout_cutoff(index: pd.DatetimeIndex) -> pd.Timestamp:
    """First timestamp of the held-out test window (``index.max() - TEST_DAYS`` days).

    Rows with ``index < holdout_cutoff(index)`` are the training portion; everything
    on or after it is the untouched test set.
    """
    return index.max() - pd.Timedelta(days=TEST_DAYS)
