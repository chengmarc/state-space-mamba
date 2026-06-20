"""The train/test boundary is date-based and matches TEST_DAYS."""

import pandas as pd

import ssm.config as config
from ssm.splits import holdout_cutoff


def test_cutoff_is_max_date_minus_test_days():
    idx = pd.date_range("2015-01-01", periods=2000, freq="D")
    assert holdout_cutoff(idx) == idx.max() - pd.Timedelta(days=config.TEST_DAYS)


def test_cutoff_is_stable_under_row_count_changes():
    """dropna trims warmup rows at the *start*; the cutoff must not move."""
    full = pd.date_range("2015-01-01", periods=2000, freq="D")
    trimmed = full[500:]  # simulate stage-3 dropna removing early rows
    assert holdout_cutoff(full) == holdout_cutoff(trimmed)
