"""Windowing shapes and the embargo leakage guard."""

import numpy as np
import pandas as pd

from ssm.data.loader import build_windows, train_val_split

SLICE_OFFSETS  = [365, 180, 90, 75, 55, 20, 10, 5, 0]
PREDICT_WINDOW = 1


def _ramp_frame(n):
    """Feature frame whose last column is the row id, so each window's target
    row indices can be recovered from its target values."""
    return pd.DataFrame({"f": np.zeros(n), "rowid": np.arange(n, dtype=float)},
                        index=pd.RangeIndex(n))


def test_build_windows_shapes():
    n   = 500
    df  = _ramp_frame(n)
    X, y = build_windows(df, SLICE_OFFSETS, PREDICT_WINDOW)
    max_offset = max(SLICE_OFFSETS)
    expected_windows = n - max_offset - PREDICT_WINDOW
    assert X.shape == (expected_windows, len(SLICE_OFFSETS), 2)
    assert y.shape == (expected_windows, 1)


def test_target_is_correct_row():
    """y[i, 0] must equal the last-column value at anchor+1."""
    df   = _ramp_frame(500)
    X, y = build_windows(df, SLICE_OFFSETS, PREDICT_WINDOW)
    max_offset = max(SLICE_OFFSETS)
    for i in range(min(10, len(y))):
        t = max_offset + i
        assert float(y[i, 0]) == df.iloc[t + 1, -1]


def test_embargo_purges_train_val_target_overlap():
    X, y = build_windows(_ramp_frame(600), SLICE_OFFSETS, PREDICT_WINDOW)
    (_, ytr), (_, yv) = train_val_split(X, y, val_split=0.2, embargo=PREDICT_WINDOW)
    train_indices = set(ytr.astype(int).ravel())
    val_indices   = set(yv.astype(int).ravel())
    assert train_indices.isdisjoint(val_indices)



def test_embargo_drops_exactly_embargo_windows():
    X, y = build_windows(_ramp_frame(600), SLICE_OFFSETS, PREDICT_WINDOW)
    embargo = PREDICT_WINDOW
    (Xtr, _), (Xv, _) = train_val_split(X, y, val_split=0.2, embargo=embargo)
    assert len(Xtr) + embargo + len(Xv) == len(X)
