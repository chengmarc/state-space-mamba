"""Windowing shapes and the embargo leakage guard."""

import numpy as np
import pandas as pd

from ssm.data.loader import build_windows, train_val_split

H, F = 30, 10


def _ramp_frame(n):
    """Feature frame whose last column is the row id, so a window's target rows
    can be recovered from its target values."""
    return pd.DataFrame({"f": np.zeros(n), "rowid": np.arange(n)}, index=pd.RangeIndex(n))


def test_build_windows_shapes():
    df = _ramp_frame(200)
    X, y = build_windows(df, H, F)
    assert X.shape == (200 - H - F + 1, H, 2)
    assert y.shape == (200 - H - F + 1, F, 1)


def test_embargo_purges_train_val_target_overlap():
    X, y = build_windows(_ramp_frame(600), H, F)
    (_, ytr), (_, yv) = train_val_split(X, y, val_split=0.2, embargo=F)
    overlap = set(ytr.astype(int).ravel()) & set(yv.astype(int).ravel())
    assert overlap == set()


def test_without_embargo_there_is_overlap():
    """Guards against a false-positive above: embargo=0 must still leak."""
    X, y = build_windows(_ramp_frame(600), H, F)
    (_, ytr), (_, yv) = train_val_split(X, y, val_split=0.2, embargo=0)
    overlap = set(ytr.astype(int).ravel()) & set(yv.astype(int).ravel())
    assert len(overlap) > 0


def test_embargo_drops_exactly_embargo_windows():
    X, y = build_windows(_ramp_frame(600), H, F)
    (Xtr, _), (Xv, _) = train_val_split(X, y, val_split=0.2, embargo=F)
    assert len(Xtr) + F + len(Xv) == len(X)
