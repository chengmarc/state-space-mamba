"""Windowed dataset construction for the forecasting pipeline.

Turns a chronological feature DataFrame into ``(X, y)`` pairs and wraps them in
PyTorch DataLoaders.

Input for each anchor ``t``:
    One consecutive ``window_len``-day window per entry in ``window_anchors``,
    ordered oldest→newest *within* each window.
    e.g. ``window_anchors=[365, 180, 30]`` with ``window_len=7`` yields
    ``(3, 7, n_features)`` inputs.

Target:
    The last column's values at ``t+1 … t+predict_window``.

The last column of the input DataFrame is the prediction target by contract
(established in stage 3's feature registry).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def build_windows(data, window_anchors, window_len: int, predict_window: int = 1):
    """Slide over ``data`` producing windowed ``(X, y)`` arrays.

    ``X[i]`` — shape ``(n_windows, window_len, n_features)``: each window is the
    ``window_len`` consecutive days ``[a .. a+window_len-1]`` before anchor t, ordered
    oldest→newest. Windows stay distinct (the model encodes each separately).
    ``y[i]`` — shape ``(1, predict_window, 1)``: 1 forward window, predict_window days,
    1 feature (the target at t+1 … t+predict_window).

    The last column is the target (e.g. the raw log-price residual target) and is NOT included among
    the inputs — the model forecasts it from the other features only.
    """
    arr        = data.values
    feats      = arr[:, :-1]   # inputs exclude the target (last column)
    tgt        = arr[:, -1]    # target = last column
    max_offset = max(window_anchors) + window_len - 1
    X, y = [], []
    for t in range(max_offset, len(data) - predict_window):
        windows = [[feats[t - a - d] for d in range(window_len - 1, -1, -1)] for a in window_anchors]
        X.append(windows)
        y.append([tgt[t + p] for p in range(1, predict_window + 1)])
    X = np.array(X, dtype=np.float32)
    # (n, 1, predict_window, 1): 1 forward window, predict_window days, 1 feature.
    y = np.array(y, dtype=np.float32)[:, np.newaxis, :, np.newaxis]
    return X, y


def train_val_split(X, y, val_split: float, embargo: int = 0, random_seed: int | None = None):
    """Train/val split.

    If ``random_seed`` is given, windows are shuffled randomly before splitting
    (appropriate when windows are not sequential in calendar order).
    Otherwise a chronological split is used with an ``embargo`` gap.
    """
    n = len(X)
    if random_seed is not None:
        rng      = np.random.default_rng(random_seed)
        idx      = rng.permutation(n)
        n_val    = int(n * val_split)
        tr_idx   = np.sort(idx[n_val:])
        val_idx  = np.sort(idx[:n_val])
        return (X[tr_idx], y[tr_idx]), (X[val_idx], y[val_idx])
    split = int(n * (1 - val_split))
    return (X[:split], y[:split]), (X[split + embargo:], y[split + embargo:])

def create_dataloader(data, window_anchors, window_len: int, predict_window: int, device,
                      val_split: float = 0.1, batch_size: int = 32, embargo: int = 0,
                      random_seed: int | None = None):
    """Build train/val DataLoaders from a feature DataFrame."""
    X, y = build_windows(data, window_anchors, window_len, predict_window)
    (X_train, y_train), (X_val, y_val) = train_val_split(X, y, val_split, embargo, random_seed)

    def to_loader(Xd, yd, shuffle):
        ds = TensorDataset(
            torch.tensor(Xd).to(device),
            torch.tensor(yd).to(device),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(X_train, y_train, shuffle=True)
    val_loader   = to_loader(X_val,   y_val,   shuffle=False)

    print(f'Windows  — train: {len(X_train)}  val: {len(X_val)}  (embargo: {embargo})')
    print(f'Input shape: {X.shape}  Target shape: {y.shape}  '
          f'Windows: {len(window_anchors)}×{window_len}')
    return train_loader, val_loader
