"""Slice-based dataset construction.

Turns a chronological feature DataFrame into (input_slices, target) pairs
and wraps them in PyTorch DataLoaders.

Input for each anchor t:
    One single-day snapshot per entry in slice_offsets, ordered oldest→newest.
    e.g. slice_offsets=[365,180,90,75,55,20,10,5,0] → 9 snapshots of shape (9, n_features).

Target:
    The last column's value at t+predict_window (default: residual at t+1).

The last column of the input DataFrame is the prediction target by contract
(established in stage 3's feature registry).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def build_windows(data, slice_offsets: list[int], predict_window: int = 1):
    """Slide over ``data`` producing ``(X, y)`` arrays.

    ``X[i]``  — shape ``(n_slices, n_features)``: one snapshot per offset.
    ``y[i]``  — shape ``(predict_window,)``: last-column value at t+1 … t+predict_window.

    Offsets should be ordered largest→smallest (oldest→newest); offset=0 = anchor day.
    """
    max_offset = max(slice_offsets)
    X, y = [], []
    for t in range(max_offset, len(data) - predict_window):
        x_slices = [data.iloc[t - offset].values for offset in slice_offsets]
        X.append(x_slices)
        y.append([data.iloc[t + 1, -1]])   # residual at t+1
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_val_split(X, y, val_split: float, embargo: int = 0, random_seed: int | None = None):
    """Train/val split.

    If ``random_seed`` is given, windows are shuffled randomly before splitting
    (appropriate when windows are not sequential, as with slice-based inputs).
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


def create_dataloader(data, slice_offsets: list[int], predict_window: int, device,
                      val_split: float = 0.1, batch_size: int = 32, embargo: int = 0,
                      random_seed: int | None = None, debug: bool = False):
    """Build train/val DataLoaders from a feature DataFrame."""
    X, y = build_windows(data, slice_offsets, predict_window)

    if debug:
        return X, y

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
    print(f'Input shape: {X.shape}  Target shape: {y.shape}  Slices: {slice_offsets}')
    return train_loader, val_loader
