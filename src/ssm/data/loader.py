"""Sliding-window dataset construction.

Turns a chronological feature DataFrame into ``(input_window, target_window)``
pairs and wraps them in PyTorch DataLoaders. The target is always the last
column of the frame (by the step-3 feature-registry contract), forecast over the
``forecast_horizon`` steps that follow each ``historic_horizon`` input window.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def build_windows(data, historic_horizon: int, forecast_horizon: int):
    """Slide over `data` producing ``(X, y)`` arrays.

    ``X[i]`` is the ``historic_horizon`` input window; ``y[i]`` is the
    ``forecast_horizon`` target taken from the last column (the contract from
    stage 3). Shapes: ``X = (n, historic_horizon, n_features)``,
    ``y = (n, forecast_horizon, 1)``.
    """
    X, y = [], []
    for i in range(len(data) - historic_horizon - forecast_horizon + 1):
        X.append(data.iloc[i : i + historic_horizon].values)
        y.append(data.iloc[i + historic_horizon : i + historic_horizon + forecast_horizon, -1].values)

    X = np.array(X, dtype=np.float32)
    y = np.expand_dims(np.array(y, dtype=np.float32), -1)
    return X, y


def train_val_split(X, y, val_split: float, embargo: int = 0):
    """Chronological train/val split with an embargo gap between the two blocks.

    Validation is the last `val_split` fraction of windows. The `embargo` windows
    immediately before the validation block are dropped (assigned to neither set).
    With ``embargo = forecast_horizon`` this guarantees no training window's target
    rows overlap any validation window's target rows — the sliding-window leak that
    otherwise makes validation (and early-stopping) optimistic.

    Note: input *context* may still overlap across the boundary (every window needs
    its full lookback). That is benign — it mirrors how evaluation legitimately
    feeds real recent context — and a full lookback-sized purge is infeasible here
    given the 730-day horizon. The target-overlap purge is the one that matters.
    """
    split = int(len(X) * (1 - val_split))
    return (X[:split], y[:split]), (X[split + embargo:], y[split + embargo:])


def create_dataloader(data, historic_horizon: int, forecast_horizon: int, device,
                      val_split: float = 0.1, batch_size: int = 32, embargo: int = 0,
                      debug: bool = False):
    """Build chronological train/val DataLoaders from a feature DataFrame.

    Both loaders keep windows in time order (no shuffle on val; shuffle on train).
    `embargo` windows are purged between train and val (see :func:`train_val_split`).
    With ``debug=True`` the raw ``(X, y)`` arrays are returned instead of loaders.
    """
    X, y = build_windows(data, historic_horizon, forecast_horizon)

    if debug:
        return X, y

    (X_train, y_train), (X_val, y_val) = train_val_split(X, y, val_split, embargo)

    def to_loader(Xd, yd, shuffle):
        ds = TensorDataset(
            torch.tensor(Xd).to(device),
            torch.tensor(yd).to(device),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(X_train, y_train, shuffle=True)
    val_loader   = to_loader(X_val,   y_val,   shuffle=False)

    print(f'Windows — train: {len(X_train)}  val: {len(X_val)}  (embargo: {embargo})')
    print(f'Input shape: {X.shape}  Target shape: {y.shape}')
    return train_loader, val_loader
