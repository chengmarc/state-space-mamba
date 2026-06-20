"""Sliding-window dataset construction.

Turns a chronological feature DataFrame into ``(input_window, target_window)``
pairs and wraps them in PyTorch DataLoaders. The target is always the last
column of the frame (by the step-3 feature-registry contract), forecast over the
``forecast_horizon`` steps that follow each ``historic_horizon`` input window.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def create_dataloader(data, historic_horizon: int, forecast_horizon: int, device,
                      val_split: float = 0.1, batch_size: int = 32, debug: bool = False):
    """Build chronological train/val DataLoaders from a feature DataFrame.

    The last `val_split` fraction of windows is held out as validation.
    Both loaders keep windows in time order (no shuffle on val; shuffle on train).
    With ``debug=True`` the raw ``(X, y)`` arrays are returned instead of loaders.
    """
    X, y = [], []
    for i in range(len(data) - historic_horizon - forecast_horizon + 1):
        X.append(data.iloc[i : i + historic_horizon].values)
        y.append(data.iloc[i + historic_horizon : i + historic_horizon + forecast_horizon, -1].values)

    X = np.array(X, dtype=np.float32)
    y = np.expand_dims(np.array(y, dtype=np.float32), -1)

    if debug:
        return X, y

    split = int(len(X) * (1 - val_split))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    def to_loader(Xd, yd, shuffle):
        ds = TensorDataset(
            torch.tensor(Xd).to(device),
            torch.tensor(yd).to(device),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(X_train, y_train, shuffle=True)
    val_loader   = to_loader(X_val,   y_val,   shuffle=False)

    print(f'Windows — train: {len(X_train)}  val: {len(X_val)}')
    print(f'Input shape: {X.shape}  Target shape: {y.shape}')
    return train_loader, val_loader
