"""Hierarchical MambaSSM maps windowed inputs to one residual per predict_window step."""

import numpy as np
import pandas as pd
import torch

from ssm.arch.mamba import create_model
from ssm.data.loader import build_windows

WINDOW_ANCHORS = [90, 30, 10]
WINDOW_LEN     = 7
PREDICT_WINDOW = 1
N_FEAT         = 9


def test_forward_pass_shape():
    idx  = pd.date_range("2020-01-01", periods=400, freq="D")
    cols = [f"f{i}" for i in range(N_FEAT - 1)] + ["log_price_residual"]
    df   = pd.DataFrame(np.random.randn(400, N_FEAT), index=idx, columns=cols)

    X, _ = build_windows(df, WINDOW_ANCHORS, WINDOW_LEN, PREDICT_WINDOW)
    model = create_model(df, n_horizons=PREDICT_WINDOW, device=torch.device("cpu"),
                         n_windows=len(WINDOW_ANCHORS), d_model=16, n_layer=2)

    out = model(torch.tensor(X[:4]))
    assert out.shape == (4, 1, PREDICT_WINDOW, 1)
