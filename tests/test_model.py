"""MambaSSM outputs one logit per predict_window step for binary direction classification."""

import numpy as np
import pandas as pd
import torch

from ssm.arch.mamba import create_model
from ssm.data.loader import build_windows

SLICE_OFFSETS  = [90, 30, 10, 5, 0]
PREDICT_WINDOW = 1
N_FEAT         = 9


def test_forward_pass_shape():
    idx  = pd.date_range("2020-01-01", periods=400, freq="D")
    cols = [f"f{i}" for i in range(N_FEAT - 1)] + ["log_price_residual"]
    df   = pd.DataFrame(np.random.randn(400, N_FEAT), index=idx, columns=cols)

    X, _ = build_windows(df, SLICE_OFFSETS, PREDICT_WINDOW)
    model = create_model(df, n_horizons=PREDICT_WINDOW, device=torch.device("cpu"), d_model=16, n_layer=2)

    logits = model(torch.tensor(X[:4]))
    assert logits.shape == (4, PREDICT_WINDOW)
