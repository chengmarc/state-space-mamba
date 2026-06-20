"""MambaSSM maps an input window to a forecast of the expected shape."""

import numpy as np
import pandas as pd
import torch

from ssm.arch.mamba import create_model
from ssm.data.loader import build_windows

H, F, N_FEAT = 60, 15, 12


def test_forward_pass_shape():
    idx = pd.date_range("2020-01-01", periods=300, freq="D")
    cols = [f"f{i}" for i in range(N_FEAT - 1)] + ["log_price_residual"]
    df = pd.DataFrame(np.random.randn(300, N_FEAT), index=idx, columns=cols)

    X, _ = build_windows(df, H, F)
    model = create_model(df, forecast_horizon=F, device=torch.device("cpu"), d_model=16, n_layer=2)

    out = model(torch.tensor(X[:4]))
    assert out.shape == (4, F, 1)
