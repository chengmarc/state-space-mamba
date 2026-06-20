"""Stage-3 normalization fits its statistics on training rows only."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

# step_3 is a pipeline script, not a package module — load it by path.
_spec = importlib.util.spec_from_file_location(
    "step_3", Path(__file__).resolve().parents[1] / "pipeline" / "step_3_model_input.py"
)
step_3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step_3)


def test_zscore_uses_only_masked_rows_for_stats():
    series = pd.Series(np.arange(100.0), index=pd.date_range("2020-01-01", periods=100))
    mask = series.index < series.index[80]  # first 80 rows are "train"

    out, params = step_3.apply_zscore(series, mask)

    assert params["mean"] == series[mask].mean()
    assert params["std"] == series[mask].std()
    # the transform applies to the full series; the held-out tail is normalized with
    # train statistics, so it is NOT re-centered to zero mean.
    assert out[~mask].mean() > 1.0
