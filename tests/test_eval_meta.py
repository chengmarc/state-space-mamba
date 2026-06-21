"""Evaluation must rely on the saved training manifest, not config fallbacks."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "step_5", Path(__file__).resolve().parents[1] / "pipeline" / "step_5_evaluate.py"
)
step_5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step_5)


def test_require_meta_returns_requested_values():
    meta = {'slice_offsets': [365, 5, 0], 'predict_window': 1, 'd_model': 32}
    assert step_5.require_meta(meta, 'slice_offsets', 'predict_window', 'd_model') == [[365, 5, 0], 1, 32]


def test_require_meta_raises_on_missing_fields():
    with pytest.raises(KeyError, match='train_meta.json missing'):
        step_5.require_meta({'slice_offsets': [365, 5, 0]}, 'slice_offsets', 'predict_window', 'd_model')
