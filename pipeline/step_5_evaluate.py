"""Stage 5 — autoregressive evaluation on the held-out test window.

Rolls the model forward day by day through the 365-day test window.  All
feature columns use real observed data; only log_price_residual is updated
with model predictions at each step (autoregressive on the residual, real on
everything else).

    Input : config.DM_CSV, config.TRAIN_META_JSON, config.CHECKPOINT,
            config.TREND_PARAMS_JSON
    Output: step5_fig_evaluation.png, step5_rollout.csv + printed metrics
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from ssm.config import OUTPUT, DM_CSV, TRAIN_META_JSON, TREND_PARAMS_JSON, CHECKPOINT
from ssm.arch.mamba import create_model


# ── helpers ───────────────────────────────────────────────────────────────────

def load_artifacts():
    data = pd.read_csv(DM_CSV, index_col='Date', parse_dates=True)
    with open(TRAIN_META_JSON) as f:
        meta = json.load(f)
    return data, meta


def load_trend_params() -> dict:
    with open(TREND_PARAMS_JSON) as f:
        return json.load(f)


def require_meta(meta, *keys):
    missing = [k for k in keys if k not in meta]
    if missing:
        raise KeyError(f'train_meta.json missing: {", ".join(missing)}. Re-run training.')
    return [meta[k] for k in keys]


# ── rollout ───────────────────────────────────────────────────────────────────

def autoregressive_rollout(model, data: pd.DataFrame, meta: dict, device) -> pd.DataFrame:
    """Roll forward through the held-out test window.

    All feature columns use real observed data except log_price_residual,
    which is replaced by the model's prediction at each step.
    """
    slice_offsets, test_boundary = require_meta(meta, 'slice_offsets', 'test_boundary')
    test_boundary  = pd.Timestamp(test_boundary)
    max_offset     = max(slice_offsets)
    residual_col   = data.columns.get_loc('log_price_residual')
    test_start_idx = data.index.searchsorted(test_boundary)

    synthetic = data.copy()
    records   = []

    model.eval()
    with torch.no_grad():
        for origin_idx in range(test_start_idx, len(data) - 1):
            if origin_idx - max_offset < 0:
                continue
            slices = [synthetic.iloc[origin_idx - off].values for off in slice_offsets]
            x      = torch.tensor(np.array(slices, dtype=np.float32)).unsqueeze(0).to(device)
            pred   = float(model(x).squeeze().cpu())

            next_idx = origin_idx + 1
            actual   = float(data.iloc[next_idx, residual_col])
            synthetic.iloc[next_idx, residual_col] = pred

            records.append({
                'date':      data.index[next_idx],
                'predicted': pred,
                'actual':    actual,
                'step':      origin_idx - test_start_idx + 1,
            })

    return pd.DataFrame(records).set_index('date')


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results: pd.DataFrame, slice_offsets: list[int]):
    actual    = results['actual'].values
    predicted = results['predicted'].values
    mae       = np.mean(np.abs(predicted - actual))
    model_mse = ((predicted - actual) ** 2).mean()

    anchor  = results['actual'].shift(1).values
    dir_acc = np.mean(
        np.sign(predicted[1:] - anchor[1:]) == np.sign(actual[1:] - anchor[1:])
    ) * 100

    copy_lag = min(slice_offsets)
    copy_mse = ((actual[copy_lag:] - actual[:-copy_lag]) ** 2).mean()

    print(f'\n{"─"*55}')
    print(f'  Test window    : {len(results)} days')
    print(f'{"─"*55}')
    print(f'  MAE            : {mae:.4f}')
    print(f'  Directional acc: {dir_acc:.1f}%')
    print(f'  Copy-lag-{copy_lag} MSE : {copy_mse:.4f}  (naive baseline)')
    print(f'  Model MSE      : {model_mse:.4f}')
    print(f'{"─"*55}\n')


# ── price reconstruction ─────────────────────────────────────────────────────

def reconstruct_prices(results: pd.DataFrame, trend_params: dict) -> pd.DataFrame:
    a, b, c  = trend_params['a'], trend_params['b'], trend_params['c']
    ref_date = pd.Timestamp(trend_params['ref_date'])

    X     = (results.index - ref_date).days.to_numpy(dtype=float)
    trend = a * np.log(X + c) + b

    out = results.copy()
    out['actual_price']    = np.exp(out['actual'].values    + trend)
    out['predicted_price'] = np.exp(out['predicted'].values + trend)
    return out


# ── plot ──────────────────────────────────────────────────────────────────────

def plot_evaluation(results: pd.DataFrame):
    dates           = results.index
    actual          = results['actual'].values
    predicted       = results['predicted'].values
    actual_price    = results['actual_price'].values
    predicted_price = results['predicted_price'].values

    fig, (ax_res, ax_px) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle('Autoregressive rollout — 365-day test window', fontsize=13)

    ax_res.plot(dates, actual,    color='steelblue', linewidth=1.2, label='actual')
    ax_res.plot(dates, predicted, color='tomato',    linewidth=1.0, linestyle='--', label='predicted')
    ax_res.axhline(0, color='grey', linewidth=0.5, linestyle=':')
    ax_res.set_ylabel('log-price residual')
    ax_res.legend(fontsize=9)
    ax_res.grid(linestyle=':', linewidth=0.5)

    ax_px.plot(dates, actual_price,    color='steelblue', linewidth=1.4, label='actual price')
    ax_px.plot(dates, predicted_price, color='tomato',    linewidth=1.0, linestyle='--', label='predicted price')
    ax_px.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax_px.set_ylabel('BTC price (USD)')
    ax_px.set_xlabel('date')
    ax_px.legend(fontsize=9)
    ax_px.grid(linestyle=':', linewidth=0.5)

    plt.tight_layout()
    path = OUTPUT / 'step5_fig_evaluation.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved → {path.name}')


# ── main ──────────────────────────────────────────────────────────────────────

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    data, meta = load_artifacts()
    slice_offsets, predict_window, test_boundary, d_model, n_layer, dropout = require_meta(
        meta, 'slice_offsets', 'predict_window', 'test_boundary', 'd_model', 'n_layer', 'dropout'
    )
    test_boundary = pd.Timestamp(test_boundary)

    model = create_model(data, predict_window, device, d_model=d_model, n_layer=n_layer, dropout=dropout)
    model.load_state_dict(torch.load(CHECKPOINT, weights_only=True, map_location=device))
    model.eval()

    print(f'Test boundary : {test_boundary.date()}')
    print(f'Data end      : {data.index[-1].date()}')
    print(f'Slices        : {slice_offsets}')
    print('Running autoregressive rollout...')

    results = autoregressive_rollout(model, data, meta, device)
    if results.empty:
        print('No predictions.')
        return

    compute_metrics(results, slice_offsets)

    trend_params = load_trend_params()
    results      = reconstruct_prices(results, trend_params)
    plot_evaluation(results)

    csv_path = OUTPUT / 'step5_rollout.csv'
    results.to_csv(csv_path, float_format='%.6f')
    print(f'Saved → {csv_path.name}')


if __name__ == '__main__':
    evaluate()
