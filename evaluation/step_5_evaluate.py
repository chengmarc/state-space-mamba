"""Stage 5 — evaluation.

Loads the trained checkpoint and produces non-overlapping rolling forecasts over
the held-out test window, reports residual and reconstructed-price metrics, and
saves comparison figures.

    Input : config.MODEL_INPUT_CSV, config.ODS_CSV, config.TREND_PARAMS_JSON,
            config.TRAIN_META_JSON, config.CHECKPOINT
    Output: step5_fig1_residuals.png, step5_fig2_price.png + printed metrics
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from ssm.config import (
    OUTPUT, MODEL_INPUT_CSV, ODS_CSV, TREND_PARAMS_JSON, TRAIN_META_JSON, CHECKPOINT,
)
from ssm.arch.mamba import create_model


# ── helpers ────────────────────────────────────────────────────────────────────

def load_artifacts():
    data  = pd.read_csv(MODEL_INPUT_CSV, index_col='Date', parse_dates=True)
    raw   = pd.read_csv(ODS_CSV,         index_col='Date', parse_dates=True)
    with open(TREND_PARAMS_JSON) as f:
        trend = json.load(f)
    with open(TRAIN_META_JSON) as f:
        meta = json.load(f)
    return data, raw, trend, meta


def reconstruct_price(residuals: pd.Series, trend: dict) -> np.ndarray:
    ref_date  = pd.Timestamp(trend['ref_date'])
    X         = (residuals.index - ref_date).days.to_numpy(dtype=float)
    log_trend = trend['a'] * np.log(X + trend['c']) + trend['b']
    return np.exp(residuals.to_numpy() + log_trend)


def rolling_predictions(model, data: pd.DataFrame, meta: dict, device) -> list[pd.Series]:
    """Return a list of non-overlapping 90-day predicted-residual Series."""
    historic_horizon = meta['historic_horizon']
    forecast_horizon = meta['forecast_horizon']
    test_boundary    = pd.Timestamp(meta['test_boundary'])

    test_start_idx = data.index.searchsorted(test_boundary)
    test_data      = data.iloc[test_start_idx:]
    n_steps        = len(test_data) // forecast_horizon

    segments = []
    model.eval()
    with torch.no_grad():
        for step in range(n_steps):
            pred_start_idx = test_start_idx + step * forecast_horizon
            ctx_start_idx  = pred_start_idx - historic_horizon

            if ctx_start_idx < 0:
                print(f'  Step {step + 1}: insufficient context, skipping.')
                continue

            context = data.iloc[ctx_start_idx : ctx_start_idx + historic_horizon]
            x       = torch.tensor(context.values, dtype=torch.float32).unsqueeze(0).to(device)
            pred    = model(x).squeeze().cpu().numpy()

            pred_dates = data.index[pred_start_idx : pred_start_idx + forecast_horizon]
            segments.append(pd.Series(pred[:len(pred_dates)], index=pred_dates, name='predicted_residual'))

    return segments


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    data, raw, trend, meta = load_artifacts()

    # ── load model ─────────────────────────────────────────────────────────────

    checkpoint_path = CHECKPOINT
    model = create_model(data, meta['forecast_horizon'], device)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))
    model.eval()
    print(f'Checkpoint loaded: {checkpoint_path}')
    print(f'Test window: {meta["test_boundary"]} → {data.index[-1].date()}  ({meta["test_days"]} days)')
    print(f'Producing {len(data.iloc[data.index.searchsorted(pd.Timestamp(meta["test_boundary"])):]) // meta["forecast_horizon"]} non-overlapping {meta["forecast_horizon"]}-day predictions\n')

    # ── predictions ────────────────────────────────────────────────────────────

    segments = rolling_predictions(model, data, meta, device)
    if not segments:
        print('No predictions produced — check test window and horizons.')
        return

    predicted_residuals = pd.concat(segments)
    actual_residuals    = data.loc[predicted_residuals.index, 'log_price_residual']

    # ── residual metrics ───────────────────────────────────────────────────────

    err  = predicted_residuals.values - actual_residuals.values
    mae  = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))

    actual_dir    = np.sign(np.diff(actual_residuals.values))
    predicted_dir = np.sign(np.diff(predicted_residuals.values))
    dir_acc       = np.mean(actual_dir == predicted_dir) * 100

    print('── Residual metrics ─────────────────────────────────')
    print(f'  MAE              : {mae:.6f}')
    print(f'  RMSE             : {rmse:.6f}')
    print(f'  Directional acc  : {dir_acc:.1f}%')

    # ── price metrics ──────────────────────────────────────────────────────────

    predicted_price = reconstruct_price(predicted_residuals, trend)
    actual_price    = raw.loc[predicted_residuals.index, 'price_usd'].values

    price_mae  = np.mean(np.abs(predicted_price - actual_price))
    price_rmse = np.sqrt(np.mean((predicted_price - actual_price) ** 2))

    print('\n── Price metrics (USD) ──────────────────────────────')
    print(f'  MAE              : ${price_mae:>10,.0f}')
    print(f'  RMSE             : ${price_rmse:>10,.0f}')

    # ── Figure 1: residuals ────────────────────────────────────────────────────

    colors = ['steelblue', 'darkorange', 'green', 'crimson']
    test_boundary = pd.Timestamp(meta['test_boundary'])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(actual_residuals.index, actual_residuals.values,
            color='black', linewidth=1.0, label='Actual residual')
    for i, seg in enumerate(segments):
        ax.plot(seg.index, seg.values,
                color=colors[i % len(colors)], linewidth=1.2, linestyle='--',
                label=f'Predicted segment {i + 1}')
    ax.axvline(test_boundary, color='gray', linestyle=':', linewidth=0.8, label='Test boundary')
    ax.set_title('Log Price Residual — Predicted vs Actual (Test Window)')
    ax.legend(fontsize=9)
    ax.grid(linestyle=':')
    plt.tight_layout()
    plt.savefig(OUTPUT / 'step5_fig1_residuals.png', dpi=150)
    plt.close()
    print('\nSaved → step5_fig1_residuals.png')

    # ── Figure 2: price ────────────────────────────────────────────────────────

    context_start = test_boundary - pd.Timedelta(days=180)
    context_price = raw.loc[context_start:test_boundary, 'price_usd']

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(context_price.index, context_price.values,
            color='steelblue', linewidth=1.0, label='Actual price (context)')
    ax.plot(raw.loc[predicted_residuals.index, 'price_usd'].index,
            raw.loc[predicted_residuals.index, 'price_usd'].values,
            color='black', linewidth=1.0, label='Actual price (test)')
    for i, seg in enumerate(segments):
        seg_price = reconstruct_price(seg, trend)
        ax.plot(seg.index, seg_price,
                color=colors[i % len(colors)], linewidth=1.2, linestyle='--',
                label=f'Predicted price segment {i + 1}')
    ax.axvline(test_boundary, color='gray', linestyle=':', linewidth=0.8, label='Test boundary')
    ax.set_title('BTC Price — Predicted vs Actual (Test Window)')
    ax.set_ylabel('USD')
    ax.legend(fontsize=9)
    ax.grid(linestyle=':')
    plt.tight_layout()
    plt.savefig(OUTPUT / 'step5_fig2_price.png', dpi=150)
    plt.close()
    print('Saved → step5_fig2_price.png')


if __name__ == '__main__':
    evaluate()
