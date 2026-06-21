"""Stage 2 — feature engineering.

Reads the ODS and builds the feature warehouse (DWD).  Only features with
empirically validated win-rate signal (proved in winrate-matrix) are computed:

  - price decomposition : log price, fitted log trend, residual
  - valuation           : MVRV ratio
  - volatility          : 30-day realized vol
  - cycle               : years since halving
  - macro               : DXY 30-day and 100-day returns
  - technical           : Williams %R (short + long), oversold composite

Also persists the fitted price-trend parameters so stage 5 can reconstruct
absolute prices from residual forecasts.

    Input : config.ODS_CSV
    Output: config.DWD_CSV, config.TREND_PARAMS_JSON, step2_fig*.png
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from ssm.config import OUTPUT, ODS_CSV, DWD_CSV, TREND_PARAMS_JSON, HALVING_DATES, ensure_dirs
from ssm.splits import holdout_cutoff


# ── config ─────────────────────────────────────────────────────────────────────

WILLIAMS_R = [
    ('short_percent_r', 21,  7),
    ('long_percent_r',  112, 3),
]

REALIZED_VOL_WINDOW = 30
TRADING_DAYS        = 252


# ── helpers ────────────────────────────────────────────────────────────────────

def _log_curve(x, a, b, c):
    return a * np.log(x + c) + b


def fit_log_trend(series: pd.Series, fit_until: pd.Timestamp | None = None):
    """Expanding-window log trend with no future data leakage.

    Step 1 — estimate c once on training rows using nonlinear curve_fit.
    Step 2 — fix c and compute a_t, b_t at every row via cumulative OLS,
              so residual[t] only uses data available up to t.

    Returns (trend_series, a_final, b_final, c) where a_final/b_final are
    the parameters at the last row (used by step 5 for future extrapolation).
    """
    valid = series.dropna()
    if fit_until is not None:
        valid = valid[valid.index < fit_until]
    X_fit = (valid.index - series.index[0]).days.to_numpy(dtype=float)
    Y_fit = valid.to_numpy(dtype=float)
    (_, _, c), _ = curve_fit(_log_curve, X_fit, Y_fit, p0=[1.0, 1.0, 1.0])

    # Expanding OLS with fixed c — O(n) via cumulative sums
    idx    = series.index
    X_days = (idx - idx[0]).days.to_numpy(dtype=float)
    log_x  = np.log(X_days + c)
    y      = series.values.copy()

    trend    = np.full(len(y), np.nan)
    a_final  = b_final = np.nan
    cum_n = cum_x = cum_y = cum_xx = cum_xy = 0.0
    MIN_PERIODS = 730   # require 2 years before trusting the trend

    for t, (lx, yt) in enumerate(zip(log_x, y)):
        if not np.isnan(yt):
            cum_n  += 1
            cum_x  += lx
            cum_y  += yt
            cum_xx += lx * lx
            cum_xy += lx * yt
        if cum_n < MIN_PERIODS:
            continue
        denom = cum_n * cum_xx - cum_x ** 2
        if abs(denom) < 1e-12:
            continue
        a = (cum_n * cum_xy - cum_x * cum_y) / denom
        b = (cum_y - a * cum_x) / cum_n
        trend[t] = a * log_x[t] + b
        a_final, b_final = a, b

    return pd.Series(trend, index=idx), a_final, b_final, c


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    hh = high.rolling(length).max()
    ll  = low.rolling(length).min()
    return 100.0 * (close - hh) / (hh - ll)


def ema(values: pd.Series, span: int) -> pd.Series:
    """EMA that skips NaN inputs rather than propagating them."""
    alpha = 2.0 / (span + 1.0)
    out   = np.full(len(values), np.nan)
    prev  = np.nan
    for i, v in enumerate(values.to_numpy(dtype=float)):
        if np.isnan(v):
            continue
        prev   = v if np.isnan(prev) else alpha * v + (1.0 - alpha) * prev
        out[i] = prev
    return pd.Series(out, index=values.index)


def years_since_halving(index: pd.DatetimeIndex) -> pd.Series:
    """Continuous years since the last BTC halving."""
    dates = pd.to_datetime(index)
    years = np.zeros(len(dates), dtype=float)
    for i, dt in enumerate(dates):
        eligible = HALVING_DATES[HALVING_DATES <= dt]
        last_halving = eligible.max() if len(eligible) else HALVING_DATES.min()
        years[i] = max(0, (dt - last_halving).days) / 365.25
    return pd.Series(years, index=index)


# ── build DWD ─────────────────────────────────────────────────────────────────

def build_dwd() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(ODS_CSV, index_col='Date', parse_dates=True)
    dwh = pd.DataFrame(index=raw.index)

    cutoff = holdout_cutoff(raw.index)
    print(f'Trend fit cutoff (test held out from): {cutoff.date()}')

    # ── price: log + log trend + residual ──────────────────────────────────────

    log_price = np.log(raw['price_usd'])
    price_trend, a, b, c = fit_log_trend(log_price, fit_until=cutoff)
    ref_date = raw.index[0].strftime('%Y-%m-%d')

    dwh['log_price_usd']      = log_price
    dwh['log_price_trend']    = price_trend
    dwh['log_price_residual'] = log_price - price_trend

    with open(TREND_PARAMS_JSON, 'w') as f:
        json.dump({'a': a, 'b': b, 'c': c, 'ref_date': ref_date}, f, indent=2)
    print(f'Trend params: a={a:.4f}  b={b:.4f}  c={c:.4f}  ref={ref_date}')

    # ── valuation ──────────────────────────────────────────────────────────────

    dwh['mvrv'] = raw['mvrv']

    # ── volatility ─────────────────────────────────────────────────────────────

    log_returns = np.log(raw['price_usd'] / raw['price_usd'].shift(1))
    dwh['realized_vol_30'] = (
        log_returns.rolling(REALIZED_VOL_WINDOW).std() * np.sqrt(TRADING_DAYS) * 100
    )

    # ── cycle ──────────────────────────────────────────────────────────────────

    dwh['years_since_halving'] = years_since_halving(raw.index)

    # ── macro: DXY returns ─────────────────────────────────────────────────────

    dwh['dxy']         = raw['dxy']
    dwh['dxy_ret_30']  = raw['dxy'].pct_change(30)
    dwh['dxy_ret_100'] = raw['dxy'].pct_change(100)

    # ── technical: Williams %R + oversold composite ────────────────────────────

    for col, length, ema_span in WILLIAMS_R:
        raw_wr   = williams_r(raw['high'], raw['low'], raw['close'], length)
        dwh[col] = (ema(raw_wr, ema_span) + 100) / 100

    dwh['wr_composite'] = (1 - dwh['short_percent_r']) * (1 - dwh['long_percent_r'])

    return dwh, raw


# ── figures ───────────────────────────────────────────────────────────────────

def plot_price(dwh: pd.DataFrame, raw: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('BTC Price Decomposition', fontsize=14)

    axes[0].plot(raw.index, raw['price_usd'], color='steelblue', linewidth=0.8)
    axes[0].set_yscale('log')
    axes[0].set_title('Close Price USD (log scale)')
    axes[0].grid(linestyle=':')

    axes[1].plot(dwh.index, dwh['log_price_usd'],   color='steelblue', linewidth=0.8, label='log price')
    axes[1].plot(dwh.index, dwh['log_price_trend'],  color='red',       linewidth=1.2, linestyle='--', label='log trend')
    axes[1].set_title('Log Price + Fitted Log Trend')
    axes[1].legend(fontsize=9)
    axes[1].grid(linestyle=':')

    axes[2].plot(dwh.index, dwh['log_price_residual'], color='green', linewidth=0.8)
    axes[2].axhline(0, color='black', linestyle=':', linewidth=0.6)
    axes[2].set_title('Log Price Residual — Forecast Target')
    axes[2].grid(linestyle=':')

    plt.tight_layout()
    plt.savefig(OUTPUT / 'step2_fig1_price.png', dpi=150)
    plt.close()
    print('Saved → step2_fig1_price.png')


def plot_signals(dwh: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Validated Signal Features', fontsize=14)

    axes[0].plot(dwh.index, dwh['realized_vol_30'], color='darkorange', linewidth=0.8)
    axes[0].axhline(18.31, color='red', linestyle='--', linewidth=0.9, label='18.31 threshold (89–99% win rate)')
    axes[0].set_title('30-Day Realized Volatility (annualised %)')
    axes[0].legend(fontsize=9)
    axes[0].grid(linestyle=':')

    axes[1].plot(dwh.index, dwh['dxy_ret_30'],  color='steelblue', linewidth=0.8, label='DXY ret 30d')
    axes[1].plot(dwh.index, dwh['dxy_ret_100'], color='navy',      linewidth=0.8, label='DXY ret 100d', alpha=0.7)
    axes[1].axhline(0, color='black', linestyle=':', linewidth=0.6)
    axes[1].set_title('DXY Returns — macro regime signal')
    axes[1].legend(fontsize=9)
    axes[1].grid(linestyle=':')

    axes[2].plot(dwh.index, dwh['years_since_halving'], color='purple', linewidth=0.8)
    axes[2].set_title('Years Since Last Halving')
    axes[2].grid(linestyle=':')

    plt.tight_layout()
    plt.savefig(OUTPUT / 'step2_fig2_signals.png', dpi=150)
    plt.close()
    print('Saved → step2_fig2_signals.png')


if __name__ == '__main__':
    ensure_dirs()
    dwh, raw = build_dwd()
    dwh.to_csv(DWD_CSV)
    print(f'\nDWD saved → {DWD_CSV}  ({dwh.shape[0]} rows × {dwh.shape[1]} cols)')
    print(f'Date range : {dwh.index[0].date()} → {dwh.index[-1].date()}')
    print(f'Columns    : {list(dwh.columns)}')

    print('\nGenerating figures...')
    plot_price(dwh, raw)
    plot_signals(dwh)
