"""Stage 2 — feature engineering.

Reads the ODS, builds the data warehouse (DWH) of model features: log-detrended
price and on-chain residuals, technical oscillators (Williams %R, NMD), realized
volatility and macro features. Also persists the fitted price-trend parameters so
stage 5 can reconstruct absolute prices from residual forecasts.

    Input : config.ODS_CSV
    Output: config.DWH_CSV, config.TREND_PARAMS_JSON, step2_fig*.png
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from ssm.config import OUTPUT, ODS_CSV, DWH_CSV, TREND_PARAMS_JSON, ensure_dirs


# ── config ─────────────────────────────────────────────────────────────────────

# On-chain features detrended against log(supply).
# Each tuple: (raw_col_in_ods, output_prefix_in_dwh)
SUPPLY_DETREND = [
    ('hash_rate',    'log_hash_rate'),
    ('tx_cnt',       'log_tx_cnt'),
    ('adr_act_cnt',  'log_adr_act_cnt'),
]

# Williams %R variants.
# Each tuple: (output_col, wr_length, ema_smoothing_span)
WILLIAMS_R = [
    ('short_percent_r', 21,  7),
    ('long_percent_r',  112, 3),
]

# Normalized Maximum Drawdown variants.
# Each tuple: (output_col, lookback_bars, normalize_by)
NMD = [
    ('nmd_730', 730, 730),
    ('nmd_365', 365, 730),
    ('nmd_90',   90, 730),
]

REALIZED_VOL_WINDOW = 30
TRADING_DAYS        = 252


# ── helpers ────────────────────────────────────────────────────────────────────

def _log_curve(x, a, b, c):
    return a * np.log(x + c) + b


def fit_log_trend(series: pd.Series):
    """Fit Y = a·log(X+c)+b where X = integer days since series.index[0].

    Returns (trend_series, a, b, c).
    """
    valid = series.dropna()
    X_fit = (valid.index - series.index[0]).days.to_numpy(dtype=float)
    Y_fit = valid.to_numpy(dtype=float)
    (a, b, c), _ = curve_fit(_log_curve, X_fit, Y_fit, p0=[1.0, 1.0, 1.0])
    X_full = (series.index - series.index[0]).days.to_numpy(dtype=float)
    trend  = pd.Series(_log_curve(X_full, a, b, c), index=series.index)
    return trend, a, b, c


def fit_supply_trend(series: pd.Series, supply: pd.Series):
    """Fit log(feature) = a·log(supply) + b — power-law baseline against issuance.

    Returns (trend_series, a, b).
    """
    valid  = series.notna() & supply.notna() & (supply > 0)
    X_fit  = np.log(supply[valid].to_numpy(dtype=float))
    Y_fit  = series[valid].to_numpy(dtype=float)
    a, b   = np.polyfit(X_fit, Y_fit, 1)
    X_full = np.where(supply > 0, np.log(supply.clip(lower=1e-10)), np.nan)
    trend  = pd.Series(a * X_full + b, index=series.index)
    return trend, a, b


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """Williams %R: 100 * (close - highest_high) / (highest_high - lowest_low). Range [-100, 0]."""
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


def compute_nmd(price: pd.Series, lookback: int, normalize_by: int) -> pd.Series:
    """Fraction of bars in `lookback` window that closed higher than current bar.

    Normalized by `normalize_by` so all variants share a common scale. Range [0, 1].
    """
    p      = price.to_numpy(dtype=float)
    n      = len(p)
    result = np.zeros(n)
    for i in range(lookback + 1, n):
        result[i] = np.sum(p[i - lookback:i] > p[i])
    return pd.Series(result / normalize_by, index=price.index)


# ── build DWH ─────────────────────────────────────────────────────────────────

def build_dwh() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(ODS_CSV, index_col='Date', parse_dates=True)
    dwh = pd.DataFrame(index=raw.index)

    # ── price: log + log trend + residual ──────────────────────────────────────

    log_price = np.log(raw['price_usd'])
    price_trend, a, b, c = fit_log_trend(log_price)
    ref_date = raw.index[0].strftime('%Y-%m-%d')

    dwh['log_price_usd']      = log_price
    dwh['log_price_trend']    = price_trend
    dwh['log_price_residual'] = log_price - price_trend

    with open(TREND_PARAMS_JSON, 'w') as f:
        json.dump({'a': a, 'b': b, 'c': c, 'ref_date': ref_date}, f, indent=2)
    print(f'Trend params: a={a:.4f}  b={b:.4f}  c={c:.4f}  ref={ref_date}')

    # ── on-chain: supply-detrended features ────────────────────────────────────

    supply = raw['sply_cur']
    for raw_col, dst in SUPPLY_DETREND:
        log_s = np.log(raw[raw_col].replace(0, np.nan))
        trend, *_ = fit_supply_trend(log_s, supply)
        dwh[dst]               = log_s
        dwh[f'{dst}_trend']    = trend
        dwh[f'{dst}_residual'] = log_s - trend

    dwh['mvrv'] = raw['mvrv']

    # ── Williams %R ────────────────────────────────────────────────────────────

    for col, length, ema_span in WILLIAMS_R:
        raw_wr  = williams_r(raw['high'], raw['low'], raw['close'], length)
        dwh[col] = (ema(raw_wr, ema_span) + 100) / 100   # rescale [-100,0] → [0,1]

    # ── NMD ───────────────────────────────────────────────────────────────────

    print('Computing NMD features...')
    for col, lookback, normalize_by in NMD:
        dwh[col] = compute_nmd(raw['price_usd'], lookback, normalize_by)

    # ── realized volatility ───────────────────────────────────────────────────

    log_returns        = np.log(raw['price_usd'] / raw['price_usd'].shift(1))
    dwh['realized_vol_30'] = log_returns.rolling(REALIZED_VOL_WINDOW).std() * np.sqrt(TRADING_DAYS) * 100

    # ── macro ─────────────────────────────────────────────────────────────────

    dwh['cpi_yoy'] = (raw['cpi'] / raw['cpi'].shift(TRADING_DAYS) - 1.0) * 100.0

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


def plot_onchain(dwh: pd.DataFrame) -> None:
    panels = [(dst, f'{dst}_residual', label) for dst, label in [
        ('log_hash_rate',   'Hash Rate'),
        ('log_tx_cnt',      'Transaction Count'),
        ('log_adr_act_cnt', 'Active Addresses'),
    ]] + [('mvrv', None, 'MVRV Ratio')]

    fig, axes = plt.subplots(len(panels), 2, figsize=(16, 4 * len(panels)), sharex=True)
    fig.suptitle('On-Chain Features: Log vs Residual', fontsize=14)

    for i, (log_col, resid_col, label) in enumerate(panels):
        ax_l, ax_r = axes[i, 0], axes[i, 1]

        ax_l.plot(dwh.index, dwh[log_col], color='steelblue', linewidth=0.8)
        trend_col = f'{log_col}_trend'
        if trend_col in dwh.columns:
            ax_l.plot(dwh.index, dwh[trend_col], color='red', linestyle='--', linewidth=1.0, label='trend')
            ax_l.legend(fontsize=8)
        ax_l.set_title(f'{label} — Log')
        ax_l.grid(linestyle=':')

        if resid_col:
            ax_r.plot(dwh.index, dwh[resid_col], color='green', linewidth=0.8)
            ax_r.axhline(0, color='black', linestyle=':', linewidth=0.6)
            ax_r.set_title(f'{label} — Residual (supply-detrended)')
            ax_r.grid(linestyle=':')
        else:
            ax_r.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT / 'step2_fig2_onchain.png', dpi=150)
    plt.close()
    print('Saved → step2_fig2_onchain.png')


def plot_oscillators(dwh: pd.DataFrame) -> None:
    panels = (
        [(col, f'Short %R  ({length}-bar, EMA-{ema_span})', (0, 1), 'purple')
         for col, length, ema_span in WILLIAMS_R] +
        [(col, f'NMD {lookback}-bar', (0, 1), c)
         for (col, lookback, _), c in zip(NMD, ['crimson', 'darkorange', 'gold'])]
    )

    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 3.5 * len(panels)), sharex=True)
    fig.suptitle('Technical Oscillators', fontsize=14)

    for ax, (col, title, ylim, color) in zip(axes, panels):
        ax.plot(dwh.index, dwh[col], color=color, linewidth=0.8)
        ax.set_ylim(*ylim)
        ax.axhline(sum(ylim) / 2, color='gray', linestyle=':', linewidth=0.6)
        ax.set_title(title)
        ax.grid(linestyle=':')

    plt.tight_layout()
    plt.savefig(OUTPUT / 'step2_fig3_oscillators.png', dpi=150)
    plt.close()
    print('Saved → step2_fig3_oscillators.png')


def plot_macro(dwh: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle('Macro & Volatility Features', fontsize=14)

    axes[0].plot(dwh.index, dwh['realized_vol_30'], color='darkorange', linewidth=0.8)
    axes[0].set_title(f'{REALIZED_VOL_WINDOW}-Day Realized Volatility (annualised %)')
    axes[0].grid(linestyle=':')

    axes[1].plot(dwh.index, dwh['cpi_yoy'], color='crimson', linewidth=0.8)
    axes[1].axhline(2.0, color='gray', linestyle=':', linewidth=0.8, label='2% target')
    axes[1].set_title('CPI YoY (%)')
    axes[1].legend(fontsize=9)
    axes[1].grid(linestyle=':')

    plt.tight_layout()
    plt.savefig(OUTPUT / 'step2_fig4_macro.png', dpi=150)
    plt.close()
    print('Saved → step2_fig4_macro.png')


if __name__ == '__main__':
    ensure_dirs()
    dwh, raw = build_dwh()

    dwh.to_csv(DWH_CSV)
    print(f'\nDWH saved → {DWH_CSV}  ({dwh.shape[0]} rows × {dwh.shape[1]} cols)')

    print('\nGenerating figures...')
    plot_price(dwh, raw)
    plot_onchain(dwh)
    plot_oscillators(dwh)
    plot_macro(dwh)
