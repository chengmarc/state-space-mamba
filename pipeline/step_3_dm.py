"""Stage 3 — DM assembly.

Selects the empirically validated features from the DWD and writes the final
DM frame plus normalization parameters.

Feature set (proven in winrate-matrix; README is the SSoT):

  - cycle      : `years_since_halving`
  - volatility : `realized_vol_30`
  - macro      : `dxy_ret_30`, `dxy_ret_100`
  - technical  : `short_percent_r`, `long_percent_r`, `wr_composite`
  - valuation  : `log_price_residual`  (input context — current distance from trend)
  - target     : `log_price_usd`  ← must be last (contract with loader + train)

    Input : config.DWD_CSV
    Output: config.DM_CSV, config.NORM_PARAMS_JSON, step3_fig_distributions.png
"""

import json

import pandas as pd
import matplotlib.pyplot as plt

from ssm.config import OUTPUT, DWD_CSV, DM_CSV, NORM_PARAMS_JSON, ensure_dirs
from ssm.splits import holdout_cutoff


# ── feature registry ───────────────────────────────────────────────────────────
# Each entry: (dwd_column, transform)
# transform options: None | 'zscore'
# Target must be the last entry.

FEATURES = [
    ('years_since_halving',  None),
    ('realized_vol_30',      'zscore'),
    ('dxy_ret_30',           'scale_10'),
    ('dxy_ret_100',          'scale_10'),
    ('short_percent_r',      None),
    ('long_percent_r',       None),
    ('wr_composite',         None),
    ('log_price_residual',   None),    # target — must be last
]


# ── transforms ─────────────────────────────────────────────────────────────────

def apply_zscore(series: pd.Series, fit_mask) -> tuple[pd.Series, dict]:
    """Z-score using mean/std estimated on training rows only."""
    ref = series[fit_mask]
    mean, std = ref.mean(), ref.std()
    return (series - mean) / std, {'mean': mean, 'std': std}


def apply_scale_10(series: pd.Series, fit_mask) -> tuple[pd.Series, dict]:
    """Multiply by 10 — fixed linear scale, no fitting required.

    Used for DXY return features whose raw values (~0.02) are ~100x smaller
    than the residual. Preserves threshold semantics (e.g. -0.023 → -0.23).
    """
    return series * 10.0, {'factor': 10.0}


TRANSFORMS = {
    'zscore':   apply_zscore,
    'scale_10': apply_scale_10,
}


# ── build ─────────────────────────────────────────────────────────────────────

def build_dm() -> tuple[pd.DataFrame, dict]:
    dwd  = pd.read_csv(DWD_CSV, index_col='Date', parse_dates=True)
    cols = [col for col, _ in FEATURES]
    data = dwd[cols].dropna().copy()

    fit_mask = data.index < holdout_cutoff(data.index)

    norm_params = {}
    for col, transform in FEATURES:
        if transform is None:
            continue
        data[col], params = TRANSFORMS[transform](data[col], fit_mask)
        norm_params[col]  = {'transform': transform, **params}

    with open(NORM_PARAMS_JSON, 'w') as f:
        json.dump(norm_params, f, indent=2)

    return data, norm_params


# ── figure ────────────────────────────────────────────────────────────────────

_DIST_COLOR = {'zscore': 'darkorange', 'scale_10': 'mediumseagreen'}


def plot_distributions(data: pd.DataFrame, norm_params: dict) -> None:
    n      = len(data.columns)
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
    fig.suptitle('DM Feature Distributions', fontsize=14)
    axes = axes.flatten()

    for ax, col in zip(axes, data.columns):
        mn, mx    = data[col].min(), data[col].max()
        transform = norm_params.get(col, {}).get('transform')
        color     = _DIST_COLOR.get(transform, 'steelblue')
        ax.hist(data[col].dropna(), bins=60, range=(mn, mx), color=color, edgecolor='none', alpha=0.85)
        suffix = '' if not transform else f' [{transform}]'
        ax.set_title(f'{col}{suffix}', fontsize=9)
        ax.set_ylabel('count', fontsize=8)
        ax.set_xlim(mn, mx)
        ax.set_xlabel(f'[{mn:.2f}, {mx:.2f}]', fontsize=8)
        ax.grid(linestyle=':', linewidth=0.5)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT / 'step3_fig_distributions.png', dpi=150)
    plt.close()
    print('Saved → step3_fig_distributions.png')


if __name__ == '__main__':
    ensure_dirs()
    dwd_len = len(pd.read_csv(DWD_CSV))
    data, norm_params = build_dm()
    data.to_csv(DM_CSV)

    print(f'DM saved → {DM_CSV}')
    print(f'Shape      : {data.shape[0]} rows × {data.shape[1]} cols')
    print(f'Date range : {data.index[0].date()} → {data.index[-1].date()}')
    print(f'Rows dropped (NaN warmup): {dwd_len - len(data)}')

    print(f'\nFeature registry ({len(FEATURES)}):')
    for i, (col, transform) in enumerate(FEATURES):
        tag = '← target' if i == len(FEATURES) - 1 else transform or ''
        print(f'  {i+1:2d}. {col:<35} {tag}')

    print('\nGenerating distribution plot...')
    plot_distributions(data, norm_params)
