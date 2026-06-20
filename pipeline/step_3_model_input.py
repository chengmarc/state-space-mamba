"""Stage 3 — model input assembly.

Selects the modelling features from the DWH, applies per-feature transforms
(z-score where configured), and writes the final model-input frame plus the
normalization parameters needed to invert them. The target column
(`log_price_residual`) is the last entry in the feature registry by contract.

    Input : config.DWH_CSV
    Output: config.MODEL_INPUT_CSV, config.NORM_PARAMS_JSON, step3_fig_distributions.png
"""

import json

import pandas as pd
import matplotlib.pyplot as plt

from ssm.config import OUTPUT, DWH_CSV, MODEL_INPUT_CSV, NORM_PARAMS_JSON, ensure_dirs


# ── feature registry ───────────────────────────────────────────────────────────
# Each entry: (dwh_column, transform)
# transform options: None | 'zscore'
# Target must be the last entry.

FEATURES = [
    ('log_hash_rate_residual',   None),
    ('log_tx_cnt_residual',      None),
    ('log_adr_act_cnt_residual', None),
    ('mvrv',                     'zscore'),
    ('short_percent_r',          None),
    ('long_percent_r',           None),
    ('nmd_730',                  None),
    ('nmd_365',                  None),
    ('nmd_90',                   None),
    ('realized_vol_30',          'zscore'),
    ('cpi_yoy',                  'zscore'),
    ('log_price_residual',       None),    # target — must be last
]


# ── transforms ─────────────────────────────────────────────────────────────────

def apply_zscore(series: pd.Series) -> tuple[pd.Series, dict]:
    mean, std = series.mean(), series.std()
    return (series - mean) / std, {'mean': mean, 'std': std}


TRANSFORMS = {
    'zscore': apply_zscore,
}


# ── build ─────────────────────────────────────────────────────────────────────

def build_model_input() -> tuple[pd.DataFrame, dict]:
    dwh  = pd.read_csv(DWH_CSV, index_col='Date', parse_dates=True)
    cols = [col for col, _ in FEATURES]
    data = dwh[cols].dropna().copy()

    norm_params = {}
    for col, transform in FEATURES:
        if transform is None:
            continue
        data[col], params = TRANSFORMS[transform](data[col])
        norm_params[col]  = {'transform': transform, **params}

    with open(NORM_PARAMS_JSON, 'w') as f:
        json.dump(norm_params, f, indent=2)

    return data, norm_params


# ── figure ────────────────────────────────────────────────────────────────────

def plot_distributions(data: pd.DataFrame, norm_params: dict) -> None:
    n    = len(data.columns)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3.5))
    fig.suptitle('Model Input Feature Distributions', fontsize=14)
    axes = axes.flatten()

    for i, col in enumerate(data.columns):
        ax    = axes[i]
        mn, mx = data[col].min(), data[col].max()
        color = 'darkorange' if col in norm_params else 'steelblue'
        ax.hist(data[col].dropna(), bins=60, range=(mn, mx), color=color, edgecolor='none', alpha=0.85)
        ax.set_title(col, fontsize=9)
        ax.set_ylabel('count', fontsize=8)
        ax.set_xlim(mn, mx)
        ax.set_xlabel(f'[{mn:.2f}, {mx:.2f}]', fontsize=8)
        ax.grid(linestyle=':', linewidth=0.5)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT / 'step3_fig_distributions.png', dpi=150)
    plt.close()
    print('Saved → step3_fig_distributions.png')


if __name__ == '__main__':
    ensure_dirs()
    dwh_len = len(pd.read_csv(DWH_CSV))
    data, norm_params = build_model_input()
    data.to_csv(MODEL_INPUT_CSV)

    print(f'Model input saved → {MODEL_INPUT_CSV}')
    print(f'Shape      : {data.shape[0]} rows × {data.shape[1]} cols')
    print(f'Date range : {data.index[0].date()} → {data.index[-1].date()}')
    print(f'Rows dropped (NaN warmup): {dwh_len - len(data)}')

    print(f'\nFeature registry ({len(FEATURES)}):')
    for i, (col, transform) in enumerate(FEATURES):
        tag = '← target' if i == len(FEATURES) - 1 else transform or ''
        print(f'  {i+1:2d}. {col:<35} {tag}')

    if norm_params:
        print(f'\nNorm params saved → norm_params.json')
        for col, p in norm_params.items():
            if p['transform'] == 'zscore':
                print(f'  {col}: mean={p["mean"]:.4f}  std={p["std"]:.4f}')

    print('\nGenerating distribution plot...')
    plot_distributions(data, norm_params)
