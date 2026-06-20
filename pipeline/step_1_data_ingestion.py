"""Stage 1 — data ingestion.

Fetches raw BTC on-chain data (CoinMetrics), OHLCV (yfinance) and macro series
(FRED), joins them on the CoinMetrics date spine, and writes the operational data
store (ODS).

    Output: config.ODS_CSV
"""

import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from ssm.config import ROOT, ODS_CSV, ensure_dirs

_YF_CACHE = Path(os.environ.get('TEMP', str(ROOT))) / 'yfinance-cache'
_YF_CACHE.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(_YF_CACHE))


def download_coinmetrics() -> pd.DataFrame:
    url = 'https://raw.githubusercontent.com/coinmetrics/data/refs/heads/master/csv/btc.csv'
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['time'])
    df = df.set_index('Date').sort_index()
    df = df.iloc[365 * 2:-1]  # skip first 2 years of sparse data; drop last incomplete row
    # Note: DiffLast / SplyAct30d / SplyAct1yr removed from the free CoinMetrics feed.
    # HashRate is used in place of DiffLast (both track mining activity; difficulty
    # adjusts to target hash rate, so the series are tightly coupled).
    # CapMVRVCur (MVRV ratio) added — available in the current feed and useful signal.
    return df[['PriceUSD', 'HashRate', 'TxCnt', 'AdrActCnt', 'SplyCur', 'CapMVRVCur']].rename(columns={
        'PriceUSD':   'price_usd',
        'HashRate':   'hash_rate',
        'TxCnt':      'tx_cnt',
        'AdrActCnt':  'adr_act_cnt',
        'SplyCur':    'sply_cur',
        'CapMVRVCur': 'mvrv',
    })


def download_ohlcv() -> pd.DataFrame:
    df = yf.download('BTC-USD', start='2010-01-01', interval='1d', auto_adjust=False, progress=False)
    if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).normalize().tz_localize(None)
    df.index.name = 'Date'
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns=str.lower)


def download_fred(series_id: str, name: str) -> pd.Series:
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
    with urllib.request.urlopen(url, timeout=60) as r:
        frame = pd.read_csv(r)
    frame['observation_date'] = pd.to_datetime(frame['observation_date'])
    values = pd.to_numeric(frame[series_id], errors='coerce')
    return pd.Series(values.to_numpy(), index=frame['observation_date'], name=name)


def build_ods() -> pd.DataFrame:
    print('Fetching CoinMetrics on-chain data...')
    cm = download_coinmetrics()

    print('Fetching yfinance BTC-USD OHLCV...')
    ohlcv = download_ohlcv()

    print('Fetching FRED UNRATE...')
    unrate = download_fred('UNRATE', 'unrate')

    print('Fetching FRED CPIAUCSL...')
    cpi = download_fred('CPIAUCSL', 'cpi')

    # CoinMetrics is the date spine; everything else is left-joined
    df = cm.copy()
    df = df.join(ohlcv, how='left')
    df = df.join(unrate, how='left')
    df = df.join(cpi, how='left')

    # Forward-fill monthly FRED series to daily
    df['unrate'] = df['unrate'].ffill()
    df['cpi']    = df['cpi'].ffill()

    return df


if __name__ == '__main__':
    ensure_dirs()
    data = build_ods()
    data.to_csv(ODS_CSV)
    print(f'\nODS saved → {ODS_CSV}')
    print(f'Shape      : {data.shape[0]} rows × {data.shape[1]} cols')
    print(f'Date range : {data.index[0].date()} → {data.index[-1].date()}')
    print(data.tail())
