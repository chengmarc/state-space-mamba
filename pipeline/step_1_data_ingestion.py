"""Stage 1 — data ingestion.

Fetches BTC OHLCV and the DXY dollar index (yfinance) plus CoinMetrics PriceUSD,
joins them on the CoinMetrics date spine, and writes the operational data store (ODS).
PriceUSD reaches back further than yfinance BTC-USD (~Sep 2014) and is used only to
backfill the early trend history in step_2; the spine must therefore stay CoinMetrics.

    Output: config.ODS_CSV
"""

import os
import json
import urllib.parse
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
    base_url = 'https://community-api.coinmetrics.io/v4/timeseries/asset-metrics'
    params = {
        'assets': 'btc',
        'metrics': 'PriceUSD',
        'frequency': '1d',
        'format': 'json',
        'page_size': '10000',
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as r:
        payload = json.load(r)

    df = pd.DataFrame(payload['data'])
    df['Date'] = pd.to_datetime(df['time'], utc=True).dt.tz_localize(None)
    df = df.set_index('Date').sort_index()
    df = df.iloc[365 * 2:]  # skip first 2 years of sparse data
    # Only PriceUSD is retained — it backfills pre-2014 trend history in step_2 where
    # yfinance BTC-USD has no data. The on-chain / MVRV metrics were never used and are dropped.
    df['PriceUSD'] = pd.to_numeric(df['PriceUSD'], errors='coerce')
    return df[['PriceUSD']].rename(columns={'PriceUSD': 'price_usd'})


def download_ohlcv() -> pd.DataFrame:
    df = yf.download('BTC-USD', start='2010-01-01', interval='1d', auto_adjust=False, progress=False)
    if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).normalize().tz_localize(None)
    df.index.name = 'Date'
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns=str.lower)


def download_dxy() -> pd.Series:
    df = yf.download('DX-Y.NYB', start='2010-01-01', interval='1d', auto_adjust=False, progress=False)
    if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).normalize().tz_localize(None)
    df.index.name = 'Date'
    return df['Close'].rename('dxy')


def build_ods() -> pd.DataFrame:
    print('Fetching CoinMetrics on-chain data...')
    cm = download_coinmetrics()

    print('Fetching yfinance BTC-USD OHLCV...')
    ohlcv = download_ohlcv()

    print('Fetching yfinance DXY...')
    dxy = download_dxy()

    # CoinMetrics is the date spine; everything else is left-joined
    df = cm.copy()
    df = df.join(ohlcv, how='left')
    df = df.join(dxy,   how='left')
    df['dxy'] = df['dxy'].ffill()

    return df


if __name__ == '__main__':
    ensure_dirs()
    data = build_ods()
    data.to_csv(ODS_CSV)
    print(f'\nODS saved → {ODS_CSV}')
    print(f'Shape      : {data.shape[0]} rows × {data.shape[1]} cols')
    print(f'Date range : {data.index[0].date()} → {data.index[-1].date()}')
    print(data.tail())
