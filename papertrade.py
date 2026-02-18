import os
import numpy as np
import pandas as pd
import yfinance as yf

DATA_WEIGHTS_LATEST = "data/weights_latest.csv"
OUT_DIR = "data/papertrade"
OUT_WEIGHTS_REB = os.path.join(OUT_DIR, "weights_rebalance.csv")
OUT_VALUE_DAILY = os.path.join(OUT_DIR, "portfolio_value_daily.csv")

START_CAPITAL_DKK = 100_000
INCEPTION_DATE = "2026-01-01"

def ensure_outputs():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(OUT_WEIGHTS_REB):
        pd.DataFrame(columns=["date", "ticker", "target_weight"]).to_csv(OUT_WEIGHTS_REB, index=False)

    if not os.path.exists(OUT_VALUE_DAILY):
        pd.DataFrame(columns=["date", "total_value"]).to_csv(OUT_VALUE_DAILY, index=False)

def load_universe():
    df = pd.read_csv(DATA_WEIGHTS_LATEST)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    tickers = df["ticker"].tolist()
    return tickers

def get_usd_to_dkk_series():
    fx = yf.download("DKK=X", period="max", interval="1d", auto_adjust=True, progress=False)
    if fx is None or fx.empty:
        raise RuntimeError("Kunne ikke hente DKK=X")
    return fx["Close"].dropna()

def download_closes_usd(tickers):
    raw = yf.download(tickers, period="max", interval="1d", auto_adjust=True, progress=False, threads=True)
    if raw is None or raw.empty:
        raise RuntimeError("Kunne ikke hente prisdata")
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = pd.DataFrame({tickers[0]: raw["Close"]})
    close = close.dropna(how="all").sort_index()
    return close

def last_trading_day_each_month(index_dt: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # Antager index_dt er handelsdage (fra Yahoo)
    s = pd.Series(index_dt, index=index_dt)
    last = s.groupby([index_dt.year, index_dt.month]).max().values
    return pd.DatetimeIndex(last)

def main():
    ensure_outputs()

    tickers = load_universe()
    closes_usd = download_closes_usd(tickers)
    fx = get_usd_to_dkk_series()

    # Align FX til aktiedage
    fx = fx.reindex(closes_usd.index).ffill()
    closes_dkk = closes_usd.mul(fx, axis=0)

    # Cut to inception+
    closes_dkk = closes_dkk[closes_dkk.index >= pd.Timestamp(INCEPTION_DATE)]

    # Rebalance dates = sidste handelsdag i måneden
    rebalance_dates = last_trading_day_each_month(closes_dkk.index)

    print(f"Universe tickers: {len(tickers)}")
    print(f"Price rows: {len(closes_dkk)}")
    print(f"Rebalance dates: {len(rebalance_dates)}")
    print("First 5 rebalance dates:", list(rebalance_dates[:5].date))

if __name__ == "__main__":
    main()
