import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from streamlit_autorefresh import st_autorefresh

TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","META",
    "NVDA","AMD","INTC","AVGO","QCOM","TXN","ADI","MU",
    "TSM","ASML","AMAT","LRCX","KLAC"
]

LOOKBACK = 252
MAX_WEIGHT = 0.08
START = "2015-01-01"

def max_sharpe(mu, cov, max_w):
    n = len(mu)
    w0 = np.ones(n)/n

    def neg_sharpe(w):
        ret = w @ mu
        vol = math.sqrt(w @ cov @ w)
        return -(ret/vol)

    bounds = [(0, max_w)] * n
    cons = {"type":"eq","fun":lambda w: w.sum()-1}

    res = minimize(neg_sharpe, w0, bounds=bounds, constraints=cons)
    w = res.x
    w = np.clip(w, 0, max_w)
    return w / w.sum()

def main():
    os.makedirs("data", exist_ok=True)

    px = yf.download(TICKERS, start=START, auto_adjust=True, progress=False)["Close"]
    rets = px.pct_change().dropna()
    window = rets.iloc[-LOOKBACK:]

    mu = window.mean()*252
    cov = window.cov()*252

    w = max_sharpe(mu.values, cov.values, MAX_WEIGHT)

    df = pd.DataFrame({
        "ticker": px.columns,
        "weight": w
    }).sort_values("weight", ascending=False)

    df.to_csv("data/weights_latest.csv", index=False)

    print("✅ weights_latest.csv created")

if __name__ == "__main__":
    main()
