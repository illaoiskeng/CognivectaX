import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

# =========================
# CONFIG
# =========================
OUT_DIR = "data/papertrade"
OUT_WEIGHTS_REB = os.path.join(OUT_DIR, "weights_rebalance.csv")
OUT_VALUE_DAILY = os.path.join(OUT_DIR, "portfolio_value_daily.csv")

# This file is what the dashboard already reads for pie/holdings:
OUT_WEIGHTS_LATEST = "data/weights_latest.csv"

START_CAPITAL_DKK = 100_000
INCEPTION_DATE = "2026-01-01"

LOOKBACK = 252
MAX_W = 0.08
ANN = 252

# Costs: bps per 100% turnover
COST_BPS = 10.0

# Universe: keep it explicit and stable (avoid accidental drift)
TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","META",
    "NVDA","AMD","INTC","AVGO","QCOM","TXN","ADI","MU","NXPI","MRVL","MCHP","ON","STM",
    "TSM","ASML","AMAT","LRCX","KLAC","TER","MPWR","SWKS","QRVO","WDC","STX","UMC",
    "CSCO","ANET","DELL","HPE","HPQ","NTAP","JNPR","ERIC","NOK",
    "ORCL","IBM","CRM","NOW","ADBE","INTU","PANW","CRWD","FTNT","SNOW","DDOG","NET",
    "PLTR","MDB","TEAM","ZS","OKTA","SHOP","SQ","PYPL","TSLA"
]

# =========================
# IO helpers
# =========================
def ensure_outputs():
    os.makedirs("data", exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(OUT_WEIGHTS_REB):
        pd.DataFrame(columns=["date", "ticker", "target_weight"]).to_csv(OUT_WEIGHTS_REB, index=False)

    if not os.path.exists(OUT_VALUE_DAILY):
        pd.DataFrame(columns=["date", "total_value"]).to_csv(OUT_VALUE_DAILY, index=False)


# =========================
# Data helpers
# =========================
def robust_download_close(tickers, start=None, period="max", interval="1d"):
    """
    Robust yfinance downloader:
    - threads=False reduces timezone/metadata flakiness
    - drops tickers that fail
    """
    # Batch first
    raw = yf.download(
        tickers=tickers,
        start=start,
        period=None if start else period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
        group_by="column"
    )

    close = None
    if raw is not None and not raw.empty:
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].copy()
        else:
            close = pd.DataFrame({tickers[0]: raw["Close"]})
        close = close.sort_index().dropna(how="all")

    out = {}
    # Fill per-ticker for missing/failed
    for t in tickers:
        if close is not None and (t in close.columns) and (not close[t].dropna().empty):
            out[t] = close[t]
            continue
        try:
            s = yf.download(
                tickers=t,
                start=start,
                period=None if start else period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if s is None or s.empty:
                raise RuntimeError("empty")
            out[t] = s["Close"].rename(t)
        except Exception:
            print(f"WARNING: Dropping ticker (download failed): {t}")

    if not out:
        raise RuntimeError("No price data downloaded successfully.")

    return pd.concat(out.values(), axis=1).sort_index()


def get_usd_to_dkk_series():
    fx = yf.download("DKK=X", period="max", interval="1d", auto_adjust=True, progress=False, threads=False)
    if fx is None or fx.empty:
        raise RuntimeError("Could not fetch DKK=X")
    return fx["Close"].dropna().sort_index()


def last_trading_day_each_month(index_dt: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index_dt, index=index_dt)
    last = s.groupby([index_dt.year, index_dt.month]).max().values
    return pd.DatetimeIndex(pd.to_datetime(last)).sort_values().unique()


# =========================
# Portfolio math
# =========================
def estimate_mu_cov_ann(returns_daily: pd.DataFrame):
    mu_daily = returns_daily.mean().values
    cov_daily = returns_daily.cov().values
    mu_ann = mu_daily * ANN
    cov_ann = cov_daily * ANN
    return mu_ann, cov_ann


def max_sharpe_weights(mu_ann: np.ndarray, cov_ann: np.ndarray, max_w: float = 0.08) -> np.ndarray:
    n = len(mu_ann)
    cov_ann = np.asarray(cov_ann, dtype=float) + np.eye(n) * 1e-8  # numerical stability

    def neg_sharpe(w):
        port_ret = float(w @ mu_ann)       # rf=0
        port_var = float(w @ cov_ann @ w)
        port_vol = math.sqrt(max(1e-12, port_var))
        return -(port_ret / port_vol)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, float(max_w))] * n

    w0 = np.ones(n) / n
    if (w0 > max_w).any():
        w0 = np.minimum(w0, max_w)
        w0 = w0 / w0.sum()

    res = minimize(
        neg_sharpe,
        x0=w0,
        bounds=bounds,
        constraints=cons,
        method="SLSQP",
        options={"maxiter": 400, "ftol": 1e-9, "disp": False},
    )

    w = res.x if res.success else w0
    w = np.clip(w, 0.0, max_w)
    w = w / w.sum()
    return w


# =========================
# Walk-forward papertrade
# =========================
def run_walkforward_papertrade(closes_dkk: pd.DataFrame):

    closes_dkk = closes_dkk.sort_index()

    start_dt = pd.Timestamp(INCEPTION_DATE)

    rets = closes_dkk.pct_change(fill_method=None)
    all_dates = rets.index

    reb_dates = last_trading_day_each_month(all_dates)
    reb_set = set(reb_dates)

    equity = START_CAPITAL_DKK
    eq_series = []
    eq_index = []

    w_current = None
    cols_current = None

    pending_w = None
    pending_cols = None
    pending_cost = 0.0

    weights_rows = []

    min_obs = max(60, int(0.7 * LOOKBACK))
    fee = COST_BPS / 10000.0

    started = False

    for dt in all_dates:

        # Activate new weights at start of day
        if pending_w is not None:
            w_current = pending_w
            cols_current = pending_cols
            pending_w = None
            pending_cols = None
            equity *= (1.0 - pending_cost)
            pending_cost = 0.0

        # Rebalance decision (month-end)
        if dt in reb_set:

            loc = all_dates.get_loc(dt)

            if loc >= LOOKBACK + 1:

                t_minus_1 = all_dates[loc - 1]

                window_prices = closes_dkk.loc[:t_minus_1].tail(LOOKBACK + 1)

                # Compute returns WITHOUT dropping rows yet
                window_rets_full = window_prices.pct_change(fill_method=None)

                # Select investable columns based on history length
                valid_cols = window_rets_full.count()
                cols = valid_cols[valid_cols >= min_obs].index.tolist()

                if len(cols) >= 10:

                    # Now drop rows only within the investable universe
                    wdw = window_rets_full[cols].dropna(axis=0, how="any")

                    if len(wdw) >= min_obs:

                        mu_ann, cov_ann = estimate_mu_cov_ann(wdw)
                        w_new = max_sharpe_weights(mu_ann, cov_ann, max_w=MAX_W)

                        if w_current is None:
                            turnover = 1.0
                        else:
                            old = pd.Series(w_current, index=cols_current).reindex(cols).fillna(0.0).values
                            turnover = float(np.sum(np.abs(w_new - old)))

                        pending_cost = turnover * fee
                        pending_w = w_new
                        pending_cols = cols

                        for tkr, wt in pd.Series(w_new, index=cols).items():
                            weights_rows.append({
                                "date": dt.date(),
                                "ticker": tkr,
                                "target_weight": float(wt)
                            })

        # Start tracking portfolio value from inception date
        if dt >= start_dt:
            started = True

        if not started:
            continue

        # Daily PnL
        if w_current is None or cols_current is None:
            eq_series.append(equity)
            eq_index.append(dt)
            continue

        r_vec = rets.loc[dt].reindex(cols_current).fillna(0.0).values
        equity *= (1.0 + float(np.dot(w_current, r_vec)))

        eq_series.append(equity)
        eq_index.append(dt)

    eq = pd.Series(eq_series, index=pd.DatetimeIndex(eq_index), name="total_value")
    w_hist = pd.DataFrame(weights_rows)

    return eq, w_hist


def main():
    ensure_outputs()

    # Download prices USD and FX, convert to DKK
    print("Downloading prices (USD) + FX (USD/DKK)...")
    closes_usd = robust_download_close(TICKERS, period="max", interval="1d")
    fx = get_usd_to_dkk_series()

    # Align FX to price dates
    fx = fx.reindex(closes_usd.index).ffill()
    closes_dkk = closes_usd.mul(fx, axis=0)

    # Run walk-forward strategy
    eq, w_hist = run_walkforward_papertrade(closes_dkk)

    # Write equity curve
    df_eq = eq.dropna().to_frame().reset_index()
    df_eq.columns = ["date", "total_value"]
    df_eq["date"] = pd.to_datetime(df_eq["date"]).dt.date
    df_eq.to_csv(OUT_VALUE_DAILY, index=False)
    print(f"Wrote: {OUT_VALUE_DAILY}")

    # Write rebalance weights history
    if not w_hist.empty:
        w_hist.to_csv(OUT_WEIGHTS_REB, index=False)
        print(f"Wrote: {OUT_WEIGHTS_REB}")

        # Write latest weights for dashboard (latest rebalance date)
        last_date = pd.to_datetime(w_hist["date"]).max()
        latest = w_hist[pd.to_datetime(w_hist["date"]) == last_date].copy()
        latest = latest.rename(columns={"target_weight": "weight"})[["ticker", "weight"]]
        latest = latest.sort_values("weight", ascending=False)
        latest.to_csv(OUT_WEIGHTS_LATEST, index=False)
        print(f"Wrote: {OUT_WEIGHTS_LATEST}")
    else:
        print("WARNING: No rebalance weights produced yet (not enough data / dates).")

    print("Done.")


if __name__ == "__main__":
    main()
