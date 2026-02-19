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
    """
    Returns a DatetimeIndex of the last trading day of each month.
    """
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
    
    # Validate that no weight exceeds MAX_W
    assert np.all(w <= max_w + 1e-8), f"Weight constraint violated: {w[w > max_w]}"
    
    return w


# =========================
# Walk-forward papertrade
# =========================
def run_walkforward_papertrade(closes_dkk: pd.DataFrame):

    closes_dkk = closes_dkk.sort_index()
    start_dt = pd.Timestamp(INCEPTION_DATE)
    
    # Compute returns using pct_change without deprecated fill_method
    rets = closes_dkk.pct_change()
    all_dates = rets.index

    equity = START_CAPITAL_DKK
    eq_series = []
    eq_index = []

    # Portfolio state: current weights and which tickers they correspond to
    w_current = None
    cols_current = None
    
    # Holdings: actual DKK value held in each stock
    holdings = {}  # ticker -> DKK value held

    weights_rows = []
    min_obs = max(60, int(0.7 * LOOKBACK))
    fee = COST_BPS / 10000.0

    started = False
    
    # Get all month-end dates for efficient checking
    month_ends = set(last_trading_day_each_month(all_dates))

    for i, dt in enumerate(all_dates):
        
        # Check if we've started (on or after inception date)
        if not started and dt >= start_dt:
            started = True
        
        # Skip tracking until inception date
        if not started:
            continue

        # ==================================================
        # REBALANCE LOGIC: Execute on the next day after month-end
        # ==================================================
        
        # Get previous trading day (if exists)
        if i > 0:
            prev_dt = all_dates[i - 1]
        else:
            prev_dt = None
        
        # Check if PREVIOUS day was a month-end
        is_day_after_month_end = prev_dt is not None and prev_dt in month_ends
        
        if is_day_after_month_end:
            print(f"\nRebalancing portfolio on {dt.date()} (day after month-end {prev_dt.date()})")
            
            loc = all_dates.get_loc(prev_dt)
            
            if loc >= LOOKBACK:
                # Get prices up to and including the month-end day (prev_dt)
                window_prices = closes_dkk.loc[:prev_dt].tail(LOOKBACK + 1)

                # Compute returns WITHOUT dropping rows yet
                window_rets_full = window_prices.pct_change()

                # Select investable columns based on history length
                valid_cols = window_rets_full.count()
                cols = valid_cols[valid_cols >= min_obs].index.tolist()
                print(f"  Eligible tickers: {len(cols)}")

                if len(cols) >= 10:
                    # Now drop rows only within the investable universe
                    wdw = window_rets_full[cols].dropna(axis=0, how="any")
                    print(f"  Rows in clean window: {len(wdw)}")

                    if len(wdw) >= min_obs:
                        mu_ann, cov_ann = estimate_mu_cov_ann(wdw)
                        w_new = max_sharpe_weights(mu_ann, cov_ann, max_w=MAX_W)

                        # Calculate turnover cost
                        if w_current is None:
                            turnover = 1.0
                        else:
                            old = pd.Series(w_current, index=cols_current).reindex(cols).fillna(0.0).values
                            turnover = float(np.sum(np.abs(w_new - old)))

                        # Deduct trading costs
                        pending_cost = turnover * fee
                        equity *= (1.0 - pending_cost)
                        print(f"  Turnover: {turnover:.4f}, Cost: {pending_cost * 100:.4f}%")

                        # Update current weights
                        w_current = w_new
                        cols_current = cols

                        # Record the rebalance
                        for tkr, wt in pd.Series(w_new, index=cols).items():
                            weights_rows.append({
                                "date": prev_dt.date(),
                                "ticker": tkr,
                                "target_weight": float(wt)
                            })

                        # EXECUTE REBALANCE: Set holdings to match target weights
                        # Holdings are expressed in DKK values
                        holdings = {tkr: equity * float(wt) for tkr, wt in zip(cols, w_new)}
                        
                        print(f"  Portfolio rebalanced to target weights:")
                        for tkr, holding_value in sorted(holdings.items()):
                            pct = (holding_value / equity) * 100
                            print(f"    {tkr}: {holding_value:,.2f} DKK ({pct:.2f}%)")

        # ==================================================
        # DAILY PnL: Apply returns to current holdings
        # ==================================================
        
        if w_current is None or cols_current is None or not holdings:
            # No position yet
            eq_series.append(equity)
            eq_index.append(dt)
            continue

        # Get today's returns for holdings
        r_today = rets.loc[dt].reindex(cols_current).fillna(0.0).values
        
        # Update each holding with today's return
        total_pnl = 0.0
        for j, tkr in enumerate(cols_current):
            if tkr in holdings:
                pnl_ticker = holdings[tkr] * r_today[j]
                holdings[tkr] += pnl_ticker
                total_pnl += pnl_ticker
        
        equity += total_pnl

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
    print(f"\nRunning walk-forward backtest from {INCEPTION_DATE}...\n")
    eq, w_hist = run_walkforward_papertrade(closes_dkk)

    # Write equity curve
    df_eq = eq.dropna().to_frame().reset_index()
    df_eq.columns = ["date", "total_value"]
    df_eq["date"] = pd.to_datetime(df_eq["date"]).dt.date
    df_eq.to_csv(OUT_VALUE_DAILY, index=False)
    print(f"\nWrote: {OUT_VALUE_DAILY}")

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
        
        print(f"\nLatest rebalance date: {last_date.date()}")
        print(f"Final portfolio value: {eq.iloc[-1]:,.2f} DKK")
        print(f"Initial capital: {START_CAPITAL_DKK:,.2f} DKK")
        print(f"Return: {((eq.iloc[-1] / START_CAPITAL_DKK) - 1) * 100:.2f}%")
    else:
        print("WARNING: No rebalance weights produced yet (not enough data / dates).")

    print("\nDone.")


if __name__ == "__main__":
    main()