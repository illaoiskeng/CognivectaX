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
OUT_WEIGHTS_LATEST = "data/weights_latest.csv"

START_CAPITAL_DKK = 100_000
DATA_START_DATE = "2023-01-01"
INCEPTION_DATE = "2026-02-19"

LOOKBACK = 252
MAX_W = 0.08
ANN = 252
TURNOVER_COST_BPS = 10.0

TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","META",
    "NVDA","AMD","INTC","AVGO","QCOM","TXN","ADI","MU","NXPI","MRVL","MCHP","ON","STM",
    "TSM","ASML","AMAT","LRCX","KLAC","TER","MPWR","SWKS","QRVO","WDC","STX","UMC",
    "CSCO","ANET","DELL","HPE","HPQ","NTAP","ERIC","NOK",
    "ORCL","IBM","CRM","NOW","ADBE","INTU","PANW","CRWD","FTNT","SNOW","DDOG","NET",
    "PLTR","MDB","TEAM","ZS","OKTA","SHOP","PYPL","TSLA"
]

def ensure_outputs():
    os.makedirs("data", exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(OUT_WEIGHTS_REB):
        pd.DataFrame(columns=["date", "ticker", "target_weight"]).to_csv(OUT_WEIGHTS_REB, index=False)
    if not os.path.exists(OUT_VALUE_DAILY):
        pd.DataFrame(columns=["date", "total_value"]).to_csv(OUT_VALUE_DAILY, index=False)

def robust_download_close(tickers, start=None, period="max", interval="1d"):
    raw = yf.download(
        tickers=tickers, start=start, period=None if start else period,
        interval=interval, auto_adjust=True, progress=False,
        threads=False, group_by="column"
    )
    close = None
    if raw is not None and not raw.empty:
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].copy()
        else:
            close = pd.DataFrame({tickers[0]: raw["Close"]})
        close = close.sort_index().dropna(how="all")
    
    out = {}
    for t in tickers:
        if close is not None and (t in close.columns) and (not close[t].dropna().empty):
            out[t] = close[t]
            continue
        try:
            s = yf.download(tickers=t, start=start, period=None if start else period,
                           interval=interval, auto_adjust=True, progress=False, threads=False)
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

def month_end_trade_dates(index_dt: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index_dt, index=index_dt)
    # Get the last trading day of each month
    last = s.groupby([index_dt.year, index_dt.month]).max().values
    last_dates = pd.DatetimeIndex(pd.to_datetime(last)).sort_values().unique()
    
    # Filter to only include dates where the next trading day is in a different month
    month_ends = []
    for dt in last_dates:
        idx = index_dt.get_loc(dt)
        if idx < len(index_dt) - 1:
            next_dt = index_dt[idx + 1]
            if next_dt.month != dt.month:
                month_ends.append(dt)
        else:
            if idx > 0:
                prev_dt = index_dt[idx - 1]
                if prev_dt.month != dt.month:
                    month_ends.append(dt)
    
    return pd.DatetimeIndex(month_ends)

def estimate_mu_cov_ann(returns_daily: pd.DataFrame):
    mu_daily = returns_daily.mean().values
    cov_daily = returns_daily.cov().values
    mu_ann = mu_daily * ANN
    cov_ann = cov_daily * ANN
    return mu_ann, cov_ann

def max_sharpe_weights(mu_ann: np.ndarray, cov_ann: np.ndarray, max_w: float = 0.08) -> np.ndarray:
    n = len(mu_ann)
    mu_ann = np.asarray(mu_ann, dtype=float)
    cov_ann = np.asarray(cov_ann, dtype=float) + np.eye(n) * 1e-8
    
    w0 = np.full(n, 1.0 / n)
    w0 = np.minimum(w0, max_w)
    w0 = w0 / w0.sum()
    
    bounds = [(0.0, float(max_w)) for _ in range(n)]
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    
    def neg_sharpe(w):
        pret = float(np.dot(w, mu_ann))
        pvar = float(np.dot(w, cov_ann @ w))
        pvol = math.sqrt(max(1e-12, pvar))
        return -(pret / pvol)
    
    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 400, "ftol": 1e-9, "disp": False})
    
    w = res.x if res.x is not None else w0
    w = np.clip(w, 0.0, max_w)
    w = w / w.sum()
    
    print(f"  DEBUG: res.success={res.success}, res.fun={res.fun:.6f}")
    print(f"  Sharpe weights: max={np.max(w):.4f}, min={np.min(w):.4f}, zeros={np.sum(w < 1e-8)}")
    return w

def run_walkforward_papertrade(closes_dkk: pd.DataFrame):
    closes_dkk = closes_dkk.sort_index()
    start_dt = pd.Timestamp(INCEPTION_DATE)
    rets = closes_dkk.pct_change(fill_method=None)
    print(f"DEBUG rets shape: {rets.shape}")
    all_dates = rets.index
    
    print(f"\n=== WALKFORWARD DEBUG ===")
    print(f"Total dates in dataset: {len(all_dates)}")
    print(f"Date range: {all_dates[0]} to {all_dates[-1]}")
    print(f"INCEPTION_DATE: {start_dt}")
    print(f"Dates >= INCEPTION_DATE: {(all_dates >= start_dt).sum()}")
    
    month_ends = set(month_end_trade_dates(all_dates))
    print(f"Month-end dates in dataset: {len(month_ends)}")
    print(f"Month-ends >= INCEPTION_DATE: {sum(1 for m in month_ends if m >= start_dt)}")
    if month_ends:
        print(f"Last 5 month-ends: {sorted(month_ends)[-5:]}")
    print(f"===\n")
    
    equity = START_CAPITAL_DKK
    eq_series = []
    eq_index = []
    
    w_current = None
    cols_current = None
    holdings = {}
    
    weights_rows = []
    min_obs = max(60, int(0.7 * LOOKBACK))
    fee = TURNOVER_COST_BPS / 10000.0
    started = False
    inception_allocated = False
    
    for i, dt in enumerate(all_dates):
        # Mark strategy start
        if not started and dt >= start_dt:
            print(f"Strategy starts on {dt}")
            started = True
        if not started:
            continue
        
        # ✅ Option B inception allocation: compute and apply on same day (t+0)
        if started and not inception_allocated and (w_current is None):
            print(f"\nOption B Inception Allocation on {dt.date()}")
            loc = all_dates.get_loc(dt)
            inception_allocated = True
            
            if loc >= LOOKBACK:
                # Lookback window ends at t-1 (yesterday)
                window_rets_full = rets.iloc[loc - LOOKBACK:loc]
                valid_cols = window_rets_full.count()
                cols = valid_cols[valid_cols >= min_obs].index.tolist()
                
                print(f"  Eligible tickers: {len(cols)}")
                
                if len(cols) >= 10:
                    wdw = window_rets_full[cols].dropna(axis=0, how="any")
                    print(f"  Clean window rows: {len(wdw)}")
                    
                    if len(wdw) >= min_obs:
                        mu_ann, cov_ann = estimate_mu_cov_ann(wdw)
                        w_new = max_sharpe_weights(mu_ann, cov_ann, max_w=MAX_W)
                        
                        # First allocation: turnover = 1.0
                        turnover = 1.0
                        turnover_cost = turnover * fee
                        equity *= (1.0 - turnover_cost)
                        
                        w_current = w_new
                        cols_current = cols
                        holdings = {tkr: equity * float(wt) for tkr, wt in zip(cols, w_new)}
                        
                        for tkr, wt in zip(cols, w_new):
                            if wt > 0.001:
                                weights_rows.append({"date": dt.date(), "ticker": tkr, "target_weight": float(wt)})
                        
                        print(f"  Turnover: {turnover:.4f}, Cost: {turnover_cost * 100:.4f}%")
                        print(f"  Portfolio active with {len(cols)} stocks")
        
        # ✅ Rebalance on month-end dates: compute and apply same day (t+0)
        is_month_end = dt in month_ends
        
        if is_month_end:
            print(f"\nRebalance decision on {dt.date()} (month-end)")
            loc = all_dates.get_loc(dt)
            
            if loc >= LOOKBACK:
                # Window ends at t-1 (yesterday)
                window_rets_full = rets.iloc[loc - LOOKBACK:loc]
                valid_cols = window_rets_full.count()
                cols = valid_cols[valid_cols >= min_obs].index.tolist()
                
                print(f"  Eligible tickers: {len(cols)}")
                
                if len(cols) >= 10:
                    wdw = window_rets_full[cols].dropna(axis=0, how="any")
                    print(f"  Clean window rows: {len(wdw)}")
                    
                    if len(wdw) >= min_obs:
                        mu_ann, cov_ann = estimate_mu_cov_ann(wdw)
                        w_new = max_sharpe_weights(mu_ann, cov_ann, max_w=MAX_W)
                        
                        # Compute turnover and apply immediately (t+0)
                        if w_current is None:
                            turnover = 1.0
                        else:
                            old = pd.Series(w_current, index=cols_current).reindex(cols).fillna(0.0).values
                            turnover = float(np.sum(np.abs(w_new - old)))
                        
                        turnover_cost = turnover * fee
                        equity *= (1.0 - turnover_cost)
                        
                        w_current = w_new
                        cols_current = cols
                        holdings = {tkr: equity * float(wt) for tkr, wt in zip(cols, w_new)}
                        
                        for tkr, wt in zip(cols, w_new):
                            if wt > 0.001:
                                weights_rows.append({"date": dt.date(), "ticker": tkr, "target_weight": float(wt)})
                        
                        print(f"  Turnover: {turnover:.4f}, Cost: {turnover_cost * 100:.4f}%")
                        print(f"  Portfolio rebalanced to {len(cols)} stocks")
        
        # Apply daily returns if portfolio is active
        if w_current is None or cols_current is None or not holdings:
            eq_series.append(equity)
            eq_index.append(dt)
            continue
        
        r_today = rets.loc[dt].reindex(cols_current).fillna(0.0).values
        total_pnl = 0.0
        for j, tkr in enumerate(cols_current):
            if tkr in holdings:
                pnl = holdings[tkr] * r_today[j]
                holdings[tkr] += pnl
                total_pnl += pnl
        
        equity += total_pnl
        eq_series.append(equity)
        eq_index.append(dt)
    
    eq = pd.Series(eq_series, index=pd.DatetimeIndex(eq_index), name="total_value")
    w_hist = pd.DataFrame(weights_rows)
    return eq, w_hist

def main():
    ensure_outputs()
    print("Downloading prices (USD) + FX (USD/DKK)...")
    closes_usd = robust_download_close(TICKERS, start=DATA_START_DATE, period=None, interval="1d")
    print(f"Closes USD shape: {closes_usd.shape}")
    print(f"Closes USD date range: {closes_usd.index[0]} to {closes_usd.index[-1]}")
    
    fx = get_usd_to_dkk_series()
    print(f"FX date range: {fx.index[0]} to {fx.index[-1]}")
    
    # Remove timezone from both BEFORE any operations
    if closes_usd.index.tz is not None:
        closes_usd.index = closes_usd.index.tz_localize(None)
    if fx.index.tz is not None:
        fx.index = fx.index.tz_localize(None)
    
    # Now reindex and forward fill
    fx = fx.reindex(closes_usd.index).ffill()
    
    closes_dkk = closes_usd.values * fx.values.reshape(-1, 1)
    closes_dkk = pd.DataFrame(closes_dkk, index=closes_usd.index, columns=closes_usd.columns)
    
    print(f"Closes DKK shape: {closes_dkk.shape}")
    print(f"Closes DKK date range: {closes_dkk.index[0]} to {closes_dkk.index[-1]}")
    print(f"closes_dkk NaN count: {closes_dkk.isna().sum().sum()}")
    
    print(f"\nRunning walk-forward backtest from {INCEPTION_DATE}...\n")
    eq, w_hist = run_walkforward_papertrade(closes_dkk)
    
    df_eq = eq.dropna().to_frame().reset_index()
    df_eq.columns = ["date", "total_value"]
    df_eq["date"] = pd.to_datetime(df_eq["date"]).dt.date
    df_eq.to_csv(OUT_VALUE_DAILY, index=False)
    print(f"\nWrote: {OUT_VALUE_DAILY}")
    
    if not w_hist.empty:
        w_hist.to_csv(OUT_WEIGHTS_REB, index=False)
        print(f"Wrote: {OUT_WEIGHTS_REB}")
        last_date = pd.to_datetime(w_hist["date"]).max()
        latest = w_hist[pd.to_datetime(w_hist["date"]) == last_date].copy()
        latest = latest.rename(columns={"target_weight": "weight"})[["ticker", "weight"]]
        latest = latest[latest["weight"] > 0.0001].copy()
        latest = latest.sort_values("weight", ascending=False)
        latest.to_csv(OUT_WEIGHTS_LATEST, index=False)
        print(f"Wrote: {OUT_WEIGHTS_LATEST}")
        print(f"\nLatest rebalance date: {last_date.date()}")
        final_value = df_eq['total_value'].iloc[-1] if len(df_eq) > 0 else equity
        print(f"Final portfolio value: {final_value:,.2f} DKK")
        print(f"Initial capital: {START_CAPITAL_DKK:,.2f} DKK")
        print(f"Return: {((final_value / START_CAPITAL_DKK) - 1) * 100:.2f}%")
    else:
        print("WARNING: No rebalance weights produced yet (not enough data / dates).")
    print("\nDone.")

if __name__ == "__main__":
    main()
