import os
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

DATA_WEIGHTS_LATEST = "data/weights_latest.csv"
OUT_DIR = "data/papertrade"
OUT_WEIGHTS_REB = os.path.join(OUT_DIR, "weights_rebalance.csv")
OUT_VALUE_DAILY = os.path.join(OUT_DIR, "portfolio_value_daily.csv")
OUT_TEST_RESULTS = os.path.join(OUT_DIR, "test_results.csv")

START_CAPITAL_DKK = 100_000
INCEPTION_DATE = "2026-01-01"

def ensure_outputs():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(OUT_WEIGHTS_REB):
        pd.DataFrame(columns=["date", "ticker", "target_weight"]).to_csv(OUT_WEIGHTS_REB, index=False)

    if not os.path.exists(OUT_VALUE_DAILY):
        pd.DataFrame(columns=["date", "total_value"]).to_csv(OUT_VALUE_DAILY, index=False)
    if not os.path.exists(OUT_TEST_RESULTS):
        pd.DataFrame(columns=["test", "metric", "value"]).to_csv(OUT_TEST_RESULTS, index=False)

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

def max_sharpe_weights(mu_ann: np.ndarray, cov_ann: np.ndarray, max_w: float = 0.08) -> np.ndarray:
    n = len(mu_ann)

    def neg_sharpe(w):
        # rf=0
        port_ret = float(w @ mu_ann)
        port_vol = float(np.sqrt(w @ cov_ann @ w))
        if port_vol <= 0:
            return 1e9
        return -(port_ret / port_vol)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, max_w)] * n
    x0 = np.ones(n) / n
    # sørg for x0 respekterer cap
    if (x0 > max_w).any():
        x0 = np.minimum(x0, max_w)
        x0 = x0 / x0.sum()

    res = minimize(neg_sharpe, x0=x0, bounds=bounds, constraints=cons, method="SLSQP")
    if not res.success:
        raise RuntimeError(f"Optimizer failed: {res.message}")
    w = res.x
    # numerisk cleanup
    w[w < 0] = 0.0
    w = np.minimum(w, max_w)
    w = w / w.sum()
    return w

def estimate_mu_cov_ann(returns_daily: pd.DataFrame):
    mu_daily = returns_daily.mean().values
    cov_daily = returns_daily.cov().values
    mu_ann = mu_daily * 252.0
    cov_ann = cov_daily * 252.0
    return mu_ann, cov_ann

def sharpe_from_daily_returns(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    vol = r.std()
    if vol == 0:
        return np.nan
    return float((r.mean() / vol) * np.sqrt(252.0))

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

# ---------------- TEST 1: Walk-forward out-of-sample Sharpe (MAX-SHARPE) ----------------
MAX_W = 0.08

rets_daily = closes_dkk.pct_change().dropna()

# rebalance-datoer hvor vi kan lave 252d lookback (t-1)
valid_reb = []
for d in rebalance_dates:
    loc = closes_dkk.index.get_loc(d)
    if loc >= 253:  # så t-1 har mindst 252 returns
        valid_reb.append(d)
valid_reb = pd.DatetimeIndex(valid_reb)
strategies = ["max_sharpe", "min_var", "equal_weight"]

wf_returns_map = {s: [] for s in strategies}
w_prev_map = {s: None for s in strategies}
turnovers_map = {s: [] for s in strategies}
wf_returns = []
turnovers = []
w_prev = None
COST_BPS = 10  # 10 bps pr 100% turnover (0.10%)

for i, reb_date in enumerate(valid_reb):
    reb_loc = closes_dkk.index.get_loc(reb_date)
    t_minus_1 = closes_dkk.index[reb_loc - 1]

    # 252d window (baseret på data op til t-1)
    window_prices = closes_dkk.loc[:t_minus_1].tail(253)  # 253 priser -> 252 returns
    window_rets = window_prices.pct_change().dropna()

    mu_ann, cov_ann = estimate_mu_cov_ann(window_rets)
    cols = window_rets.columns.tolist()

    w_ms = pd.Series(max_sharpe_weights(mu_ann, cov_ann, max_w=MAX_W), index=cols)
    w_mv = pd.Series(min_var_weights(cov_ann, max_w=MAX_W), index=cols)
    w_eq = pd.Series(equal_weight(len(cols), max_w=MAX_W), index=cols)

    w_dict = {
    "max_sharpe": w_ms,
    "min_var": w_mv,
    "equal_weight": w_eq,
    }
for s in strategies:
    w_new = w_dict[s]

    # turnover
    if w_prev_map[s] is None:
        to = float(np.abs(w_new).sum())
    else:
        aligned = w_prev_map[s].reindex(cols).fillna(0.0)
        to = float(np.abs(w_new - aligned).sum())

    turnovers_map[s].append(to)

    port_ret = (period_rets @ w_new).astype(float)

    # træk trading cost på første dag i perioden
    cost = to * (COST_BPS / 10000.0)
    if len(port_ret) > 0:
        port_ret.iloc[0] = port_ret.iloc[0] - cost

    wf_returns_map[s].append(port_ret)

    w_prev_map[s] = w_new.copy()

results = {}

for s in strategies:
    r = pd.concat(wf_returns_map[s]).sort_index()
    r = r[~r.index.duplicated(keep="first")]
    results[s] = r
wf = wf[~wf.index.duplicated(keep="first")]

print("\n--- TEST 3: STRATEGY COMPARISON ---")

for s in strategies:
    print(
        s,
        "Sharpe:",
        round(sharpe_from_daily_returns(results[s]), 3),
        "Avg TO:",
        round(float(np.mean(turnovers_map[s])), 3),
    )
results_rows = []

for s in strategies:
    results_rows.append({
        "test": "strategy_comparison",
        "metric": f"{s}_sharpe",
        "value": float(sharpe_from_daily_returns(results[s]))
    })
    results_rows.append({
        "test": "strategy_comparison",
        "metric": f"{s}_avg_turnover",
        "value": float(np.mean(turnovers_map[s]))
    })
]

df_out = pd.DataFrame(results_rows)

# overwrite (så den altid viser seneste run)
df_out.to_csv(OUT_TEST_RESULTS, index=False)
print(f"Wrote: {OUT_TEST_RESULTS}")
    # ---- TEST: first rebalance date weights using data up to t-1 ----
first_reb = rebalance_dates[0]
t_minus_1 = closes_dkk.index[closes_dkk.index.get_loc(first_reb) - 1]

prices_window = closes_dkk.loc[:t_minus_1].tail(252)
rets_window = prices_window.pct_change().dropna()

mu_ann, cov_ann = estimate_mu_cov_ann(rets_window)
w_opt = max_sharpe_weights(mu_ann, cov_ann, max_w=0.08)

w_series = pd.Series(w_opt, index=rets_window.columns).sort_values(ascending=False)
print("\n--- OPT TEST ---")
print("Rebalance date:", first_reb.date())
print("Using data up to:", t_minus_1.date())
print("Sum weights:", float(w_series.sum()))
print("Max weight:", float(w_series.max()))
print("Top 10 weights:\n", w_series.head(10))

    print(f"Universe tickers: {len(tickers)}")
    print(f"Price rows: {len(closes_dkk)}")
    print(f"Rebalance dates: {len(rebalance_dates)}")
    print("First 5 rebalance dates:", list(rebalance_dates[:5].date))

if __name__ == "__main__":
    main()
