import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

# ---------------- CONFIG ----------------
st.set_page_config(layout="wide")
st.title("CognivectaX – Portfolio Dashboard")

# Refresh every 5 seconds (real-world cadence will be ~5s + runtime)
st_autorefresh(interval=5000, key="cognivectax_refresh_5s")
import time
t0 = time.time()

DATA = "data/weights_latest.csv"
START_CAPITAL_DKK = 100_000
INCEPTION_DATE = "2026-01-01"

st.sidebar.header("Visning")
view = st.sidebar.selectbox(
    "Graf-interval",
    ["15 min", "1 time", "12 timer", "1 dag", "1 uge", "1 måned", "3 måneder", "6 måneder", "12 måneder", "3 år", "5 år", "Historisk"],
    index=3
)    
st.sidebar.caption("Horisont påvirker kun grafen.")

# ---------------- Load weights (cache hard) ----------------
@st.cache_data(ttl=300)
def load_weights(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["weight"] = df["weight"].astype(float)
    return df

try:
    weights = load_weights(DATA)
except Exception:
    st.warning("No weights file found. Run runner.py first.")
    st.stop()

tickers = weights["ticker"].tolist()
benchmark_ticker = "QQQ"

# ---------------- FX: USD -> DKK (hourly) ----------------
@st.cache_data(ttl=3600)
def get_usd_to_dkk() -> float:
    fx = yf.download("DKK=X", period="5d", interval="1d", auto_adjust=True, progress=False)
    if fx is None or fx.empty:
        return np.nan
    if "Close" in fx.columns:
        return float(fx["Close"].dropna().iloc[-1])
    return float(fx.dropna().iloc[-1].values[-1])

usd_to_dkk = get_usd_to_dkk()
if np.isnan(usd_to_dkk):
    st.warning("Kunne ikke hente USD/DKK (DKK=X). Viser værdier som 'USD≈DKK'.")
    usd_to_dkk = 1.0

# ---------------- DAILY layer (heavy) ----------------
# We keep daily data cached strongly so reruns are fast.
@st.cache_data(ttl=86400)  # 1 day cache; heavy call
def download_daily_prices_usd(tickers: list[str]) -> pd.DataFrame:
    raw = yf.download(
        tickers,
        period="max",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = pd.DataFrame({tickers[0]: raw["Close"]})

    close = close.dropna(how="all").sort_index()
    return close

@st.cache_data(ttl=86400)  # 1 day cache; deterministic calc
def compute_equity_dkk(prices_dkk: pd.DataFrame, weights_df: pd.DataFrame,
                       start_capital: float, inception_date: str):
    rets = prices_dkk.pct_change().fillna(0.0)

    w = weights_df.set_index("ticker")["weight"].reindex(prices_dkk.columns).fillna(0.0)

    port_ret = rets @ w
    port_ret = port_ret[port_ret.index >= inception_date]

    equity = start_capital * (1 + port_ret).cumprod()
    if len(equity) > 0:
        equity = equity / float(equity.iloc[0]) * start_capital

    equity = equity[equity.index >= inception_date]
    return equity, w

prices_usd = download_daily_prices_usd(tickers)
benchmark_usd = download_daily_prices_usd([benchmark_ticker])
benchmark_dkk = benchmark_usd * usd_to_dkk
if prices_usd.empty:
    st.error("Ingen daily prisdata kunne hentes fra Yahoo.")
    st.stop()

prices_dkk = prices_usd * usd_to_dkk

equity_dkk, w = compute_equity_dkk(prices_dkk, weights, START_CAPITAL_DKK, INCEPTION_DATE)
benchmark_ret = benchmark_dkk.pct_change().fillna(0.0)
benchmark_equity = START_CAPITAL_DKK * (1 + benchmark_ret).cumprod()
benchmark_equity = benchmark_equity / benchmark_equity.iloc[0] * START_CAPITAL_DKK
benchmark_equity = benchmark_equity[benchmark_equity.index >= INCEPTION_DATE]
if equity_dkk is None or len(equity_dkk) == 0:
    st.warning("Ingen data endnu efter launch-dato.")
    st.stop()

last_daily_close_dkk = prices_dkk.iloc[-1]

# ---------------- LIVE layer (light, refreshed) ----------------
@st.cache_data(ttl=5)
def download_live_last_prices_dkk(tickers: list[str], usd_to_dkk: float) -> pd.Series:
    # threads=False is often faster/more stable on Streamlit Cloud
    raw = yf.download(
        tickers,
        period="1d",
        interval="1m",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        return pd.Series(dtype=float)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = pd.DataFrame({tickers[0]: raw["Close"]})

    close = close.dropna(how="all")
    if close.empty:
        return pd.Series(dtype=float)

    return close.iloc[-1] * usd_to_dkk

@st.cache_data(ttl=5)
def download_intraday_close_dkk(tickers, usd_to_dkk):
    raw = yf.download(
        tickers,
        period="1d",
        interval="1m",
        auto_adjust=True,
        progress=False,
        threads=False
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = pd.DataFrame({tickers[0]: raw["Close"]})

    close = close.dropna(how="all").sort_index()
    return close * usd_to_dkk

live_last_dkk = download_live_last_prices_dkk(tickers, usd_to_dkk)

# Live portfolio value = last equity * intraday multiplier (vs last daily close)
common = live_last_dkk.index.intersection(last_daily_close_dkk.index)
if len(common) > 0:
    intraday_rets = (live_last_dkk[common] / last_daily_close_dkk[common] - 1.0).fillna(0.0)
    w_common = w.reindex(common).fillna(0.0)
    live_multiplier = 1.0 + float(intraday_rets @ w_common)
    portfolio_value_live = float(equity_dkk.iloc[-1] * live_multiplier)
else:
    portfolio_value_live = float(equity_dkk.iloc[-1])

# ---------------- METRICS ----------------
st.subheader("Portfolio Performance")

col0, col1, col2, col3, col4, col5, col6 = st.columns(7)

def calc_return(days: int) -> float:
    if len(equity_dkk) <= days:
        return np.nan
    return float(equity_dkk.iloc[-1] / equity_dkk.iloc[-days] - 1.0)

col0.metric("Portfolio Value (DKK)", f"{portfolio_value_live:,.0f} kr")
st.caption(f"Live opdateret: {pd.Timestamp.now(tz='Europe/Copenhagen').strftime('%H:%M:%S')}")
col1.metric("1D", f"{calc_return(1)*100:.2f}%")
col2.metric("1W", f"{calc_return(5)*100:.2f}%")
col3.metric("1M", f"{calc_return(21)*100:.2f}%")
col4.metric("3M", f"{calc_return(63)*100:.2f}%")
col5.metric("6M", f"{calc_return(126)*100:.2f}%")
col6.metric("Since inception", f"{(float(equity_dkk.iloc[-1]) / START_CAPITAL_DKK - 1)*100:.2f}%")

# ---------------- CHARTS ----------------
c1, c2 = st.columns([2, 1])

with c1:
    # Mapping til hvor meget vi viser på grafen
    days_map = {
        "1 uge": 7,
        "1 måned": 31,
        "3 måneder": 93,
        "6 måneder": 186,
        "12 måneder": 366,
        "3 år": 1096,
        "5 år": 1826,
    }

    # --- KORTE intervaller: intraday (opdaterer ~som value) ---
    if view in ["15 min", "1 time", "12 timer", "1 dag"]:
        intraday_dkk = download_intraday_close_dkk(tickers, usd_to_dkk)

        if intraday_dkk.empty:
            st.warning("Ingen intraday-data lige nu.")
        else:
            # Porteføljeværdi (DKK) pr minut baseret på vægte
            intraday_port = intraday_dkk.reindex(columns=w.index).fillna(method="ffill").fillna(0.0) @ w

            # Skær til valgt interval
            if view == "15 min":
                cutoff = intraday_port.index.max() - pd.Timedelta(minutes=15)
            elif view == "1 time":
                cutoff = intraday_port.index.max() - pd.Timedelta(hours=1)
            elif view == "12 timer":
                cutoff = intraday_port.index.max() - pd.Timedelta(hours=12)
            else:  # "1 dag"
                cutoff = intraday_port.index.max() - pd.Timedelta(days=1)

            intraday_port = intraday_port[intraday_port.index >= cutoff]

            eq_df = intraday_port.reset_index()
            eq_df.columns = ["Date", "CognivectaX"]

            fig = px.line(eq_df, x="Date", y="CognivectaX", title="Equity Curve (DKK) – Live")
            fig.update_traces(hovertemplate="Tid: %{x}<br>Værdi: %{y:,.0f} kr")
            fig.update_yaxes(tickformat=",.0f", title="DKK")
            st.plotly_chart(fig, use_container_width=True)

    # --- LANGE intervaller: daily equity (din eksisterende) ---
    else:
        eq_plot = equity_dkk

        if view != "Historisk":
            cutoff = eq_plot.index.max() - pd.Timedelta(days=days_map[view])
            eq_plot = eq_plot[eq_plot.index >= cutoff]

eq_df = eq_plot.reset_index()
eq_df.columns = ["Date", "CognivectaX"]

benchmark_plot = benchmark_equity.reindex(eq_plot.index)

eq_df["Benchmark"] = benchmark_plot.values

fig = px.line(
    eq_df,
    x="Date",
    y=["CognivectaX", "Benchmark"],
    title="Equity Curve (DKK)"
)

fig.update_traces(hovertemplate="Dato: %{x}<br>Værdi: %{y:,.0f} kr")
fig.update_yaxes(tickformat=",.0f", title="DKK")

st.plotly_chart(fig, use_container_width=True)

with c2:
    fig2 = px.pie(
        weights,
        names="ticker",
        values="weight",
        title="Weight Allocation"
    )
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(f"Run time: {time.time() - t0:.2f}s")
