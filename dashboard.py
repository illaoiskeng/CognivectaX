import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

# ---------------- CONFIG ----------------
st.set_page_config(layout="wide")
st.title("CognivectaX – Portfolio Dashboard")

# Auto refresh every 5 seconds
st_autorefresh(interval=5000, key="main_refresh")

DATA = "data/weights_latest.csv"
START_CAPITAL_DKK = 100_000
INCEPTION_DATE = "2026-01-01"
interval = "1d"

# ---------------- Sidebar ----------------
st.sidebar.header("Visning")
period = st.sidebar.selectbox(
    "Horisont",
    ["5d", "1mo", "3mo", "6mo", "1y", "5y", "max"],
    index=6,
)

# ---------------- Load weights ----------------
try:
    weights = pd.read_csv(DATA)
except:
    st.warning("No weights file found. Run runner.py first.")
    st.stop()

weights["ticker"] = weights["ticker"].astype(str).str.upper()
weights["weight"] = weights["weight"].astype(float)
tickers = weights["ticker"].tolist()

# ---------------- FX ----------------
@st.cache_data
def get_usd_to_dkk():
    fx = yf.download("DKK=X", period="5d", interval="1d", auto_adjust=True, progress=False)
    if fx.empty:
        return 1.0
    return float(fx["Close"].dropna().iloc[-1])

usd_to_dkk = get_usd_to_dkk()

# ---------------- Daily Prices (for equity curve) ----------------
@st.cache_data(ttl=1800)
def download_prices(tickers):
    raw = yf.download(
        tickers,
        period="max",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True
    )
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = pd.DataFrame({tickers[0]: raw["Close"]})

    return close.dropna(how="all")
    
@st.cache_data
def compute_equity(prices, weights, start_capital, inception_date):

    rets = prices.pct_change().fillna(0.0)

    w = weights.set_index("ticker")["weight"].reindex(prices.columns).fillna(0.0)

    port_ret = rets @ w
    port_ret = port_ret[port_ret.index >= inception_date]

    equity = start_capital * (1 + port_ret).cumprod()
    equity = equity / equity.iloc[0] * start_capital
    equity = equity[equity.index >= inception_date]

    return equity, w
    
prices = download_prices(tickers) * usd_to_dkk
if prices.empty:
    st.error("No price data.")
    st.stop()

rets = prices.pct_change().fillna(0.0)
w = weights.set_index("ticker")["weight"].reindex(prices.columns).fillna(0.0)

port_ret = rets @ w
port_ret = port_ret[port_ret.index >= INCEPTION_DATE]

equity_dkk = START_CAPITAL_DKK * (1 + port_ret).cumprod()
equity_dkk = equity_dkk / equity_dkk.iloc[0] * START_CAPITAL_DKK
equity_dkk = equity_dkk[equity_dkk.index >= INCEPTION_DATE]

if equity_dkk.empty:
    st.warning("Ingen data endnu efter launch-dato")
    st.stop()

# ---------------- Live Prices (for Portfolio Value only) ----------------
@st.cache_data(ttl=5)
def download_live_last_prices(tickers):
    raw = yf.download(
        tickers,
        period="1d",
        interval="1m",
        auto_adjust=True,
        progress=False,
        threads=True
    )
    if raw.empty:
        return pd.Series(dtype=float)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = pd.DataFrame({tickers[0]: raw["Close"]})

    close = close.dropna(how="all")
    if close.empty:
        return pd.Series(dtype=float)

    return close.iloc[-1] * usd_to_dkk

live_last = download_live_last_prices(tickers)

last_daily = prices.iloc[-1]
common = live_last.index.intersection(last_daily.index)

if len(common) > 0:
    intraday_rets = (live_last[common] / last_daily[common] - 1).fillna(0.0)
    live_multiplier = 1 + float(intraday_rets @ w.reindex(common).fillna(0.0))
    portfolio_value_live = float(equity_dkk.iloc[-1] * live_multiplier)
else:
    portfolio_value_live = float(equity_dkk.iloc[-1])

# ---------------- Metrics ----------------
st.subheader("Portfolio Performance")

col0, col1, col2, col3, col4, col5, col6 = st.columns(7)

def calc_return(days):
    if len(equity_dkk) <= days:
        return np.nan
    return equity_dkk.iloc[-1] / equity_dkk.iloc[-days] - 1

col0.metric("Portfolio Value (DKK)", f"{portfolio_value_live:,.0f} kr")
col1.metric("1D", f"{calc_return(1)*100:.2f}%")
col2.metric("1W", f"{calc_return(5)*100:.2f}%")
col3.metric("1M", f"{calc_return(21)*100:.2f}%")
col4.metric("3M", f"{calc_return(63)*100:.2f}%")
col5.metric("6M", f"{calc_return(126)*100:.2f}%")
col6.metric("Since inception", f"{(equity_dkk.iloc[-1]/START_CAPITAL_DKK - 1)*100:.2f}%")

# ---------------- Charts ----------------
c1, c2 = st.columns([2, 1])

with c1:
    eq_df = equity_dkk.reset_index()
    eq_df.columns = ["Date", "CognivectaX"]
    fig = px.line(eq_df, x="Date", y="CognivectaX", title="Equity Curve (DKK)")
    fig.update_yaxes(tickformat=",.0f")
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
