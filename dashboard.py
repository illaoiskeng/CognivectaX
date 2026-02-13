import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px

st.set_page_config(layout="wide")
st.title("CognivectaX – Portfolio Dashboard")

DATA = "data/weights_latest.csv"

# Load weights
try:
    weights = pd.read_csv(DATA)
except:
    st.warning("No weights file found. Run runner.py first.")
    st.stop()

tickers = weights["ticker"].tolist()

# Download price data
prices = yf.download(tickers, start="2020-01-01", auto_adjust=True, progress=False)["Close"]
rets = prices.pct_change().dropna()

# Portfolio returns
w = weights.set_index("ticker")["weight"]
port_ret = rets @ w
equity = (1 + port_ret).cumprod()

# Performance metrics
st.subheader("Portfolio Performance")

col0, col1, col2, col3, col4, col5, col6 = st.columns(7)

col0.metric("Portfolio Value", f"${equity.iloc[-1]:,.0f}")

def calc_return(days):
    if len(equity) <= days:
        return np.nan
    return equity.iloc[-1] / equity.iloc[-days] - 1

col1.metric("1D", f"{calc_return(1)*100:.2f}%")
col2.metric("1W", f"{calc_return(5)*100:.2f}%")
col3.metric("1M", f"{calc_return(21)*100:.2f}%")
col4.metric("3M", f"{calc_return(63)*100:.2f}%")
col5.metric("6M", f"{calc_return(126)*100:.2f}%")
col6.metric("Total", f"{(equity.iloc[-1]-1)*100:.2f}%")

# Charts
c1, c2 = st.columns([2, 1])

with c1:
    fig = px.line(equity, title="Equity Curve")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig2 = px.pie(weights, names="ticker", values="weight", title="Weight Allocation")
    st.plotly_chart(fig2, use_container_width=True)

# Sortable table
st.subheader("Holdings")
weights_display = weights.copy()
weights_display["weight"] = (weights_display["weight"] * 100).round(2).astype(str) + " %"

st.dataframe(
    weights_display.sort_values("weight", ascending=False),
    use_container_width=True
)
