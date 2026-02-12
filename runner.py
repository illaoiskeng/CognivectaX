import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px

st.set_page_config(layout="wide")
st.title("CognivectaX – Portfolio Dashboard")

DATA = "data/weights_latest.csv"

if not st.session_state.get("loaded"):
    st.session_state.loaded = True

if not st.session_state.get("weights"):

    try:
        weights = pd.read_csv(DATA)
    except:
        st.warning("No weights file found. Run runner.py first.")
        st.stop()

    st.session_state.weights = weights

weights = st.session_state.weights

tickers = weights["ticker"].tolist()

prices = yf.download(tickers, start="2020-01-01", auto_adjust=True, progress=False)["Close"]
rets = prices.pct_change().dropna()

w = weights.set_index("ticker")["weight"]
port_ret = rets @ w
equity = (1+port_ret).cumprod()

# Metrics
st.subheader("Portfolio Performance")

col1,col2,col3,col4,col5,col6 = st.columns(6)

def ret(n):
    if len(equity) <= n:
        return np.nan
    return equity.iloc[-1]/equity.iloc[-n]-1

col1.metric("1D", f"{ret(1)*100:.2f}%")
col2.metric("1W", f"{ret(5)*100:.2f}%")
col3.metric("1M", f"{ret(21)*100:.2f}%")
col4.metric("3M", f"{ret(63)*100:.2f}%")
col5.metric("6M", f"{ret(126)*100:.2f}%")
col6.metric("Total", f"{(equity.iloc[-1]-1)*100:.2f}%")

# Charts
c1,c2 = st.columns([2,1])

with c1:
    st.plotly_chart(px.line(equity, title="Equity Curve"), use_container_width=True)

with c2:
    st.plotly_chart(px.pie(weights, names="ticker", values="weight", title="Weight Allocation"),
                    use_container_width=True)

# Sortable table
st.subheader("Holdings")
st.dataframe(weights.sort_values("weight", ascending=False), use_container_width=True)
