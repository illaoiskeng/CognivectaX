import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide")
st.title("CognivectaX – Portfolio Dashboard")
st_autorefresh(interval=5000, key="cognivectax_refresh_5s")

# ===== CONFIG =====
DATA_WEIGHTS = "data/weights_latest.csv"
DATA_EQUITY = "data/papertrade/portfolio_value_daily.csv"
START_CAPITAL_DKK = 100_000
INCEPTION_DATE = "2026-01-01"

# ===== LOAD DATA =====
@st.cache_data(ttl=5)
def load_weights(path: str) -> pd.DataFrame:
    """Load the latest portfolio weights (only selected stocks with meaningful weight)"""
    df = pd.read_csv(path)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["weight"] = df["weight"].astype(float)
    # Filter to only stocks with weight > 0.001 (ignore numerical noise)
    df = df[df["weight"] > 0.001].copy()
    # Normalize weights to sum to 1.0
    s = float(df["weight"].sum())
    if s > 0:
        df["weight"] = df["weight"] / s
    return df.sort_values("weight", ascending=False)

@st.cache_data(ttl=5)
def load_equity(path: str) -> pd.Series:
    """Load the daily equity curve"""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    eq = pd.Series(df["total_value"].values, index=df["date"], name="equity")
    eq = eq[eq.index >= pd.Timestamp(INCEPTION_DATE)]
    return eq

# Load data
try:
    weights = load_weights(DATA_WEIGHTS)
except Exception as e:
    st.error(f"Could not load weights: {e}")
    st.stop()

try:
    equity_dkk = load_equity(DATA_EQUITY)
except Exception as e:
    st.error(f"Could not load equity: {e}")
    st.stop()

if equity_dkk.empty:
    st.warning("No equity data yet. Run papertrader first.")
    st.stop()

# ===== SIDEBAR =====
st.sidebar.header("Visning")
view = st.sidebar.selectbox(
    "Graf-interval",
    ["1 dag", "1 uge", "1 måned", "3 måneder", "6 måneder", "12 måneder", "Historisk"],
    index=2
)
st.sidebar.caption("Horisont påvirker kun grafen.")

# ===== CALCULATE METRICS =====
def get_return(trading_days: int) -> float:
    """Calculate return over N trading days"""
    if len(equity_dkk) <= trading_days:
        return np.nan
    return float(equity_dkk.iloc[-1] / equity_dkk.iloc[-trading_days] - 1.0)

current_value = float(equity_dkk.iloc[-1])
inception_return = (current_value / START_CAPITAL_DKK - 1.0)

# ===== METRICS ROW =====
st.subheader("Portfolio Performance")
col0, col1, col2, col3, col4, col5, col6 = st.columns(7)

col0.metric("Portfolio Value (DKK)", f"{current_value:,.0f} kr")
col1.metric("1D", f"{get_return(1)*100:.2f}%")
col2.metric("1W", f"{get_return(5)*100:.2f}%")
col3.metric("1M", f"{get_return(21)*100:.2f}%")
col4.metric("3M", f"{get_return(63)*100:.2f}%")
col5.metric("6M", f"{get_return(126)*100:.2f}%")
col6.metric("Since Inception", f"{inception_return*100:.2f}%")

st.caption(f"Last updated: {pd.Timestamp.now(tz='Europe/Copenhagen').strftime('%Y-%m-%d %H:%M:%S')}")

# ===== EQUITY CURVE =====
c1, c2 = st.columns([2, 1])

with c1:
    days_map = {
        "1 uge": 7,
        "1 måned": 31,
        "3 måneder": 93,
        "6 måneder": 186,
        "12 måneder": 366,
    }

    eq_plot = equity_dkk.copy()
    
    if view != "Historisk" and view in days_map:
        cutoff = eq_plot.index.max() - pd.Timedelta(days=days_map[view])
        eq_plot = eq_plot[eq_plot.index >= cutoff]

    eq_df = eq_plot.to_frame("CognivectaX").reset_index()
    eq_df = eq_df.rename(columns={eq_df.columns[0]: "Date"})

    fig = px.line(
        eq_df,
        x="Date",
        y="CognivectaX",
        title="Equity Curve (DKK)",
        labels={"CognivectaX": "Portfolio Value"}
    )
    fig.update_traces(
        hovertemplate="Dato: %{x}<br>Værdi: %{y:,.0f} kr<extra></extra>",
        line=dict(color="#1f77b4", width=2)
    )
    fig.update_yaxes(tickformat=",.0f", title="DKK")
    fig.update_xaxes(title="Date")
    st.plotly_chart(fig, use_container_width=True)

# ===== PORTFOLIO WEIGHTS =====
with c2:
    st.subheader("Current Allocation")
    
    if len(weights) > 0:
        # Pie chart of current weights
        fig_pie = px.pie(
            weights,
            names="ticker",
            values="weight",
            title=f"Portfolio Weights ({len(weights)} stocks)"
        )
        fig_pie.update_traces(
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Weight: %{value:.2%}<extra></extra>"
        )
        fig_pie.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No holdings yet")

# ===== HOLDINGS TABLE =====
st.subheader("Holdings")

if len(weights) > 0:
    display_weights = weights.copy()
    display_weights["Weight %"] = (display_weights["weight"] * 100).round(2)
    display_weights = display_weights[["ticker", "Weight %"]].reset_index(drop=True)
    display_weights.index = display_weights.index + 1
    display_weights = display_weights.rename(columns={"ticker": "Ticker"})
    
    st.dataframe(display_weights, use_container_width=True)
    
    st.caption(f"Total holdings: {len(weights)} stocks")
    st.caption(f"Total weight: {weights['weight'].sum()*100:.2f}%")
else:
    st.info("No holdings")

# ===== DEBUG INFO =====
with st.expander("Debug Info"):
    st.write(f"**Equity curve length:** {len(equity_dkk)} days")
    st.write(f"**Date range:** {equity_dkk.index[0]} to {equity_dkk.index[-1]}")
    st.write(f"**Number of holdings:** {len(weights)}")
    st.write(f"**Weights sum:** {weights['weight'].sum():.4f}")
