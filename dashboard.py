Copilot said: # Redesigned Dashboard for CognivectaX Here's the updated
Redesigned Dashboard for CognivectaX
Here's the updated dashboard.py with a professional financial dashboard design like the images you showed:

Python
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide")
st_autorefresh(interval=5000, key="cognivectax_refresh_5s")

# ===== CONFIG =====
DATA_WEIGHTS = "data/weights_latest.csv"
DATA_EQUITY = "data/papertrade/portfolio_value_daily.csv"
START_CAPITAL_DKK = 100_000
INCEPTION_DATE = "2026-01-01"

# ===== LOAD DATA =====
@st.cache_data(ttl=5, show_spinner=False)
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

@st.cache_data(ttl=5, show_spinner=False)
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

# ===== CALCULATE METRICS =====
def get_return(trading_days: int) -> float:
    """Calculate return over N trading days"""
    if len(equity_dkk) <= trading_days:
        return np.nan
    return float(equity_dkk.iloc[-1] / equity_dkk.iloc[-trading_days] - 1.0)

def get_return_color(ret: float) -> str:
    """Return color based on return value"""
    if np.isnan(ret):
        return "#999999"
    return "#00AA00" if ret >= 0 else "#FF4444"

def get_return_sign(ret: float) -> str:
    """Return sign for positive/negative"""
    if np.isnan(ret):
        return "N/A"
    return "+" if ret >= 0 else ""

current_value = float(equity_dkk.iloc[-1])
inception_return = (current_value / START_CAPITAL_DKK - 1.0)

ret_1d = get_return(1)
ret_1w = get_return(5)
ret_1m = get_return(21)
ret_3m = get_return(63)
ret_6m = get_return(126)
ret_12m = get_return(252)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    .metric-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .header-title {
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .header-subtitle {
        font-size: 14px;
        color: #666;
    }
    .return-positive {
        color: #00AA00;
        font-weight: bold;
    }
    .return-negative {
        color: #FF4444;
        font-weight: bold;
    }
    .time-button {
        padding: 8px 12px;
        margin: 5px;
        border: 1px solid #ddd;
        border-radius: 5px;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
col_header_left, col_header_right = st.columns([2, 1])

with col_header_left:
    st.markdown(f"""
    <div class="metric-box">
        <div class="header-title">🚀 CognivectaX</div>
        <div class="header-subtitle">AI-Powered Portfolio Management</div>
    </div>
    """, unsafe_allow_html=True)

with col_header_right:
    st.markdown(f"""
    <div style="text-align: right; padding: 20px;">
        <div style="font-size: 24px; font-weight: bold; color: #1f77b4;">
            {current_value:,.0f} kr
        </div>
        <div style="font-size: 12px; color: #666;">
            Portfolio Value
        </div>
    </div>
    """, unsafe_allow_html=True)

# ===== TIME PERIOD SELECTOR =====
st.markdown("---")

col_time_left, col_time_right = st.columns([0.3, 0.7])

with col_time_left:
    view = st.selectbox(
        "Tidshorisont:",
        ["1 dag", "1 uge", "1 måned", "3 måneder", "6 måneder", "12 måneder", "Historisk"],
        index=2,
        label_visibility="collapsed"
    )

with col_time_right:
    st.caption(f"Seneste opdatering: {pd.Timestamp.now(tz='Europe/Copenhagen').strftime('%d. %b. %Y – %H:%M:%S')}")

# ===== EQUITY CURVE CHART =====
days_map = {
    "1 uge": 7,
    "1 måned": 31,
    "3 måneder": 93,
    "6 måneder": 186,
    "12 måneder": 252,
}

eq_plot = equity_dkk.copy()

if view != "Historisk" and view in days_map:
    cutoff = eq_plot.index.max() - pd.Timedelta(days=days_map[view])
    eq_plot = eq_plot[eq_plot.index >= cutoff]

eq_df = eq_plot.to_frame("CognivectaX").reset_index()
eq_df = eq_df.rename(columns={eq_df.columns[0]: "Date"})

# Create professional chart
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=eq_df["Date"],
    y=eq_df["CognivectaX"],
    fill='tozeroy',
    name='CognivectaX',
    line=dict(color='#1f77b4', width=2),
    fillcolor='rgba(31, 119, 180, 0.15)',
    hovertemplate="<b>%{x|%d. %b. %Y}</b><br>Værdi: %{y:,.0f} kr<extra></extra>"
))

fig.update_layout(
    title="",
    xaxis_title="Dato",
    yaxis_title="Portfolio Værdi (DKK)",
    hovermode='x unified',
    template='plotly_white',
    height=500,
    margin=dict(l=50, r=50, t=50, b=50),
    xaxis=dict(
        gridcolor='#f0f0f0',
        showgrid=True,
    ),
    yaxis=dict(
        gridcolor='#f0f0f0',
        showgrid=True,
        tickformat=',.0f',
    )
)

st.plotly_chart(fig, use_container_width=True)

# ===== RETURNS METRICS ROW =====
st.markdown("---")

ret_cols = st.columns(7)

returns_data = [
    ("1d", ret_1d),
    ("1u", ret_1w),
    ("1m", ret_1m),
    ("3m", ret_3m),
    ("6m", ret_6m),
    ("1 år", ret_12m),
    ("max", inception_return)
]

for col, (label, ret) in zip(ret_cols, returns_data):
    with col:
        if np.isnan(ret):
            display_ret = "N/A"
            color = "#999999"
        else:
            display_ret = f"{get_return_sign(ret)}{ret*100:.2f}%"
            color = get_return_color(ret)
        
        st.markdown(f"""
        <div style="text-align: center; padding: 15px; border-radius: 8px; background-color: #f9f9f9; border-left: 4px solid {color};">
            <div style="font-size: 10px; color: #666; margin-bottom: 5px; text-transform: uppercase; font-weight: bold;">
                {label}
            </div>
            <div style="font-size: 18px; font-weight: bold; color: {color};">
                {display_ret}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ===== MAIN CONTENT AREA =====
st.markdown("---")

col_main, col_side = st.columns([2, 1])

# ===== LEFT COLUMN: DETAILS =====
with col_main:
    st.subheader("📊 Portfolio Information")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.metric("Startkapital", f"{START_CAPITAL_DKK:,.0f} kr")
    
    with info_col2:
        st.metric("Nuværende Værdi", f"{current_value:,.0f} kr")
    
    with info_col3:
        profit = current_value - START_CAPITAL_DKK
        profit_pct = (profit / START_CAPITAL_DKK) * 100
        st.metric("Samlet Profit", f"{profit:,.0f} kr", f"{profit_pct:.2f}%")

# ===== RIGHT COLUMN: ALLOCATION PIE =====
with col_side:
    st.subheader("📈 Allokeringer")
    
    if len(weights) > 0:
        fig_pie = px.pie(
            weights,
            names="ticker",
            values="weight",
            title=""
        )
        fig_pie.update_traces(
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Vægt: %{value:.2%}<extra></extra>"
        )
        fig_pie.update_layout(
            showlegend=False,
            height=350,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Ingen aktier endnu")

# ===== HOLDINGS TABLE =====
st.markdown("---")
st.subheader("📋 Beholdinger")

if len(weights) > 0:
    display_weights = weights.copy()
    display_weights["Weight %"] = (display_weights["weight"] * 100).round(2)
    display_weights = display_weights[["ticker", "Weight %"]].reset_index(drop=True)
    display_weights.index = display_weights.index + 1
    display_weights = display_weights.rename(columns={"ticker": "Ticker"})
    
    st.dataframe(
        display_weights,
        use_container_width=True,
        height=300
    )
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.caption(f"**Total beholdinger:** {len(weights)} aktier")
    with col_info2:
        st.caption(f"**Total vægt:** {weights['weight'].sum()*100:.2f}%")
else:
    st.info("Ingen beholdinger")

# ===== DEBUG INFO =====
with st.expander("🔧 Debug Info"):
    col_debug1, col_debug2 = st.columns(2)
    
    with col_debug1:
        st.write(f"**Equity curve længde:** {len(equity_dkk)} dage")
        st.write(f"**Dato interval:** {equity_dkk.index[0].date()} til {equity_dkk.index[-1].date()}")
    
    with col_debug2:
        st.write(f"**Antal beholdinger:** {len(weights)}")
        st.write(f"**Vægt sum:** {weights['weight'].sum():.4f}")
