import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import timedelta, datetime

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
    df = df[df["weight"] > 0.001].copy()
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

def resample_data(data: pd.Series, interval: str) -> pd.Series:
    """Resample data to specified interval"""
    return data.resample(interval).last()

def calculate_return_at_point(equity_series: pd.Series, start_value: float) -> pd.Series:
    """Calculate return percentage at each point"""
    return (equity_series / start_value - 1.0) * 100

def generate_tick_positions(data_df: pd.DataFrame, period: str) -> tuple:
    """Generate appropriate tick positions and format based on period"""
    dates = data_df["Date"].values
    
    if period == "1D":
        # Show every hour (hourly ticks)
        tick_format = "%H:%M"
        # Get unique hours
        tick_positions = pd.to_datetime(dates).floor('H').unique()
        tick_positions = sorted(tick_positions)
        return tick_positions, tick_format
    
    elif period == "1U":
        # Show every day
        tick_format = "%d. %b"
        tick_positions = pd.to_datetime(dates).floor('D').unique()
        tick_positions = sorted(tick_positions)
        return tick_positions, tick_format
    
    elif period == "1M":
        # Show every 5 days
        tick_format = "%d. %b"
        tick_positions = pd.to_datetime(dates).floor('D').unique()
        tick_positions = sorted(tick_positions)
        # Filter to every 5 days
        tick_positions = [t for i, t in enumerate(tick_positions) if i % 5 == 0]
        if len(tick_positions) > 0 and tick_positions[-1] != pd.to_datetime(dates[-1]).floor('D'):
            tick_positions.append(pd.to_datetime(dates[-1]).floor('D'))
        return tick_positions, tick_format
    
    elif period == "3M":
        # Show every 10 days
        tick_format = "%d. %b"
        tick_positions = pd.to_datetime(dates).floor('D').unique()
        tick_positions = sorted(tick_positions)
        # Filter to every 10 days
        tick_positions = [t for i, t in enumerate(tick_positions) if i % 10 == 0]
        if len(tick_positions) > 0 and tick_positions[-1] != pd.to_datetime(dates[-1]).floor('D'):
            tick_positions.append(pd.to_datetime(dates[-1]).floor('D'))
        return tick_positions, tick_format
    
    elif period == "6M":
        # Show every 15 days
        tick_format = "%d. %b"
        tick_positions = pd.to_datetime(dates).floor('D').unique()
        tick_positions = sorted(tick_positions)
        # Filter to every 15 days
        tick_positions = [t for i, t in enumerate(tick_positions) if i % 15 == 0]
        if len(tick_positions) > 0 and tick_positions[-1] != pd.to_datetime(dates[-1]).floor('D'):
            tick_positions.append(pd.to_datetime(dates[-1]).floor('D'))
        return tick_positions, tick_format
    
    else:  # 1 ÅR or max
        # Show every month
        tick_format = "%b %Y"
        tick_positions = pd.to_datetime(dates).to_period('M').unique()
        tick_positions = [p.to_timestamp() for p in tick_positions]
        return sorted(tick_positions), tick_format

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
    .return-box {
        text-align: center;
        padding: 15px;
        border-radius: 8px;
        background-color: #f9f9f9;
        border-left: 4px solid;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 2px solid #e0e0e0;
    }
    .return-box:hover {
        background-color: #f0f7ff;
        border: 2px solid #1f77b4;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(31, 119, 180, 0.2);
    }
    .return-box.active {
        background-color: #e3f2fd;
        border: 2px solid #1f77b4;
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.3);
    }
    .return-label {
        font-size: 10px;
        color: #666;
        margin-bottom: 5px;
        text-transform: uppercase;
        font-weight: bold;
    }
    .return-value {
        font-size: 18px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
col_header_left, col_header_right = st.columns([2, 1])

with col_header_left:
    st.markdown("""
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

st.caption(f"Seneste opdatering: {pd.Timestamp.now(tz='Europe/Copenhagen').strftime('%d. %b. %Y – %H:%M:%S')}")

# Initialize session state for time period
if "selected_time_period" not in st.session_state:
    st.session_state.selected_time_period = "1M"

# Time period configuration
time_periods = {
    "1D": {
        "days": 1,
        "intervals": ["5min", "15min", "30min"],
        "description": "24 hours"
    },
    "1U": {
        "days": 7,
        "intervals": ["15min", "30min", "1H"],
        "description": "7 days"
    },
    "1M": {
        "days": 30,
        "intervals": ["1H", "4H", "1D"],
        "description": "30 days"
    },
    "3M": {
        "days": 90,
        "intervals": ["4H", "1D", "1W"],
        "description": "90 days"
    },
    "6M": {
        "days": 180,
        "intervals": ["4H", "1D", "1W"],
        "description": "180 days"
    },
    "1 ÅR": {
        "days": 365,
        "intervals": ["4H", "1D", "1W"],
        "description": "365 days"
    },
    "max": {
        "days": None,
        "intervals": ["1D", "1W", "1M"],
        "description": "All data"
    },
}

current_period = st.session_state.selected_time_period

# Get current config, default to 1M if not found
if current_period not in time_periods:
    current_period = "1M"
    st.session_state.selected_time_period = "1M"

current_config = time_periods[current_period]

# ===== EQUITY CURVE CHART WITH INTERVAL SELECTOR =====
col_chart_header, col_interval = st.columns([0.85, 0.15])

with col_interval:
    selected_interval = st.selectbox(
        "Interval",
        options=current_config["intervals"],
        key=f"interval_{current_period}",
        label_visibility="collapsed"
    )

# Get current time rounded to nearest hour (using timezone-naive datetime)
current_time = pd.Timestamp.now()
rounded_now = current_time.replace(minute=0, second=0, microsecond=0)

# Prepare data based on selected time period
eq_plot = equity_dkk.copy()

if current_config["days"] is not None:
    # Calculate cutoff: go back N days from the rounded current hour (timezone-naive)
    cutoff = rounded_now - timedelta(days=current_config["days"])
    eq_plot = eq_plot[eq_plot.index >= cutoff]
else:
    # For 'max', use all data
    pass

# Resample data based on selected interval
eq_resampled = resample_data(eq_plot, selected_interval)

# Remove NaN values that might appear from resampling
eq_resampled = eq_resampled.dropna()

eq_df = eq_resampled.to_frame("CognivectaX").reset_index()
eq_df = eq_df.rename(columns={eq_df.columns[0]: "Date"})

# Calculate return at each point for hover info
if len(eq_df) > 0:
    start_value = eq_resampled.iloc[0]
    eq_df["Return %"] = calculate_return_at_point(eq_resampled.values, start_value)
else:
    start_value = 0

# Debug: Check if we have data
if len(eq_df) == 0:
    st.warning(f"No data for period {current_period} with interval {selected_interval}")
else:
    # Calculate y-axis range with padding at the top
    min_value = eq_df["CognivectaX"].min()
    max_value = eq_df["CognivectaX"].max()
    value_range = max_value - min_value
    
    # Add 15% padding above the maximum value
    y_max = max_value + (value_range * 0.15)
    y_min = min_value - (value_range * 0.05)
    
    # Generate tick positions
    tick_positions, tick_format = generate_tick_positions(eq_df, current_period)
    
    # Create professional chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=eq_df["Date"],
        y=eq_df["CognivectaX"],
        fill='tozeroy',
        name='CognivectaX',
        mode='lines',
        line=dict(color='#1f77b4', width=3),
        fillcolor='rgba(31, 119, 180, 0.15)',
        hovertemplate="<b>%{x|%d. %b. %Y %H:%M}</b><br>Værdi: %{y:,.0f} kr<br>Gain: %{customdata:.2f}%<extra></extra>",
        customdata=eq_df["Return %"]
    ))

    fig.update_layout(
        title="",
        xaxis_title="",
        yaxis_title="Portfolio Værdi (DKK)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        margin=dict(l=50, r=50, t=20, b=50),
        xaxis=dict(
            gridcolor='#f0f0f0',
            showgrid=True,
            tickformat=tick_format,
            tickvals=tick_positions,
            ticktext=[t.strftime(tick_format) if isinstance(t, pd.Timestamp) else str(t) for t in tick_positions],
            tickmode='array',
        ),
        yaxis=dict(
            gridcolor='#f0f0f0',
            showgrid=True,
            tickformat=',.0f',
            range=[y_min, y_max]
        )
    )

    st.plotly_chart(fig, use_container_width=True)

# ===== CLICKABLE RETURNS METRICS ROW (Bottom) =====
st.markdown("---")

returns_data = [
    ("1D", ret_1d),
    ("1U", ret_1w),
    ("1M", ret_1m),
    ("3M", ret_3m),
    ("6M", ret_6m),
    ("1 ÅR", ret_12m),
    ("max", inception_return)
]

# Single row - clickable buttons with returns displayed inside
ret_cols = st.columns(7)

for col, (label, ret) in zip(ret_cols, returns_data):
    with col:
        # Check if this button is active
        is_active = st.session_state.selected_time_period == label
        
        if np.isnan(ret):
            display_ret = "N/A"
            color = "#999999"
        else:
            display_ret = f"{get_return_sign(ret)}{ret*100:.2f}%"
            color = get_return_color(ret)
        
        # Determine if button is clicked
        if st.button(
            label,
            key=f"ret_btn_{label}",
            use_container_width=True
        ):
            st.session_state.selected_time_period = label
            st.rerun()
        
        # Apply active styling with return value inside
        active_class = "active" if is_active else ""
        st.markdown(f"""
        <div class="return-box {active_class}" style="border-left-color: {color}; margin-top: -40px;">
            <div class="return-label">{label}</div>
            <div class="return-value" style="color: {color};">{display_ret}</div>
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
        st.write(f"**Valgt periode:** {current_period}")
        st.write(f"**Valgt interval:** {selected_interval}")
        st.write(f"**Nuværende tid (afrundet):** {rounded_now}")
    
    with col_debug2:
        st.write(f"**Antal beholdinger:** {len(weights)}")
        st.write(f"**Vægt sum:** {weights['weight'].sum():.4f}")
        st.write(f"**Data points på chart:** {len(eq_df)}")
        if start_value > 0:
            st.write(f"**Start værdi:** {start_value:,.0f} kr")
        st.write(f"**Tick positions:** {len(tick_positions)} ticks")
