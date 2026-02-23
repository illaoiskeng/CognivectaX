import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import timedelta, datetime, time

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

def is_market_hours(timestamp: pd.Timestamp) -> bool:
    """Check if timestamp is within US market hours (9:30 AM - 4:00 PM EST)"""
    try:
        # Handle both timezone-aware and timezone-naive timestamps
        if timestamp.tz is None:
            ts_est = timestamp.tz_localize('UTC').tz_convert('US/Eastern')
        else:
            ts_est = timestamp.tz_convert('US/Eastern')
        
        time_only = ts_est.time()
        weekday = ts_est.weekday()
        
        # Weekday 0-4 = Monday-Friday
        if weekday > 4:
            return False
        
        # 9:30 AM to 4:00 PM
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        return market_open <= time_only < market_close
    except Exception:
        return False

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

def generate_intraday_data(daily_data: pd.Series, interval: str) -> pd.Series:
    """Generate intraday data from daily data by interpolating"""
    if len(daily_data) == 0:
        return daily_data
    
    # Create time range for the day (9:30 AM - 4:00 PM EST)
    last_date = daily_data.index[-1]
    start_time = last_date.replace(hour=9, minute=30, second=0, microsecond=0)
    end_time = last_date.replace(hour=16, minute=0, second=0, microsecond=0)
    
    # Create range based on interval
    intraday_index = pd.date_range(start=start_time, end=end_time, freq=interval)
    
    # For now, use the daily value for all intraday points (linear interpolation)
    intraday_data = pd.Series(
        np.full(len(intraday_index), daily_data.iloc[-1]),
        index=intraday_index
    )
    
    return intraday_data

def resample_data(data: pd.Series, interval: str) -> pd.Series:
    """Resample data to specified interval, with fallback"""
    try:
        resampled = data.resample(interval).last()
        # If resampling resulted in empty data, return original
        if len(resampled) == 0:
            return data
        return resampled
    except Exception:
        # If resample fails, return original data
        return data

def calculate_return_from_now(equity_value: float, current_value: float) -> float:
    """Calculate return percentage from a point to now"""
    if current_value == 0 or equity_value == 0:
        return 0
    return (current_value / equity_value - 1.0) * 100

def filter_market_hours(data: pd.Series, period: str) -> pd.Series:
    """Filter data to only include US market hours (only for intraday periods)"""
    # Only apply market hours filter for intraday periods (1D, 1U)
    if period not in ["1D", "1U"]:
        return data
    
    # Apply market hours filter
    market_data = data[data.index.map(is_market_hours)]
    # If filtering resulted in empty data, return original
    if len(market_data) == 0:
        return data
    return market_data

def generate_tick_positions(data_df: pd.DataFrame, period: str) -> tuple:
    """Generate appropriate tick positions and format based on period"""
    dates = data_df["Date"].values
    
    if period == "1D":
        # Show every hour (hourly ticks)
        tick_format = "%H:%M"
        tick_positions = pd.to_datetime(dates).floor('H').unique()
        tick_positions = sorted(tick_positions)
        # Fallback: if no hourly data, show all points
        if len(tick_positions) == 0:
            tick_positions = pd.to_datetime(dates).unique()
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
        tick_positions = [t for i, t in enumerate(tick_positions) if i % 5 == 0]
        if len(tick_positions) > 0 and tick_positions[-1] != pd.to_datetime(dates[-1]).floor('D'):
            tick_positions.append(pd.to_datetime(dates[-1]).floor('D'))
        return tick_positions, tick_format
    
    elif period == "3M":
        # Show every 10 days
        tick_format = "%d. %b"
        tick_positions = pd.to_datetime(dates).floor('D').unique()
        tick_positions = sorted(tick_positions)
        tick_positions = [t for i, t in enumerate(tick_positions) if i % 10 == 0]
        if len(tick_positions) > 0 and tick_positions[-1] != pd.to_datetime(dates[-1]).floor('D'):
            tick_positions.append(pd.to_datetime(dates[-1]).floor('D'))
        return tick_positions, tick_format
    
    elif period == "6M":
        # Show every 15 days
        tick_format = "%d. %b"
        tick_positions = pd.to_datetime(dates).floor('D').unique()
        tick_positions = sorted(tick_positions)
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
current_time = pd.Timestamp.now()
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

st.caption(f"Seneste opdatering: {pd.Timestamp.now().strftime('%d. %b. %Y – %H:%M:%S')}")

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

# Get current time rounded to nearest hour
rounded_now = current_time.replace(minute=0, second=0, microsecond=0)

# Prepare data based on selected time period
eq_plot = equity_dkk.copy()

if current_config["days"] is not None:
    cutoff = rounded_now - timedelta(days=current_config["days"])
    eq_plot = eq_plot[eq_plot.index >= cutoff]

# Special handling for 1D: generate intraday data
if current_period == "1D" and len(eq_plot) > 0:
    # Generate intraday data points for the last day
    eq_plot_intraday = generate_intraday_data(eq_plot, selected_interval)
    eq_plot_filtered = eq_plot_intraday
else:
    # Filter to market hours only (only for intraday periods)
    eq_plot_filtered = filter_market_hours(eq_plot, current_period)
    # Resample data based on selected interval (with fallback)
    eq_plot_filtered = resample_data(eq_plot_filtered, selected_interval)

# Remove NaN values
eq_plot_filtered = eq_plot_filtered.dropna()

eq_df = eq_plot_filtered.to_frame("CognivectaX").reset_index()
eq_df = eq_df.rename(columns={eq_df.columns[0]: "Date"})

# Calculate return from EACH point to NOW (current_value)
if len(eq_df) > 0:
    eq_df["Change %"] = eq_df["CognivectaX"].apply(
        lambda x: calculate_return_from_now(x, current_value)
    )
    # Format change percentage as string for hover
    eq_df["Change Text"] = eq_df["Change %"].apply(lambda x: f"{x:+.2f}")

# Check if we have data
if len(eq_df) == 0:
    st.warning(f"No data available for period {current_period}.")
else:
    # Calculate y-axis range with padding
    min_value = eq_df["CognivectaX"].min()
    max_value = eq_df["CognivectaX"].max()
    value_range = max_value - min_value
    
    if value_range == 0:
        value_range = 1
    
    y_max = max_value + (value_range * 0.15)
    y_min = min_value - (value_range * 0.05)
    
    # Generate tick positions
    tick_positions, tick_format = generate_tick_positions(eq_df, current_period)
    
    # Create chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=eq_df["Date"],
        y=eq_df["CognivectaX"],
        fill='tozeroy',
        name='CognivectaX',
        mode='lines',
        line=dict(color='#1f77b4', width=3),
        fillcolor='rgba(31, 119, 180, 0.15)',
        hovertemplate="<b>%{x|%d. %b. %Y %H:%M}</b><br>Værdi: %{y:,.0f} kr<br>Change till now: %{text}%<extra></extra>",
        text=eq_df["Change Text"]
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

ret_cols = st.columns(7)

for col, (label, ret) in zip(ret_cols, returns_data):
    with col:
        is_active = st.session_state.selected_time_period == label
        
        if np.isnan(ret):
            display_ret = "N/A"
            color = "#999999"
        else:
            display_ret = f"{get_return_sign(ret)}{ret*100:.2f}%"
            color = get_return_color(ret)
        
        # Create button with label and return value
        button_label = f"{label}\n{display_ret}"
        
        if st.button(
            button_label,
            key=f"ret_btn_{label}",
            use_container_width=True,
            help=f"Click to view {label} period"
        ):
            st.session_state.selected_time_period = label
            st.rerun()

# ===== MAIN CONTENT AREA =====
st.markdown("---")

col_main, col_side = st.columns([2, 1])

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
        st.write(f"**Equity curve længde:** {len(equity_dkk)}")
        st.write(f"**Dato interval:** {equity_dkk.index[0]} til {equity_dkk.index[-1]}")
        st.write(f"**Valgt periode:** {current_period}")
        st.write(f"**Valgt interval:** {selected_interval}")
    
    with col_debug2:
        st.write(f"**Antal beholdinger:** {len(weights)}")
        st.write(f"**Data points på chart:** {len(eq_df)}")
        st.write(f"**Aktuel værdi (now):** {current_value:,.0f} kr")
        st.write(f"**Tick positions:** {len(tick_positions)}")
