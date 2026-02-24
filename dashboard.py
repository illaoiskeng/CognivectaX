import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import timedelta, datetime, time
import time as time_module
from dashboard_metrics import (
    init_tracking, track_intraday, calculate_and_save_daily_metrics,
    calculate_and_save_monthly_metrics,
    display_daily_metrics, display_monthly_metrics, display_rebalance_history
)
from metrics_tracker import get_intraday_values

st.set_page_config(layout="wide")
st_autorefresh(interval=5000, key="cognivectax_refresh_5s")

# ===== CONFIG =====
DATA_WEIGHTS = "data/weights_latest.csv"
DATA_EQUITY = "data/papertrade/portfolio_value_daily.csv"
DATA_METRICS_DB = "data/portfolio_metrics.db"
START_CAPITAL_DKK = 100_000
INCEPTION_DATE = "2026-02-19"

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

@st.cache_data(ttl=1, show_spinner=False)
def load_intraday_data_from_db(db_path: str, days: int = None) -> pd.DataFrame:
    """Load intraday data from database as DataFrame - cached for 1 second to always get latest data"""
    intraday_df = get_intraday_values(db_path, days=days if days else 365)
    
    if intraday_df is None or len(intraday_df) == 0:
        return pd.DataFrame()
    
    # Convert timestamp to datetime and sort
    intraday_df['timestamp'] = pd.to_datetime(intraday_df['timestamp'])
    intraday_df = intraday_df.sort_values('timestamp')
    
    return intraday_df

def remove_weekends(data: pd.DataFrame) -> pd.DataFrame:
    """Remove Saturday (5) and Sunday (6) data"""
    if len(data) == 0:
        return data
    
    mask = data['timestamp'].dt.weekday < 5  # 0-4 = Mon-Fri
    return data[mask].copy()

def get_trading_days_back(data: pd.DataFrame, num_trading_days: int) -> pd.Timestamp:
    """
    Get the cutoff timestamp for N trading days back from latest data.
    If fewer than N trading days exist, return the earliest available date.
    """
    if len(data) == 0:
        return pd.Timestamp.now()
    
    # Get unique trading days (dates) in the data, sorted newest first
    unique_dates = sorted(data['timestamp'].dt.date.unique(), reverse=True)
    
    # Determine how many days to go back
    days_to_use = min(num_trading_days, len(unique_dates))
    
    # Get the cutoff date (Nth day back, or earliest if fewer days exist)
    cutoff_date = unique_dates[days_to_use - 1]
    cutoff_timestamp = pd.Timestamp(cutoff_date).tz_localize(None)
    
    # Set to market open time (9:30 AM)
    cutoff_timestamp = cutoff_timestamp.replace(hour=0, minute=0, second=0)
    
    return cutoff_timestamp

def resample_intraday_data(data: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Resample intraday data to specified interval"""
    if len(data) == 0:
        return data
    
    try:
        # Set timestamp as index for resampling
        data_indexed = data.set_index('timestamp')
        resampled = data_indexed['portfolio_value'].resample(interval).last()
        
        # If resampling resulted in empty data, return original
        if len(resampled) == 0:
            return data
        
        result = resampled.reset_index()
        result.columns = ['timestamp', 'portfolio_value']
        return result
    except Exception as e:
        print(f"Error resampling: {e}")
        return data

def calculate_return_from_now(equity_value: float, current_value: float) -> float:
    """Calculate return percentage from a point to now"""
    if current_value == 0 or equity_value == 0:
        return 0
    return (current_value / equity_value - 1.0) * 100

def generate_tick_positions(data_df: pd.DataFrame, period: str) -> tuple:
    """Generate appropriate tick positions and format based on period"""
    if len(data_df) == 0:
        return [], ""
    
    dates = data_df["Date"].values
    
    if period == "1D":
        # Show every hour (hourly ticks)
        tick_format = "%H:%M"
        tick_positions = pd.to_datetime(dates).floor('H').unique()
        tick_positions = sorted(tick_positions)
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

# ===== METRICS TRACKING INITIALIZATION =====
init_tracking()

current_value = float(equity_dkk.iloc[-1])

# Track intraday value every 3 minutes
daily_return_pct = ((current_value / START_CAPITAL_DKK) - 1) * 100
track_intraday(current_value, daily_return_pct)

# Calculate daily metrics (will only save once per day at close)
calculate_and_save_daily_metrics(weights['ticker'].tolist(), weights['weight'].tolist(), 
                                 current_value, equity_dkk)

# ===== LOAD INTRADAY DATA FIRST =====
all_intraday_df = load_intraday_data_from_db(DATA_METRICS_DB, days=365)

# Use latest data timestamp, or now if no data
current_time = all_intraday_df['timestamp'].max() if len(all_intraday_df) > 0 else pd.Timestamp.now()

# ===== CALCULATE METRICS =====
def get_return(trading_days: int) -> float:
    """Calculate return over N trading days"""
    if len(equity_dkk) <= trading_days:
        return np.nan
    return float(equity_dkk.iloc[-1] / equity_dkk.iloc[-trading_days] - 1.0)

def get_return_sign(ret: float) -> str:
    """Return sign for positive/negative"""
    if np.isnan(ret):
        return "N/A"
    return "+" if ret >= 0 else ""

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

# Time period configuration - in TRADING DAYS
time_periods = {
    "1D": {
        "trading_days": 1,
        "intervals": ["5min", "15min", "30min"],
        "description": "1 trading day"
    },
    "1U": {
        "trading_days": 5,
        "intervals": ["15min", "30min", "1H"],
        "description": "5 trading days"
    },
    "1M": {
        "trading_days": 20,
        "intervals": ["1H", "4H", "1D"],
        "description": "20 trading days"
    },
    "3M": {
        "trading_days": 60,
        "intervals": ["4H", "1D", "1W"],
        "description": "60 trading days"
    },
    "6M": {
        "trading_days": 120,
        "intervals": ["4H", "1D", "1W"],
        "description": "120 trading days"
    },
    "1 ÅR": {
        "trading_days": 252,
        "intervals": ["4H", "1D", "1W"],
        "description": "252 trading days"
    },
    "max": {
        "trading_days": None,
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

# ===== LOAD AND PROCESS INTRADAY DATA =====
if len(all_intraday_df) == 0:
    st.warning("No intraday data available in database. Please check if tracking is running.")
else:
    # Step 1: Remove weekends
    weekday_data = remove_weekends(all_intraday_df)
    
    # Step 2: Calculate cutoff based on trading days (no market hours filter)
    if current_config["trading_days"] is not None:
        cutoff_time = get_trading_days_back(weekday_data, current_config["trading_days"])
    else:
        cutoff_time = weekday_data['timestamp'].min()
    
    # Step 3: Filter to time period (trading days) - from cutoff to NOW
    period_data = weekday_data[
        (weekday_data['timestamp'] >= cutoff_time) & 
        (weekday_data['timestamp'] <= current_time)
    ].copy()
    
    # Step 4: Resample to selected interval
    eq_plot_resampled = resample_intraday_data(period_data, selected_interval)
    
    # Remove NaN values
    eq_plot_filtered = eq_plot_resampled.dropna()
    
    if len(eq_plot_filtered) > 0:
        eq_df = eq_plot_filtered.copy()
        eq_df = eq_df.rename(columns={'timestamp': 'Date', 'portfolio_value': 'CognivectaX'})
        
        # Calculate return from EACH point to NOW (current_value)
        eq_df["Change %"] = eq_df["CognivectaX"].apply(
            lambda x: calculate_return_from_now(x, current_value)
        )
        # Format change percentage as string for hover
        eq_df["Change Text"] = eq_df["Change %"].apply(lambda x: f"{x:+.2f}")
        
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
    else:
        st.warning(f"No data available for period {current_period} with interval {selected_interval}.")

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
        if np.isnan(ret):
            display_ret = "N/A"
            color = "#999999"
        else:
            display_ret = f"{get_return_sign(ret)}{ret*100:.2f}%"
            if ret >= 0:
                color = "#00AA00"
            else:
                color = "#FF4444"
        
        # Create a container for layering
        container = st.container()
        
        with container:
            # Hidden button (actual clickable element)
            if st.button(" ", key=f"ret_btn_{label}", use_container_width=True, help=f"Select {label}"):
                st.session_state.selected_time_period = label
                st.rerun()
            
            # Styled overlay (sits on top, pointer-events allows clicks to pass through)
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                padding: 15px 10px;
                border-radius: 8px;
                border: 2px solid #e0e0e0;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s ease;
                margin-top: -50px;
                position: relative;
                z-index: 10;
                pointer-events: none;
            " onmouseover="this.style.backgroundColor='#f0f7ff'; this.style.border='2px solid #1f77b4'; this.style.boxShadow='0 4px 8px rgba(31, 119, 180, 0.2)';" onmouseout="this.style.backgroundColor=''; this.style.border='2px solid #e0e0e0'; this.style.boxShadow='none';">
                <div style="font-size: 16px; font-weight: 900; color: #000000; margin-bottom: 8px;">
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
        @st.cache_data(ttl=3600, show_spinner=False)
        def fetch_company_names(tickers):
            """Fetch company names from yfinance"""
            names = {}
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    names[ticker] = stock.info.get('longName', ticker)
                    time_module.sleep(0.2)
                except Exception:
                    names[ticker] = ticker
            return names
        
        tickers_list = weights['ticker'].tolist()
        company_names = fetch_company_names(tickers_list)
        
        pie_data = weights.copy()
        pie_data['company_name'] = pie_data['ticker'].map(company_names)
        
        fig_pie = go.Figure()
        fig_pie.add_trace(go.Pie(
            labels=pie_data['ticker'],
            values=pie_data['weight'],
            hovertemplate="<b>%{customdata}</b><br>Vægt: %{value:.2%}<extra></extra>",
            customdata=pie_data['company_name'].values,
            textinfo="label+percent",
            textposition="auto",
            textfont=dict(size=12)
        ))
        
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

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data_batch(tickers):
    """Fetch stock data with rate limiting and batching"""
    data = {}
    
    batch_size = 5
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        for ticker in batch:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                data[ticker] = {
                    'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                    'pe_ratio': info.get('trailingPE', 'N/A'),
                    'avg_buy_price': info.get('regularMarketPreviousClose', 0)
                }
                time_module.sleep(0.2)
            except Exception as e:
                data[ticker] = {
                    'current_price': 0,
                    'pe_ratio': 'N/A',
                    'avg_buy_price': 0
                }
        
        if i + batch_size < len(tickers):
            time_module.sleep(1)
    
    return data

if len(weights) > 0:
    tickers = weights['ticker'].tolist()
    stock_data = fetch_stock_data_batch(tickers)
    
    display_weights = weights.copy()
    display_weights["Weight %"] = (display_weights["weight"] * 100).round(2)
    
    display_weights["Avg Buy Price"] = display_weights["ticker"].apply(
        lambda x: f"${stock_data[x]['avg_buy_price']:.2f}" if stock_data[x]['avg_buy_price'] > 0 else "N/A"
    )
    
    display_weights["Current Price"] = display_weights["ticker"].apply(
        lambda x: f"${stock_data[x]['current_price']:.2f}" if stock_data[x]['current_price'] > 0 else "N/A"
    )
    
    def calc_gain_loss(row):
        ticker = row['ticker']
        if stock_data[ticker]['avg_buy_price'] > 0 and stock_data[ticker]['current_price'] > 0:
            gain_loss = ((stock_data[ticker]['current_price'] - stock_data[ticker]['avg_buy_price']) / 
                        stock_data[ticker]['avg_buy_price']) * 100
            return f"{gain_loss:+.2f}%"
        return "N/A"
    
    display_weights["Gain/Loss %"] = display_weights.apply(calc_gain_loss, axis=1)
    
    display_weights["P/E Ratio"] = display_weights["ticker"].apply(
        lambda x: f"{stock_data[x]['pe_ratio']:.2f}" if isinstance(stock_data[x]['pe_ratio'], (int, float)) else "N/A"
    )
    
    display_table = display_weights[["ticker", "Weight %", "Avg Buy Price", "Current Price", "Gain/Loss %", "P/E Ratio"]].reset_index(drop=True)
    display_table.index = display_table.index + 1
    display_table = display_table.rename(columns={"ticker": "Ticker"})
    
    st.dataframe(
        display_table,
        use_container_width=True,
        height=300,
        hide_index=False
    )
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.caption(f"**Total beholdinger:** {len(weights)} aktier")
    with col_info2:
        st.caption(f"**Total vægt:** {weights['weight'].sum()*100:.2f}%")
else:
    st.info("Ingen beholdinger")

# ===== ANALYTICS & TRACKING SECTION =====
st.markdown("---")
st.subheader("📊 Analytics & Tracking")

tab1, tab2, tab3 = st.tabs(["Daily Metrics", "Monthly Summary", "Rebalance History"])

with tab1:
    display_daily_metrics()

with tab2:
    display_monthly_metrics()

with tab3:
    display_rebalance_history()

# ===== DEBUG INFO =====
with st.expander("🔧 Debug Info"):
    st.write("**=== RAW DATA ===**")
    st.write(f"all_intraday_df length: {len(all_intraday_df)}")
    if len(all_intraday_df) > 0:
        st.write(f"all_intraday_df date range: {all_intraday_df['timestamp'].min()} to {all_intraday_df['timestamp'].max()}")
        st.write(f"all_intraday_df unique dates: {sorted(all_intraday_df['timestamp'].dt.date.unique())}")
    
    st.write("**=== AFTER WEEKENDS REMOVED ===**")
    st.write(f"weekday_data length: {len(weekday_data) if 'weekday_data' in locals() else 'N/A'}")
    if 'weekday_data' in locals() and len(weekday_data) > 0:
        st.write(f"weekday_data unique dates: {sorted(weekday_data['timestamp'].dt.date.unique())}")
    
    st.write("**=== AFTER TIME PERIOD FILTER ===**")
    st.write(f"cutoff_time: {cutoff_time if 'cutoff_time' in locals() else 'N/A'}")
    st.write(f"current_time: {current_time}")
    st.write(f"period_data length: {len(period_data) if 'period_data' in locals() else 'N/A'}")
    if 'period_data' in locals() and len(period_data) > 0:
        st.write(f"period_data date range: {period_data['timestamp'].min()} to {period_data['timestamp'].max()}")
        st.write(f"period_data unique dates: {sorted(period_data['timestamp'].dt.date.unique())}")
    
    st.write("**=== AFTER RESAMPLING ===**")
    st.write(f"eq_plot_filtered length: {len(eq_plot_filtered) if 'eq_plot_filtered' in locals() else 'N/A'}")
