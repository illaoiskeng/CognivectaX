import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import timedelta, datetime, time
import time as time_module

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
        if timestamp.tz is None:
            ts_est = timestamp.tz_localize('UTC').tz_convert('US/Eastern')
        else:
            ts_est = timestamp.tz_convert('US/Eastern')
        
        time_only = ts_est.time()
        weekday = ts_est.weekday()
        
        if weekday > 4:
            return False
        
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

current_time = pd.Timestamp.now()
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

current_value = equity_dkk.iloc[-1]
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

if "selected_time_period" not in st.session_state:
    st.session_state.selected_time_period = "1M"

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

# ===== TIME PERIOD BUTTONS (ORIGINAL CODE) =====
time_cols = st.columns(len(time_periods))
for i, (period_name, period_config) in enumerate(time_periods.items()):
    with time_cols[i]:
        if st.button(period_name, key=f"btn_{period_name}", use_container_width=True):
            st.session_state.selected_time_period = period_name
            st.rerun()

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

rounded_now = current_time.replace(minute=0, second=0, microsecond=0)

eq_plot = equity_dkk.copy()

if current_config["days"] is not None:
    cutoff = rounded_now - timedelta(days=current_config["days"])
    eq_plot = eq_plot[eq_plot.index >= cutoff]

def resample_data(data: pd.Series, interval: str) -> pd.DataFrame:
    """Resample equity data to specified interval"""
    if interval == "5min":
        resampled = data.resample("5T").last()
    elif interval == "15min":
        resampled = data.resample("15T").last()
    elif interval == "30min":
        resampled = data.resample("30T").last()
    elif interval == "1H":
        resampled = data.resample("1H").last()
    elif interval == "4H":
        resampled = data.resample("4H").last()
    elif interval == "1D":
        resampled = data.resample("1D").last()
    elif interval == "1W":
        resampled = data.resample("1W").last()
    elif interval == "1M":
        resampled = data.resample("1M").last()
    else:
        resampled = data
    
    resampled = resampled.dropna()
    return resampled

resampled_eq = resample_data(eq_plot, selected_interval)

eq_df = pd.DataFrame({
    "Date": resampled_eq.index,
    "CognivectaX": resampled_eq.values
})

eq_df["Change %"] = eq_df["CognivectaX"].pct_change() * 100
eq_df.loc[0, "Change %"] = (eq_df.loc[0, "CognivectaX"] / equity_dkk.iloc[0] - 1) * 100
eq_df["Change Text"] = eq_df["Change %"].apply(lambda x: f"{x:+.2f}")

if len(eq_df) == 0:
    st.warning(f"No data available for period {current_period}.")
else:
    min_value = eq_df["CognivectaX"].min()
    max_value = eq_df["CognivectaX"].max()
    value_range = max_value - min_value
    
    if value_range == 0:
        value_range = 1
    
    y_max = max_value + (value_range * 0.15)
    y_min = min_value - (value_range * 0.05)

    def generate_tick_positions(data: pd.DataFrame, period: str):
        """Generate smart tick positions based on time period"""
        n_ticks = 6
        indices = np.linspace(0, len(data) - 1, n_ticks, dtype=int)
        tick_positions = [data.iloc[i]["Date"] for i in indices]
        
        if period == "1D":
            tick_format = "%H:%M"
        elif period in ["1U", "1M"]:
            tick_format = "%d %b"
        else:
            tick_format = "%d %b"
        
        return tick_positions, tick_format

    tick_positions, tick_format = generate_tick_positions(eq_df, current_period)
    
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

inception_return = (equity_dkk.iloc[-1] / equity_dkk.iloc[0] - 1.0)

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
            display_ret = f"{get_return_sign(ret)}{abs(ret)*100:.2f}%"
            color = get_return_color(ret)
        
        st.markdown(f"""
            <div style="background-color: white; border: 2px solid #e0e0e0; border-radius: 8px; padding: 12px; text-align: center; margin-bottom: 15px; margin-top: -50px; position: relative; z-index: 10; pointer-events: none;" onmouseover="this.style.backgroundColor='#f0f7ff'; this.style.border='2px solid #1f77b4'; this.style.boxShadow='0 4px 8px rgba(31, 119, 180, 0.2)';" onmouseout="this.style.backgroundColor=''; this.style.border='2px solid #e0e0e0'; this.style.boxShadow='none';">
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
        # Fetch company names for pie chart hover
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
        
        # Create pie chart with company names
        pie_data = weights.copy()
        pie_data['company_name'] = pie_data['ticker'].map(company_names)
        
        fig_pie = go.Figure()
        fig_pie.add_trace(go.Pie(
            labels=pie_data['ticker'],
            values=pie_data['weight'],
            hovertemplate="<b>%{customdata[0]}</b><br>Ticker: %{label}<br>Vægt: %{value:.2%}<extra></extra>",
            customdata=np.array(pie_data['company_name']).reshape(-1, 1),
            textinfo="label+percent",
            textposition="auto"
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
