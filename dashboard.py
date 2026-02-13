import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px

@st.cache_data(ttl=86400)
def get_full_names(tickers):
    names = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).get_info()
            names[t] = info.get("shortName") or info.get("longName") or t
        except:
            names[t] = t
    return names

st.set_page_config(layout="wide")
st.title("CognivectaX – Portfolio Dashboard")

DATA = "data/weights_latest.csv"

# ---------- Sidebar controls ----------
st.sidebar.header("Visning")

START_CAPITAL_DKK = 100_000

INCEPTION_DATE = "2026-01-01"

interval = st.sidebar.selectbox("Interval (granularitet)", ["15m", "30m", "60m", "1d"], index=3)
# Yahoo-begrænsning: intraday har typisk begrænset historik
period = st.sidebar.selectbox("Horisont", ["5d", "1mo", "3mo", "6mo", "1y", "5y", "max"], index=6)

st.sidebar.caption("OBS: 15m/30m/60m har begrænset historik hos Yahoo (ofte ~60 dage).")

# ---------- Load weights ----------
try:
    weights = pd.read_csv(DATA)
except:
    st.warning("No weights file found. Run runner.py first.")
    st.stop()

weights["ticker"] = weights["ticker"].astype(str).str.upper()
weights["weight"] = weights["weight"].astype(float)

tickers = weights["ticker"].tolist()

# ---------- FX: USD -> DKK ----------
@st.cache_data(ttl=3600)
def get_usd_to_dkk() -> float:
    fx = yf.download("DKK=X", period="5d", interval="1d", auto_adjust=True, progress=False)
    if fx is None or fx.empty:
        return np.nan
    # Close kolonne kan hedde 'Close' eller være multiindex alt efter yfinance-version
    if "Close" in fx.columns:
        val = float(fx["Close"].dropna().iloc[-1])
    else:
        # fallback: tag sidste kolonne sidste værdi
        val = float(fx.dropna().iloc[-1].values[-1])
    return val

usd_to_dkk = get_usd_to_dkk()
if np.isnan(usd_to_dkk):
    st.warning("Kunne ikke hente USD/DKK (DKK=X). Viser værdier som 'USD≈DKK'.")
    usd_to_dkk = 1.0

# ---------- Download prices ----------
@st.cache_data(ttl=1800)
def download_prices(tickers, period, interval):
    raw = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=True
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    # MultiIndex -> Close
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        # single ticker fallback
        close = pd.DataFrame({tickers[0]: raw["Close"]})
    close = close.dropna(how="all").sort_index()
    return close

prices = download_prices(tickers, "max", interval)
if prices.empty:
    st.error("Ingen prisdata (måske pga. Yahoo-begrænsning for intraday). Prøv større horisont/interval, fx 1d + max.")
    st.stop()

rets = prices.pct_change().fillna(0.0)

# Align weights to available columns
w = weights.set_index("ticker")["weight"].reindex(prices.columns).fillna(0.0)

# Portfolio returns + equity (i DKK)
port_ret = rets @ w
equity_usd = START_CAPITAL_DKK / usd_to_dkk * (1 + port_ret).cumprod()
equity_dkk = equity_usd * usd_to_dkk
equity_dkk = equity_dkk[equity_dkk.index >= INCEPTION_DATE]
if equity_dkk.empty:
    st.warning("Ingen data endnu efter launch-dato")
    st.stop()
equity_dkk.name = "CognivectaX"

# ---------- Metrics ----------
st.subheader("Portfolio Performance")

col0, col1, col2, col3, col4, col5, col6 = st.columns(7)

def calc_return(days):
    if len(equity_dkk) <= days:
        return np.nan
    return equity_dkk.iloc[-1] / equity_dkk.iloc[-days] - 1

col0.metric("Portfolio Value (DKK)", f"{equity_dkk.iloc[-1]:,.0f} kr")
col1.metric("1D", f"{calc_return(1)*100:.2f}%")
col2.metric("1W", f"{calc_return(5)*100:.2f}%")
col3.metric("1M", f"{calc_return(21)*100:.2f}%")
col4.metric("3M", f"{calc_return(63)*100:.2f}%")
col5.metric("6M", f"{calc_return(126)*100:.2f}%")
col6.metric("Total", f"{(equity_dkk.iloc[-1]/equity_dkk.iloc[0]-1)*100:.2f}%")

# ---------- Charts ----------
pie_df = weights.copy()

name_map = get_full_names(pie_df["ticker"].tolist())
pie_df["full_name"] = pie_df["ticker"].map(name_map)

c1, c2 = st.columns([2, 1])

with c1:
    eq_df = equity_dkk.reset_index()
    eq_df.columns = ["Date", "CognivectaX"]
    fig = px.line(eq_df, x="Date", y="CognivectaX", title="Equity Curve (DKK)")
    fig.update_traces(hovertemplate="Dato: %{x}<br>Værdi: %{y:,.0f} kr")
    fig.update_yaxes(tickformat=",.0f", title="DKK")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig2 = px.pie(
        pie_df,
        names="ticker",
        values="weight",
        title="Weight Allocation"
    )

    # kun ticker på selve kagen
    fig2.update_traces(
        textinfo="label",
        customdata=pie_df["full_name"],
        hovertemplate="<b>%{label}</b><br>%{customdata}<br>Vægt: %{percent}<extra></extra>"
    )

    # fjern farvelisten til højre
    fig2.update_layout(showlegend=False)

    st.plotly_chart(fig2, use_container_width=True)
