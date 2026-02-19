import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide")
st.title("CognivectaX – Portfolio Dashboard")
st_autorefresh(interval=5000, key="cognivectax_refresh_5s")

DATA_WEIGHTS = "data/weights_latest.csv"
DATA_EQUITY = "data/papertrade/portfolio_value_daily.csv"

START_CAPITAL_DKK = 100_000
INCEPTION_DATE = "2026-01-01"

st.sidebar.header("Visning")
view = st.sidebar.selectbox(
    "Graf-interval",
    ["15 min", "1 time", "12 timer", "1 dag", "1 uge", "1 måned", "3 måneder", "6 måneder", "12 måneder", "3 år", "5 år", "Historisk"],
    index=3
)
st.sidebar.caption("Horisont påvirker kun grafen.")

@st.cache_data(ttl=5)
def load_weights(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["weight"] = df["weight"].astype(float)
    df.loc[df["weight"] < 1e-8, "weight"] = 0.0
    s = float(df["weight"].sum())
    if s > 0:
        df["weight"] = df["weight"] / s
    return df

@st.cache_data(ttl=5)
def load_equity(path: str) -> pd.Series:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    eq = pd.Series(df["total_value"].values, index=df["date"], name="equity")
    eq = eq[eq.index >= pd.Timestamp(INCEPTION_DATE)]
    return eq

try:
    weights = load_weights(DATA_WEIGHTS)
except Exception:
    st.warning("No weights file found yet. Run papertrade/runner first.")
    st.stop()

try:
    equity_dkk = load_equity(DATA_EQUITY)
except Exception:
    st.warning("No equity curve found yet. Run papertrade/runner first.")
    st.stop()

tickers = weights["ticker"].tolist()

@st.cache_data(ttl=86400)
def get_full_names(tickers: list[str]) -> dict:
    names = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).get_info()
            names[t] = info.get("shortName") or info.get("longName") or t
        except Exception:
            names[t] = t
    return names

@st.cache_data(ttl=3600)
def get_usd_to_dkk() -> float:
    fx = yf.download("DKK=X", period="5d", interval="1d", auto_adjust=True, progress=False, threads=False)
    if fx is None or fx.empty:
        return np.nan
    return float(fx["Close"].dropna().iloc[-1])

usd_to_dkk = get_usd_to_dkk()
if np.isnan(usd_to_dkk):
    st.warning("Kunne ikke hente USD/DKK (DKK=X). Viser værdier som 'USD≈DKK'.")
    usd_to_dkk = 1.0

@st.cache_data(ttl=86400)
def download_daily_prices_usd(tickers: list[str]) -> pd.DataFrame:
    raw = yf.download(
        tickers,
        period="max",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else pd.DataFrame({tickers[0]: raw["Close"]})
    return close.dropna(how="all").sort_index()

prices_usd = download_daily_prices_usd(tickers)
if prices_usd.empty:
    st.error("Ingen daily prisdata kunne hentes fra Yahoo.")
    st.stop()

prices_dkk = prices_usd * usd_to_dkk

last_daily_close_dkk = prices_dkk.iloc[-1]

@st.cache_data(ttl=5)
def download_live_last_prices_dkk(tickers: list[str], usd_to_dkk: float) -> pd.Series:
    raw = yf.download(
        tickers,
        period="1d",
        interval="1m",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        return pd.Series(dtype=float)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else pd.DataFrame({tickers[0]: raw["Close"]})
    close = close.dropna(how="all")
    if close.empty:
        return pd.Series(dtype=float)
    return close.iloc[-1] * usd_to_dkk

@st.cache_data(ttl=5)
def download_intraday_close_dkk(tickers, usd_to_dkk):
    raw = yf.download(
        tickers,
        period="1d",
        interval="1m",
        auto_adjust=True,
        progress=False,
        threads=False
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else pd.DataFrame({tickers[0]: raw["Close"]})
    return close.dropna(how="all").sort_index() * usd_to_dkk

w = weights.set_index("ticker")["weight"].reindex(prices_dkk.columns).fillna(0.0)

live_last_dkk = download_live_last_prices_dkk(tickers, usd_to_dkk)
common = live_last_dkk.index.intersection(last_daily_close_dkk.index)

if len(common) > 0:
    intraday_rets = (live_last_dkk[common] / last_daily_close_dkk[common] - 1.0).fillna(0.0)
    w_common = w.reindex(common).fillna(0.0)
    live_multiplier = 1.0 + float(intraday_rets @ w_common)
    portfolio_value_live = float(equity_dkk.iloc[-1] * live_multiplier)
else:
    portfolio_value_live = float(equity_dkk.iloc[-1])

st.subheader("Portfolio Performance")

col0, col1, col2, col3, col4, col5, col6 = st.columns(7)

def calc_return(trading_days: int) -> float:
    if len(equity_dkk) <= trading_days:
        return np.nan
    return float(equity_dkk.iloc[-1] / equity_dkk.iloc[-trading_days] - 1.0)

col0.metric("Portfolio Value (DKK)", f"{portfolio_value_live:,.0f} kr")
st.caption(f"Live opdateret: {pd.Timestamp.now(tz='Europe/Copenhagen').strftime('%H:%M:%S')}")
col1.metric("1D", f"{calc_return(1)*100:.2f}%")
col2.metric("1W", f"{calc_return(5)*100:.2f}%")
col3.metric("1M", f"{calc_return(21)*100:.2f}%")
col4.metric("3M", f"{calc_return(63)*100:.2f}%")
col5.metric("6M", f"{calc_return(126)*100:.2f}%")
col6.metric("Since inception", f"{(float(equity_dkk.iloc[-1]) / START_CAPITAL_DKK - 1)*100:.2f}%")

c1, c2 = st.columns([2, 1])

with c1:
    days_map = {
        "1 uge": 7,
        "1 måned": 31,
        "3 måneder": 93,
        "6 måneder": 186,
        "12 måneder": 366,
        "3 år": 1096,
        "5 år": 1826,
    }

    if view in ["15 min", "1 time", "12 timer", "1 dag"]:
        intraday_dkk = download_intraday_close_dkk(tickers, usd_to_dkk)

        if intraday_dkk.empty:
            st.warning("Ingen intraday-data lige nu.")
        else:
            common_cols = intraday_dkk.columns.intersection(last_daily_close_dkk.index)
            intraday_dkk = intraday_dkk[common_cols].ffill()

            base = last_daily_close_dkk[common_cols]
            r_t = intraday_dkk.div(base, axis=1) - 1.0

            w_common = w.reindex(common_cols).fillna(0.0)
            multiplier = 1.0 + (r_t @ w_common)

            intraday_equity = equity_dkk.iloc[-1] * multiplier

            if view == "15 min":
                cutoff = intraday_equity.index.max() - pd.Timedelta(minutes=15)
            elif view == "1 time":
                cutoff = intraday_equity.index.max() - pd.Timedelta(hours=1)
            elif view == "12 timer":
                cutoff = intraday_equity.index.max() - pd.Timedelta(hours=12)
            else:
                cutoff = intraday_equity.index.max() - pd.Timedelta(days=1)

            intraday_equity = intraday_equity[intraday_equity.index >= cutoff]

            eq_df = intraday_equity.reset_index()
            eq_df.columns = ["Date", "CognivectaX"]

            fig = px.line(eq_df, x="Date", y="CognivectaX", title="Equity Curve (DKK) – Live")
            fig.update_traces(hovertemplate="Tid: %{x}<br>Værdi: %{y:,.0f} kr")
            fig.update_yaxes(tickformat=",.0f", title="DKK")
            st.plotly_chart(fig, use_container_width=True)
    else:
        eq_plot = equity_dkk.copy()
        if view != "Historisk":
            cutoff = eq_plot.index.max() - pd.Timedelta(days=days_map[view])
            eq_plot = eq_plot[eq_plot.index >= cutoff]

        eq_df = eq_plot.to_frame("CognivectaX").reset_index()
        eq_df = eq_df.rename(columns={eq_df.columns[0]: "Date"})

        fig = px.line(eq_df, x="Date", y=["CognivectaX"], title="Equity Curve (DKK)")
        fig.update_traces(hovertemplate="Dato: %{x}<br>Værdi: %{y:,.0f} kr")
        fig.update_yaxes(tickformat=",.0f", title="DKK")
        st.plotly_chart(fig, use_container_width=True)

with c2:
    pie_df = weights.copy()
    w_target = pie_df.set_index("ticker")["weight"].astype(float)

    common = live_last_dkk.index.intersection(last_daily_close_dkk.index).intersection(w_target.index)
    drift = (live_last_dkk[common] / last_daily_close_dkk[common]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    w_live = (w_target[common] * drift).clip(lower=0.0)
    if float(w_live.sum()) > 0:
        w_live = w_live / float(w_live.sum())

    pie_df = w_live.reset_index()
    pie_df.columns = ["ticker", "weight"]

    name_map = get_full_names(pie_df["ticker"].tolist())
    pie_df["full_name"] = pie_df["ticker"].map(name_map).fillna(pie_df["ticker"])

    fig2 = px.pie(
        pie_df[pie_df["weight"] > 0],
        names="ticker",
        values="weight",
        title="Weight Allocation"
    )
    fig2.update_traces(
        textinfo="label",
        customdata=pie_df.loc[pie_df["weight"] > 0, "full_name"],
        hovertemplate="<b>%{label}</b><br>%{customdata}<br>Vægt: %{percent}<extra></extra>"
    )
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Holdings")

holdings = weights.copy()
holdings["Name"] = holdings["ticker"].map(get_full_names(holdings["ticker"].tolist())).fillna(holdings["ticker"])
holdings["Weight %"] = (holdings["weight"] * 100).round(2)
holdings = holdings.loc[holdings["weight"] > 0, ["ticker", "Weight %", "Name"]].sort_values("Weight %", ascending=False)
holdings = holdings.rename(columns={"ticker": "Ticker"})
st.dataframe(holdings, use_container_width=True, hide_index=True)
