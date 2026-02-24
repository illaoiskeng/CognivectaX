"""
Metrics tracking integration for Streamlit dashboard
Call these functions during dashboard load to automatically track metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time as time_module
from metrics_tracker import (
    init_sqlite_db, init_json_reports,
    insert_intraday_value, get_intraday_values,
    calculate_sharpe_ratio, calculate_alpha_beta, calculate_max_drawdown_from_tickers,
    insert_daily_metrics, get_daily_metrics,
    get_lowest_highest_for_month, insert_ticker_monthly_stats, insert_monthly_metrics,
    get_monthly_metrics, get_rebalance_history,
    save_before_open_snapshot, save_after_close_snapshot, insert_rebalance_stock_change,
    save_monthly_report, load_json_report
)

def init_tracking():
    """Initialize tracking system on app startup"""
    if "tracking_initialized" not in st.session_state:
        init_sqlite_db("data/portfolio_metrics.db")
        init_json_reports("data/reports")
        st.session_state.tracking_initialized = True
        print("✅ Tracking system initialized")

def track_intraday(portfolio_value: float, daily_return_pct: float):
    """Track intraday portfolio value (call every 3 minutes)"""
    try:
        insert_intraday_value("data/portfolio_metrics.db", portfolio_value, daily_return_pct)
    except Exception as e:
        print(f"Error tracking intraday: {e}")

def calculate_and_save_daily_metrics(tickers: list, weights: list, 
                                    portfolio_value: float, equity_series: pd.Series):
    """Calculate and save daily metrics (call once per day at market close)"""
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Check if already saved today
        existing = get_daily_metrics("data/portfolio_metrics.db", days=1)
        if len(existing) > 0 and existing.iloc[-1]['date'] == today:
            return
        
        # Calculate metrics
        returns = equity_series.pct_change().dropna()
        daily_return = float(returns.iloc[-1]) if len(returns) > 0 else 0
        sharpe = calculate_sharpe_ratio(returns)
        
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
        end_date = today
        alpha, beta = calculate_alpha_beta(tickers, weights, start_date, end_date)
        
        volatility = float(returns.std() * np.sqrt(252)) if len(returns) > 0 else 0
        max_dd = calculate_max_drawdown_from_tickers(tickers, weights, start_date, end_date)
        
        # Insert into database
        insert_daily_metrics(
            "data/portfolio_metrics.db",
            today,
            portfolio_value,
            daily_return,
            sharpe,
            alpha,
            beta,
            volatility,
            max_dd
        )
        
        print(f"✅ Daily metrics calculated and saved for {today}")
        
    except Exception as e:
        print(f"Error calculating daily metrics: {e}")

def capture_rebalance_before_open(rebalance_date: str, portfolio_value: float,
                                  tickers: list, weights: list, prices: dict):
    """
    Capture pre-market snapshot on rebalance day
    Call this before market opens (9:30 AM ET)
    """
    try:
        holdings = {}
        for ticker, weight in zip(tickers, weights):
            if ticker in prices:
                holdings[ticker] = {
                    "weight": weight,
                    "price": prices[ticker],
                    "position_value": portfolio_value * weight
                }
        
        save_before_open_snapshot("data/reports", rebalance_date, portfolio_value, holdings)
        print(f"✅ Pre-market snapshot captured for {rebalance_date}")
        
    except Exception as e:
        print(f"Error capturing pre-market snapshot: {e}")

def capture_rebalance_after_close(rebalance_date: str, portfolio_value: float,
                                  tickers: list, weights: list, prices: dict):
    """
    Capture post-market snapshot on rebalance day
    Call this after market closes (4:00 PM ET)
    """
    try:
        # Load before snapshot
        before_file = f"data/reports/rebalance/{rebalance_date}_before_open.json"
        before_data = load_json_report(before_file)
        before_holdings = before_data['holdings']
        before_value = before_data['portfolio_value']
        
        # Create after holdings and calculate changes
        after_holdings = {}
        stock_changes = []
        
        for ticker, weight in zip(tickers, weights):
            if ticker in prices:
                after_holdings[ticker] = {
                    "weight": weight,
                    "price": prices[ticker],
                    "position_value": portfolio_value * weight
                }
                
                # Compare before and after
                if ticker in before_holdings:
                    before_price = before_holdings[ticker]['price']
                    before_weight = before_holdings[ticker]['weight']
                    
                    price_change_pct = ((prices[ticker] - before_price) / before_price * 100) if before_price > 0 else 0
                    weight_change_pct = ((weight - before_weight) / before_weight * 100) if before_weight > 0 else 0
                    
                    stock_changes.append({
                        "ticker": ticker,
                        "pre_open_price": before_price,
                        "post_close_price": prices[ticker],
                        "price_change_pct": price_change_pct,
                        "pre_open_weight": before_weight,
                        "post_close_weight": weight,
                        "weight_change_pct": weight_change_pct
                    })
                    
                    # Insert into database
                    insert_rebalance_stock_change(
                        "data/portfolio_metrics.db",
                        rebalance_date,
                        ticker,
                        before_price,
                        prices[ticker],
                        before_weight,
                        weight
                    )
        
        save_after_close_snapshot("data/reports", rebalance_date, portfolio_value, 
                                 after_holdings, stock_changes)
        print(f"✅ Post-market snapshot captured for {rebalance_date}")
        
    except Exception as e:
        print(f"Error capturing post-market snapshot: {e}")

def calculate_and_save_monthly_metrics(tickers: list, weights: list,
                                      portfolio_value: float, equity_series: pd.Series):
    """Calculate and save monthly metrics (call once per month at month end)"""
    try:
        now = pd.Timestamp.now()
        month = now.strftime('%Y-%m')
        
        # Get ticker stats
        ticker_stats = get_lowest_highest_for_month(tickers, now.year, now.month)
        
        for ticker, stats in ticker_stats.items():
            if stats:
                insert_ticker_monthly_stats("data/portfolio_metrics.db", month, ticker, stats)
        
        # Calculate portfolio metrics
        returns = equity_series.pct_change().dropna()
        monthly_return = (equity_series.iloc[-1] / equity_series.iloc[0] - 1) if len(equity_series) > 0 else 0
        sharpe = calculate_sharpe_ratio(returns)
        
        start_date = f"{now.year}-{now.month:02d}-01"
        end_date = now.strftime('%Y-%m-%d')
        alpha, beta = calculate_alpha_beta(tickers, weights, start_date, end_date)
        max_dd = calculate_max_drawdown_from_tickers(tickers, weights, start_date, end_date)
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        # Insert monthly summary
        monthly_metrics = {
            "monthly_return": monthly_return,
            "sharpe_ratio": sharpe,
            "alpha": alpha,
            "beta": beta,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "ticker_stats": ticker_stats
        }
        
        insert_monthly_metrics("data/portfolio_metrics.db", month, monthly_metrics)
        save_monthly_report("data/reports", month, monthly_metrics)
        
        print(f"✅ Monthly metrics calculated and saved for {month}")
        
    except Exception as e:
        print(f"Error calculating monthly metrics: {e}")

# ===== DASHBOARD DISPLAY FUNCTIONS =====

def safe_format(value, format_str=".2f", suffix=""):
    """Safely format values, handling None and NaN"""
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):{format_str}}{suffix}"
    except:
        return "N/A"

def display_daily_metrics():
    """Display daily metrics in dashboard"""
    st.subheader("📊 Daily Performance Metrics (Last 30 Days)")
    
    daily_df = get_daily_metrics("data/portfolio_metrics.db", days=30)
    
    if len(daily_df) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            latest_sharpe = daily_df.iloc[-1]['sharpe_ratio']
            delta = None
            if len(daily_df) > 1:
                prev_sharpe = daily_df.iloc[-2]['sharpe_ratio']
                if not pd.isna(prev_sharpe) and not pd.isna(latest_sharpe):
                    delta = latest_sharpe - prev_sharpe
            st.metric(
                "Sharpe Ratio", 
                safe_format(latest_sharpe, ".2f"),
                delta=delta
            )
        
        with col2:
            latest_alpha = daily_df.iloc[-1]['alpha']
            delta = None
            if len(daily_df) > 1:
                prev_alpha = daily_df.iloc[-2]['alpha']
                if not pd.isna(prev_alpha) and not pd.isna(latest_alpha):
                    delta = latest_alpha - prev_alpha
            st.metric(
                "Alpha", 
                safe_format(latest_alpha, ".4f"),
                delta=delta
            )
        
        with col3:
            latest_beta = daily_df.iloc[-1]['beta']
            delta = None
            if len(daily_df) > 1:
                prev_beta = daily_df.iloc[-2]['beta']
                if not pd.isna(prev_beta) and not pd.isna(latest_beta):
                    delta = latest_beta - prev_beta
            st.metric(
                "Beta", 
                safe_format(latest_beta, ".2f"),
                delta=delta
            )
        
        with col4:
            latest_dd = daily_df.iloc[-1]['max_drawdown']
            if not pd.isna(latest_dd):
                dd_display = f"{latest_dd*100:.2f}%"
            else:
                dd_display = "N/A"
            st.metric(
                "Max Drawdown", 
                dd_display
            )
        
        # Charts
        st.write("**Sharpe Ratio Trend**")
        sharpe_data = daily_df[['date', 'sharpe_ratio']].dropna()
        if len(sharpe_data) > 0:
            st.line_chart(sharpe_data.set_index('date'))
        else:
            st.info("No sharpe ratio data available yet")
        
        st.write("**Alpha & Beta Trend**")
        ab_data = daily_df[['date', 'alpha', 'beta']].dropna()
        if len(ab_data) > 0:
            st.line_chart(ab_data.set_index('date'))
        else:
            st.info("No alpha/beta data available yet")
    else:
        st.info("⏳ No daily metrics yet. Check back after market close.")

def display_monthly_metrics():
    """Display monthly metrics in dashboard"""
    st.subheader("📈 Monthly Summary")
    
    monthly_df = get_monthly_metrics("data/portfolio_metrics.db", months=12)
    
    if len(monthly_df) > 0:
        # Format for display
        display_df = monthly_df.copy()
        display_df['monthly_return'] = display_df['monthly_return'].apply(lambda x: safe_format(x * 100, ".2f", "%"))
        display_df['sharpe_ratio'] = display_df['sharpe_ratio'].apply(lambda x: safe_format(x, ".2f"))
        display_df['alpha'] = display_df['alpha'].apply(lambda x: safe_format(x, ".4f"))
        display_df['beta'] = display_df['beta'].apply(lambda x: safe_format(x, ".2f"))
        display_df['max_drawdown'] = display_df['max_drawdown'].apply(lambda x: safe_format(x * 100, ".2f", "%"))
        display_df['win_rate'] = display_df['win_rate'].apply(lambda x: safe_format(x * 100, ".1f", "%"))
        
        st.dataframe(
            display_df[['month', 'monthly_return', 'sharpe_ratio', 'alpha', 'beta', 'max_drawdown', 'win_rate']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("⏳ No monthly metrics yet. Check back at month end.")

def display_rebalance_history():
    """Display rebalance history"""
    st.subheader("🔄 Rebalance History")
    
    rebalance_df = get_rebalance_history("data/portfolio_metrics.db")
    
    if len(rebalance_df) > 0:
        # Format for display
        display_df = rebalance_df.copy()
        display_df['pre_open_portfolio_value'] = display_df['pre_open_portfolio_value'].apply(lambda x: safe_format(x, ",.0f", " kr"))
        display_df['post_close_portfolio_value'] = display_df['post_close_portfolio_value'].apply(lambda x: safe_format(x, ",.0f", " kr"))
        display_df['portfolio_change_pct'] = display_df['portfolio_change_pct'].apply(lambda x: safe_format(x, "+.2f", "%"))
        
        st.dataframe(
            display_df[['rebalance_date', 'pre_open_portfolio_value', 'post_close_portfolio_value', 'portfolio_change_pct']],
            use_container_width=True,
            hide_index=True
        )
        
        # Show detailed stock changes for latest rebalance
        if len(rebalance_df) > 0:
            latest_date = rebalance_df.iloc[0]['rebalance_date']
            st.write(f"**Stock Changes on {latest_date}**")
            
            try:
                after_file = f"data/reports/rebalance/{latest_date}_after_close.json"
                after_data = load_json_report(after_file)
                stock_changes = after_data.get('stock_changes', [])
                
                if stock_changes:
                    changes_df = pd.DataFrame(stock_changes)
                    st.dataframe(changes_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No stock changes available for this rebalance")
            except Exception as e:
                st.info("No detailed stock changes available yet")
    else:
        st.info("⏳ No rebalance events recorded yet.")
