# CognivectaX

A systematic, Sharpe-optimized technology equity strategy.

## Overview

CognivectaX constructs a diversified portfolio of 40–70 global tech and semiconductor stocks, optimizing weights for maximum risk-adjusted return using rolling covariance estimation.

The strategy is fully automated with real-time monitoring, comprehensive metrics tracking, and detailed performance reporting.

## Features

### Portfolio Management
- **Sharpe-optimized portfolio construction** - Maximizes risk-adjusted returns
- **Monthly/Quarterly rebalancing** - Systematic weight adjustments
- **Stress-tested cost assumptions** - Realistic fee modeling
- **40-70 tech stocks** - Diversified across global tech & semiconductors

### Real-Time Monitoring
- **Interactive dashboard** - Live portfolio analytics and charts
- **3-minute tracking** - Intraday portfolio value snapshots
- **Daily metrics** - Sharpe ratio, Alpha, Beta, Volatility, Max Drawdown
- **Monthly summaries** - Complete performance analysis with per-stock breakdowns

### Comprehensive Reporting
- **SQLite database** - Permanent storage of all metrics
- **JSON reports** - Monthly summaries and rebalance snapshots
- **Before/After snapshots** - Track rebalance impact
- **Historical data** - Query and analyze any period

## Dashboard

The interactive Streamlit dashboard provides:

**📈 Main Chart**
- Real-time portfolio value with multiple timeframes (1D, 1W, 1M, 3M, 6M, 1Y, max)
- Customizable intervals for detailed analysis
- Hover for exact values and daily returns

**📊 Portfolio Information**
- Start capital vs current value
- Total profit and return percentage
- Current allocation pie chart
- Holdings table with live prices, P/E ratios, and gain/loss

**📋 Analytics & Tracking** (3 tabs)
- **Daily Metrics** - Sharpe ratio, Alpha, Beta, Max Drawdown with 30-day trends
- **Monthly Summary** - Full month performance comparison table
- **Rebalance History** - Before/after snapshots, impact analysis, stock changes

## Data Tracking

### Every 3 Minutes (Market Hours)
