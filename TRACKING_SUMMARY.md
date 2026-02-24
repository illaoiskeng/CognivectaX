# CognivectaX - Data Tracking Summary

## Every 3 minutes:
┌─────────────────────┐
│ Dashboard refreshes  │ ← Auto-refresh every 5 seconds
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Saves portfolio      │ ← metrics_tracker saves intraday data
│ value to database    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ User sees 3 new     │ ← dashboard_metrics displays it
│ tabs with charts    │
└─────────────────────┘

**What we track:** Portfolio value (current amount of money)
**Stored in:** SQLite database (intraday_values table)
**You see:** Live charts updating in real-time

---

## Every day at 4 PM:
┌─────────────────────┐
│ Calculates Sharpe,  │ ← metrics_tracker computes metrics
│ Alpha, Beta, etc    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Saves to database   │ ← Stores in SQLite
│ and JSON file       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Dashboard shows it  │ ← dashboard_metrics displays
│ in "Daily Metrics"  │   in charts
└─────────────────────┘

**What we track:** Sharpe Ratio, Alpha, Beta, Volatility, Max Drawdown
**Stored in:** SQLite database + JSON file (one file per day)
**You see:** "Daily Metrics" tab with charts and numbers

---

## Every month (last trading day):
┌─────────────────────┐
│ Collects all ticker │ ← metrics_tracker pulls from yfinance
│ data for the month  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Calculates monthly  │ ← Computes all monthly metrics
│ summary and stats   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Saves to database   │ ← Stores in SQLite
│ and JSON file       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Dashboard shows it  │ ← dashboard_metrics displays
│ in "Monthly        │   monthly data
│ Summary" tab        │
└─────────────────────┘

**What we track:** Monthly return, Sharpe, Alpha, Beta, per-stock lowest/highest prices, win rate
**Stored in:** SQLite database + JSON file (2026-02.json, 2026-03.json, etc.)
**You see:** "Monthly Summary" tab with table of all months

---

## On rebalance days:
┌──────────────────────┐
│ Before market opens  │ ← Captures snapshot at 9:30 AM ET
│ (9:30 AM ET)        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Saves portfolio      │ ← Stores in JSON file
│ and stock snapshot   │   (YYYY-MM-DD_before_open.json)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ After market closes  │ ← Captures snapshot at 4:00 PM ET
│ (4:00 PM ET)        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Saves updated       │ ← Stores in SQLite +
│ portfolio and       │   JSON file (YYYY-MM-DD_after_close.json)
│ calculates changes  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Dashboard shows it  │ ← dashboard_metrics displays
│ in "Rebalance      │   in Rebalance History tab
│ History" tab        │
└──────────────────────┘

**What we track:** Portfolio value before & after, each stock's price change, weight changes, impact of rebalance
**Stored in:** SQLite database + Two JSON files (before_open and after_close)
**You see:** "Rebalance History" tab showing all rebalance details

---

## Data Storage Locations

**SQLite Database:** `data/portfolio_metrics.db`
- All metrics stored permanently here

**JSON Report Files:** `data/reports/`
