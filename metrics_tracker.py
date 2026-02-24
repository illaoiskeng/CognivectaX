import sqlite3
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time as time_module
from typing import Dict, List, Tuple
import os

# ===== DATABASE INITIALIZATION =====
def init_sqlite_db(db_path: str = "data/portfolio_metrics.db"):
    """Initialize SQLite database with all tracking tables"""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Daily intraday values (3-minute snapshots)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS intraday_values (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL UNIQUE,
        portfolio_value REAL NOT NULL,
        daily_return_pct REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Daily metrics (calculated once per day at close)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS daily_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date DATE NOT NULL UNIQUE,
        portfolio_value REAL NOT NULL,
        daily_return REAL,
        sharpe_ratio REAL,
        alpha REAL,
        beta REAL,
        volatility REAL,
        max_drawdown REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Monthly metrics
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS monthly_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        month TEXT NOT NULL UNIQUE,
        monthly_return REAL,
        sharpe_ratio REAL,
        alpha REAL,
        beta REAL,
        max_drawdown REAL,
        win_rate REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Ticker monthly statistics
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ticker_monthly_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        month TEXT NOT NULL,
        ticker TEXT NOT NULL,
        lowest_low REAL,
        lowest_low_date DATE,
        highest_high REAL,
        highest_high_date DATE,
        opening_price REAL,
        closing_price REAL,
        monthly_return REAL,
        max_drawdown REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(month, ticker)
    )
    ''')
    
    # Rebalance events (before and after market)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS rebalance_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        rebalance_date DATE NOT NULL UNIQUE,
        pre_open_portfolio_value REAL,
        post_close_portfolio_value REAL,
        pre_open_time DATETIME,
        post_close_time DATETIME,
        portfolio_change_pct REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Stock changes on rebalance (before and after)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS rebalance_stock_changes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        rebalance_date DATE NOT NULL,
        ticker TEXT NOT NULL,
        pre_open_price REAL,
        post_close_price REAL,
        pre_open_weight REAL,
        post_close_weight REAL,
        price_change_pct REAL,
        weight_change_pct REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(rebalance_date, ticker)
    )
    ''')
    
    # Holdings snapshots
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS holdings_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_date DATE NOT NULL,
        snapshot_type TEXT NOT NULL,
        ticker TEXT NOT NULL,
        weight REAL,
        price REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(snapshot_date, snapshot_type, ticker)
    )
    ''')
    
    conn.commit()
    conn.close()
    print(f"✅ SQLite database initialized at {db_path}")

# ===== JSON REPORTS INITIALIZATION =====
def init_json_reports(report_dir: str = "data/reports"):
    """Initialize JSON report directory structure"""
    os.makedirs(f"{report_dir}/rebalance", exist_ok=True)
    os.makedirs(f"{report_dir}/monthly", exist_ok=True)
    print(f"✅ JSON report directories initialized at {report_dir}")

# ===== INTRADAY TRACKING =====
def insert_intraday_value(db_path: str, portfolio_value: float, daily_return_pct: float):
    """Insert 3-minute intraday portfolio snapshot"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        cursor.execute('''
        INSERT INTO intraday_values (timestamp, portfolio_value, daily_return_pct)
        VALUES (?, ?, ?)
        ''', (timestamp, portfolio_value, daily_return_pct))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    finally:
        conn.close()

def get_intraday_values(db_path: str, days: int = 5):
    """Retrieve intraday values for last N trading days"""
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT * FROM intraday_values 
    WHERE timestamp >= datetime('now', '-{days} days')
    ORDER BY timestamp DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# ===== DAILY METRICS CALCULATION =====
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio from returns"""
    if len(returns) < 2:
        return np.nan
    
    excess_return = returns.mean() - (risk_free_rate / 252)
    volatility = returns.std()
    
    if volatility == 0:
        return np.nan
    
    sharpe = (excess_return / volatility) * np.sqrt(252)
    return sharpe

def calculate_alpha_beta(tickers: List[str], weights: List[float], 
                         start_date: str, end_date: str, 
                         benchmark_ticker: str = "^GSPC") -> Tuple[float, float]:
    """Calculate alpha and beta vs benchmark (S&P 500)"""
    try:
        portfolio_returns = np.array([0.0])
        
        for ticker, weight in zip(tickers, weights):
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                if len(hist) > 0:
                    returns = hist['Adj Close'].pct_change().dropna()
                    portfolio_returns = np.append(portfolio_returns, returns.values * weight)
                time_module.sleep(0.1)
            except Exception:
                continue
        
        benchmark = yf.Ticker(benchmark_ticker)
        bench_hist = benchmark.history(start=start_date, end=end_date)
        benchmark_returns = bench_hist['Adj Close'].pct_change().dropna()
        
        if len(portfolio_returns) < 2 or len(benchmark_returns) < 2:
            return np.nan, np.nan
        
        min_len = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[-min_len:]
        benchmark_returns = benchmark_returns.values[-min_len:]
        
        covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
        benchmark_variance = np.var(benchmark_returns)
        
        if benchmark_variance == 0:
            return np.nan, np.nan
        
        beta = covariance / benchmark_variance
        
        portfolio_return = np.mean(portfolio_returns) * 252
        benchmark_return = np.mean(benchmark_returns) * 252
        risk_free_rate = 0.02
        
        alpha = portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
        
        return alpha, beta
        
    except Exception as e:
        print(f"Error calculating alpha/beta: {e}")
        return np.nan, np.nan

def calculate_max_drawdown_from_tickers(tickers: List[str], weights: List[float],
                                       start_date: str, end_date: str) -> float:
    """Calculate max drawdown from actual ticker data"""
    try:
        max_dd = 0.0
        
        for ticker, weight in zip(tickers, weights):
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                
                if len(hist) == 0:
                    continue
                
                running_max = hist['Adj Close'].expanding().max()
                drawdown = (hist['Adj Close'] - running_max) / running_max
                ticker_max_dd = drawdown.min()
                
                if ticker_max_dd < max_dd:
                    max_dd = ticker_max_dd
                
                time_module.sleep(0.1)
            except Exception:
                continue
        
        return max_dd
        
    except Exception as e:
        print(f"Error calculating max drawdown: {e}")
        return np.nan

def insert_daily_metrics(db_path: str, date: str, portfolio_value: float,
                        daily_return: float, sharpe: float, alpha: float,
                        beta: float, volatility: float, max_drawdown: float):
    """Insert daily metrics into database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
        INSERT INTO daily_metrics
        (date, portfolio_value, daily_return, sharpe_ratio, alpha, beta, volatility, max_drawdown)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (date, portfolio_value, daily_return, sharpe, alpha, beta, volatility, max_drawdown))
        conn.commit()
        print(f"✅ Daily metrics inserted for {date}")
    except sqlite3.IntegrityError:
        print(f"⚠️ Daily metrics for {date} already exist")
    finally:
        conn.close()

def get_daily_metrics(db_path: str, days: int = 30) -> pd.DataFrame:
    """Retrieve daily metrics for last N days"""
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT * FROM daily_metrics
    ORDER BY date DESC
    LIMIT {days}
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df.sort_values('date')

# ===== MONTHLY METRICS =====
def get_lowest_highest_for_month(tickers: List[str], year: int, month: int) -> Dict:
    """Get lowest low and highest high for each ticker for a specific month"""
    stats = {}
    
    start_date = f"{year}-{month:02d}-01"
    if month == 12:
        end_date = f"{year+1}-01-01"
    else:
        end_date = f"{year}-{month+1:02d}-01"
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if len(hist) > 0:
                stats[ticker] = {
                    "lowest_low": float(hist['Low'].min()),
                    "lowest_low_date": hist['Low'].idxmin().strftime('%Y-%m-%d'),
                    "highest_high": float(hist['High'].max()),
                    "highest_high_date": hist['High'].idxmax().strftime('%Y-%m-%d'),
                    "opening_price": float(hist['Open'].iloc[0]),
                    "closing_price": float(hist['Close'].iloc[-1]),
                    "monthly_return": float((hist['Close'].iloc[-1] - hist['Open'].iloc[0]) / hist['Open'].iloc[0])
                }
            
            time_module.sleep(0.2)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            stats[ticker] = None
    
    return stats

def insert_ticker_monthly_stats(db_path: str, month: str, ticker: str, stats: Dict):
    """Insert ticker monthly statistics"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
        INSERT INTO ticker_monthly_stats
        (month, ticker, lowest_low, lowest_low_date, highest_high, highest_high_date, 
         opening_price, closing_price, monthly_return, max_drawdown)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            month,
            ticker,
            stats.get('lowest_low'),
            stats.get('lowest_low_date'),
            stats.get('highest_high'),
            stats.get('highest_high_date'),
            stats.get('opening_price'),
            stats.get('closing_price'),
            stats.get('monthly_return'),
            stats.get('max_drawdown')
        ))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    finally:
        conn.close()

def insert_monthly_metrics(db_path: str, month: str, metrics: Dict):
    """Insert monthly summary metrics"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
        INSERT INTO monthly_metrics
        (month, monthly_return, sharpe_ratio, alpha, beta, max_drawdown, win_rate)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            month,
            metrics.get('monthly_return'),
            metrics.get('sharpe_ratio'),
            metrics.get('alpha'),
            metrics.get('beta'),
            metrics.get('max_drawdown'),
            metrics.get('win_rate')
        ))
        conn.commit()
        print(f"✅ Monthly metrics inserted for {month}")
    except sqlite3.IntegrityError:
        print(f"⚠️ Monthly metrics for {month} already exist")
    finally:
        conn.close()

def get_monthly_metrics(db_path: str, months: int = 12) -> pd.DataFrame:
    """Retrieve monthly metrics for last N months"""
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT * FROM monthly_metrics
    ORDER BY month DESC
    LIMIT {months}
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df.sort_values('month')

# ===== REBALANCE TRACKING (BEFORE & AFTER MARKET) =====
def save_before_open_snapshot(report_dir: str, rebalance_date: str, 
                             portfolio_value: float, holdings: Dict[str, Dict]):
    """Save pre-market open snapshot on rebalance day"""
    filename = f"{report_dir}/rebalance/{rebalance_date}_before_open.json"
    
    report = {
        "rebalance_date": rebalance_date,
        "snapshot_type": "before_open",
        "timestamp": datetime.now().isoformat(),
        "portfolio_value": portfolio_value,
        "holdings": holdings
    }
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✅ Pre-market snapshot saved: {filename}")

def save_after_close_snapshot(report_dir: str, rebalance_date: str,
                             portfolio_value: float, holdings: Dict[str, Dict],
                             stock_changes: List[Dict]):
    """Save post-market close snapshot on rebalance day"""
    filename = f"{report_dir}/rebalance/{rebalance_date}_after_close.json"
    
    # Load before_open snapshot to compare
    before_file = f"{report_dir}/rebalance/{rebalance_date}_before_open.json"
    try:
        with open(before_file, 'r') as f:
            before_data = json.load(f)
        before_value = before_data['portfolio_value']
    except:
        before_value = portfolio_value
    
    portfolio_change_pct = ((portfolio_value - before_value) / before_value * 100) if before_value > 0 else 0
    
    report = {
        "rebalance_date": rebalance_date,
        "snapshot_type": "after_close",
        "timestamp": datetime.now().isoformat(),
        "portfolio_value": portfolio_value,
        "portfolio_change_pct": portfolio_change_pct,
        "holdings": holdings,
        "stock_changes": stock_changes
    }
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✅ Post-market snapshot saved: {filename}")
    
    # Also store in database
    insert_rebalance_event("data/portfolio_metrics.db", rebalance_date, 
                          before_value, portfolio_value, portfolio_change_pct)

def insert_rebalance_event(db_path: str, rebalance_date: str,
                          pre_open_value: float, post_close_value: float,
                          change_pct: float):
    """Insert rebalance event into database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
        INSERT INTO rebalance_events
        (rebalance_date, pre_open_portfolio_value, post_close_portfolio_value, 
         pre_open_time, post_close_time, portfolio_change_pct)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (rebalance_date, pre_open_value, post_close_value,
              datetime.now().replace(hour=9, minute=30).isoformat(),
              datetime.now().replace(hour=16, minute=0).isoformat(),
              change_pct))
        conn.commit()
        print(f"✅ Rebalance event recorded for {rebalance_date}")
    except sqlite3.IntegrityError:
        print(f"⚠️ Rebalance event for {rebalance_date} already exists")
    finally:
        conn.close()

def insert_rebalance_stock_change(db_path: str, rebalance_date: str, ticker: str,
                                  pre_price: float, post_price: float,
                                  pre_weight: float, post_weight: float):
    """Insert stock change during rebalance"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    price_change_pct = ((post_price - pre_price) / pre_price) * 100 if pre_price > 0 else 0
    weight_change_pct = ((post_weight - pre_weight) / pre_weight) * 100 if pre_weight > 0 else 0
    
    try:
        cursor.execute('''
        INSERT INTO rebalance_stock_changes
        (rebalance_date, ticker, pre_open_price, post_close_price, 
         pre_open_weight, post_close_weight, price_change_pct, weight_change_pct)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (rebalance_date, ticker, pre_price, post_price, pre_weight, post_weight,
              price_change_pct, weight_change_pct))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    finally:
        conn.close()

def get_rebalance_history(db_path: str) -> pd.DataFrame:
    """Get all rebalance events"""
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM rebalance_events ORDER BY rebalance_date DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# ===== MONTHLY REPORTS =====
def save_monthly_report(report_dir: str, month: str, summary: Dict):
    """Save monthly metrics as JSON"""
    filename = f"{report_dir}/monthly/{month}.json"
    
    report = {
        "month": month,
        "timestamp": datetime.now().isoformat(),
        "summary": summary
    }
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✅ Monthly report saved: {filename}")

def load_json_report(filename: str) -> Dict:
    """Load JSON report"""
    with open(filename, 'r') as f:
        return json.load(f)
