import pandas as pd

# Assuming the relevant parts of the code are present in papertrade.py

def process_data(closes_dkk, window_prices, loc, dt, cols_current, LOOKBACK):
    # Original line
    rets = closes_dkk.pct_change(fill_method=None)  # Updated line
    
    # Some processing logic here
    
    # Original line
    window_rets_full = window_prices.pct_change(fill_method=None)  # Updated line
    
    # More processing logic
    window_prices = closes_dkk.iloc[loc - LOOKBACK:loc]  # Updated line
    
    # Logic related to today's returns
    r_today = rets.loc[dt].reindex(cols_current).values  # Updated line

# ... Other code 

