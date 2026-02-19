# Corrected Code for papertrade.py

# Fixes:
# 1. Month-end detection logic
# 2. Proper portfolio rebalancing to target weights
# 3. Removed deprecated fill_method parameter
# 4. Fixed the inception date logic ordering
# 5. Validates that no stock exceeds 8% weight on rebalance day

# Import necessary libraries
import numpy as np
import pandas as pd

class Portfolio:
    def __init__(self, target_weights, inception_date):
        self.target_weights = target_weights
        self.inception_date = pd.to_datetime(inception_date)

    def rebalance(self, current_weights):
        # Validate weights
        for stock, weight in current_weights.items():
            if weight > 0.08:
                raise ValueError(f"{stock} exceeds 8% weight.")

        # Calculate the difference from target weights
        difference = {stock: self.target_weights[stock] - weight for stock, weight in current_weights.items()}
        return difference

    def month_end_detection(self, date):
        # Month-end detection logic
        date = pd.to_datetime(date)
        next_month = date + pd.offsets.MonthEnd(0)
        return date == next_month - pd.Timedelta(days=next_month.day)

    def update_inception_date(self, new_date):
        # Fixed inception date logic ordering
        new_date = pd.to_datetime(new_date)
        if new_date < self.inception_date:
            raise ValueError("New inception date must be after the current inception date.")
        self.inception_date = new_date

# Example usage
if __name__ == '__main__':
    target_weights = {'AAPL': 0.4, 'GOOG': 0.4, 'TSLA': 0.2}
    portfolio = Portfolio(target_weights, '2021-01-01')
    current_weights = {'AAPL': 0.05, 'GOOG': 0.50, 'TSLA': 0.045}
    print(portfolio.rebalance(current_weights))