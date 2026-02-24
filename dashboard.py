import yfinance as yf

# Function to update the holdings table

def update_holdings_table(holdings):
    for holding in holdings:
        # Fetch data from yfinance
        stock_info = yf.Ticker(holding['symbol']).info
        holding['average_buy_price'] = stock_info.get('regularMarketPreviousClose', 0)
        holding['gain_loss_percentage'] = ((stock_info['regularMarketPrice'] - holding['average_buy_price']) / holding['average_buy_price']) * 100
        holding['pe_ratio'] = stock_info.get('trailingPE', 'N/A')

        # Color coding for gain/loss percentage
        if holding['gain_loss_percentage'] > 0:
            holding['gain_loss_color'] = 'green'  # Gain
        elif holding['gain_loss_percentage'] < 0:
            holding['gain_loss_color'] = 'red'    # Loss
        else:
            holding['gain_loss_color'] = 'gray'   # No change

    return holdings

# Example usage
holdings = [
    {'symbol': 'AAPL', 'name': 'Apple', 'shares': 10},
    {'symbol': 'GOOGL', 'name': 'Alphabet', 'shares': 5},
]

updated_holdings = update_holdings_table(holdings)
# Output the updated holdings table
print(updated_holdings)