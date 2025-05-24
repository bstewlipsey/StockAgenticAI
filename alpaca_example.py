import alpaca_trade_api as tradeapi
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY

# Base URL for paper trading
base_url = "https://paper-api.alpaca.markets"

# Initialize the Alpaca API client
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url)

# Now you can use the 'api' object to interact with the Alpaca API.
# For example, to get your account information:
try:
    account = api.get_account()
    print(account)
except Exception as e:
    print(f"Error getting account information: {e}")

# Example: To get a quote for a stock (e.g., Apple - AAPL)
try:
    quote = api.get_last_quote("AAPL")
    print(quote)
except Exception as e:
    print(f"Error getting quote: {e}")

# Example: Submit a market order to buy 1 share of AAPL
try:
    api.submit_order(
        symbol="AAPL",
        qty=1,
        side="buy",
        type="market",
        time_in_force="gtc"
    )
    print("Market order submitted to buy 1 share of AAPL")
except Exception as e:
    print(f"Error submitting order: {e}")