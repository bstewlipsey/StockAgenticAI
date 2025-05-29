from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
import os
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_SECRET_KEY')

# Test stock data fetch for MSFT
def fetch_msft_bars():
    client = StockHistoricalDataClient(API_KEY, API_SECRET)
    timeframe = TimeFrame(1, TimeFrameUnit.Day)
    request_params = StockBarsRequest(
        symbol_or_symbols="MSFT",
        timeframe=timeframe,
        limit=5,
        feed=DataFeed.IEX  # Use free IEX feed
    )
    bars = client.get_stock_bars(request_params)
    print("MSFT bars:", bars)

client = CryptoHistoricalDataClient(API_KEY, API_SECRET)

tf_value = 1
# For daily bars, use 'Day' as the unit
# This matches the pattern: TimeFrame(tf_value, TimeFrameUnit[tf_unit])
timeframe = TimeFrame(tf_value, TimeFrameUnit.Day)

request_params = CryptoBarsRequest(
    symbol_or_symbols="BTC/USD",
    timeframe=timeframe,
    limit=5
)

bars = client.get_crypto_bars(request_params)
print(bars)

if __name__ == "__main__":
    fetch_msft_bars()
