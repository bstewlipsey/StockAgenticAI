# filepath: stock_agent/config.py
import os
from dotenv import load_dotenv
import logging.config

# Load environment variables
load_dotenv()

# === API Keys and Environment Configuration ===
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PAPER_TRADING = os.getenv('PAPER_TRADING', 'True').lower() == 'true'

# Determine Alpaca API base URL based on paper trading status
if PAPER_TRADING:
    ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Alpaca paper trading endpoint
else:
    ALPACA_BASE_URL = "https://api.alpaca.markets"      # Alpaca live trading endpoint

# === System Settings ===
# API and Connection Settings
MAX_RETRIES = 3                     # Number of times to retry if an API call fails
RETRY_DELAY = 5                     # Seconds to wait between general retries
RATE_LIMIT_DELAY_SECONDS = 3        # Seconds to wait between API calls (e.g., for ccxt rate limiting)
ERROR_RETRY_DELAY = 60              # Seconds to wait after an error before retrying a major operation

# === Retry Strategy Configuration ===
BASE_RETRY_DELAY = 5                # Base delay in seconds for exponential backoff
MAX_RETRY_BACKOFF_DELAY = 300       # Maximum delay in seconds for exponential backoff
JITTER_DELAY = 1                    # Maximum jitter in seconds to add to retry delay

# === Market Data Configuration ===
# For Alpaca, valid TimeFrameUnit enum names are: 'Minute', 'Hour', 'Day', 'Week', 'Month'
# For Kraken/CCXT, valid timeframes are typically: '1m', '5m', '15m', '1h', '4h', '1d', etc.
# Example enum-like usage:
#   STOCK: DEFAULT_STOCK_TIMEFRAME = (1, "Day")   # 1 day bars
#   CRYPTO: DEFAULT_CRYPTO_TIMEFRAME = ("1", "d") # 1 day bars ("m"=minute, "h"=hour, "d"=day)
#   # To use: timeframe = f"{value}{unit}"  # e.g., "1d", "15m"
DEFAULT_STOCK_TIMEFRAME = (1, "Day")       # (value, unit) for Alpaca TimeFrame
DEFAULT_CRYPTO_TIMEFRAME = ("1", "d")   # (value, unit) for Kraken/CCXT or similar
LOOKBACK_PERIOD = 100         # Number of candlesticks to analyze
TRADING_CYCLE_INTERVAL = 60  # How often to check for trades (1 minute for testing)

# === LLM (AI) Configuration ===
GEMINI_MODEL = "gemini-1.5-flash"  # Which Gemini model to use (updated to working model)
MAX_TOKENS = 1000             # Maximum tokens for AI responses
TEMPERATURE = 0.7             # AI creativity level (0.0 to 1.0)

# Enable or disable continuous trading bot loop
ENABLE_TRADING_BOT = False   # Set to False to disable continuous trading

# === AI Analysis Templates ===
# Templates for generating prompts for the AI model
# These control how we ask the AI to analyze different types of assets

# Base JSON schema that all analyses must follow
ANALYSIS_SCHEMA = """{
    \"action\": \"buy\"|\"sell\"|\"hold\",
    \"reasoning\": \"<1 brief sentence>\",
    \"confidence\": <number between 0.0 and 1.0>
}"""

# Template for cryptocurrency analysis prompts
CRYPTO_ANALYSIS_TEMPLATE = """Analyze this cryptocurrency and respond ONLY with valid JSON matching this schema exactly:
{schema}

Crypto Data:
Symbol: {symbol}
Current Price: ${current_price}
24h Volume: {volume}
24h Change: {price_change}%
24h High: ${high_24h}
24h Low: ${low_24h}

Technical Indicators:
RSI (14): {rsi:.2f}
MACD: {macd:.2f}
SMA20: ${sma_20:.2f}
Price vs SMA20: {sma_diff_prefix}{sma_diff:.2f}

Technical Trading Signals:
{signal_summary}
"""

# Template for stock analysis prompts
STOCK_ANALYSIS_TEMPLATE = """Analyze this stock data and respond ONLY with valid JSON matching this schema exactly:
{schema}

Stock Data:
Symbol: {symbol}
Current Price: ${current_price}
Position: {position_qty} shares
Entry Price: ${entry_price}
Price Change: {price_change:+.2f}%
Volume Activity: {volume_ratio:.1f}x average

Market Context:
Overall Market: {market_context}
S&P 500 Change: {market_change:+.2f}%

Technical Indicators:
RSI (14): {rsi:.2f}
MACD: {macd:.2f}
SMA20: ${sma_20:.2f}
EMA20: ${ema_20:.2f}
Bollinger Bands: ${bb_low:.2f} - ${bb_high:.2f}

Technical Signals:
{signal_summary}

Risk Metrics:
Investment: ${investment:.2f}
Current Value: ${current_value:.2f}
P&L: {pnl:+.2f}%
Risk Level: {risk_level}
"""

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'trading.log',
            'formatter': 'standard',
            'level': 'INFO',
        },
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO',
        },
    },
    'root': {
        'handlers': ['file', 'console'],
        'level': 'INFO',
    },
}

# Initialize logging
logging.config.dictConfig(LOGGING_CONFIG)

# Check Python version
import sys
print("Python version:", sys.version)