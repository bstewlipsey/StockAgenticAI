# filepath: stock_agent/config_system.py

# === Alpaca Endpoint Documentation ===
# Trading endpoint: https://paper-api.alpaca.markets (for placing trades)
# Market data endpoint: https://data.alpaca.markets (for historical/streaming data)
# Note: SIP (full market) data requires a paid subscription (Algo Trader Plus or higher).
#       IEX (free) data is limited and may return empty/incomplete bars for many stocks.
#       If you only have IEX access, you can specify feed="iex" in API requests, but data will be limited.

import os
from dotenv import load_dotenv
import logging.config

# Load environment variables
load_dotenv()

# === API Keys and Environment Configuration ===
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
GEMINI_API_KEY_1 = os.getenv("GEMINI_API_KEY_1")  # Over quota
GEMINI_API_KEY_2 = os.getenv("GEMINI_API_KEY_2")  # Over quota
GEMINI_API_KEY_3 = os.getenv("GEMINI_API_KEY_3")  # Fresh, unused
GEMINI_API_KEYS = [GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3]
PAPER_TRADING = os.getenv('PAPER_TRADING', 'True').lower() == 'true'

# Add your News API key here for NewsRetrieverBot
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Determine Alpaca API base URL based on paper trading status
if PAPER_TRADING:
    ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Alpaca paper trading endpoint
else:
    ALPACA_BASE_URL = "https://api.alpaca.markets"      # Alpaca live trading endpoint

# Alpaca data feed (IEX is free, SIP/Polygon require paid subscription)
ALPACA_DATA_FEED = "iex"

# === System Settings ===
# API and Connection Settings
MAX_RETRIES = 3                     # Number of times to retry if an API call fails
RETRY_DELAY = 5                     # Seconds to wait between retries (standard default)
RATE_LIMIT_DELAY_SECONDS = 3        # Seconds to wait between API calls (e.g., for ccxt rate limiting)
ERROR_RETRY_DELAY = 60              # Seconds to wait after an error before retrying a major operation

# Set a higher interval to avoid Gemini quota exhaustion during testing
# === TEST MODE AND TRADING INTERVAL (Section 2.3 Paper Trading Test) ===
TEST_MODE_ENABLED = True  # Section 2.3: Ensure paper trading mode is enabled for debugging
TRADING_CYCLE_INTERVAL = 1800  # Section 6.1: 30 minutes, quota-safe for multiple assets

# Remove or comment out the quick test override to enforce quota-safe interval
# if 'TEST_MODE_ENABLED' in globals() and TEST_MODE_ENABLED:
#     TRADING_CYCLE_INTERVAL = 5  # 5 seconds for rapid test mode

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
DEFAULT_CRYPTO_TIMEFRAME = ("1", "h")   # (value, unit) for Kraken/CCXT or similar
LOOKBACK_PERIOD = 100         # Number of candlesticks to analyze

# === LLM (AI) Configuration ===
GEMINI_MODEL = "gemini-1.5-flash"  # Which Gemini model to use (updated to working model)
MAX_TOKENS = 500             # Lowered for quota conservation
TEMPERATURE = 0.5            # Lowered for more deterministic, shorter responses

# Enable or disable continuous trading bot loop
ENABLE_TRADING_BOT = False   # Set to False to disable continuous trading

# === AI Analysis Templates (System-wide) ===
ANALYSIS_SCHEMA = """{
    "action": "buy"|"sell"|"hold",
    "reasoning": "<1 brief sentence>",
    "confidence": <number between 0.0 and 1.0>
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

---
Your primary goal is to identify actionable trading opportunities. If conditions warrant, recommend BUY or SELL. Only recommend HOLD if there is truly no actionable signal or if all indicators are ambiguous. HOLD is only acceptable if there is significant uncertainty or no clear signal. If in doubt, prefer BUY or SELL with reasoning and a confidence estimate. Be explicit about profit opportunities or risk mitigation needs.

IMPORTANT: If all indicators are present and not directly contradictory, you MUST choose BUY or SELL. If you are unsure, default to BUY or SELL with a low confidence, not HOLD. HOLD is only allowed if data is missing or all indicators are truly ambiguous. If you output HOLD when there is any actionable signal, you are making a mistake. If you are not sure, you must still choose BUY or SELL with low confidence.

# FEW-SHOT EXAMPLES:
Example 1:
{"action": "buy", "reasoning": "Strong bullish momentum, RSI not overbought, MACD positive", "confidence": 0.82}
Example 2:
{"action": "sell", "reasoning": "Bearish divergence, RSI overbought, price below SMA20", "confidence": 0.76}
Example 3:
{"action": "buy", "reasoning": "MACD crossover, high volume, price above SMA20", "confidence": 0.79}
Example 4:
{"action": "sell", "reasoning": "Sharp drop in price, negative news, RSI falling", "confidence": 0.81}
Example 5:
{"action": "buy", "reasoning": "Indicators mixed but no clear reason to hold, so buying with low confidence", "confidence": 0.32}
Example 6:
{"action": "hold", "reasoning": "No clear trend, mixed indicators, and missing data", "confidence": 0.28}
"""

# Template for stock analysis prompts
STOCK_ANALYSIS_TEMPLATE = """Analyze this stock data and respond ONLY with valid JSON matching this schema exactly:
{schema}

Stock Data:
Symbol: {symbol}
Current Price: ${current_price}
Position: {position_qty} shares
Entry Price: {entry_price}
Price Change: {price_change:+.2f}%
Volume Activity: {volume_ratio:.1f}x average

Market Context:
Overall Market: {market_context}
S&P 500 Change: {market_change:+.2f}%

Technical Indicators:
RSI (14): {rsi:.2f}
MACD: {macd:.2f}
SMA20: ${sma_20:.2f}
Price vs SMA20: {sma_diff_prefix}{sma_diff:.2f}

Technical Trading Signals:
{signal_summary}

---
Your primary goal is to identify actionable trading opportunities. If conditions warrant, recommend BUY or SELL. Only recommend HOLD if there is truly no actionable signal or if all indicators are ambiguous. HOLD is only acceptable if there is significant uncertainty or no clear signal. If in doubt, prefer BUY or SELL with reasoning and a confidence estimate. Be explicit about profit opportunities or risk mitigation needs.

IMPORTANT: If all indicators are present and not directly contradictory, you MUST choose BUY or SELL. If you are unsure, default to BUY or SELL with a low confidence, not HOLD. HOLD is only allowed if data is missing or all indicators are truly ambiguous. If you output HOLD when there is any actionable signal, you are making a mistake. If you are not sure, you must still choose BUY or SELL with low confidence.

# FEW-SHOT EXAMPLES:
Example 1:
{"action": "buy", "reasoning": "Breakout above resistance, strong volume, RSI rising", "confidence": 0.88}
Example 2:
{"action": "sell", "reasoning": "Bearish reversal, MACD negative, RSI overbought", "confidence": 0.73}
Example 3:
{"action": "buy", "reasoning": "Golden cross, high volume, positive news", "confidence": 0.81}
Example 4:
{"action": "sell", "reasoning": "Earnings miss, price below SMA20, negative sentiment", "confidence": 0.77}
Example 5:
{"action": "buy", "reasoning": "Indicators mixed but no clear reason to hold, so buying with low confidence", "confidence": 0.29}
Example 6:
{"action": "hold", "reasoning": "Sideways price action, no strong signals, and missing data", "confidence": 0.22}
"""

# === Prompt Experimentation Flag ===
USE_EXPERIMENTAL_PROMPT = True  # Toggle to switch prompt templates for A/B testing

EXPERIMENTAL_STOCK_ANALYSIS_TEMPLATE = """Analyze this stock data and respond ONLY with valid JSON matching this schema exactly:
{schema}

Stock Data:
Symbol: {symbol}
Current Price: ${current_price}
Position: {position_qty} shares
Entry Price: {entry_price}
Price Change: {price_change:+.2f}%
Volume Activity: {volume_ratio:.1f}x average

Market Context:
Overall Market: {market_context}
S&P 500 Change: {market_change:+.2f}%

Technical Indicators:
RSI (14): {rsi:.2f}
MACD: {macd:.2f}
SMA20: ${sma_20:.2f}
Price vs SMA20: {sma_diff_prefix}{sma_diff:.2f}

Technical Trading Signals:
{signal_summary}

---
Your primary goal is to identify actionable trading opportunities. If conditions warrant, recommend BUY or SELL. Only recommend HOLD if there is truly no actionable signal or if all indicators are ambiguous. HOLD is only acceptable if there is significant uncertainty or no clear signal. If in doubt, prefer BUY or SELL with reasoning and a confidence estimate. Be explicit about profit opportunities or risk mitigation needs.

IMPORTANT: If all indicators are present and not directly contradictory, you MUST choose BUY or SELL. If you are unsure, default to BUY or SELL with a low confidence, not HOLD. HOLD is only allowed if data is missing or all indicators are truly ambiguous. If you output HOLD when there is any actionable signal, you are making a mistake. If you are not sure, you must still choose BUY or SELL with low confidence.

*** If you output HOLD, you must explain exactly why no actionable trade is possible, and you will be penalized for unjustified HOLDs. If there is any plausible BUY or SELL case, you must choose it, even with low confidence. ***

# FEW-SHOT EXAMPLES:
Example 1:
{"action": "buy", "reasoning": "Breakout above resistance, strong volume, RSI rising", "confidence": 0.88}
Example 2:
{"action": "sell", "reasoning": "Bearish reversal, MACD negative, RSI overbought", "confidence": 0.73}
Example 3:
{"action": "buy", "reasoning": "Golden cross, high volume, positive news", "confidence": 0.81}
Example 4:
{"action": "sell", "reasoning": "Earnings miss, price below SMA20, negative sentiment", "confidence": 0.77}
Example 5:
{"action": "buy", "reasoning": "Indicators mixed but no clear reason to hold, so buying with low confidence", "confidence": 0.29}
Example 6:
{"action": "sell", "reasoning": "MACD negative, but RSI neutral and price above SMA20, so selling with low confidence", "confidence": 0.31}
Example 7:
{"action": "buy", "reasoning": "No strong signal, but price above SMA20 and positive news, so buying with low confidence", "confidence": 0.27}
Example 8:
{"action": "hold", "reasoning": "All indicators are ambiguous and data is missing, so no actionable trade is possible", "confidence": 0.18}
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
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        },
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'formatter': 'standard',
            'filename': 'trading.log',
            'mode': 'a',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    },
}

# Initialize logging
logging.config.dictConfig(LOGGING_CONFIG)

# Check Python version
import sys
print("Python version:", sys.version)

def validate_startup_config():
    import sys
    import types
    import logging
    logger = logging.getLogger(__name__)
    errors = []
    # Validate config_system.py essentials
    try:
        import config_system as cs
        if not cs.ALPACA_API_KEY or not cs.ALPACA_SECRET_KEY:
            errors.append('Missing Alpaca API keys.')
        if not cs.GEMINI_API_KEYS or not any(cs.GEMINI_API_KEYS):
            errors.append('No Gemini API keys configured.')
        if not isinstance(cs.TRADING_CYCLE_INTERVAL, int) or cs.TRADING_CYCLE_INTERVAL <= 0:
            errors.append('TRADING_CYCLE_INTERVAL must be a positive integer.')
        if not hasattr(cs, 'ENABLE_TRADING_BOT'):
            errors.append('ENABLE_TRADING_BOT missing.')
        if not hasattr(cs, 'PAPER_TRADING'):
            errors.append('PAPER_TRADING missing.')
    except Exception as e:
        errors.append(f'config_system.py import/validation error: {e}')
    # Validate config_trading.py essentials
    try:
        import config_trading as ct
        if not hasattr(ct, 'TOTAL_CAPITAL') or ct.TOTAL_CAPITAL <= 0:
            errors.append('TOTAL_CAPITAL must be > 0.')
        if not hasattr(ct, 'TRADING_ASSETS') or not isinstance(ct.TRADING_ASSETS, list) or not ct.TRADING_ASSETS:
            errors.append('TRADING_ASSETS must be a non-empty list.')
        if not hasattr(ct, 'MIN_CONFIDENCE') or not (0 <= ct.MIN_CONFIDENCE <= 1):
            errors.append('MIN_CONFIDENCE must be between 0 and 1.')
    except Exception as e:
        errors.append(f'config_trading.py import/validation error: {e}')
    if errors:
        for err in errors:
            logger.error(f'[CONFIG VALIDATION ERROR] {err}')
        print('\n'.join(['[CONFIG VALIDATION ERROR] ' + e for e in errors]), file=sys.stderr)
        sys.exit(1)
    else:
        logger.info('Startup configuration validation passed.')

# Run validation at import
validate_startup_config()