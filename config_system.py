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

FILE_LOG_LEVEL = "DEBUG"

# Load environment variables
load_dotenv()

# =========================
# [API_KEYS]
# =========================
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
# Gemini API keys: Ensure these are valid and not over quota.
GEMINI_API_KEY_1 = os.getenv("GEMINI_API_KEY_1")  # Over quota
GEMINI_API_KEY_2 = os.getenv("GEMINI_API_KEY_2")  # Over quota
GEMINI_API_KEY_3 = os.getenv("GEMINI_API_KEY_3")  # Fresh, unused
GEMINI_API_KEYS = [GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3]
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# =========================
# [SYSTEM_SETTINGS]
# =========================
PAPER_TRADING = os.getenv("PAPER_TRADING", "True").lower() == "true"
if PAPER_TRADING:
    ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
else:
    ALPACA_BASE_URL = "https://api.alpaca.markets"
ALPACA_DATA_FEED = "iex"
MAX_RETRIES = 3
RETRY_DELAY = 5
RATE_LIMIT_DELAY_SECONDS = 3
ERROR_RETRY_DELAY = 60
TRADING_CYCLE_INTERVAL = 3600  # Increased to 1 hour to reduce API call frequency and avoid rate limits
BASE_RETRY_DELAY = 5
MAX_RETRY_BACKOFF_DELAY = 300
JITTER_DELAY = 1
DEFAULT_STOCK_TIMEFRAME = (1, "Day")
DEFAULT_CRYPTO_TIMEFRAME = ("1", "h")
LOOKBACK_PERIOD = 100
ENABLE_TRADING_BOT = True

# =========================
# [LLM_SETTINGS]
# =========================
GEMINI_MODEL = "gemini-1.5-flash"
MAX_TOKENS = 500
TEMPERATURE = 0.5
ANALYSIS_SCHEMA = """{\n    \"action\": \"buy\"|\"sell\"|\"hold\",\n    \"reasoning\": \"<1 brief sentence>\",\n    \"confidence\": <number between 0.0 and 1.0>\n}"""

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

# =========================
# [LOGGING_SETTINGS]
# =========================
# Remove old LOGGING_CONFIG and logging.config.dictConfig usage
# Logging is now handled by utils/logging_setup.py
LOG_FULL_PROMPT = False  # Only log full LLM prompts if True or log level is DEBUG

# =========================
# [TEST_SETTINGS]
# =========================
TEST_MODE_ENABLED = True  # Enables paper trading and test logic
SELFTEST_LIVE_API_CALLS_ENABLED = (
    False  # Controls if selftest() methods make live API calls
)


# =========================
# [QUOTA_SETTINGS]
# =========================
NEWS_API_QUOTA_PER_MINUTE = int(os.getenv("NEWS_API_QUOTA_PER_MINUTE", 20))  # Default: 20 req/min
GEMINI_API_QUOTA_PER_MINUTE = 60  # Gemini: 60 requests/minute (adjust as needed)

# =========================
# [NEWS_API_SETTINGS]
# =========================
NEWS_API_MAX_RETRIES = int(os.getenv("NEWS_API_MAX_RETRIES", 5))
NEWS_API_RETRY_DELAY = int(os.getenv("NEWS_API_RETRY_DELAY", 5))
NEWS_API_BACKOFF_FACTOR = float(os.getenv("NEWS_API_BACKOFF_FACTOR", 2.0))
NEWS_API_MAX_BACKOFF = int(os.getenv("NEWS_API_MAX_BACKOFF", 120))

# === Validate Startup Configuration ===
# (Retained from original code)
def validate_startup_config():
    import sys
    import logging

    logger = logging.getLogger(__name__)
    errors = []
    # Validate config_system.py essentials
    try:
        import config_system as cs

        if not cs.ALPACA_API_KEY or not cs.ALPACA_SECRET_KEY:
            errors.append("Missing Alpaca API keys.")
        if not cs.GEMINI_API_KEYS or not any(cs.GEMINI_API_KEYS):
            errors.append("No Gemini API keys configured.")
        if (
            not isinstance(cs.TRADING_CYCLE_INTERVAL, int)
            or cs.TRADING_CYCLE_INTERVAL <= 0
        ):
            errors.append("TRADING_CYCLE_INTERVAL must be a positive integer.")
        if not hasattr(cs, "ENABLE_TRADING_BOT"):
            errors.append("ENABLE_TRADING_BOT missing.")
        if not hasattr(cs, "PAPER_TRADING"):
            errors.append("PAPER_TRADING missing.")
    except Exception as e:
        errors.append(f"config_system.py import/validation error: {e}")
    # Validate config_trading.py essentials
    try:
        import config_trading as ct

        if not hasattr(ct, "TOTAL_CAPITAL") or ct.TOTAL_CAPITAL <= 0:
            errors.append("TOTAL_CAPITAL must be > 0.")
        if (
            not hasattr(ct, "TRADING_ASSETS")
            or not isinstance(ct.TRADING_ASSETS, list)
            or not ct.TRADING_ASSETS
        ):
            errors.append("TRADING_ASSETS must be a non-empty list.")
        if not hasattr(ct, "MIN_CONFIDENCE") or not (0 <= ct.MIN_CONFIDENCE <= 1):
            errors.append("MIN_CONFIDENCE must be between 0 and 1.")
    except Exception as e:
        errors.append(f"config_trading.py import/validation error: {e}")
    if errors:
        for err in errors:
            logger.error(f"[CONFIG VALIDATION ERROR] {err}")
        sys.exit(1)
    else:
        logger.info("Startup configuration validation passed.")


# Run validation at import
validate_startup_config()
