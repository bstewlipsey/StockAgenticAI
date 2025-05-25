"""
Trading Strategy Configuration

This file contains the settings that control your trading strategy.
These are the values you'll want to adjust as you fine-tune your trading approach.
Think of these as the 'knobs and dials' that control how aggressive or conservative your trading is.
"""

# === Portfolio Risk Settings ===
# Your starting capital - how much money you're trading with
# This is a paper trading amount - no real money is used
TOTAL_CAPITAL = 100000.0

# Risk Management Settings - How much you're willing to risk
# These are the most important settings for protecting your capital
# Conservative values are used by default - adjust carefully!
MAX_PORTFOLIO_RISK = 0.02     # Never risk more than 2% of total capital
MAX_POSITION_SIZE = 0.02      # No single trade uses more than 2% of capital
MAX_POSITION_RISK = 0.01      # Don't risk more than 1% on any trade

STOCK_STOP_LOSS_PCT = 0.02    # Sell if a stock trade loses 2% of its value
CRYPTO_STOP_LOSS_PCT = 0.05   # Sell if a crypto trade loses 5% of its value (Adjust as needed)

# === Trading Strategy Settings ===
# How confident should the AI be before making a trade?
# Higher = fewer but potentially better trades
# Lower = more trades but potentially more risky ones
MIN_CONFIDENCE = 0.7          # 70% confidence required for trades

# === What to Trade ===
# List of assets the bot will analyze and potentially trade
# Format: (symbol, type)
TRADING_ASSETS = [
    ("AAPL", "stock"),        # Apple Inc.
    ("MSFT", "stock"),        # Microsoft
    ("XXBTZUSD", "crypto")    # Bitcoin in USD
]

# === Technical Analysis Settings ===
# RSI (Relative Strength Index) Settings
# These help identify if a stock is overbought or oversold
RSI_OVERSOLD = 30            # Consider buying below this RSI value
RSI_OVERBOUGHT = 70          # Consider selling above this RSI value

# Moving Average Settings
# How many days to look back when calculating average price
# 20 is a common setting for short-term trading
SMA_WINDOW = 20              # Short-term trend indicator

# === AI Analysis Templates ===
# Templates for generating prompts for the AI model
# These control how we ask the AI to analyze different types of assets

# Base JSON schema that all analyses must follow
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
