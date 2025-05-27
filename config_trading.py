"""
Trading Strategy Configuration

This file contains the settings that control your trading strategy.
These are the values you'll want to adjust as you fine-tune your trading approach.
Think of these as the 'knobs and dials' that control how aggressive or conservative your trading is.
"""

# === Portfolio Risk Settings ===
TOTAL_CAPITAL = 100000.0

# === Assets to Trade & Per-Asset USD Allocation (Combined) ===
TRADING_ASSETS = [
    ("AAPL",    "stock", 500),
    ("BTC/USD", "crypto", 200)
]

DEFAULT_TRADE_AMOUNT_USD = 500

# Risk Management Settings
MAX_PORTFOLIO_RISK = 0.02
MAX_POSITION_SIZE = 0.02
MAX_POSITION_RISK = 0.01
STOCK_STOP_LOSS_PCT = 0.02
CRYPTO_STOP_LOSS_PCT = 0.05

# === Trading Strategy Settings ===
MIN_CONFIDENCE = 0.7
TRADING_CYCLE_INTERVAL = 5  # seconds

# === Technical Indicator Thresholds ===
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
SMA_WINDOW = 20

# === AI Analysis Templates ===
ANALYSIS_SCHEMA = """{
    "action": "buy"|"sell"|"hold",
    "reasoning": "<1 brief sentence>",
    "confidence": <number between 0.0 and 1.0>
}"""

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

LLM_TIMEFRAME_PROMPT_EXAMPLES = """
- For Alpaca, 1 hour bars: '1H'
- For Kraken, 1 hour bars: '1h'
- For Alpaca, 1 day bars: '1D'
- For Kraken, 1 day bars: '1d'
- For Alpaca, 1 minute bars: '1Min'
- For Kraken, 1 minute bars: '1m'
"""

# --- DecisionMakerBot Parameters ---
# These factors are tuned separately for stocks and crypto to avoid bias.
CONFIDENCE_BOOST_FACTORS = {
    'news': 1.2,         # Stock news boost
    'technical': 1.1,    # Stock technicals boost
    'fundamental': 1.15  # Stock fundamentals boost
}
CONFIDENCE_BOOST_FACTORS_CRYPTO = {
    'news': 1.1,         # Crypto news is more volatile
    'technical': 1.05,   # Technicals less reliable for thinly traded pairs
    'on_chain': 1.2      # On-chain signals for crypto
}
RISK_PENALTY_FACTORS = {
    'volatility': 0.9,   # Stock volatility penalty
    'drawdown': 0.85     # Stock drawdown penalty
}
RISK_PENALTY_FACTORS_CRYPTO = {
    'volatility': 0.7,   # Crypto is more volatile, penalize more
    'drawdown': 0.8,     # Crypto drawdown penalty
    'illiquidity': 0.7   # Penalize illiquid crypto pairs
}
FILTERS_ENABLED = {
    'min_confidence': True, # Enforce min confidence for stocks
    'max_risk': True        # Enforce max risk for stocks
}
FILTERS_ENABLED_CRYPTO = {
    'min_confidence': True, # Enforce min confidence for crypto
    'max_risk': True,      # Enforce max risk for crypto
    'min_volume': True     # Require min volume for crypto
}

# === Screening Parameters ===
MIN_MARKET_CAP = 1_000_000_000  # For stocks
MIN_MARKET_CAP_CRYPTO = 100_000_000  # Lower for crypto
MIN_VOLUME = 1_000_000  # For stocks
MIN_VOLUME_CRYPTO = 100_000  # Lower for crypto pairs

# === Sizing and Risk Parameters (Crypto & Stock) ===
MIN_TRADE_VALUE = 10  # Minimum trade value for stocks (USD)
MIN_TRADE_VALUE_CRYPTO = 5  # Minimum trade value for crypto (USD)
MAX_ASSET_ALLOCATION = 0.2  # Max allocation per asset (20% of capital) for stocks
MAX_ASSET_ALLOCATION_CRYPTO = 0.1  # Max allocation per asset (10% of capital) for crypto
DAILY_RISK_LIMIT = 0.05  # Max daily risk (5% of capital) for stocks
DAILY_RISK_LIMIT_CRYPTO = 0.1  # Max daily risk (10% of capital) for crypto
TRANSACTION_FEES = 0.001  # 0.1% per trade for stocks
TRANSACTION_FEES_CRYPTO = 0.002  # 0.2% per trade for crypto

# --- ReflectionBot Parameters ---
MIN_REFLECTION_PNL = 10.0
MAX_REFLECTION_AGE_DAYS = 7

# --- End of Configuration ---
