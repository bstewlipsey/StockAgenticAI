"""
Trading Strategy Configuration

This file contains the settings that control your trading strategy.
These are the values you'll want to adjust as you fine-tune your trading approach.
Think of these as the 'knobs and dials' that control how aggressive or conservative your trading is.

NOTE: All variables in this file are referenced by the trading system, bots, or tests. Avoid duplicating system/API variables from config.py. Keep only trading/strategy variables here for clarity and maintainability.
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
# ANALYSIS_SCHEMA and STOCK_ANALYSIS_TEMPLATE are now defined in config_system.py for system-wide use.

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
# All variables above are used by the trading system. If you add new strategy variables, document them and ensure they are not duplicated in config.py.
