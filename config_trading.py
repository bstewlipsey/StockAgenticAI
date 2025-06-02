"""
Trading Strategy Configuration

This file contains the settings that control your trading strategy.
These are the values you'll want to adjust as you fine-tune your trading approach.
Think of these as the 'knobs and dials' that control how aggressive or conservative your trading is.

NOTE: All variables in this file are referenced by the trading system, bots, or tests. Avoid duplicating system/API variables from config_system.py. Keep only trading/strategy variables here for clarity and maintainability.
"""

# =========================
# [ASSET_SETTINGS]
# =========================
TRADING_ASSETS = [
    ("AAPL", "stock", 500),
]  # Reduced to 1 asset for Section 6.1 test cycle
DEFAULT_TRADE_AMOUNT_USD = 500

# =========================
# [RISK_MANAGEMENT]
# =========================
TOTAL_CAPITAL = 100000.0
MAX_PORTFOLIO_RISK = 0.02
MAX_POSITION_SIZE = 0.02
MAX_POSITION_RISK = 0.01  # 1% per position for realistic risk
STOCK_STOP_LOSS_PCT = 0.02
CRYPTO_STOP_LOSS_PCT = 0.05
MIN_CONFIDENCE = (
    0.1  # Aggressively lowered for HOLD debugging (restore to 0.3+ for production)
)
HOLD_OVERRIDE_THRESHOLD = 0.7  # If AI confidence is below this and action is HOLD, override to BUY/SELL. Documented: If AI returns HOLD with confidence < threshold, system may override to a more decisive action. Adjust as needed for production.

# =========================
# [POSITION_SIZING]
# =========================
MAX_POSITION_SIZE = 0.02
MIN_TRADE_VALUE = 1.0  # Lowered for test mode to ensure trades can be placed
MIN_TRADE_VALUE_CRYPTO = 1.0  # Lowered for test mode to ensure trades can be placed
MAX_ASSET_ALLOCATION = 0.2
MAX_ASSET_ALLOCATION_CRYPTO = 0.1
DAILY_RISK_LIMIT = 0.05
DAILY_RISK_LIMIT_CRYPTO = 0.1
TRANSACTION_FEES = 0.001
TRANSACTION_FEES_CRYPTO = 0.002

# =========================
# [TECHNICAL_INDICATORS]
# =========================
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65
SMA_WINDOW = 10

# =========================
# [LLM_STRATEGY]
# =========================
LLM_TIMEFRAME_PROMPT_EXAMPLES = """
- For Alpaca, 1 hour bars: '1H'
- For Kraken, 1 hour bars: '1h'
- For Alpaca, 1 day bars: '1D'
- For Kraken, 1 day bars: '1d'
- For Alpaca, 1 minute bars: '1Min'
- For Kraken, 1 minute bars: '1m'
"""

# --- DecisionMakerBot Parameters ---
CONFIDENCE_BOOST_FACTORS = {"news": 1.2, "technical": 1.1, "fundamental": 1.15}
CONFIDENCE_BOOST_FACTORS_CRYPTO = {"news": 1.1, "technical": 1.05, "on_chain": 1.2}
RISK_PENALTY_FACTORS = {"volatility": 0.9, "drawdown": 0.85}
RISK_PENALTY_FACTORS_CRYPTO = {"volatility": 0.7, "drawdown": 0.8, "illiquidity": 0.7}
FILTERS_ENABLED = {"min_confidence": True, "max_risk": True}
FILTERS_ENABLED_CRYPTO = {"min_confidence": True, "max_risk": True, "min_volume": True}

# =========================
# [SCREENING]
# =========================
MIN_MARKET_CAP = 1_000_000_000
MIN_MARKET_CAP_CRYPTO = 100_000_000
MIN_VOLUME = 1_000_000
MIN_VOLUME_CRYPTO = 100_000

# --- ReflectionBot Parameters ---
MIN_REFLECTION_PNL = 10.0
MAX_REFLECTION_AGE_DAYS = 7

# --- End of Configuration ---
# All variables above are used by the trading system. If you add new strategy variables, document them and ensure they are not duplicated in config_system.py.
