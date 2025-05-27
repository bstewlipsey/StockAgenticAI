# bot_crypto.py
"""
CryptoBot: Handles all crypto-specific analysis, trading, and integration with Kraken/ccxt for the trading system.
- Fetches and analyzes crypto market data
- Generates AI-driven trading signals
- Executes trades and manages crypto positions
"""

# === Imports ===
import logging
import time
import ccxt
from config_system import RATE_LIMIT_DELAY_SECONDS, MAX_RETRIES, RETRY_DELAY, ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, PAPER_TRADING, CRYPTO_ANALYSIS_TEMPLATE, ANALYSIS_SCHEMA, DEFAULT_CRYPTO_TIMEFRAME
from config_trading import (
    TRADING_ASSETS, RSI_OVERSOLD,
    DEFAULT_TRADE_AMOUNT_USD, MIN_CONFIDENCE, CRYPTO_STOP_LOSS_PCT
)
from bot_indicators import IndicatorBot
from bot_risk_manager import RiskManager, Position
from bot_trade_executor import TradeExecutorBot
from bot_ai import generate_ai_analysis

logger = logging.getLogger(__name__)

# === CryptoBot Class ===
class CryptoBot:
    """
    CryptoBot encapsulates all crypto trading logic for the system.
    - Fetches and analyzes crypto data
    - Generates AI-driven trading signals
    - Executes trades and manages crypto positions
    """
    def __init__(self, exchange=None):
        self.exchange = exchange or ccxt.kraken({'enableRateLimit': True, 'rateLimit': RATE_LIMIT_DELAY_SECONDS * 1000})
        self.risk_manager = RiskManager()
        self.executor = TradeExecutorBot(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper_trading=PAPER_TRADING)
        self.asset_type_map = {symbol: asset_type for symbol, asset_type, _ in TRADING_ASSETS}

    def get_crypto_data(self, symbol, max_retries=5):
        """
        Fetch crypto data for a given symbol using ccxt or other exchange API.
        Returns a dict with price, volume, and other metrics, or None on failure.
        Uses DEFAULT_CRYPTO_TIMEFRAME from config.py for all OHLCV fetches.
        Implements aggressive error handling, exponential backoff, and retry logic.
        Logs rate limits, empty data, and network issues.
        """
        logger = logging.getLogger(__name__)
        retries = 0
        delay = 2
        while retries < max_retries:
            try:
                time.sleep(self.exchange.rateLimit / 1000)
                if '/' not in symbol:
                    logger.warning(f"Invalid symbol format: {symbol}. Expected format: BASE/QUOTE (e.g., BTC/USD)")
                    return None
                base, quote = symbol.split('/')
                currency_mapping = {'BTC': 'XBT', 'DOGE': 'XDG', 'LUNA': 'LUNA2'}
                base = currency_mapping.get(base, base)
                if base and isinstance(base, str):
                    base_prefix = 'X' if len(base) <= 4 and base.isalpha() else ''
                else:
                    base_prefix = ''
                quote_prefix = 'Z' if quote in ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF'] else ''
                kraken_symbol = f"{base_prefix}{base}{quote_prefix}{quote}"
                try:
                    # RateLimitExceeded must be checked before NetworkError
                    try:
                        ticker = self.exchange.fetch_ticker(kraken_symbol)
                    except ccxt.RateLimitExceeded as rle:
                        logger.warning(f"ccxt rate limit exceeded for {symbol}: {rle}")
                        raise
                    except ccxt.NetworkError as ne:
                        logger.error(f"Network error fetching ticker for {symbol}: {ne}")
                        raise
                except Exception as e:
                    logger.error(f"Unknown error fetching ticker for {symbol}: {e}")
                    raise
                tf_value, tf_unit = DEFAULT_CRYPTO_TIMEFRAME
                timeframe = f"{tf_value}{tf_unit}"
                try:
                    try:
                        ohlcv = self.exchange.fetch_ohlcv(kraken_symbol, timeframe, limit=2)
                    except ccxt.RateLimitExceeded as rle:
                        logger.warning(f"ccxt rate limit exceeded for OHLCV {symbol}: {rle}")
                        raise
                    except ccxt.NetworkError as ne:
                        logger.error(f"Network error fetching OHLCV for {symbol}: {ne}")
                        raise
                except Exception as e:
                    logger.error(f"Unknown error fetching OHLCV for {symbol}: {e}")
                    raise
                if not ticker or 'last' not in ticker:
                    logger.warning(f"Empty or incomplete ticker data for {symbol}: {ticker}")
                    raise ValueError("Empty ticker data")
                if not ohlcv or len(ohlcv) == 0:
                    logger.warning(f"Empty OHLCV data for {symbol}")
                    raise ValueError("Empty OHLCV data")
                return {
                    'current_price': ticker['last'],
                    'volume': ticker.get('quoteVolume', ticker.get('baseVolume', 0)),
                    'price_change': ticker.get('percentage', 0),
                    'high_24h': ticker.get('high'),
                    'low_24h': ticker.get('low'),
                    'yesterday_close': ohlcv[0][4] if len(ohlcv) > 1 else None
                }
            except ccxt.RateLimitExceeded:
                wait_time = min(60, delay)
                logger.warning(f"Rate limit hit for {symbol}, retrying in {wait_time} seconds (attempt {retries+1}/{max_retries})...")
                time.sleep(wait_time)
            except ccxt.NetworkError as ne:
                if isinstance(ne, ccxt.ExchangeNotAvailable):
                    wait_time = min(60, delay)
                    logger.error(f"Exchange not available for {symbol}: {ne}. Retrying in {wait_time} seconds (attempt {retries+1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    wait_time = min(60, delay)
                    logger.error(f"Network error for {symbol}: {ne}. Retrying in {wait_time} seconds (attempt {retries+1}/{max_retries})...")
                    time.sleep(wait_time)
            except ccxt.InsufficientFunds as inf:
                logger.error(f"Insufficient funds error for {symbol}: {inf}. Not retrying.")
                return None
            except ccxt.OrderNotFound as onf:
                logger.error(f"Order not found error for {symbol}: {onf}. Not retrying.")
                return None
            except ValueError as ve:
                logger.error(f"Data error for {symbol}: {ve}. Not retrying.")
                return None
            except Exception as e:
                wait_time = min(60, delay)
                logger.error(f"Unexpected error fetching crypto data for {symbol}: {e}. Retrying in {wait_time} seconds (attempt {retries+1}/{max_retries})...")
                time.sleep(wait_time)
            retries += 1
        logger.error(f"Failed to fetch crypto data for {symbol} after {max_retries} retries.")
        return None

    def analyze_crypto(self, symbol, prompt_note=None):
        """
        Analyze a cryptocurrency using market data, technical indicators, and AI.
        Optionally include a prompt_note for adaptive AI learning.
        Returns the AI's analysis response or an error dict.
        Uses DEFAULT_CRYPTO_TIMEFRAME from config.py for all OHLCV fetches.
        """
        try:
            data = self.get_crypto_data(symbol)
            if not data:
                return {"error": "Could not fetch crypto data"}
            # Use DEFAULT_CRYPTO_TIMEFRAME for ohlcv
            tf_value, tf_unit = DEFAULT_CRYPTO_TIMEFRAME
            timeframe = f"{tf_value}{tf_unit}"
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=50)
            prices = [candle[4] for candle in ohlcv]
            tech_analysis = IndicatorBot(prices)
            signals, indicators = tech_analysis.get_signals()
            signal_summary = "\n".join([f"- {signal}: {reason} (Confidence: {confidence*100:.0f}%)" for signal, reason, confidence in signals])
            template_vars = {
                'schema': ANALYSIS_SCHEMA,
                'symbol': symbol,
                'current_price': data['current_price'],
                'volume': data['volume'],
                'price_change': data['price_change'],
                'high_24h': data['high_24h'],
                'low_24h': data['low_24h'],
                'rsi': indicators['rsi'],
                'macd': indicators['macd'],
                'sma_20': indicators['sma_20'],
                'sma_diff': abs(data['current_price'] - indicators['sma_20']),
                'sma_diff_prefix': '+' if data['current_price'] > indicators['sma_20'] else '-',
                'signal_summary': signal_summary
            }
            # If prompt_note is provided, append it to the prompt
            if prompt_note:
                prompt = CRYPTO_ANALYSIS_TEMPLATE + f"\n{prompt_note}"
            else:
                prompt = CRYPTO_ANALYSIS_TEMPLATE
            response = generate_ai_analysis(prompt, variables=template_vars)
            if 'error' in response:
                return response
            return response
        except Exception as e:
            return {"error": f"Error analyzing crypto: {str(e)}"}

    def monitor_positions(self):
        """
        Monitor all open crypto positions and enforce stop loss rules.
        For each open position, checks if the current price has fallen below the stop loss threshold.
        If so, triggers a sell order via the TradeExecutorBot and logs the event.
        """
        try:
            # Fetch open positions from the executor (assumes executor has get_open_positions for crypto)
            open_positions = self.executor.get_open_positions(asset_type='crypto')
            if not open_positions:
                logger.info("No open crypto positions to monitor.")
                return
            for pos in open_positions:
                symbol = pos['symbol']
                entry_price = pos['entry_price']
                quantity = pos['quantity']
                data = self.get_crypto_data(symbol)
                if not data:
                    logger.warning(f"Could not fetch data for {symbol} while monitoring positions.")
                    continue
                current_price = data['current_price']
                stop_loss_price = entry_price * (1 - CRYPTO_STOP_LOSS_PCT)
                if current_price <= stop_loss_price:
                    logger.warning(f"Stop loss triggered for {symbol}: entry={entry_price}, current={current_price}, stop={stop_loss_price}")
                    # Attempt to close the position
                    result = self.executor.close_position(symbol, quantity, asset_type='crypto')
                    logger.info(f"Closed {symbol} position: {result}")
        except Exception as e:
            logger.error(f"Error monitoring crypto positions: {e}")

    def print_performance_summary(self, symbol, asset_type, timeframe=None):
        """Prints a summary of crypto performance for logging/reporting."""
        print(f"\n=== Performance Summary for {symbol} ({asset_type}) ===")
        # Placeholder: Add more detailed metrics as you implement them
        # Example: print last analysis, open positions, etc.
        print(f"No detailed crypto performance summary implemented yet for {symbol}.")

# === End of bot_crypto.py ===
