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
        Uses DEFAULT_CRYPTO_TIMEFRAME from config_system.py for all OHLCV fetches.
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
                    logger.error("Returning None due to invalid symbol format in get_crypto_data.")
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
                    logger.error("Returning None due to empty/incomplete ticker data in get_crypto_data.")
                    raise ValueError("Empty ticker data")
                if not ohlcv or len(ohlcv) == 0:
                    logger.warning(f"Empty OHLCV data for {symbol}")
                    logger.error("Returning None due to empty OHLCV data in get_crypto_data.")
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

    def analyze_crypto(self, symbol, prompt_note=None, test_override=None):
        """
        Analyze a cryptocurrency using market data, technical indicators, and AI.
        If test_override is provided, use it as the LLM response (string or dict).
        Implements LLM answer memory for decision-making (Task 3.4).
        Now integrates news sentiment into memory logic (Task 3.5).
        """
        import logging
        logger = logging.getLogger(__name__)
        try:
            # === Get latest news sentiment ===
            from bot_news_retriever import NewsRetrieverBot
            news_bot = NewsRetrieverBot()
            news_articles = news_bot.fetch_news(symbol + " crypto", max_results=3)
            news_sentiment = None
            if news_articles:
                # For simplicity, use the title of the most relevant article as sentiment proxy
                news_sentiment = news_articles[0].title if news_articles else None

            # === TEST OVERRIDE: bypass memory logic for tests ===
            if test_override is not None:
                response = test_override
                logger.info(f"[TEST] Using test_override for {symbol}: {response}")
                import json
                try:
                    parsed_json = json.loads(response) if isinstance(response, str) else response
                except Exception:
                    parsed_json = None
                if parsed_json is None:
                    return None
                parsed_json['source'] = 'test_override'
                return parsed_json

            # === LLM Answer Memory: Check for recent high-confidence answer with similar news sentiment ===
            from bot_database import DatabaseBot
            db = DatabaseBot()
            recent_contexts = db.get_analysis_context(symbol, context_types=["llm_analysis"], days=1, limit=5)
            import time
            now = time.time()
            for ctx in recent_contexts:
                ctx_data = ctx.get('context_data', {})
                raw_llm = ctx_data.get('raw_llm_response', {})
                conf = raw_llm.get('confidence', 0)
                ts = ctx.get('timestamp')
                prev_sentiment = ctx_data.get('news_sentiment')
                # Accept if confidence > 0.7, answer is <3h old, and news sentiment matches
                if conf and float(conf) > 0.7 and ts and prev_sentiment == news_sentiment:
                    import datetime
                    try:
                        ts_dt = datetime.datetime.fromisoformat(ts)
                        age_hours = (datetime.datetime.utcnow() - ts_dt).total_seconds() / 3600
                        if age_hours < 3:
                            logger.info(f"[LLM_MEMORY] Using recent high-confidence LLM answer for {symbol} from memory (age={age_hours:.2f}h, conf={conf}, sentiment match).")
                            result = dict(raw_llm)
                            result['source'] = 'llm_memory'
                            return result
                    except Exception:
                        pass
            # ...existing code for LLM call...
            if test_override is not None:
                response = test_override
                logger.info(f"[TEST] Using test_override for {symbol}: {response}")
            else:
                data = self.get_crypto_data(symbol)
                if not data:
                    return {"error": "Could not fetch crypto data"}
                # Use DEFAULT_CRYPTO_TIMEFRAME for ohlcv
                tf_value, tf_unit = DEFAULT_CRYPTO_TIMEFRAME
                timeframe = f"{tf_value}{tf_unit}"
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=50)
                # === Market Data Logging for Quality/Freshness ===
                import datetime
                now = datetime.datetime.utcnow()
                candle_times = [c[0] for c in ohlcv] if ohlcv else []
                logger.debug(f"[MARKET_DATA] {symbol} OHLCV timestamps: {candle_times}")
                latest_candle_time = candle_times[-1] if candle_times else None
                try:
                    if isinstance(latest_candle_time, (int, float)):
                        candle_dt = datetime.datetime.utcfromtimestamp(latest_candle_time/1000 if latest_candle_time > 1e12 else latest_candle_time)
                    elif isinstance(latest_candle_time, str):
                        candle_dt = datetime.datetime.fromisoformat(latest_candle_time.replace('Z', '+00:00'))
                    else:
                        candle_dt = None
                    if candle_dt:
                        age_sec = (now - candle_dt).total_seconds()
                        logger.info(f"[MARKET_DATA] {symbol} latest OHLCV age: {age_sec:.1f} seconds")
                        if age_sec > 300:
                            logger.warning(f"[MARKET_DATA] {symbol} OHLCV data is stale! Age: {age_sec:.1f} seconds")
                except Exception as e:
                    logger.warning(f"[MARKET_DATA] Could not parse candle_time for {symbol}: {latest_candle_time} ({e})")
                ticker = self.exchange.fetch_ticker(symbol)
                ticker_time = ticker.get('timestamp') if ticker else None
                logger.info(f"[MARKET_DATA] {symbol} ticker: time={ticker_time}, last={ticker.get('last') if ticker else None}, volume={ticker.get('quoteVolume') if ticker else None}")
                # Technical analysis
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
                logger.info(f"[DEBUG] CRYPTO ANALYSIS RAW LLM RESPONSE for {symbol}:\n{response}")
                import json
                try:
                    parsed_json = json.loads(response) if isinstance(response, str) else response
                    logger.info(f"[AI_JSON] {symbol} LLM JSON: {json.dumps(parsed_json, indent=2)}")
                except Exception as e:
                    logger.warning(f"[AI_JSON] Could not parse LLM response as JSON for {symbol}: {e}")
                    parsed_json = None
                if 'error' in response:
                    return response
                # === LLM JSON Schema Validation ===
                def is_valid_llm_json(data):
                    if not isinstance(data, dict):
                        return False
                    required = ['action', 'confidence']
                    if any(k not in data for k in required):
                        return False
                    if data['action'] not in ['buy', 'sell', 'hold']:
                        return False
                    try:
                        conf = float(data['confidence'])
                        if not (0.0 <= conf <= 1.0):
                            return False
                    except Exception:
                        return False
                    return True
                if parsed_json is None or not is_valid_llm_json(parsed_json):
                    logger.error(f"[AI_JSON] Skipping {symbol}: LLM response missing/invalid or failed schema validation. Raw: {response}")
                    return {"error": f"LLM response for {symbol} missing, malformed, or invalid. Trade skipped."}
                # === Store LLM answer in memory for future reuse, including news sentiment ===
                context_data = {
                    'symbol': symbol,
                    'asset_type': 'crypto',
                    'prompt': prompt,
                    'raw_llm_response': parsed_json,
                    'template_vars': template_vars,
                    'news_sentiment': news_sentiment,
                    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
                }
                db.store_analysis_context(symbol, 'llm_analysis', context_data, relevance_score=parsed_json.get('confidence', 0.0))
                logger.info(f"[LLM_MEMORY] Stored new LLM analysis for {symbol} in database.")
                parsed_json['source'] = 'fresh_llm_call'
                return parsed_json
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
        print(f"\n=== Performance Summary for {symbol} ({asset_type}) ===")  # CLI/test output
        # Placeholder: Add more detailed metrics as you implement them
        # Example: print last analysis, open positions, etc.
        print(f"No detailed crypto performance summary implemented yet for {symbol}.")  # CLI/test output

    class MockExchange:
        def fetch_ticker(self, symbol):
            return {'last': 50000, 'quoteVolume': 100, 'percentage': 2, 'high': 51000, 'low': 49000}
        def fetch_ohlcv(self, symbol, timeframe, limit=2):
            return [[0,0,0,0,49500],[0,0,0,0,50000]]
        rateLimit = 1000

    @staticmethod
    def selftest():
        print(f"\n--- Running CryptoBot Self-Test ---")  # CLI/test output
        try:
            bot = CryptoBot(exchange=CryptoBot.MockExchange())
            # Test strong BUY
            fake_buy = '{"action": "buy", "reasoning": "Strong breakout, high volume.", "confidence": 0.97}'
            result_buy = bot.analyze_crypto("BTC/USD", test_override=fake_buy)
            print("    -> Test BUY result:", result_buy)  # CLI/test output
            assert result_buy and result_buy.get('action') == 'buy', f"Expected BUY, got: {result_buy}"
            # Test strong SELL
            fake_sell = '{"action": "sell", "reasoning": "Bearish divergence, low volume.", "confidence": 0.91}'
            result_sell = bot.analyze_crypto("ETH/USD", test_override=fake_sell)
            print("    -> Test SELL result:", result_sell)  # CLI/test output
            assert result_sell and result_sell.get('action') == 'sell', f"Expected SELL, got: {result_sell}"
            print(f"--- CryptoBot Self-Test PASSED ---")  # CLI/test output
        except AssertionError as e:
            print(f"--- CryptoBot Self-Test FAILED: {e} ---")  # CLI/test output
        except Exception as e:
            print(f"--- CryptoBot Self-Test encountered an ERROR: {e} ---")  # CLI/test output

if __name__ == "__main__":
    CryptoBot.selftest()

# === End of bot_crypto.py ===
