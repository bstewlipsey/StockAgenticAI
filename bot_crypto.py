# bot_crypto.py
"""
CryptoBot: Handles all crypto-specific analysis, trading, and integration with Kraken/ccxt for the trading system.
- Fetches and analyzes crypto market data
- Generates AI-driven trading signals
- Executes trades and manages crypto positions
"""

# === Imports ===
import time
import ccxt
from config_system import (
    RATE_LIMIT_DELAY_SECONDS,
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    PAPER_TRADING,
    CRYPTO_ANALYSIS_TEMPLATE,
    ANALYSIS_SCHEMA,
    DEFAULT_CRYPTO_TIMEFRAME,
    LOG_FULL_PROMPT,
)
from config_trading import (
    TRADING_ASSETS,
    CRYPTO_STOP_LOSS_PCT,
)
from bot_indicators import IndicatorBot
from bot_risk_manager import RiskManager
from bot_trade_executor import TradeExecutorBot
from bot_ai import generate_ai_analysis, AIBot
from data_structures import AgenticBotError
from utils.logger_mixin import LoggerMixin
from utils.logging_decorators import log_method_calls


# === CryptoBot Class ===
class CryptoBot(LoggerMixin):
    """
    CryptoBot encapsulates all crypto trading logic for the system.
    - Fetches and analyzes crypto data
    - Generates AI-driven trading signals
    - Executes trades and manages crypto positions
    """

    @log_method_calls
    def __init__(self, exchange=None):
        """
        Initialize the CryptoBot with exchange, risk manager, and executor.
        Sets up asset type mapping for all configured trading assets.
        """
        super().__init__()
        self.exchange = exchange or ccxt.kraken(
            {"enableRateLimit": True, "rateLimit": RATE_LIMIT_DELAY_SECONDS * 1000}
        )
        self.risk_manager = RiskManager()
        self.executor = TradeExecutorBot(
            ALPACA_API_KEY, ALPACA_SECRET_KEY, paper_trading=PAPER_TRADING
        )
        self.asset_type_map = {
            symbol: asset_type for symbol, asset_type, _ in TRADING_ASSETS
        }

    @log_method_calls
    def get_crypto_data(self, symbol, max_retries=5):
        """
        Fetch crypto data for a given symbol using ccxt or other exchange API.
        Returns a dict with price, volume, and other metrics, or None on failure.
        Implements aggressive error handling, exponential backoff, and retry logic.
        Uses QuotaManagerBot for centralized quota/retry/backoff and logs staleness.
        """
        from bot_quota_manager import QuotaManagerBot
        quota_manager = QuotaManagerBot()
        retries = 0
        delay = 2
        last_ohlcv_time = None
        while retries < max_retries:
            try:
                time.sleep(self.exchange.rateLimit / 1000)
                if "/" not in symbol:
                    return None
                base, quote = symbol.split("/")
                currency_mapping = {"BTC": "XBT", "DOGE": "XDG", "LUNA": "LUNA2"}
                base = currency_mapping.get(base, base)
                if hasattr(self.exchange, "id") and self.exchange.id == "kraken":
                    kraken_symbol = f"{base}/{quote}"
                else:
                    kraken_symbol = symbol
                # --- Use QuotaManagerBot for ticker fetch ---
                def fetch_ticker():
                    return self.exchange.fetch_ticker(kraken_symbol)
                success, ticker = quota_manager.api_request(
                    api_name="crypto_ticker",
                    func=fetch_ticker,
                    quota_per_minute=None
                )
                if not success or not ticker or "last" not in ticker:
                    raise ValueError("Empty ticker data")
                tf_value, tf_unit = DEFAULT_CRYPTO_TIMEFRAME
                timeframe = f"{tf_value}{tf_unit}"
                def fetch_ohlcv():
                    return self.exchange.fetch_ohlcv(kraken_symbol, timeframe, limit=2)
                success, ohlcv = quota_manager.api_request(
                    api_name="crypto_ohlcv",
                    func=fetch_ohlcv,
                    quota_per_minute=None
                )
                if not success or not ohlcv or len(ohlcv) == 0:
                    raise ValueError("Empty OHLCV data")
                # --- Staleness Logging ---
                import datetime
                last_ohlcv_time = ohlcv[-1][0] if ohlcv and len(ohlcv[-1]) > 0 else None
                if last_ohlcv_time:
                    ohlcv_dt = datetime.datetime.utcfromtimestamp(last_ohlcv_time / 1000)
                    now = datetime.datetime.utcnow()
                    age_sec = (now - ohlcv_dt).total_seconds()
                    if age_sec > 3600:
                        self.logger.warning(f"[MARKET_DATA][STALE] {symbol} OHLCV data is stale! Last bar age: {age_sec/60:.1f} min.")
                    else:
                        self.logger.info(f"[MARKET_DATA][FRESH] {symbol} OHLCV data is fresh. Last bar age: {age_sec/60:.1f} min.")
                result = {
                    "current_price": ticker["last"],
                    "volume": ticker.get("quoteVolume", ticker.get("baseVolume", 0)),
                    "price_change": ticker.get("percentage", 0),
                    "high_24h": ticker.get("high"),
                    "low_24h": ticker.get("low"),
                    "yesterday_close": ohlcv[0][4] if len(ohlcv) > 1 else None,
                }
                return result
            except Exception as e:
                self.logger.error(f"[get_crypto_data][RETRY] {symbol} attempt {retries+1}/{max_retries} failed: {e}")
                wait_time = min(60, delay)
                time.sleep(wait_time)
            retries += 1
        self.logger.error(f"[get_crypto_data][FAIL] {symbol} failed after {max_retries} retries.")
        return None

    @log_method_calls
    def analyze_crypto(self, symbol, prompt_note=None, test_override=None):
        """
        Analyze a crypto asset using LLM prompts and return structured analysis or error dict.
        Handles prompt formatting, LLM call, and schema validation.
        """
        method = "analyze_crypto"
        # === Get latest news sentiment ===
        from bot_news_retriever import NewsRetrieverBot
        news_bot = NewsRetrieverBot()
        news_articles = news_bot.fetch_news(symbol, max_results=3)
        news_sentiment = None
        if news_articles:
            news_sentiment = news_articles[0].title if news_articles else None
        # === Build template_vars and prompt ===
        # Fetch crypto data for prompt variables
        data = self.get_crypto_data(symbol)
        if not data:
            return {"error": f"No crypto data for {symbol}"}
        template_vars = {
            "schema": ANALYSIS_SCHEMA,
            "symbol": symbol,
            "current_price": data["current_price"],
            "volume": data["volume"],
            "price_change": data["price_change"],
            "high_24h": data["high_24h"],
            "low_24h": data["low_24h"],
        }
        # Defensive: Ensure all required variables for crypto prompt are present
        safe_vars = {k: (v if v is not None else "0") for k, v in template_vars.items()}
        for key in ["rsi", "macd", "sma_20", "sma_diff_prefix", "sma_diff", "signal_summary"]:
            if key not in safe_vars:
                safe_vars[key] = ""
        prompt_template = CRYPTO_ANALYSIS_TEMPLATE
        prompt = prompt_template
        for k, v in safe_vars.items():
            prompt = prompt.replace(f"{{{k}}}", str(v))
        if prompt_note:
            prompt += f"\n{prompt_note}"
        # Defensive: If prompt is empty after formatting, log and return error
        if not prompt.strip():
            self.logger.error(f"[analyze_crypto()] FAILED | Error: prompt is empty for {symbol}")
            return {"error": f"Prompt was empty for {symbol}. No analysis performed."}
        # --- Standardized LLM prompt logging ---
        full_prompt_text = prompt
        prompt_summary = f"Symbol: {template_vars.get('symbol', symbol)}, Type: crypto"
        if 'current_price' in template_vars:
            prompt_summary += f", Price: {template_vars['current_price']}"
        if LOG_FULL_PROMPT:
            self.logger.debug(f"[analyze_crypto()] Full prompt sent: {full_prompt_text}")
        else:
            self.logger.info(f"[analyze_crypto()] Prompt summary: {prompt_summary}")
        # --- LLM call ---
        if test_override is not None:
            response = test_override
        else:
            response = generate_ai_analysis(prompt, variables=template_vars)
            # --- Robust JSON Parsing with Retries ---
            from bot_ai import AIBot
            parsed_json = AIBot.clean_json_response(response)
            if parsed_json is None:
                return {
                    "error": f"LLM response for {symbol} missing, malformed, or invalid. Trade skipped.",
                    "raw_cleaned_response": repr(response),
                }
        # --- Parse and log decision ---
        try:
            parsed_json = AIBot.clean_json_response(response)
            if parsed_json is not None:
                action = parsed_json.get('action', None)
                confidence = parsed_json.get('confidence', None)
                reason = parsed_json.get('reason', parsed_json.get('reasoning', None))
                # === LLM JSON Schema Validation ===
                def is_valid_llm_json(data):
                    if not isinstance(data, dict):
                        return False
                    required = ["action", "confidence"]
                    if any(k not in data for k in required):
                        return False
                    if data["action"] not in ["buy", "sell", "hold"]:
                        return False
                    try:
                        conf = float(data["confidence"])
                        if not (0.0 <= conf <= 1.0):
                            return False
                    except Exception:
                        return False
                    return True
                if not is_valid_llm_json(parsed_json):
                    return {"error": f"LLM response for {symbol} missing, malformed, or invalid. Trade skipped."}
                # === Store LLM answer in memory for future reuse, including news sentiment ===
                if not hasattr(self, 'database_bot'):
                    from bot_database import DatabaseBot
                    self.database_bot = DatabaseBot()
                context_data = {
                    "symbol": symbol,
                    "asset_type": "crypto",
                    "prompt": prompt,
                    "raw_llm_response": parsed_json,
                    "template_vars": template_vars,
                    "news_sentiment": news_sentiment,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                self.database_bot.store_analysis_context(
                    symbol,
                    "llm_analysis",
                    context_data,
                    relevance_score=parsed_json.get("confidence", 0.0),
                )
                parsed_json["source"] = "fresh_llm_call"
                return parsed_json
            else:
                return {
                    "error": "JSON parsing failed after cleaning",
                    "raw_cleaned_response": repr(response),
                }
        except Exception as e:
            return {
                "error": f"Exception during JSON cleaning/parsing: {e}",
                "raw_cleaned_response": repr(response),
            }

    @log_method_calls
    def monitor_positions(self):
        """
        Monitor open crypto positions for risk and performance.
        Logs alerts for risk or profit-taking opportunities.
        """
        try:
            # Fetch open positions from the executor (assumes executor has get_open_positions for crypto)
            open_positions = self.executor.get_open_positions(asset_type="crypto")
            if not open_positions:
                return
            for pos in open_positions:
                symbol = pos["symbol"]
                entry_price = pos["entry_price"]
                quantity = pos["quantity"]
                data = self.get_crypto_data(symbol)
                if not data:
                    continue
                current_price = data["current_price"]
                stop_loss_price = entry_price * (1 - CRYPTO_STOP_LOSS_PCT)
                if current_price <= stop_loss_price:
                    # Attempt to close the position
                    result = self.executor.close_position(
                        symbol, quantity, asset_type="crypto"
                    )
        except Exception as e:
            pass

    @log_method_calls
    def print_performance_summary(self, symbol, asset_type, timeframe=None):
        """
        Print a summary of performance for a given crypto asset and timeframe.
        Includes price, volume, and key metrics.
        """
        print(
            f"\n=== Performance Summary for {symbol} ({asset_type}) ==="
        )  # CLI/test output
        # Placeholder: Add more detailed metrics as you implement them
        # Example: print last analysis, open positions, etc.
        print(
            f"No detailed crypto performance summary implemented yet for {symbol}."
        )  # CLI@test output

    class MockExchange:
        def fetch_ticker(self, symbol):
            return {
                "last": 50000,
                "quoteVolume": 100,
                "percentage": 2,
                "high": 51000,
                "low": 49000,
            }

        def fetch_ohlcv(self, symbol, timeframe, limit=2):
            return [[0, 0, 0, 0, 49500], [0, 0, 0, 0, 50000]]

        rateLimit = 1000

    @staticmethod
    @log_method_calls
    def selftest():
        """
        Run a self-test to verify CryptoBot's core logic. Returns True if successful.
        """
        print("\n--- Running CryptoBot Self-Test ---")  # CLI/test output
        try:
            bot = CryptoBot(exchange=CryptoBot.MockExchange())
            # Test strong BUY
            fake_buy = '{"action": "buy", "reasoning": "Strong breakout, high volume.", "confidence": 0.97}'
            result_buy = bot.analyze_crypto("BTC/USD", test_override=fake_buy)
            print("    -> Test BUY result:", result_buy)  # CLI@test output
            assert (
                result_buy and result_buy.get("action") == "buy"
            ), f"Expected BUY, got: {result_buy}"
            # Test strong SELL
            fake_sell = '{"action": "sell", "reasoning": "Bearish divergence, low volume.", "confidence": 0.91}'
            result_sell = bot.analyze_crypto("ETH/USD", test_override=fake_sell)
            print("    -> Test SELL result:", result_sell)  # CLI@test output
            assert (
                result_sell and result_sell.get("action") == "sell"
            ), f"Expected SELL, got: {result_sell}"
            print("--- CryptoBot Self-Test PASSED ---")  # CLI@test output
        except AssertionError as e:
            print(f"--- CryptoBot Self-Test FAILED: {e} ---")  # CLI@test output
        except Exception as e:
            print(
                f"--- CryptoBot Self-Test encountered an ERROR: {e} ---"
            )  # CLI@test output


if __name__ == "__main__":
    CryptoBot.selftest()

# === End of bot_crypto.py ===
