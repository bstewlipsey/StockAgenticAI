# bot_stock.py
"""
StockBot: Stock-specific analysis and trading logic for agentic trading systems.
- Fetches and analyzes stock market data using Alpaca
- Generates AI-driven trading signals and risk metrics
- Designed for modular integration with other trading bots
"""

# === STOCK BOT LOGIC ===
# Extracted from agent.py

import time
import json
from config_system import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_BASE_URL,
    PAPER_TRADING,
    DEFAULT_STOCK_TIMEFRAME,
    LOG_FULL_PROMPT,
)
from alpaca_trade_api import REST
from alpaca_trade_api.rest import URL, TimeFrame, TimeFrameUnit
from bot_indicators import IndicatorBot
from bot_trade_executor import TradeExecutorBot
from bot_risk_manager import RiskManager, Position
from config_trading import TRADING_ASSETS
from config_system import (
    ANALYSIS_SCHEMA,
    STOCK_ANALYSIS_TEMPLATE,
    USE_EXPERIMENTAL_PROMPT,
    EXPERIMENTAL_STOCK_ANALYSIS_TEMPLATE,
)

from bot_ai import generate_ai_analysis
from bot_database import DatabaseBot
from bot_news_retriever import NewsRetrieverBot
from data_structures import AgenticBotError
from utils.logger_mixin import LoggerMixin
from utils.logging_decorators import log_method_calls


class MockAlpacaAPI:
    @log_method_calls
    def get_latest_quote(self, symbol):
        class Quote:
            ap = 123.45

        return Quote()

    @log_method_calls
    def get_bars(self, symbol, timeframe, limit=50):
        class Bar:
            c = 123.45
            v = 1000
            t = "2023-01-01T00:00:00Z"
            o = 120.00

        return [Bar() for _ in range(limit)]

    @log_method_calls
    def get_position(self, symbol):
        class Position:
            qty = 10
            avg_entry_price = 120.00

        return Position()


class StockBot(LoggerMixin):
    """
    StockBot: Stock-specific analysis and trading logic for agentic trading systems.
    - Fetches and analyzes stock market data using Alpaca
    - Generates AI-driven trading signals and risk metrics
    - Designed for modular integration with other trading bots
    """

    @log_method_calls
    def __init__(self):
        """
        Initialize StockBot with Alpaca API, executor, risk manager, and database bot.
        Sets up asset type mapping and default timeframe.
        """
        super().__init__()
        self.api = REST(
            key_id=ALPACA_API_KEY or "",
            secret_key=ALPACA_SECRET_KEY or "",
            base_url=URL(ALPACA_BASE_URL),
            api_version="v2",
        )
        self.executor = TradeExecutorBot(
            ALPACA_API_KEY, ALPACA_SECRET_KEY, paper_trading=PAPER_TRADING
        )
        self.risk_manager = RiskManager()
        self.asset_type_map = {
            symbol: asset_type for symbol, asset_type, _ in TRADING_ASSETS
        }
        self.tf_value, self.tf_unit = DEFAULT_STOCK_TIMEFRAME
        self.database_bot = DatabaseBot()

    @log_method_calls
    def get_current_price(self, symbol):
        """
        Fetch the current price for a stock symbol using Alpaca API.
        Uses QuotaManagerBot for rate limiting and retries.
        Returns the price as float, or None if unavailable.
        """
        method = "get_current_price"
        from bot_quota_manager import QuotaManagerBot
        quota_manager = QuotaManagerBot()
        for _ in range(3):
            try:
                def fetch_quote():
                    return self.api.get_latest_quote(symbol)
                success, quote = quota_manager.api_request(
                    api_name="stock_quote",
                    func=fetch_quote,
                    quota_per_minute=None
                )
                if success and quote and hasattr(quote, "ap"):
                    return float(quote.ap)
            except Exception as e:
                self.logger.error(f"[get_current_price][RETRY] {symbol} failed: {e}")
                time.sleep(2)
        self.logger.error(f"[get_current_price][FAIL] {symbol} failed after retries.")
        return {"error": f"Failed to get current price for {symbol}", "status": "failed"}

    @log_method_calls
    def analyze_stock(self, symbol, prompt_note=None, test_override=None):
        """
        Analyze a stock using LLM prompts and return structured analysis or error dict.
        Handles prompt formatting, LLM call, and schema validation.
        """
        method = "analyze_stock"
        """
        Analyze a stock using market data, technical analysis, and AI predictions.
        Optionally include a prompt_note for adaptive AI learning.
        If test_override is provided, use it as the LLM response (string or dict).
        Implements robust LLM output/JSON parsing with retries and logging.
        Implements LLM answer memory for decision-making (Task 3.4).
        """
        try:
            # === Get latest news sentiment ===
            news_bot = NewsRetrieverBot()
            news_articles = news_bot.fetch_news(symbol, max_results=3)
            news_sentiment = None
            if news_articles:
                # For simplicity, use the title of the most relevant article as sentiment proxy
                news_sentiment = news_articles[0].title if news_articles else None
            # === LLM Answer Memory: Check for recent high-confidence answer with similar news sentiment ===
            recent_contexts = self.database_bot.get_analysis_context(
                symbol, context_types=["llm_analysis"], days=1, limit=5
            )
            now = time.time()
            for ctx in recent_contexts:
                ctx_data = ctx.get("context_data", {})
                raw_llm = ctx_data.get("raw_llm_response", {})
                conf = raw_llm.get("confidence", 0)
                ts = ctx.get("timestamp")
                prev_sentiment = ctx_data.get("news_sentiment")
                # Accept if confidence > 0.7, answer is <3h old, and news sentiment matches
                if conf and float(conf) > 0.7 and ts:
                    import datetime

                    try:
                        ts_dt = datetime.datetime.fromisoformat(ts)
                        age_hours = (
                            datetime.datetime.utcnow() - ts_dt
                        ).total_seconds() / 3600
                        if prev_sentiment == news_sentiment:
                            if age_hours < 3:
                                result = dict(raw_llm)
                                result["source"] = "llm_memory"
                                return result
                        else:
                            pass
                    except Exception:
                        pass
            # ...existing code for test_override and LLM call...
            if test_override is not None:
                response = test_override
            else:
                tf_value, tf_unit = self.tf_value, self.tf_unit
                timeframe = TimeFrame(tf_value, TimeFrameUnit[tf_unit])
                quote = self.api.get_latest_quote(symbol)
                # --- Robust Alpaca get_bars with Error Handling ---
                try:
                    bars = self.api.get_bars(symbol, timeframe, limit=50)
                except Exception:
                    bars = []
                if not bars:
                    # Optionally, retry once if this is a transient issue
                    time.sleep(1)
                    try:
                        bars = self.api.get_bars(symbol, timeframe, limit=50)
                    except Exception:
                        bars = []
                    if not bars:
                        return {
                            "error": f"No bars returned for {symbol} from Alpaca. Data unavailable or API issue."
                        }
                # === Market Data Logging for Quality/Freshness ===
                import datetime
                now = datetime.datetime.utcnow()
                bar_times = [
                    getattr(bar, "t", None)
                    or getattr(bar, "timestamp", None)
                    or getattr(bar, "start", None)
                    or getattr(bar, "time", None)
                    for bar in bars
                ]
                # Log quote info if available
                quote_time = (
                    getattr(quote, "t", None)
                    or getattr(quote, "timestamp", None)
                    or getattr(quote, "time", None)
                )
                # Check for staleness (bar_time should be recent)
                try:
                    latest_bar_time = bar_times[-1] if bar_times else None
                    if latest_bar_time:
                        if isinstance(latest_bar_time, str):
                            bar_dt = datetime.datetime.fromisoformat(
                                latest_bar_time.replace("Z", "+00:00")
                            )
                        elif isinstance(latest_bar_time, (int, float)):
                            bar_dt = datetime.datetime.utcfromtimestamp(
                                latest_bar_time / 1000
                                if latest_bar_time > 1e12
                                else latest_bar_time
                            )
                        else:
                            bar_dt = None
                        if bar_dt:
                            age_sec = (now - bar_dt).total_seconds()
                            if age_sec > 3600:
                                self.logger.warning(f"[MARKET_DATA][STALE] {symbol} OHLCV data is stale! Last bar age: {age_sec/60:.1f} min.")
                            else:
                                self.logger.info(f"[MARKET_DATA][FRESH] {symbol} OHLCV data is fresh. Last bar age: {age_sec/60:.1f} min.")
                except Exception as e:
                    self.logger.error(f"[MARKET_DATA][STALE_CHECK_FAIL] {symbol} failed to check staleness: {e}")
                prices = [bar.c for bar in bars]
                volumes = [bar.v for bar in bars]
                tech_analysis = IndicatorBot(prices)
                signals, indicators = tech_analysis.get_signals()
                signal_summary = "\n".join(
                    [
                        f"- {signal}: {reason} (Confidence: {confidence*100:.0f}%)"
                        for signal, reason, confidence in signals
                    ]
                )
                # Market context (S&P 500)
                try:
                    spy_timeframe = TimeFrame(tf_value, TimeFrameUnit[tf_unit])
                    spy_bars = self.api.get_bars("SPY", spy_timeframe, limit=1)
                    if not spy_bars:
                        market_change = 0
                        market_context = "Neutral"
                    else:
                        market_change = (
                            (spy_bars[-1].c - spy_bars[-1].o) / spy_bars[-1].o
                        ) * 100
                        market_context = "Bullish" if market_change > 0 else "Bearish"
                except Exception:
                    market_change = 0
                    market_context = "Neutral"
                # Position and risk
                position_qty = "0"
                avg_entry_price = "0.00"
                try:
                    position = self.api.get_position(symbol)
                    position_qty = str(position.qty)
                    avg_entry_price = str(position.avg_entry_price)
                except Exception:
                    pass
                current_price = float(quote.ap)
                position_obj = Position(
                    symbol=symbol,
                    quantity=float(position_qty),
                    entry_price=float(avg_entry_price),
                    current_price=current_price,
                    asset_type="stock",
                )
                risk_metrics = self.risk_manager.calculate_position_risk(position_obj)
                price_change = ((current_price - prices[0]) / prices[0]) * 100
                avg_volume = sum(volumes) / len(volumes)
                current_volume = volumes[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                template_vars = {
                    "schema": ANALYSIS_SCHEMA,
                    "symbol": symbol,
                    "current_price": quote.ap,
                    "position_qty": position_qty,
                    "entry_price": avg_entry_price,
                    "investment": risk_metrics["investment"],
                    "current_value": risk_metrics["current_value"],
                    "pnl": risk_metrics["pnl_percent"] * 100,
                    "risk_level": risk_metrics["risk_level"],
                    "rsi": indicators["rsi"],
                    "macd": indicators["macd"],
                    "sma_20": indicators["sma_20"],
                    "ema_20": indicators.get("ema_20", 0),
                    "bb_high": indicators.get("bb_high", 0),
                    "bb_low": indicators.get("bb_low", 0),
                    "price_change": price_change,
                    "volume_ratio": volume_ratio,
                    "market_context": market_context,
                    "market_change": market_change,
                    "signal_summary": signal_summary,
                }
                # === RAG: Retrieve historical AI context and reflection insights ===
                historical_context = self.database_bot.get_analysis_history(symbol)
                reflection_insights = self.database_bot.get_reflection_insights(symbol)
                rag_note = ""
                if historical_context:
                    rag_note += "\nRecent AI decisions:"
                    for row in historical_context[:5]:
                        rag_note += (
                            f"\n- {row[1]}: {row[3].upper()} (confidence: {row[4]:.2f})"
                        )
                if reflection_insights:
                    rag_note += "\nRecent Reflection Insights:"
                    for insight in reflection_insights[:3]:
                        rag_note += (
                            f"\n- {insight['timestamp']}: {insight['key_insights']}"
                        )
                # Combine prompt_note and RAG note
                full_prompt_note = (prompt_note or "") + rag_note
                # --- ENFORCE JSON-ONLY OUTPUT IN PROMPT ---
                prompt_template = STOCK_ANALYSIS_TEMPLATE
                # Fill all required variables, fallback to safe defaults if missing
                safe_vars = {k: (v if v is not None else "0") for k, v in template_vars.items()}
                # Add missing keys for template
                for key in ["sma_diff_prefix", "sma_diff"]:
                    if key not in safe_vars:
                        safe_vars[key] = ""
                prompt = prompt_template
                for k, v in safe_vars.items():
                    prompt = prompt.replace(f"{{{k}}}", str(v))
                if full_prompt_note:
                    prompt += f"\n{full_prompt_note}"
                # Defensive: If prompt is empty after formatting, log and return error
                if not prompt.strip():
                    self.logger.error(f"[analyze_stock()] FAILED | Error: prompt is empty for {symbol}")
                    return {"error": f"Prompt was empty for {symbol}. No analysis performed."}
                response = generate_ai_analysis(prompt, variables=template_vars)
                # --- Robust JSON Parsing with Retries ---
                from bot_ai import AIBot
                parsed_json = AIBot.clean_json_response(response)
                if parsed_json is None:
                    return {
                        "error": f"LLM response for {symbol} missing, malformed, or invalid. Trade skipped.",
                        "raw_cleaned_response": repr(response),
                    }
                action = parsed_json.get('action', None)
                confidence = parsed_json.get('confidence', None)
                reason = parsed_json.get('reason', parsed_json.get('reasoning', None))
                if "error" in response:
                    return response

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

                if parsed_json is None or not is_valid_llm_json(parsed_json):
                    return {
                        "error": f"LLM response for {symbol} missing, malformed, or invalid. Trade skipped."
                    }
                # === Store LLM answer in memory for future reuse, including news sentiment ===
                context_data = {
                    "symbol": symbol,
                    "asset_type": "stock",
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
            # --- Robust JSON Parsing with Retries ---
            from bot_ai import AIBot

            parsed_json = None
            last_error = None
            for attempt in range(3):
                try:
                    # Try direct JSON parse first
                    if isinstance(response, dict):
                        parsed_json = response
                        break
                    parsed_json = json.loads(response)
                    break
                except Exception as e1:
                    # Try cleaning and parsing
                    try:
                        parsed_json = AIBot.clean_json_response(response)
                        if parsed_json is not None:
                            break
                    except Exception as e2:
                        last_error = e2
                    last_error = e1
                    time.sleep(0.5)
            if parsed_json is not None:
                if isinstance(parsed_json, dict) and "error" in parsed_json:
                    return parsed_json
                return parsed_json
            return {
                "error": f"Could not parse LLM response as JSON for {symbol}. See logs for details."
            }
        except Exception as e:
            return {"error": f"Error analyzing stock: {str(e)}", "status": "failed"}

    # === Stock Trading Logic (migrated from agent.py) ===
    # Migrate TradingAgent methods related to stock trading here.

    @log_method_calls
    def print_performance_summary(self, symbol, asset_type, timeframe, days_back=50):
        """
        Print a summary of performance for a given stock and timeframe.
        Includes price, volume, and key metrics.
        """
        # This would use DatabaseBot and VisualizerBot for reporting
        # Example stub:
        print(f"Performance summary for {symbol} ({asset_type}) over {days_back} days:")
        # ...fetch and print metrics...

    @staticmethod
    @log_method_calls
    def selftest() -> bool:
        """
        Run a self-test to verify StockBot's core logic. Returns True if successful.
        """
        print("\n--- Running StockBot Self-Test ---")  # CLI/test output only
        try:
            bot = StockBot()
            bot.api = MockAlpacaAPI()  # Inject mock API
            # Test strong BUY
            fake_buy = '{"action": "buy", "reasoning": "Strong uptrend, high volume.", "confidence": 0.95}'
            result_buy = bot.analyze_stock("AAPL", test_override=fake_buy)
            print("    -> Test BUY result:", result_buy)  # CLI/test output only
            assert (
                result_buy and result_buy.get("action") == "buy"
            ), f"Expected BUY, got: {result_buy}"
            # Test strong SELL
            fake_sell = '{"action": "sell", "reasoning": "Bearish reversal, negative news.", "confidence": 0.92}'
            result_sell = bot.analyze_stock("TSLA", test_override=fake_sell)
            print("    -> Test SELL result:", result_sell)  # CLI@test output only
            assert (
                result_sell and result_sell.get("action") == "sell"
            ), f"Expected SELL, got: {result_sell}"
            print("--- StockBot Self-Test PASSED ---")  # CLI@test output only
            return True
        except AssertionError as e:
            print(f"--- StockBot Self-Test FAILED: {e} ---")  # CLI@test output only
        except Exception as e:
            print(
                f"--- StockBot Self-Test encountered an ERROR: {e} ---"
            )  # CLI@test output only
        return False


# === Usage Example ===
if __name__ == "__main__":
    StockBot.selftest()
