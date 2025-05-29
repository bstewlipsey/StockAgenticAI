# bot_stock.py
"""
StockBot: Stock-specific analysis and trading logic for agentic trading systems.
- Fetches and analyzes stock market data using Alpaca
- Generates AI-driven trading signals and risk metrics
- Designed for modular integration with other trading bots
"""

# === STOCK BOT LOGIC ===
# Extracted from agent.py

import logging
import time
import re
import json
from config_system import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, PAPER_TRADING, DEFAULT_STOCK_TIMEFRAME
from alpaca_trade_api import REST
from alpaca_trade_api.rest import URL, TimeFrame, TimeFrameUnit
from bot_indicators import IndicatorBot
from bot_trade_executor import TradeExecutorBot
from bot_risk_manager import RiskManager, Position
from config_trading import TRADING_ASSETS, RSI_OVERSOLD, RSI_OVERBOUGHT
from config_system import (
    ANALYSIS_SCHEMA, STOCK_ANALYSIS_TEMPLATE, USE_EXPERIMENTAL_PROMPT, EXPERIMENTAL_STOCK_ANALYSIS_TEMPLATE
)

from bot_ai import generate_ai_analysis
from bot_database import DatabaseBot
from bot_news_retriever import NewsRetrieverBot

            

logger = logging.getLogger(__name__)

class MockAlpacaAPI:
    def get_latest_quote(self, symbol):
        class Quote:
            ap = 123.45
        return Quote()
    def get_bars(self, symbol, timeframe, limit=50):
        class Bar:
            c = 123.45
            v = 1000
            t = '2023-01-01T00:00:00Z'
            o = 120.00
        return [Bar() for _ in range(limit)]
    def get_position(self, symbol):
        class Position:
            qty = 10
            avg_entry_price = 120.00
        return Position()

class StockBot:
    def __init__(self):
        self.api = REST(
            key_id=ALPACA_API_KEY or "",
            secret_key=ALPACA_SECRET_KEY or "",
            base_url=URL(ALPACA_BASE_URL),
            api_version='v2'
        )
        self.executor = TradeExecutorBot(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper_trading=PAPER_TRADING)
        self.risk_manager = RiskManager()
        self.asset_type_map = {symbol: asset_type for symbol, asset_type, _ in TRADING_ASSETS}
        self.tf_value, self.tf_unit = DEFAULT_STOCK_TIMEFRAME  
        self.database_bot = DatabaseBot()
        
    def get_current_price(self, symbol):
        for _ in range(3):
            try:
                quote = self.api.get_latest_quote(symbol)
                if quote and hasattr(quote, 'ap'):
                    return float(quote.ap)
            except Exception as e:
                logger.error(f"Error getting price for {symbol}: {e}")
                time.sleep(2)
        logger.error(f"Returning None for get_current_price({symbol}) after 3 failed attempts.")
        return None

    def analyze_stock(self, symbol, prompt_note=None, test_override=None):
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
            recent_contexts = self.database_bot.get_analysis_context(symbol, context_types=["llm_analysis"], days=1, limit=5)
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
            # ...existing code for test_override and LLM call...
            if test_override is not None:
                response = test_override
                logger.info(f"[TEST] Using test_override for {symbol}: {response}")
            else:
                tf_value, tf_unit = self.tf_value, self.tf_unit
                timeframe = TimeFrame(tf_value, TimeFrameUnit[tf_unit])
                quote = self.api.get_latest_quote(symbol)
                bars = self.api.get_bars(symbol, timeframe, limit=50)
                # === Market Data Logging for Quality/Freshness ===
                import datetime
                now = datetime.datetime.utcnow()
                bar_times = [getattr(bar, 't', None) or getattr(bar, 'timestamp', None) or getattr(bar, 'start', None) or getattr(bar, 'time', None) for bar in bars]
                logger.debug(f"[MARKET_DATA] {symbol} bar timestamps: {bar_times}")
                # Log quote info if available
                quote_time = getattr(quote, 't', None) or getattr(quote, 'timestamp', None) or getattr(quote, 'time', None)
                logger.info(f"[MARKET_DATA] {symbol} quote: time={quote_time}, ask_price={getattr(quote, 'ap', None)}")
                # Check for staleness (bar_time should be recent)
                try:
                    latest_bar_time = bar_times[-1] if bar_times else None
                    if isinstance(latest_bar_time, str):
                        bar_dt = datetime.datetime.fromisoformat(latest_bar_time.replace('Z', '+00:00'))
                    elif isinstance(latest_bar_time, (int, float)):
                        bar_dt = datetime.datetime.utcfromtimestamp(latest_bar_time/1000 if latest_bar_time > 1e12 else latest_bar_time)
                    else:
                        bar_dt = None
                    if bar_dt:
                        age_sec = (now - bar_dt).total_seconds()
                        logger.info(f"[MARKET_DATA] {symbol} latest bar age: {age_sec:.1f} seconds")
                        if age_sec > 120:
                            logger.warning(f"[MARKET_DATA] {symbol} bar data is stale! Age: {age_sec:.1f} seconds")
                except Exception as e:
                    logger.warning(f"[MARKET_DATA] Could not parse bar_time for {symbol}: {latest_bar_time} ({e})")
                prices = [bar.c for bar in bars]
                volumes = [bar.v for bar in bars]
                timestamps = [getattr(bar, 't', None) for bar in bars]
                tech_analysis = IndicatorBot(prices)
                signals, indicators = tech_analysis.get_signals()
                signal_summary = "\n".join([f"- {signal}: {reason} (Confidence: {confidence*100:.0f}%)" for signal, reason, confidence in signals])
                # Market context (S&P 500)
                try:
                    spy_timeframe = TimeFrame(tf_value, TimeFrameUnit[tf_unit])
                    spy_bars = self.api.get_bars('SPY', spy_timeframe, limit=1)
                    if not spy_bars:
                        logger.warning(f"No bars returned for SPY with timeframe {spy_timeframe}. Skipping market context.")
                        market_change = 0
                        market_context = "Neutral"
                    else:
                        market_change = ((spy_bars[-1].c - spy_bars[-1].o) / spy_bars[-1].o) * 100
                        market_context = "Bullish" if market_change > 0 else "Bearish"
                except Exception as e:
                    logger.warning(f"Could not fetch market context: {e}")
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
                    asset_type='stock'
                )
                risk_metrics = self.risk_manager.calculate_position_risk(position_obj)
                price_change = ((current_price - prices[0]) / prices[0]) * 100
                avg_volume = sum(volumes) / len(volumes)
                current_volume = volumes[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                template_vars = {
                    'schema': ANALYSIS_SCHEMA,
                    'symbol': symbol,
                    'current_price': quote.ap,
                    'position_qty': position_qty,
                    'entry_price': avg_entry_price,
                    'investment': risk_metrics['investment'],
                    'current_value': risk_metrics['current_value'],
                    'pnl': risk_metrics['pnl_percent'] * 100,
                    'risk_level': risk_metrics['risk_level'],
                    'rsi': indicators['rsi'],
                    'macd': indicators['macd'],
                    'sma_20': indicators['sma_20'],
                    'ema_20': indicators.get('ema_20', 0),
                    'bb_high': indicators.get('bb_high', 0),
                    'bb_low': indicators.get('bb_low', 0),
                    'price_change': price_change,
                    'volume_ratio': volume_ratio,
                    'market_context': market_context,
                    'market_change': market_change,
                    'signal_summary': signal_summary
                }
                # === RAG: Retrieve historical AI context and reflection insights ===
                historical_context = self.database_bot.get_analysis_history(symbol)
                reflection_insights = self.database_bot.get_reflection_insights(symbol)
                rag_note = ''
                if historical_context:
                    rag_note += '\nRecent AI decisions:'
                    for row in historical_context[:5]:
                        rag_note += f"\n- {row[1]}: {row[3].upper()} (confidence: {row[4]:.2f})"
                if reflection_insights:
                    rag_note += '\nRecent Reflection Insights:'
                    for insight in reflection_insights[:3]:
                        rag_note += f"\n- {insight['timestamp']}: {insight['key_insights']}"
                # Combine prompt_note and RAG note
                full_prompt_note = (prompt_note or '') + rag_note
                # --- ENFORCE JSON-ONLY OUTPUT IN PROMPT ---
                prompt_template = EXPERIMENTAL_STOCK_ANALYSIS_TEMPLATE if USE_EXPERIMENTAL_PROMPT else STOCK_ANALYSIS_TEMPLATE
                prompt = prompt_template + "\nIMPORTANT: Respond ONLY with valid JSON matching the schema above. Do NOT include any explanation, markdown, or extra text."
                if full_prompt_note:
                    prompt += f"\n{full_prompt_note}"
                logger.info(f"[DEBUG] STOCK ANALYSIS PROMPT for {symbol}:\n{prompt}\nVariables: {template_vars}")
                response = generate_ai_analysis(prompt, variables=template_vars)
                logger.info(f"[DEBUG] STOCK ANALYSIS RAW LLM RESPONSE for {symbol}:\n{response}")
                # Try to parse as JSON and log the attempt
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
                    'asset_type': 'stock',
                    'prompt': prompt,
                    'raw_llm_response': parsed_json,
                    'template_vars': template_vars,
                    'news_sentiment': news_sentiment,
                    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
                }
                self.database_bot.store_analysis_context(symbol, 'llm_analysis', context_data, relevance_score=parsed_json.get('confidence', 0.0))
                logger.info(f"[LLM_MEMORY] Stored new LLM analysis for {symbol} in database.")
                parsed_json['source'] = 'fresh_llm_call'
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
                logger.info(f"[AI_JSON] {symbol} LLM JSON: {json.dumps(parsed_json, indent=2)}")
                if isinstance(parsed_json, dict) and 'error' in parsed_json:
                    return parsed_json
                return parsed_json
            logger.error(f"[AI_JSON] Could not parse LLM response as JSON for {symbol} after retries: {last_error}")
            return {"error": f"Could not parse LLM response as JSON for {symbol}. See logs for details."}
        except Exception as e:
            logger.error(f"[AI_ANALYSIS_ERROR] Exception in analyze_stock for {symbol}: {e}")
            return {"error": f"Error analyzing stock: {str(e)}"}

    # === Stock Trading Logic (migrated from agent.py) ===
    # Migrate TradingAgent methods related to stock trading here.

    def print_performance_summary(self, symbol, asset_type, timeframe, days_back=50):
        """
        Print a summary of performance for a given symbol and asset type.
        """
        # This would use DatabaseBot and VisualizerBot for reporting
        # Example stub:
        print(f"Performance summary for {symbol} ({asset_type}) over {days_back} days:")
        # ...fetch and print metrics...

    @staticmethod
    def selftest():
        print(f"\n--- Running StockBot Self-Test ---")  # CLI/test output only
        try:
            bot = StockBot()
            bot.api = MockAlpacaAPI()  # Inject mock API
            # Test strong BUY
            fake_buy = '{"action": "buy", "reasoning": "Strong uptrend, high volume.", "confidence": 0.95}'
            result_buy = bot.analyze_stock("AAPL", test_override=fake_buy)
            print("    -> Test BUY result:", result_buy)  # CLI/test output only
            assert result_buy and result_buy.get('action') == 'buy', f"Expected BUY, got: {result_buy}"
            # Test strong SELL
            fake_sell = '{"action": "sell", "reasoning": "Bearish reversal, negative news.", "confidence": 0.92}'
            result_sell = bot.analyze_stock("TSLA", test_override=fake_sell)
            print("    -> Test SELL result:", result_sell)  # CLI@test output only
            assert result_sell and result_sell.get('action') == 'sell', f"Expected SELL, got: {result_sell}"
            print(f"--- StockBot Self-Test PASSED ---")  # CLI@test output only
        except AssertionError as e:
            print(f"--- StockBot Self-Test FAILED: {e} ---")  # CLI@test output only
        except Exception as e:
            print(f"--- StockBot Self-Test encountered an ERROR: {e} ---")  # CLI@test output only

# === Usage Example ===
if __name__ == "__main__":
    StockBot.selftest()
