# bot_orchestrator.py
"""
OrchestratorBot: Central coordinator for the agentic trading system.
- Manages the main trading loop and workflow orchestration
- Handles inter-bot communication and data flow
- Implements error handling and recovery mechanisms
- Coordinates the complete trading lifecycle
"""

import time
from typing import Dict, List, Any, Optional, Tuple

# Import all specialized bots
from bot_stock import StockBot
from bot_crypto import CryptoBot
from bot_portfolio import PortfolioBot
from bot_ai import AIBot
from bot_risk_manager import RiskManager, Position
from bot_trade_executor import TradeExecutorBot
from bot_position_sizer import PositionSizerBot
from bot_database import DatabaseBot
from bot_decision_maker import DecisionMakerBot
from bot_reflection import ReflectionBot, TradeOutcome
from bot_asset_screener import AssetScreenerBot
from bot_news_retriever import NewsRetrieverBot
from data_structures import (
    TradingDecision as CoreTradingDecision,
    AssetAnalysisInput,
)
from utils.logger_mixin import LoggerMixin
from utils.logging_decorators import log_method_calls

# Configuration imports
from config_system import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    TRADING_CYCLE_INTERVAL,
)
from config_trading import (
    TRADING_ASSETS,
    MIN_CONFIDENCE,
    MAX_PORTFOLIO_RISK,
    MAX_POSITION_RISK,
)


class OrchestratorBot(LoggerMixin):
    """
    MAIN TRADING ORCHESTRATOR

    Coordinates the complete agentic trading workflow:
    1. Portfolio-level risk assessment
    2. AI learning and adaptation
    3. Asset analysis and decision making
    4. Position sizing and risk checks
    5. Trade execution and portfolio updates
    6. Post-trade reflection and learning
    """

    @log_method_calls
    def __init__(self):
        """
        Initialize all specialized bots and components for orchestration.
        Sets up trading configuration and state.
        """
        super().__init__()
        """Initialize all specialized bots and components"""
        self.running = False

        # Initialize all specialized bots
        self.stock_bot = StockBot()
        self.crypto_bot = CryptoBot()
        self.portfolio_bot = PortfolioBot()
        self.ai_bot = AIBot()
        self.position_sizer = PositionSizerBot()
        self.risk_manager = RiskManager()
        self.database_bot = DatabaseBot()
        self.decision_maker = DecisionMakerBot()
        self.reflection_bot = ReflectionBot()
        # Initialize AssetScreenerBot with AI and Database bots
        self.asset_screener = AssetScreenerBot(self.ai_bot, self.database_bot)
        # Initialize NewsRetrieverBot
        self.news_retriever = NewsRetrieverBot()
        # Optional: Knowledge Graph integration
        self.knowledge_graph = None  # Set externally if needed

        # Initialize trade executor
        self.trade_executor = TradeExecutorBot(
            api_key=ALPACA_API_KEY,
            api_secret=ALPACA_SECRET_KEY,
            paper_trading=True,  # Always start with paper trading for safety
        )

        # Trading configuration
        self.assets = TRADING_ASSETS
        self.cycle_interval = TRADING_CYCLE_INTERVAL

    @log_method_calls
    def start_trading_loop(self):
        """
        Start the main agentic trading loop.
        Runs the full trading cycle repeatedly until stopped.
        """
        print("\n=== Starting Full Agentic Trading Loop...===")
        print(
            f"Configuration: MIN_CONFIDENCE={MIN_CONFIDENCE}, MAX_PORTFOLIO_RISK={MAX_PORTFOLIO_RISK}"
        )

        self.running = True

        try:
            while self.running:
                print("\n=== New Trading Cycle ===")

                # Execute one complete trading cycle
                cycle_successful = self._execute_trading_cycle()

                if cycle_successful:
                    print("Trading cycle completed successfully")
                else:
                    print("Trading cycle completed with some issues")

                # Sleep until next cycle
                print(f"\nSleeping {self.cycle_interval} seconds until next cycle...")
                time.sleep(self.cycle_interval)

        except KeyboardInterrupt:
            print("\nTrading loop stopped by user.")
            self.running = False
        except Exception as e:
            print(f"\nCritical error in trading loop: {e}")
            self.running = False
            raise

    @log_method_calls
    def stop_trading_loop(self):
        """Stop the trading loop gracefully"""
        self.running = False

    @log_method_calls
    def _execute_trading_cycle(self) -> bool:
        """
        Execute one complete trading cycle

        Returns:
            bool: True if cycle completed successfully, False if there were issues
        """
        cycle_success = True

        try:
            # === STEP 0: PORTFOLIO-LEVEL RISK ASSESSMENT ===
            portfolio_safe = self._assess_portfolio_risk()
            if not portfolio_safe:
                print("Portfolio risk too high - halting new trades this cycle")
                return False

            # === STEP 1: AI LEARNING & ADAPTATION ===
            min_confidence, prompt_note = self._perform_ai_adaptation()

            # === STEP 2: DYNAMIC ASSET SCREENING ===
            print("\nScreening for promising assets...")
            try:
                # Use AssetScreenerBot to get prioritized list of symbols
                screened_assets = self.asset_screener.screen_assets()
                print(
                    f"Screened assets for this cycle: {[a.symbol for a in screened_assets]}"
                )
            except Exception as e:
                import traceback
                self.logger.error(f"[ASSET_SCREENING][ERROR] Exception during asset screening: {e}\n{traceback.format_exc()}")
                print("Asset screening failed. See log for details.")
                screened_assets = []

            # === STEP 3: ANALYZE SCREENED ASSETS & MAKE DECISIONS ===
            trading_decisions = self._analyze_assets_and_decide(
                min_confidence, prompt_note, screened_assets
            )

            # === STEP 4: EXECUTE APPROVED TRADES ===
            execution_results = self._execute_approved_trades(trading_decisions)

            # === STEP 5: UPDATE PORTFOLIO STATE ===
            self._update_portfolio_state(execution_results)

            # === STEP 6: POST-TRADE REFLECTION ===
            self._run_post_trade_reflection()

            # === STEP 7: PORTFOLIO SUMMARY ===
            self._print_cycle_summary()

        except Exception as e:
            cycle_success = False

        return cycle_success

    @log_method_calls
    def _assess_portfolio_risk(self) -> bool:
        """
        STEP 0: Portfolio-level risk assessment

        Returns:
            bool: True if portfolio risk is acceptable, False otherwise
        """
        print("\nPortfolio Risk Assessment...")  # CLI/test output only

        try:
            # Get current positions from portfolio bot
            open_positions = self.portfolio_bot.get_open_positions()

            # Convert to Position objects for risk manager
            position_objects = []
            for symbol, trade in open_positions.items():
                # Fetch live price for each open position (always use live price, fallback to 0.0 if unavailable)
                if trade.asset_type == "stock":
                    current_price = self.stock_bot.get_current_price(symbol)
                    if not isinstance(current_price, (int, float)) or current_price is None:
                        current_price = 0.0
                elif trade.asset_type == "crypto":
                    data = self.crypto_bot.get_crypto_data(symbol)
                    if isinstance(data, dict) and "current_price" in data and isinstance(data["current_price"], (int, float)):
                        current_price = data["current_price"]
                    else:
                        current_price = 0.0
                else:
                    current_price = 0.0
                position_obj = Position(
                    symbol=symbol,
                    quantity=trade.quantity,
                    entry_price=trade.entry_price,
                    current_price=current_price,
                    asset_type=trade.asset_type,
                )
                position_objects.append(position_obj)

            # Calculate portfolio-level risk
            portfolio_risk = self.risk_manager.calculate_portfolio_risk(
                position_objects
            )
            portfolio_risk_pct = portfolio_risk.get("portfolio_pnl_percent", 0)

            print(
                f"   Portfolio Risk Level: {abs(portfolio_risk_pct)*100:.1f}%"
            )  # CLI/test output only

            # Check if we should halt trading due to high portfolio risk
            if abs(portfolio_risk_pct) > MAX_PORTFOLIO_RISK:
                print(
                    f"   PORTFOLIO RISK TOO HIGH ({abs(portfolio_risk_pct)*100:.1f}% > {MAX_PORTFOLIO_RISK*100:.1f}%)"
                )  # CLI/test output only
                return False
            else:
                print(
                    "   Portfolio risk within acceptable limits"
                )  # CLI/test output only
                return True

        except Exception as e:
            print(
                "   Portfolio risk assessment failed, proceeding with caution"
            )  # CLI/test output only
            return True  # Proceed with caution if assessment fails

    @log_method_calls
    def _perform_ai_adaptation(self) -> Tuple[float, Optional[str]]:
        """
        STEP 1: AI Learning & Adaptation

        Returns:
            Tuple[float, Optional[str]]: (min_confidence, prompt_note)
        """
        print("\nAI Learning & Adaptation...")

        try:
            # Get trade history and performance metrics for AI learning
            trade_history = self.portfolio_bot.get_trade_history()
            metrics = self.portfolio_bot.get_portfolio_metrics()
            win_rate = metrics.get("win_rate", 0.0)

            # AI adapts its confidence threshold based on performance
            min_confidence, prompt_note = self.ai_bot.adapt_with_performance(
                trade_history, win_rate
            )
            print(
                f"   Win Rate: {win_rate*100:.1f}% | Adapted Min Confidence: {min_confidence:.1f}"
            )
            print(
                f"   AI Prompt Note: {prompt_note[:100]}..."
                if prompt_note
                else "   No specific prompt adaptations"
            )

            return min_confidence, prompt_note

        except Exception as e:
            return MIN_CONFIDENCE, None

    @staticmethod
    @log_method_calls
    def _clean_context_field(field):
        """
        Recursively clean a context field by removing empty strings, empty dicts, and empty lists.
        For dicts: remove keys with empty values.
        For lists: remove empty/blank items.
        For strings: return only if non-empty after strip.
        """
        if field is None:
            return None
        if isinstance(field, str):
            return field.strip() if field.strip() else None
        if isinstance(field, dict):
            cleaned = {k: OrchestratorBot._clean_context_field(v) for k, v in field.items()}
            # Remove keys with None or empty values
            return {k: v for k, v in cleaned.items() if v not in (None, '', [], {})}
        if isinstance(field, list):
            cleaned = [OrchestratorBot._clean_context_field(v) for v in field]
            # Remove None or empty items
            return [v for v in cleaned if v not in (None, '', [], {})]
        return field

    @log_method_calls
    def _analyze_assets_and_decide(
        self,
        min_confidence: float,
        prompt_note: Optional[str],
        screened_assets: List[Any],
    ) -> List[Tuple[CoreTradingDecision, dict]]:
        """
        STEP 2: Analyze each asset and make trading decisions

        Args:
            min_confidence: Minimum confidence threshold from AI adaptation
            prompt_note: Enhanced prompt note from AI adaptation
            screened_assets: List of assets screened as promising by AssetScreenerBot

        Returns:
            List[TradingDecision]: List of trading decisions for approved trades
        """
        print("\nAsset Analysis & Trading Decisions...")

        trading_decisions = []

        # Calculate current portfolio risk once for this cycle
        try:
            open_positions = self.portfolio_bot.get_open_positions()
            position_objects = []
            for symbol, trade in open_positions.items():
                if trade.asset_type == "stock":
                    current_price = self.stock_bot.get_current_price(symbol)
                    if not isinstance(current_price, (int, float)) or current_price is None:
                        current_price = 0.0
                elif trade.asset_type == "crypto":
                    data = self.crypto_bot.get_crypto_data(symbol)
                    if isinstance(data, dict) and "current_price" in data and isinstance(data["current_price"], (int, float)):
                        current_price = data["current_price"]
                    else:
                        current_price = 0.0
                else:
                    current_price = 0.0
                position_obj = Position(
                    symbol=symbol,
                    quantity=trade.quantity,
                    entry_price=trade.entry_price,
                    current_price=current_price,
                    asset_type=trade.asset_type,
                )
                position_objects.append(position_obj)
            portfolio_risk = self.risk_manager.calculate_portfolio_risk(position_objects)
            current_portfolio_risk = portfolio_risk.get("portfolio_pnl_percent", 0)
        except Exception:
            current_portfolio_risk = 0

        for asset in screened_assets:
            symbol = asset.symbol
            asset_type = asset.asset_type
            allocation_usd = asset.allocation_usd

            try:
                print(f"\n   Analyzing {symbol} ({asset_type.upper()})...")
                self.logger.info(f"[ANALYSIS][START] {symbol} ({asset_type}) - allocation: {allocation_usd}")

                # === GET AI ANALYSIS ===
                analysis = self._get_asset_analysis(symbol, asset_type, prompt_note)
                if not analysis or "error" in analysis:
                    self.logger.error(f"[ANALYSIS][FAIL] {symbol} ({asset_type}) - AI analysis failed: {analysis.get('error') if analysis else 'No analysis'}")
                    print(f"      Analysis failed for {symbol}")
                    continue

                # Fetch live price for each asset
                if asset_type == "stock":
                    current_price = self.stock_bot.get_current_price(symbol)
                    self.logger.info(f"[STOCK_DATA][FETCH] {symbol} - current_price: {current_price}")
                    if current_price is None:
                        current_price = 0.0
                elif asset_type == "crypto":
                    data = self.crypto_bot.get_crypto_data(symbol)
                    self.logger.info(f"[CRYPTO_DATA][FETCH] {symbol} - data: {data}")
                    current_price = (
                        data["current_price"]
                        if data and data["current_price"] is not None
                        else 0.0
                    )
                else:
                    current_price = 0.0
                # === Gather RAG context for DecisionMakerBot ===
                historical_context = self._get_historical_analysis_context(symbol)
                self.logger.debug(f"[DB][HISTORICAL_CONTEXT][FETCH] {symbol} - context: {historical_context}")
                if not historical_context or (isinstance(historical_context, str) and not historical_context.strip()):
                    historical_context = []
                if isinstance(historical_context, str):
                    historical_context = [{"text": historical_context}] if historical_context.strip() else []
                reflection_insights = self.reflection_bot.generate_enhanced_prompt_note(symbol)
                self.logger.debug(f"[REFLECTION][FETCH] {symbol} - insights: {reflection_insights}")
                if not reflection_insights or (isinstance(reflection_insights, str) and not reflection_insights.strip()):
                    reflection_insights = []
                if isinstance(reflection_insights, str):
                    reflection_insights = [reflection_insights] if reflection_insights.strip() else []

                # --- Log news fetch and summary ---
                news_sentiment = locals().get('news_summary', None)
                try:
                    if hasattr(self, 'news_retriever'):
                        last_news_query = f"{symbol} {'stock' if asset_type=='stock' else 'crypto'} news"
                        news_articles = self.news_retriever.fetch_news(last_news_query, max_results=5)
                        self.logger.info(f"[NEWS][FETCH] {symbol} - query: '{last_news_query}', articles: {[getattr(a, 'title', str(a)) for a in news_articles] if news_articles else 'None'}")
                        if not news_articles:
                            print(f"      No news articles fetched for {symbol} ({asset_type}) with query '{last_news_query}'")
                        else:
                            print(f"      Fetched news articles for {symbol}: {[getattr(a, 'title', str(a)) for a in news_articles]}")
                        # Try to get summary
                        try:
                            news_chunks = self.news_retriever.preprocess_and_chunk(news_articles)
                            self.news_retriever.generate_embeddings(news_chunks)
                            news_summary_debug = self.news_retriever.augment_context_and_llm(last_news_query)
                            self.logger.info(f"[NEWS][SUMMARY] {symbol} - summary: {news_summary_debug}")
                            print(f"      News summary debug for {symbol}: {news_summary_debug if news_summary_debug else 'BLANK'}")
                        except Exception as e2:
                            self.logger.error(f"[NEWS][SUMMARY][FAIL] {symbol} - {e2}")
                            print(f"      Could not summarize news for {symbol}: {e2}")
                except Exception as e:
                    self.logger.error(f"[NEWS][FETCH][FAIL] {symbol} - {e}")
                    print(f"      Could not fetch/log news for {symbol}: {e}")

                # --- DEBUG: Log all context fields before cleaning ---
                # REMOVE: logger.debug(...)

                # Clean up context fields
                if news_sentiment is None or (isinstance(news_sentiment, str) and not news_sentiment.strip()):
                    news_sentiment = {}
                else:
                    news_sentiment = news_sentiment if not isinstance(news_sentiment, str) else {"summary": news_sentiment}
                # Clean all context fields
                cleaned_news_sentiment = self._clean_context_field(news_sentiment)
                cleaned_reflection_insights = self._clean_context_field(reflection_insights)
                cleaned_historical_context = self._clean_context_field(historical_context)

                # --- DEBUG: Log all context fields after cleaning ---
                # REMOVE: logger.debug(...)

                # Warn if all context is empty after cleaning
                if not analysis.get("indicators") and not cleaned_news_sentiment and not cleaned_reflection_insights and not cleaned_historical_context:
                    print(f"      Warning: All context empty for {symbol} (after cleaning). Decision may default to HOLD.")
                # Prepare analysis input for DecisionMakerBot with full RAG context
                # Fix types for AssetAnalysisInput dataclass
                def _to_str_list(val):
                    if val is None:
                        return None
                    if isinstance(val, list):
                        return [str(x) for x in val if isinstance(x, str)]
                    if isinstance(val, str):
                        return [val]
                    return None
                def _to_dict_list(val):
                    if val is None:
                        return None
                    if isinstance(val, list):
                        return [x for x in val if isinstance(x, dict)]
                    if isinstance(val, dict):
                        return [val]
                    return None
                cleaned_news_sentiment = cleaned_news_sentiment if isinstance(cleaned_news_sentiment, dict) else None
                cleaned_reflection_insights = _to_str_list(cleaned_reflection_insights)
                cleaned_historical_context = _to_dict_list(cleaned_historical_context)
                analysis_input = AssetAnalysisInput(
                    symbol=symbol,
                    market_data={
                        "action": analysis.get("action", "").lower(),
                        "confidence": analysis.get("confidence", 0.0),
                    },
                    technical_indicators=analysis.get("indicators", {}),
                    news_sentiment=cleaned_news_sentiment,
                    reflection_insights=cleaned_reflection_insights,
                    historical_ai_context=cleaned_historical_context,
                    asset_type=asset_type
                )
                # --- Explicit BLANK logging for context fields ---
                def _show_blank(val):
                    if val is None or val == {} or val == [] or val == "":
                        return "BLANK"
                    return val
                print(
                    f"      Decision input for {symbol} | market_data={analysis_input.market_data} | "
                    f"technical_indicators={_show_blank(analysis_input.technical_indicators)} | "
                    f"news_sentiment={_show_blank(cleaned_news_sentiment)} | "
                    f"reflection_insights={_show_blank(cleaned_reflection_insights)} | "
                    f"historical_ai_context={_show_blank(cleaned_historical_context)}"
                )
                self.logger.info(f"[DECISION_MAKER][INPUT] {symbol} | market_data={analysis_input.market_data} | technical_indicators={_show_blank(analysis_input.technical_indicators)} | news_sentiment={_show_blank(cleaned_news_sentiment)} | reflection_insights={_show_blank(cleaned_reflection_insights)} | historical_ai_context={_show_blank(cleaned_historical_context)}")

                # Use DecisionMakerBot for final decision
                try:
                    decision_obj = self.decision_maker.make_trading_decision(
                        analysis_input,
                        min_confidence=min_confidence,
                        current_portfolio_risk=current_portfolio_risk,
                    )
                    self.logger.info(f"[DECISION_MAKER][OUTPUT] {symbol} | action={getattr(decision_obj.signal, 'value', decision_obj.signal)} | confidence={decision_obj.confidence} | rationale={getattr(decision_obj, 'rationale', 'N/A')}")
                except Exception as e:
                    self.logger.error(f"[DECISION_MAKER][ERROR] {symbol} - {e}")
                    print(f"      DecisionMakerBot error for {symbol}: {e}")
                    continue
                # Map ActionSignal to string for action
                action_str = (
                    str(decision_obj.signal.value).lower()
                    if hasattr(decision_obj.signal, "value")
                    else str(decision_obj.signal).lower()
                )
                if action_str not in ["buy", "sell"]:
                    print("      DecisionMakerBot recommends HOLD - no action taken")
                    self.logger.info(f"[DECISION_MAKER][HOLD] {symbol} - No actionable signal.")
                    continue
                # Create trading decision for execution (using shared structure)
                decision = CoreTradingDecision(
                    symbol=symbol,
                    signal=decision_obj.signal,
                    confidence=decision_obj.confidence,
                    rationale=(
                        decision_obj.rationale
                        if hasattr(decision_obj, "rationale") and decision_obj.rationale is not None
                        else "No rationale provided."
                    ),
                    metadata=(
                        decision_obj.metadata
                        if hasattr(decision_obj, "metadata")
                        else {}
                    ),
                    # Remove any parameters not in TradingDecision dataclass
                )
                # Store extra execution details in a dict
                extra = {
                    "asset_type": asset_type,
                    "current_price": current_price,
                    "allocation_usd": allocation_usd,
                }
                trading_decisions.append((decision, extra))
                print(
                    f"      Trade decision approved for {symbol}"
                )  # CLI/test output only
                self.logger.info(f"[DECISION_APPROVED] {symbol} | action={action_str} | confidence={decision.confidence} | position_allocation={allocation_usd}")

            except Exception as e:
                print(f"   Error processing {symbol}: {e}")  # CLI/test output only
                self.logger.error(f"[ANALYSIS][ERROR] {symbol} - {e}")
                continue

        print(
            f"\nGenerated {len(trading_decisions)} trading decisions"
        )  # CLI/test output only
        return trading_decisions

    @log_method_calls
    def _get_asset_analysis(
        self, symbol: str, asset_type: str, prompt_note: Optional[str]
    ) -> Dict[str, Any]:
        """Get AI analysis for a specific asset, including news insights and reflection insights, with LLM answer memory/reuse and news sentiment similarity."""
        try:
            cycle_id = time.strftime('%Y%m%d%H%M%S')
            # === Gather current news sentiment and embedding ===
            news_query = (
                f"{symbol} stock news"
                if asset_type == "stock"
                else f"{symbol} crypto news"
            )
            news_articles = self.news_retriever.fetch_news(news_query, max_results=5)
            news_chunks = self.news_retriever.preprocess_and_chunk(news_articles)
            self.news_retriever.generate_embeddings(news_chunks)
            news_summary = self.news_retriever.augment_context_and_llm(news_query)
            # Use the first embedding as a summary embedding for similarity (could be improved)
            current_news_embedding = (
                news_chunks[0].embedding
                if news_chunks and hasattr(news_chunks[0], "embedding")
                else None
            )

            # === LLM ANSWER MEMORY/REUSE LOGIC with news sentiment similarity ===
            recent_analyses = self.database_bot.get_analysis_context(
                symbol, context_types=["llm_analysis"], days=3, limit=10
            )
            similar_past = None
            sim = None
            similarity_threshold = 0.85  # Cosine similarity threshold for news context
            if current_news_embedding:
                import numpy as np
                for entry in recent_analyses:
                    ctx = entry.get("context_data", {})
                    past_embedding = ctx.get("news_embedding")
                    if past_embedding:
                        v1, v2 = np.array(current_news_embedding), np.array(past_embedding)
                        sim = float(
                            np.dot(v1, v2)
                            / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                        )
                        if sim > similarity_threshold:
                            similar_past = ctx
                            break
            if similar_past:
                return {
                    "action": similar_past.get("raw_llm_response", {}).get("action"),
                    "confidence": similar_past.get("raw_llm_response", {}).get("confidence"),
                    "reasoning": similar_past.get("raw_llm_response", {}).get("reasoning"),
                    "source": "memory",
                    "timestamp": similar_past.get("timestamp"),
                    "news_similarity": sim,
                }
            # === CONTEXT GATHERING ===
            historical_context = self._get_historical_analysis_context(symbol)
            reflection_insights = self.reflection_bot.generate_enhanced_prompt_note(symbol)
            enhanced_prompt = prompt_note or ""
            if historical_context:
                enhanced_prompt += f"\n\nHistorical Analysis Context:\n{historical_context}"
            if reflection_insights:
                enhanced_prompt += f"\n\n{reflection_insights}"
            if news_summary:
                enhanced_prompt += f"\n\nNews Insights:\n{news_summary}"
            # --- LOG: Full prompt/context sent to LLM ---
            # 3. Make new LLM call if no suitable prior exists
            if asset_type == "stock":
                analysis = self.stock_bot.analyze_stock(symbol, prompt_note=enhanced_prompt)
            elif asset_type == "crypto":
                analysis = self.crypto_bot.analyze_crypto(symbol, prompt_note=enhanced_prompt)
            else:
                return {"error": f"Unknown asset type: {asset_type}"}
            # --- LOG: Raw LLM response ---
            # 4. Store full LLM response context in database, including news embedding
            context_data = {
                "symbol": symbol,
                "asset_type": asset_type,
                "prompt": enhanced_prompt,
                "news_summary": news_summary,
                "news_embedding": current_news_embedding,
                "reflection_insights": reflection_insights,
                "historical_context": historical_context,
                "raw_llm_response": analysis,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            # Fix: ensure relevance_score is always a float
            confidence_val = analysis.get("confidence", 0.0) if isinstance(analysis, dict) else 0.0
            try:
                relevance_score = float(confidence_val)
            except Exception:
                relevance_score = 0.0
            self.database_bot.store_analysis_context(
                symbol,
                "llm_analysis",
                context_data,
                relevance_score=relevance_score,
            )
            # 5. Mark source as fresh LLM call
            if isinstance(analysis, dict):
                analysis["source"] = "fresh_llm_call"
                return analysis
            elif isinstance(analysis, str):
                return {"result": analysis, "source": "fresh_llm_call"}
            else:
                return {
                    "error": "Unknown analysis result type",
                    "source": "fresh_llm_call",
                }
        except Exception as e:
            return {"error": str(e), "source": "error"}

    @log_method_calls
    def _get_historical_analysis_context(self, symbol: str) -> str:
        """
        RAG: Get historical analysis context for better AI decisions

        Args:
            symbol: Asset symbol to get history for

        Returns:
            str: Formatted historical context
        """
        try:
            # Get recent analysis history from database
            analysis_history = self.database_bot.get_analysis_history(symbol)

            if not analysis_history:
                return ""

            context_lines = []
            for analysis in analysis_history:
                date = analysis.get("timestamp", "Unknown")
                action = analysis.get("action", "Unknown")
                confidence = analysis.get("confidence", 0.0)
                context_lines.append(
                    f"- {date}: {action.upper()} (confidence: {confidence:.1f})"
                )

            return f"Recent AI decisions for {symbol}:\n" + "\n".join(context_lines)

        except Exception as e:
            return ""

    @log_method_calls
    def _execute_approved_trades(
        self, trading_decisions: List[Tuple[CoreTradingDecision, dict]]
    ) -> List[dict]:
        """
        STEP 3: Execute approved trades with position sizing and risk checks

        Args:
            trading_decisions: List of approved trading decisions

        Returns:
            List[Dict[str, Any]]: Execution results
        """
        print("\nExecuting Approved Trades...")

        execution_results = []

        for decision, extra in trading_decisions:
            try:
                print(f"\n   Processing {decision.symbol}...")
                self.logger.info(f"[TRADE_EXECUTION][START] {decision.symbol} | action={getattr(decision.signal, 'value', decision.signal)} | confidence={decision.confidence}")

                # === POSITION SIZING ===
                shares = self._calculate_position_size(decision, extra)
                if shares <= 0:
                    print(
                        "      Position size calculation resulted in 0 shares - skipping"
                    )
                    self.logger.warning(f"[POSITION_SIZER][SKIP] {decision.symbol} - Calculated shares: {shares}")
                    continue
                extra["position_size"] = shares
                print(
                    f"      Position Size: {shares} shares @ ${extra['current_price']:.2f}"
                )
                self.logger.info(f"[POSITION_SIZER][RESULT] {decision.symbol} - shares: {shares}, price: {extra['current_price']}, allocation: {extra['allocation_usd']}")

                # === PRE-TRADE RISK CHECKS ===
                if not self._check_position_risk(decision, extra):
                    print("      Position risk too high - skipping trade")
                    self.logger.warning(f"[RISK_MANAGER][SKIP] {decision.symbol} - Position risk too high.")
                    continue

                # === TRADE EXECUTION ===
                execution_result = self._execute_trade(decision, extra)
                execution_results.append(execution_result)
                self.logger.info(f"[TRADE_EXECUTION][RESULT] {decision.symbol} - success: {execution_result.get('success', False)} | order_response: {execution_result.get('order_response', 'N/A')}")

            except Exception as e:
                print(f"   Error executing {decision.symbol}: {e}")
                self.logger.error(f"[TRADE_EXECUTION][ERROR] {decision.symbol} - {e}")
                continue
        return execution_results

    @log_method_calls
    def _calculate_position_size(
        self, decision: CoreTradingDecision, extra: dict
    ) -> float:
        """Calculate position size using PositionSizerBot"""
        try:
            shares = self.position_sizer.calculate_position_size(
                price=extra["current_price"],
                confidence=decision.confidence,
                volatility=None,  # Could add volatility calculation
                available_cash=extra["allocation_usd"],
                min_shares=1,
                allow_fractional=(extra["asset_type"] == "crypto"),
            )
            self.logger.info(f"[POSITION_SIZER][CALL] {decision.symbol} - price: {extra['current_price']}, confidence: {decision.confidence}, allocation: {extra['allocation_usd']}, shares: {shares}")
            return max(0, shares)
        except Exception as e:
            self.logger.error(f"[POSITION_SIZER][ERROR] {decision.symbol} - {e}")
            return 0

    @log_method_calls
    def _check_position_risk(self, decision: CoreTradingDecision, extra: dict) -> bool:
        """Check if position risk is acceptable"""
        try:
            # Create hypothetical position for risk calculation
            hypothetical_position = Position(
                symbol=decision.symbol,
                quantity=extra["position_size"],
                entry_price=extra["current_price"],
                current_price=extra["current_price"],
                asset_type=extra["asset_type"],
            )
            self.logger.info(f"[RISK_MANAGER][CALL] {decision.symbol} - position: {hypothetical_position}")
            # Calculate position risk
            position_risk = self.risk_manager.calculate_position_risk(
                hypothetical_position
            )
            position_investment = position_risk["investment"]
            max_loss = position_risk["max_loss"]
            self.logger.info(f"[RISK_MANAGER][RESULT] {decision.symbol} - investment: {position_investment}, max_loss: {max_loss}")
            # Check if position risk exceeds limits
            if (
                position_investment
                > self.portfolio_bot.current_capital * MAX_POSITION_RISK
            ):
                print(
                    f"      Position risk too high: ${position_investment:.2f} > ${self.portfolio_bot.current_capital * MAX_POSITION_RISK:.2f}"
                )
                self.logger.warning(f"[RISK_MANAGER][LIMIT] {decision.symbol} - investment: {position_investment}, max allowed: {self.portfolio_bot.current_capital * MAX_POSITION_RISK}")
                return False
            print(f"      Risk acceptable - Max Loss: ${max_loss:.2f}")
            extra["risk_assessment"] = position_risk
            self.logger.info(f"[RISK_MANAGER][PASS] {decision.symbol} - investment: {position_investment}, max_loss: {max_loss}")
            return True
        except Exception as e:
            self.logger.error(f"[RISK_MANAGER][ERROR] {decision.symbol} - {e}")
            return False

    @log_method_calls
    def _execute_trade(self, decision: CoreTradingDecision, extra: dict) -> dict:
        """Execute a single trade"""
        try:
            action_str = (
                str(decision.signal.value).lower()
                if hasattr(decision.signal, "value")
                else str(decision.signal).lower()
            )
            print(f"      Executing {action_str} order for {decision.symbol}...")
            self.logger.info(f"[TRADE_EXECUTION][ORDER] {decision.symbol} - action: {action_str}, qty: {extra['position_size']}, price: {extra['current_price']}")
            success, order_response = self.trade_executor.execute_trade(
                symbol=decision.symbol,
                side=action_str,
                quantity=extra["position_size"],
                confidence=decision.confidence,
            )
            self.logger.info(f"[TRADE_EXECUTION][RESPONSE] {decision.symbol} - success: {success}, order_response: {order_response}")
            execution_result = {
                "decision": decision,
                "extra": extra,
                "success": success,
                "order_response": order_response,
                "timestamp": time.time(),
            }
            if success:
                print("      Trade executed successfully!")
                print(f"         Order ID: {getattr(order_response, 'id', 'N/A')}")
                self.logger.info(f"[TRADE_EXECUTION][SUCCESS] {decision.symbol} - order_id: {getattr(order_response, 'id', 'N/A')}")
            else:
                print(f"      Trade execution failed: {order_response}")
                self.logger.warning(f"[TRADE_EXECUTION][FAIL] {decision.symbol} - {order_response}")
            return execution_result
        except Exception as e:
            self.logger.error(f"[TRADE_EXECUTION][ERROR] {decision.symbol} - {e}")
            return {
                "decision": decision,
                "extra": extra,
                "success": False,
                "error": str(e),
                "timestamp": time.time(),
            }

    @log_method_calls
    def _update_portfolio_state(self, execution_results: List[dict]):
        """
        STEP 4: Update portfolio state based on execution results
        """
        print("\nUpdating Portfolio State...")
        for result in execution_results:
            if not result["success"]:
                self.logger.warning(f"[PORTFOLIO][SKIP] {result['decision'].symbol} - Not updating portfolio due to failed execution.")
                continue
            try:
                decision = result["decision"]
                extra = result["extra"]
                action_str = (
                    str(decision.signal.value).lower()
                    if hasattr(decision.signal, "value")
                    else str(decision.signal).lower()
                )
                if action_str == "buy":
                    self.portfolio_bot.add_or_update_position(
                        symbol=decision.symbol,
                        asset_type=extra["asset_type"],
                        quantity=extra["position_size"],
                        entry_price=extra["current_price"],
                    )
                    self.portfolio_bot.current_capital -= (
                        extra["position_size"] * extra["current_price"]
                    )
                    print(f"   Added {decision.symbol} position to portfolio")
                    self.logger.info(f"[PORTFOLIO][ADD] {decision.symbol} - qty: {extra['position_size']}, price: {extra['current_price']}")
                    try:
                        self.database_bot.store_trade_outcome(
                            symbol=decision.symbol,
                            asset_type=extra["asset_type"],
                            trade_type="buy",
                            entry_price=extra["current_price"],
                            exit_price=0.0,  # Not closed yet
                            quantity=extra["position_size"],
                            hold_duration_minutes=0,  # Open trade
                            original_confidence=decision.confidence,
                            analysis_id=-1,  # No analysis_id available
                            execution_status="executed",
                            fees=0.0,  # Optional: add if available
                        )
                        self.logger.info(f"[DATABASE][RECORD] BUY {decision.symbol} - qty: {extra['position_size']}, price: {extra['current_price']}")
                    except Exception as e:
                        print(f"   Failed to record BUY trade for {decision.symbol}: {e}")
                        self.logger.error(f"[DATABASE][ERROR] BUY {decision.symbol} - {e}")
                elif action_str == "sell":
                    self.portfolio_bot.close_position(
                        decision.symbol, extra["current_price"]
                    )
                    print(f"   Closed {decision.symbol} position in portfolio")
                    self.logger.info(f"[PORTFOLIO][CLOSE] {decision.symbol} - price: {extra['current_price']}")
                    try:
                        closed_trade = None
                        for t in reversed(self.portfolio_bot.get_trade_history()):
                            if (
                                t.symbol == decision.symbol
                                and t.exit_price == extra["current_price"]
                            ):
                                closed_trade = t
                                break
                        if closed_trade:
                            hold_duration = (
                                (
                                    closed_trade.exit_date - closed_trade.entry_date
                                ).total_seconds()
                                / 60.0
                                if closed_trade.exit_date and closed_trade.entry_date
                                else 0
                            )
                            self.database_bot.store_trade_outcome(
                                symbol=closed_trade.symbol,
                                asset_type=closed_trade.asset_type,
                                trade_type="sell",
                                entry_price=closed_trade.entry_price,
                                exit_price=closed_trade.exit_price,
                                quantity=closed_trade.quantity,
                                hold_duration_minutes=int(hold_duration),
                                original_confidence=closed_trade.confidence,
                                analysis_id=-1,  # No analysis_id available
                                execution_status="executed",
                                fees=0.0,  # Optional: add if available
                            )
                            self.logger.info(f"[DATABASE][RECORD] SELL {closed_trade.symbol} - qty: {closed_trade.quantity}, price: {closed_trade.exit_price}")
                        else:
                            print(
                                f"   Could not find closed trade for {decision.symbol} to record in DatabaseBot."
                            )
                            self.logger.warning(f"[DATABASE][ERROR] SELL {decision.symbol} - Could not find closed trade to record.")
                    except Exception as e:
                        print(f"   Failed to record SELL trade for {decision.symbol}: {e}")
                        self.logger.error(f"[DATABASE][ERROR] SELL {decision.symbol} - {e}")
            except Exception as e:
                print(f"   Error updating portfolio for {decision.symbol}: {e}")
                self.logger.error(f"[PORTFOLIO][ERROR] {decision.symbol} - {e}")
        print(f"   Updated capital: ${self.portfolio_bot.current_capital:.2f}")
        self.logger.info(f"[PORTFOLIO][CAPITAL] Updated capital: ${self.portfolio_bot.current_capital:.2f}")

    @log_method_calls
    def _run_post_trade_reflection(self):
        """Analyze any trades closed in this cycle using ReflectionBot and update knowledge graph if enabled"""
        print("\nRunning Post-Trade Reflection...")
        closed_trades = self.portfolio_bot.get_trade_history()
        if not closed_trades:
            print("   No closed trades to reflect on.")
            self.logger.info("[REFLECTION][SKIP] No closed trades to reflect on.")
            return
        for trade in closed_trades[-5:]:  # Only reflect on the most recent trades (adjust as needed)
            if trade.exit_date is None:
                continue  # Only reflect on closed trades
            trade_outcome = TradeOutcome(
                trade_id=f"{trade.symbol}_{trade.entry_date.strftime('%Y%m%d%H%M%S')}",
                symbol=trade.symbol,
                asset_type=trade.asset_type,
                action=trade.action,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price if trade.exit_price is not None else 0.0,
                quantity=trade.quantity,
                entry_time=trade.entry_date,
                exit_time=trade.exit_date,
                pnl=(
                    (trade.exit_price - trade.entry_price) * trade.quantity
                    if trade.exit_price is not None
                    else 0.0
                ),
                pnl_percent=(
                    ((trade.exit_price - trade.entry_price) / trade.entry_price)
                    if trade.exit_price is not None and trade.entry_price != 0
                    else 0.0
                ),
                duration_hours=(
                    (trade.exit_date - trade.entry_date).total_seconds() / 3600
                    if trade.exit_date is not None
                    else 0.0
                ),
                original_analysis=None,  # Optionally fetch/store original analysis if available
                market_conditions_entry=None,
                market_conditions_exit=None,
            )
            try:
                self.logger.info(f"[REFLECTION][START] {trade.symbol} - trade_outcome: {trade_outcome}")
                reflection_result = self.reflection_bot.analyze_completed_trade(trade_outcome)
                self.logger.info(f"[REFLECTION][RESULT] {trade.symbol} - {reflection_result}")
                # If knowledge graph integration is present, log update
                if hasattr(self, 'knowledge_graph') and self.knowledge_graph:
                    try:
                        kg_result = self.knowledge_graph.update_from_reflection(trade_outcome, reflection_result)
                        self.logger.info(f"[KNOWLEDGE_GRAPH][UPDATE] {trade.symbol} - {kg_result}")
                    except Exception as kg_e:
                        self.logger.error(f"[KNOWLEDGE_GRAPH][ERROR] {trade.symbol} - {kg_e}")
            except Exception as e:
                print(f"   Reflection error for {trade.symbol}: {e}")
                self.logger.error(f"[REFLECTION][ERROR] {trade.symbol} - {e}")
        print("   Reflection complete.")
        self.logger.info("[REFLECTION][COMPLETE] Post-trade reflection complete.")

    @log_method_calls
    def _print_cycle_summary(self):
        """Print summary of the trading cycle"""
        print("\nTrading Cycle Summary...")
        self.portfolio_bot.print_portfolio_summary()

    @log_method_calls
    def process_cycle(self):
        method = "process_cycle"
        # ...existing code...
