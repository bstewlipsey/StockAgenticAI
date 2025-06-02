# bot_decision_maker.py
"""
DecisionMakerBot: Centralizes final trading decisions based on multiple inputs.
- Receives analysis data from StockBot, CryptoBot, and AIBot
- Applies dynamically adapted confidence thresholds
- Incorporates strategic rules and market sentiment filters
- Outputs clear, actionable trading decisions with rationale
"""

from typing import Dict, Any, Optional

# Import configuration
from config_trading import (
    MIN_CONFIDENCE,
    CONFIDENCE_BOOST_FACTORS,
    RISK_PENALTY_FACTORS,
    FILTERS_ENABLED,
)
from data_structures import AssetAnalysisInput, TradingDecision, ActionSignal
from utils.logger_mixin import LoggerMixin
from utils.logging_decorators import log_method_calls


class DecisionMakerBot(LoggerMixin):
    """
    CENTRALIZED TRADING DECISION ENGINE

    Consolidates analysis from multiple sources and applies strategic filters
    to make final trading decisions. Acts as the decision layer between
    analysis and execution.
    """

    @log_method_calls
    def __init__(self):
        """
        Initialize the DecisionMakerBot with confidence, risk, and filter parameters.
        Logs successful initialization.
        """
        super().__init__()
        # Decision-making parameters
        self.confidence_boost_factors = CONFIDENCE_BOOST_FACTORS
        # Risk adjustment factors
        self.risk_penalty_factors = RISK_PENALTY_FACTORS
        # Strategic filters
        self.filters_enabled = FILTERS_ENABLED
        self.logger.info("DecisionMakerBot initialized successfully")

    @log_method_calls
    def log_prompt_context(self, analysis_input: AssetAnalysisInput, prompt: str):
        """
        Log LLM prompt context in a standardized, configurable way.
        - If LOG_FULL_PROMPT is False: log only a summary/identifier at INFO level.
        - If LOG_FULL_PROMPT is True: log the full prompt at DEBUG level.
        All logs use the standardized format.
        """
        from config_system import LOG_FULL_PROMPT

        method = "log_prompt_context"
        prompt_type = getattr(analysis_input, "prompt_type", "unknown")
        asset = getattr(analysis_input, "symbol", "unknown")
        # Always log a concise summary at INFO
        if not LOG_FULL_PROMPT:
            self.logger.info(
                f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}()] LLM_PROMPT_CONTEXT | prompt_type={prompt_type} | asset={asset}"
            )
        # Log the full prompt at DEBUG if enabled
        if LOG_FULL_PROMPT:
            self.logger.debug(
                f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}()] LLM_PROMPT_FULL | prompt_type={prompt_type} | asset={asset} | prompt=\n{prompt}"
            )
            # Optionally log RAG sections if present
            if getattr(analysis_input, "reflection_insights", None):
                self.logger.debug(
                    f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}()] LLM_PROMPT_RAG_REFLECTION | asset={asset} | insights={str(analysis_input.reflection_insights)}"
                )
            if getattr(analysis_input, "historical_ai_context", None):
                self.logger.debug(
                    f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}()] LLM_PROMPT_RAG_HISTORY | asset={asset} | context={str(analysis_input.historical_ai_context)}"
                )

    @log_method_calls
    def make_trading_decision(
        self,
        analysis_input: AssetAnalysisInput,
        min_confidence: float,
        current_portfolio_risk: float = 0.0,
        market_conditions: Optional[Dict[str, Any]] = None,
    ) -> TradingDecision:
        """
        Make final trading decision based on analysis input and filters.
        Logs detailed info for crypto assets: symbol, action, confidence, and reasoning.
        Implements post-processing override for unjustified 'hold' actions if all indicators are present.

        Args:
            analysis_input: Analysis from StockBot/CryptoBot/AIBot
            min_confidence: Dynamically adapted minimum confidence threshold
            current_portfolio_risk: Current portfolio risk percentage
            market_conditions: Optional market condition data

        Returns:
            TradingDecision: Final trading decision with rationale
        """
        method = "make_trading_decision"
        self.logger.debug(
            f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}(asset_type='{analysis_input.asset_type}', hold_override_threshold={min_confidence})] START"
        )
        self.logger.info(f"Making trading decision for {analysis_input.symbol}")
        # Try to extract action/confidence/reasoning from market_data or technical_indicators
        action = (
            analysis_input.market_data.get("action")
            or analysis_input.technical_indicators.get("action")
            or "hold"
        ).lower()
        confidence = (
            analysis_input.market_data.get("confidence")
            or analysis_input.technical_indicators.get("confidence")
            or 0.0
        )
        rationale = (
            analysis_input.market_data.get("reasoning")
            or analysis_input.technical_indicators.get("reasoning")
            or "No rationale provided."
        )
        # Enhanced logging for crypto assets
        asset_type = getattr(analysis_input, "asset_type", None)
        if asset_type is None:
            # Try to infer asset_type from symbol
            symbol = getattr(analysis_input, "symbol", "")
            if "/" in symbol or symbol.endswith("USD"):
                asset_type = "crypto"
            else:
                asset_type = "stock"
        if asset_type == "crypto":
            self.logger.info(f"Analyzing {analysis_input.symbol} (CRYPTO)...")
            self.logger.info(f"Action: {action.upper()} | Confidence: {confidence:.2f}")
            self.logger.info(f"Reasoning: {rationale}")
            self.logger.info(
                f"Crypto Decision: action={action}, confidence={confidence}, reasoning={rationale}"
            )
        # Map action string to ActionSignal enum
        try:
            signal = ActionSignal[action.upper()]
        except Exception:
            signal = ActionSignal.HOLD
        # Apply confidence adjustments and filters as needed (simplified)
        adjusted_confidence = confidence
        # === POST-PROCESSING OVERRIDE FOR UNJUSTIFIED 'HOLD' ===
        # If action is 'hold', confidence is low, and all required indicators are present and not contradictory, override to BUY/SELL with low confidence
        override_triggered = False
        if signal == ActionSignal.HOLD:
            indicators = analysis_input.technical_indicators or {}
            # Check for presence of all required indicators (RSI, MACD, SMA20, price vs SMA20)
            required_keys = ["rsi", "macd", "sma_20", "price_vs_sma20"]
            all_present = all(
                k in indicators and indicators[k] is not None for k in required_keys
            )
            contradictory = False
            # Simple contradiction check: e.g., RSI overbought + MACD positive = mixed
            try:
                rsi = float(indicators.get("rsi", 0))
                macd = float(indicators.get("macd", 0))
                price_vs_sma20 = float(indicators.get("price_vs_sma20", 0))
                # Example: if all are neutral (not extreme), not contradictory
                if (
                    (30 < rsi < 70)
                    and (-0.5 < macd < 0.5)
                    and (-1 < price_vs_sma20 < 1)
                ):
                    contradictory = False
                # If RSI > 70 and MACD > 0, that's not contradictory (bullish)
                # If RSI < 30 and MACD < 0, that's not contradictory (bearish)
                # If RSI > 70 and MACD < 0, that's contradictory
                elif (rsi > 70 and macd < 0) or (rsi < 30 and macd > 0):
                    contradictory = True
            except Exception:
                pass
            # If all indicators present, not contradictory, and confidence is low, override
            if (
                all_present
                and not contradictory
                and adjusted_confidence < min_confidence
            ):
                # Choose BUY or SELL based on indicator direction
                if rsi < 40 and macd < 0:
                    override_action = "sell"
                else:
                    override_action = "buy"
                self.logger.warning(
                    f"[HOLD_OVERRIDE] Overriding unjustified HOLD for {analysis_input.symbol}: All indicators present, not contradictory, low confidence. Forcing {override_action.upper()} with confidence {min_confidence:.2f}. Rationale: {rationale}"
                )
                override_triggered = True
                signal = ActionSignal[override_action.upper()]
                adjusted_confidence = min_confidence
                rationale = f"[OVERRIDE] Forced {override_action.upper()} due to unjustified HOLD with all indicators present. Original rationale: {rationale}"
        # Always include final_action for downstream compatibility
        final_action = (
            signal.name.lower() if hasattr(signal, "name") else str(signal).lower()
        )
        # Return actionable decision if confidence and filters are met
        if adjusted_confidence >= min_confidence and signal in [
            ActionSignal.BUY,
            ActionSignal.SELL,
        ]:
            decision = TradingDecision(
                symbol=analysis_input.symbol,
                signal=signal,
                confidence=adjusted_confidence,
                rationale=rationale,
                metadata={
                    "source": "DecisionMakerBot",
                    "raw_market_data": analysis_input.market_data,
                    "raw_technical_indicators": analysis_input.technical_indicators,
                    "final_action": final_action,
                    "hold_override": override_triggered,
                },
                final_action=final_action,
            )
            self.logger.info(
                f"Decision made for {analysis_input.symbol}: {decision.signal.name} (confidence: {decision.confidence:.2f})"
            )
            self.logger.debug(
                f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}(asset_type='{analysis_input.asset_type}', hold_override_threshold={min_confidence})] END with decision: {decision}"
            )
            return decision
        else:
            decision = TradingDecision(
                symbol=analysis_input.symbol,
                signal=ActionSignal.HOLD,
                confidence=adjusted_confidence,
                rationale="Did not meet confidence or filter requirements.",
                metadata={
                    "source": "DecisionMakerBot",
                    "raw_market_data": analysis_input.market_data,
                    "raw_technical_indicators": analysis_input.technical_indicators,
                    "final_action": "hold",
                    "hold_override": override_triggered,
                },
                final_action="hold",
            )
            self.logger.info(
                f"Decision did not meet criteria for {analysis_input.symbol}: HOLD (confidence: {adjusted_confidence:.2f})"
            )
            self.logger.debug(
                f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}(asset_type='{analysis_input.asset_type}', hold_override_threshold={min_confidence})] END with decision: {decision}"
            )
            return decision
        # Before returning, log the prompt context if available
        prompt = getattr(analysis_input, "full_prompt", None)
        if prompt:
            self.log_prompt_context(analysis_input, prompt)

    @log_method_calls
    def batch_make_decisions(
        self,
        analyses: list[AssetAnalysisInput],
        min_confidence: float,
        current_portfolio_risk: float = 0.0,
        market_conditions: Optional[dict] = None,
    ) -> list[TradingDecision]:
        """
        Make trading decisions for a batch of analyses.
        Returns a list of TradingDecision objects.
        """
        # Log batch decision start
        self.logger.info(f"Batch making decisions for {len(analyses)} assets.")
        results = []
        for a in analyses:
            decision = self.make_trading_decision(
                a, min_confidence, current_portfolio_risk, market_conditions
            )
            # Log if hold override was triggered for any asset
            if getattr(decision, "metadata", {}).get("hold_override"):
                self.logger.info(
                    f"[BATCH_HOLD_OVERRIDE] {decision.symbol}: Hold override triggered in batch decision."
                )
            results.append(decision)
        return results

    @staticmethod
    @log_method_calls
    def selftest() -> bool:
        """
        Run a self-test to verify decision logic. Returns True if successful.
        """
        print("\n--- Running DecisionMakerBot Self-Test ---")
        from data_structures import AssetAnalysisInput

        try:
            bot = DecisionMakerBot()
            # Test case 1: High confidence, BUY
            input_buy = AssetAnalysisInput(
                symbol="AAPL",
                market_data={
                    "action": "buy",
                    "confidence": MIN_CONFIDENCE + 0.1,
                    "reasoning": "Strong uptrend.",
                },
                technical_indicators={},
                reflection_insights=None,
                historical_ai_context=None,
            )
            decision_buy = bot.make_trading_decision(
                input_buy, min_confidence=MIN_CONFIDENCE
            )
            assert (
                decision_buy.final_action == "buy"
            ), f"Expected BUY, got {decision_buy.final_action}"
            print("    -> BUY decision logic passed.")
            # Test case 2: High confidence, SELL
            input_sell = AssetAnalysisInput(
                symbol="TSLA",
                market_data={
                    "action": "sell",
                    "confidence": MIN_CONFIDENCE + 0.2,
                    "reasoning": "Bearish reversal.",
                },
                technical_indicators={},
                reflection_insights=None,
                historical_ai_context=None,
            )
            decision_sell = bot.make_trading_decision(
                input_sell, min_confidence=MIN_CONFIDENCE
            )
            assert (
                decision_sell.final_action == "sell"
            ), f"Expected SELL, got {decision_sell.final_action}"
            print("    -> SELL decision logic passed.")
            # Test case 3: Low confidence, should HOLD
            input_hold = AssetAnalysisInput(
                symbol="GOOGL",
                market_data={
                    "action": "buy",
                    "confidence": MIN_CONFIDENCE - 0.05,
                    "reasoning": "Weak signal.",
                },
                technical_indicators={},
                reflection_insights=None,
                historical_ai_context=None,
            )
            decision_hold = bot.make_trading_decision(
                input_hold, min_confidence=MIN_CONFIDENCE
            )
            assert (
                decision_hold.final_action == "hold"
            ), f"Expected HOLD, got {decision_hold.final_action}"
            print("    -> HOLD (low confidence) logic passed.")
            print("--- DecisionMakerBot Self-Test PASSED ---")
            return True
        except AssertionError as e:
            print(f"--- DecisionMakerBot Self-Test FAILED: {e} ---")
        except Exception as e:
            print(f"--- DecisionMakerBot Self-Test encountered an ERROR: {e} ---")
        return False


if __name__ == "__main__":
    DecisionMakerBot.selftest()
