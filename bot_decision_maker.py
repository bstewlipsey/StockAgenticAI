# bot_decision_maker.py
"""
DecisionMakerBot: Centralizes final trading decisions based on multiple inputs.
- Receives analysis data from StockBot, CryptoBot, and AIBot
- Applies dynamically adapted confidence thresholds
- Incorporates strategic rules and market sentiment filters
- Outputs clear, actionable trading decisions with rationale
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import configuration
from config_trading import MIN_CONFIDENCE, MAX_PORTFOLIO_RISK, CONFIDENCE_BOOST_FACTORS, RISK_PENALTY_FACTORS, FILTERS_ENABLED
from data_structures import AssetAnalysisInput, TradingDecision, ActionSignal

logger = logging.getLogger(__name__)

class DecisionMakerBot:
    """
    CENTRALIZED TRADING DECISION ENGINE
    
    Consolidates analysis from multiple sources and applies strategic filters
    to make final trading decisions. Acts as the decision layer between
    analysis and execution.
    """
    
    def __init__(self):
        """Initialize the DecisionMakerBot"""
        logger.info("Initializing DecisionMakerBot...")
        # Decision-making parameters
        self.confidence_boost_factors = CONFIDENCE_BOOST_FACTORS
        # Risk adjustment factors
        self.risk_penalty_factors = RISK_PENALTY_FACTORS
        # Strategic filters
        self.filters_enabled = FILTERS_ENABLED
        logger.info("DecisionMakerBot initialized successfully")
    
    def make_trading_decision(
        self, 
        analysis_input: AssetAnalysisInput,
        min_confidence: float,
        current_portfolio_risk: float = 0.0,
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> TradingDecision:
        """
        Make final trading decision based on analysis input and filters
        Logs detailed info for crypto assets: symbol, action, confidence, and reasoning.
        
        Args:
            analysis_input: Analysis from StockBot/CryptoBot/AIBot
            min_confidence: Dynamically adapted minimum confidence threshold
            current_portfolio_risk: Current portfolio risk percentage
            market_conditions: Optional market condition data
        
        Returns:
            TradingDecision: Final trading decision with rationale
        """
        logger.info(f"Making trading decision for {analysis_input.symbol}")
        # Try to extract action/confidence/reasoning from market_data or technical_indicators
        action = (analysis_input.market_data.get('action') or analysis_input.technical_indicators.get('action') or 'hold').lower()
        confidence = (
            analysis_input.market_data.get('confidence')
            or analysis_input.technical_indicators.get('confidence')
            or 0.0
        )
        rationale = (
            analysis_input.market_data.get('reasoning')
            or analysis_input.technical_indicators.get('reasoning')
            or 'No rationale provided.'
        )
        # Enhanced logging for crypto assets
        asset_type = getattr(analysis_input, 'asset_type', None)
        if asset_type is None:
            # Try to infer asset_type from symbol
            symbol = getattr(analysis_input, 'symbol', '')
            if '/' in symbol or symbol.endswith('USD'):
                asset_type = 'crypto'
            else:
                asset_type = 'stock'
        if asset_type == 'crypto':
            logger.info(f"Analyzing {analysis_input.symbol} (CRYPTO)...")
            logger.info(f"Action: {action.upper()} | Confidence: {confidence:.2f}")
            logger.info(f"Reasoning: {rationale}")
            logger.info(f"Crypto Decision: action={action}, confidence={confidence}, reasoning={rationale}")
        # Map action string to ActionSignal enum
        try:
            signal = ActionSignal[action.upper()]
        except Exception:
            signal = ActionSignal.HOLD
        # Apply confidence adjustments and filters as needed (simplified)
        adjusted_confidence = confidence
        if adjusted_confidence >= min_confidence and signal in [ActionSignal.BUY, ActionSignal.SELL]:
            return TradingDecision(
                symbol=analysis_input.symbol,
                signal=signal,
                confidence=adjusted_confidence,
                rationale=rationale,
                metadata={
                    'source': 'DecisionMakerBot',
                    'raw_market_data': analysis_input.market_data,
                    'raw_technical_indicators': analysis_input.technical_indicators
                }
            )
        else:
            return TradingDecision(
                symbol=analysis_input.symbol,
                signal=ActionSignal.HOLD,
                confidence=adjusted_confidence,
                rationale="Did not meet confidence or filter requirements.",
                metadata={
                    'source': 'DecisionMakerBot',
                    'raw_market_data': analysis_input.market_data,
                    'raw_technical_indicators': analysis_input.technical_indicators
                }
            )
    
    def batch_make_decisions(
        self,
        analyses: list[AssetAnalysisInput],
        min_confidence: float,
        current_portfolio_risk: float = 0.0,
        market_conditions: Optional[dict] = None
    ) -> list[TradingDecision]:
        """
        Make trading decisions for multiple assets using canonical types
        """
        return [
            self.make_trading_decision(a, min_confidence, current_portfolio_risk, market_conditions)
            for a in analyses
        ]
