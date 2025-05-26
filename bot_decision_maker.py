# bot_decision_maker.py
"""
DecisionMakerBot: Centralizes final trading decisions based on multiple inputs.
- Receives analysis data from StockBot, CryptoBot, and AIBot
- Applies dynamically adapted confidence thresholds
- Incorporates strategic rules and market sentiment filters
- Outputs clear, actionable trading decisions with rationale
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Import configuration
from config_trading_variables import MIN_CONFIDENCE, MAX_PORTFOLIO_RISK

logger = logging.getLogger(__name__)

@dataclass
class AnalysisInput:
    """Structure for analysis inputs from various bots"""
    symbol: str
    asset_type: str
    action: str
    confidence: float
    reasoning: str
    technical_indicators: Optional[Dict[str, Any]] = None
    market_sentiment: Optional[str] = None
    risk_metrics: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None

@dataclass
class TradingDecision:
    """Structure for final trading decisions"""
    symbol: str
    asset_type: str
    final_action: str  # 'buy', 'sell', 'hold'
    final_confidence: float
    decision_rationale: str
    from dataclasses import field
    original_analyses: List[AnalysisInput] = field(default_factory=list)
    filters_applied: List[str] = field(default_factory=list)
    risk_assessment: Optional[str] = None
    timestamp: Optional[datetime] = None

class DecisionMakerBot:
    """
    CENTRALIZED TRADING DECISION ENGINE
    
    Consolidates analysis from multiple sources and applies strategic filters
    to make final trading decisions. Acts as the decision layer between
    analysis and execution.
    """
    
    def __init__(self):
        """Initialize the DecisionMakerBot"""
        logger.info("🎯 Initializing DecisionMakerBot...")
        
        # Decision-making parameters
        self.confidence_boost_factors = {
            'technical_alignment': 0.1,  # Boost if technical indicators align
            'high_volume': 0.05,         # Boost for high volume confirmation
            'market_sentiment': 0.08,    # Boost for positive market sentiment
        }
        
        # Risk adjustment factors
        self.risk_penalty_factors = {
            'high_volatility': -0.15,    # Reduce confidence for high volatility
            'poor_liquidity': -0.1,      # Reduce for poor liquidity
            'market_stress': -0.2,       # Reduce during market stress
        }
        
        # Strategic filters
        self.filters_enabled = {
            'min_confidence_filter': True,
            'portfolio_risk_filter': True,
            'market_hours_filter': True,
            'volatility_filter': True,
            'correlation_filter': False,  # Advanced filter, disabled by default
        }
        
        logger.info("✅ DecisionMakerBot initialized successfully")
    
    def make_trading_decision(
        self, 
        analysis_input: AnalysisInput,
        min_confidence: float,
        current_portfolio_risk: float = 0.0,
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> TradingDecision:
        """
        Make final trading decision based on analysis input and filters
        
        Args:
            analysis_input: Analysis from StockBot/CryptoBot/AIBot
            min_confidence: Dynamically adapted minimum confidence threshold
            current_portfolio_risk: Current portfolio risk percentage
            market_conditions: Optional market condition data
        
        Returns:
            TradingDecision: Final trading decision with rationale
        """
        logger.info(f"🎯 Making trading decision for {analysis_input.symbol}")
        
        # Initialize decision structure
        decision = TradingDecision(
            symbol=analysis_input.symbol,
            asset_type=analysis_input.asset_type,
            final_action='hold',  # Default to hold
            final_confidence=analysis_input.confidence,
            decision_rationale="",
            original_analyses=[analysis_input],
            filters_applied=[],
            timestamp=datetime.now()
        )
        
        try:
            # === STEP 1: APPLY CONFIDENCE ADJUSTMENTS ===
            adjusted_confidence = self._apply_confidence_adjustments(
                analysis_input, market_conditions
            )
            decision.final_confidence = adjusted_confidence
            
            # === STEP 2: APPLY STRATEGIC FILTERS ===
            passed_filters, filter_results = self._apply_strategic_filters(
                analysis_input, 
                adjusted_confidence, 
                min_confidence,
                current_portfolio_risk
            )
            
            decision.filters_applied = list(filter_results.keys())
            
            # === STEP 3: MAKE FINAL DECISION ===
            if passed_filters and analysis_input.action in ['buy', 'sell']:
                decision.final_action = analysis_input.action
                decision.decision_rationale = self._generate_decision_rationale(
                    analysis_input, adjusted_confidence, filter_results, True
                )
                decision.risk_assessment = "APPROVED"
            else:
                decision.final_action = 'hold'
                decision.decision_rationale = self._generate_decision_rationale(
                    analysis_input, adjusted_confidence, filter_results, False
                )
                decision.risk_assessment = "REJECTED" if not passed_filters else "HOLD_RECOMMENDED"
            
            logger.info(
                f"Decision for {analysis_input.symbol}: {decision.final_action.upper()} "
                f"(confidence: {decision.final_confidence:.1f})"
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Error making decision for {analysis_input.symbol}: {e}")
            
            # Return safe hold decision on error
            decision.final_action = 'hold'
            decision.decision_rationale = f"Decision error: {str(e)}"
            decision.risk_assessment = "ERROR"
            return decision
    
    def batch_make_decisions(
        self,
        analyses: List[AnalysisInput],
        min_confidence: float,
        current_portfolio_risk: float = 0.0,
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> List[TradingDecision]:
        """
        Make trading decisions for multiple assets
        
        Args:
            analyses: List of analysis inputs
            min_confidence: Minimum confidence threshold
            current_portfolio_risk: Current portfolio risk
            market_conditions: Market condition data
        
        Returns:
            List[TradingDecision]: Final trading decisions
        """
        logger.info(f"🎯 Making batch decisions for {len(analyses)} assets")
        
        decisions = []
        
        for analysis in analyses:
            try:
                decision = self.make_trading_decision(
                    analysis, min_confidence, current_portfolio_risk, market_conditions
                )
                decisions.append(decision)
                
            except Exception as e:
                logger.error(f"Error in batch decision for {analysis.symbol}: {e}")
                # Add error decision
                error_decision = TradingDecision(
                    symbol=analysis.symbol,
                    asset_type=analysis.asset_type,
                    final_action='hold',
                    final_confidence=0.0,
                    decision_rationale=f"Batch processing error: {str(e)}",
                    risk_assessment="ERROR",
                    timestamp=datetime.now()
                )
                decisions.append(error_decision)
        
        # Apply portfolio-level decision filters
        decisions = self._apply_portfolio_level_filters(decisions, current_portfolio_risk)
        
        approved_decisions = [d for d in decisions if d.final_action in ['buy', 'sell']]
        logger.info(f"📋 Batch decisions: {len(approved_decisions)}/{len(decisions)} approved")
        
        return decisions
    
    def _apply_confidence_adjustments(
        self, 
        analysis: AnalysisInput, 
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Apply confidence boosts and penalties based on various factors
        
        Args:
            analysis: Analysis input data
            market_conditions: Market condition data
        
        Returns:
            float: Adjusted confidence level
        """
        adjusted_confidence = analysis.confidence
        adjustments = []
        
        try:
            # === TECHNICAL INDICATOR ALIGNMENT ===
            if analysis.technical_indicators:
                rsi = analysis.technical_indicators.get('rsi', 50)
                macd_signal = analysis.technical_indicators.get('macd_signal', 'neutral')
                
                # Check for technical alignment
                if analysis.action == 'buy' and rsi < 40 and macd_signal == 'bullish':
                    boost = self.confidence_boost_factors['technical_alignment']
                    adjusted_confidence += boost
                    adjustments.append(f"Technical alignment boost: +{boost:.2f}")
                elif analysis.action == 'sell' and rsi > 60 and macd_signal == 'bearish':
                    boost = self.confidence_boost_factors['technical_alignment']
                    adjusted_confidence += boost
                    adjustments.append(f"Technical alignment boost: +{boost:.2f}")
            
            # === VOLUME CONFIRMATION ===
            if analysis.technical_indicators and analysis.technical_indicators.get('volume_above_average', False):
                boost = self.confidence_boost_factors['high_volume']
                adjusted_confidence += boost
                adjustments.append(f"High volume boost: +{boost:.2f}")
            
            # === MARKET SENTIMENT ===
            if market_conditions:
                market_sentiment = market_conditions.get('sentiment', 'neutral')
                if ((analysis.action == 'buy' and market_sentiment == 'bullish') or
                    (analysis.action == 'sell' and market_sentiment == 'bearish')):
                    boost = self.confidence_boost_factors['market_sentiment']
                    adjusted_confidence += boost
                    adjustments.append(f"Market sentiment boost: +{boost:.2f}")
            
            # === RISK PENALTIES ===
            
            # High volatility penalty
            if analysis.risk_metrics:
                volatility = analysis.risk_metrics.get('volatility', 0)
                if volatility > 0.3:  # 30% volatility threshold
                    penalty = self.risk_penalty_factors['high_volatility']
                    adjusted_confidence += penalty
                    adjustments.append(f"High volatility penalty: {penalty:.2f}")
            
            # Market stress penalty
            if market_conditions and market_conditions.get('vix', 0) > 25:
                penalty = self.risk_penalty_factors['market_stress']
                adjusted_confidence += penalty
                adjustments.append(f"Market stress penalty: {penalty:.2f}")
            
            # Ensure confidence stays within valid range
            adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
            
            if adjustments:
                logger.debug(f"Confidence adjustments for {analysis.symbol}: {', '.join(adjustments)}")
            
            return adjusted_confidence
            
        except Exception as e:
            logger.error(f"Error applying confidence adjustments for {analysis.symbol}: {e}")
            return analysis.confidence
    
    def _apply_strategic_filters(
        self,
        analysis: AnalysisInput,
        adjusted_confidence: float,
        min_confidence: float,
        current_portfolio_risk: float
    ) -> Tuple[bool, Dict[str, bool]]:
        """
        Apply strategic filters to determine if trade should proceed
        
        Args:
            analysis: Analysis input
            adjusted_confidence: Confidence after adjustments
            min_confidence: Minimum confidence threshold
            current_portfolio_risk: Current portfolio risk level
        
        Returns:
            Tuple[bool, Dict[str, bool]]: (passed_all_filters, filter_results)
        """
        filter_results = {}
        
        try:
            # === MINIMUM CONFIDENCE FILTER ===
            if self.filters_enabled['min_confidence_filter']:
                passed = adjusted_confidence >= min_confidence
                filter_results['min_confidence_filter'] = passed
                if not passed:
                    logger.debug(f"Failed min confidence filter: {adjusted_confidence:.2f} < {min_confidence:.2f}")
            
            # === PORTFOLIO RISK FILTER ===
            if self.filters_enabled['portfolio_risk_filter']:
                passed = current_portfolio_risk <= MAX_PORTFOLIO_RISK
                filter_results['portfolio_risk_filter'] = passed
                if not passed:
                    logger.debug(f"Failed portfolio risk filter: {current_portfolio_risk:.2f} > {MAX_PORTFOLIO_RISK:.2f}")
            
            # === MARKET HOURS FILTER ===
            if self.filters_enabled['market_hours_filter']:
                # Simplified market hours check (extend as needed)
                current_hour = datetime.now().hour
                is_market_hours = 9 <= current_hour <= 16  # Basic US market hours
                
                # Crypto trades 24/7, stocks need market hours
                if analysis.asset_type == 'stock':
                    passed = is_market_hours
                else:
                    passed = True  # Crypto always passes
                
                filter_results['market_hours_filter'] = passed
                if not passed:
                    logger.debug(f"Failed market hours filter: current hour {current_hour}")
            
            # === VOLATILITY FILTER ===
            if self.filters_enabled['volatility_filter']:
                # Check if volatility is too extreme
                if analysis.risk_metrics:
                    volatility = analysis.risk_metrics.get('volatility', 0)
                    passed = volatility <= 0.5  # 50% max volatility
                else:
                    passed = True  # Pass if no volatility data
                
                filter_results['volatility_filter'] = passed
                if not passed:
                    logger.debug(f"Failed volatility filter: {volatility:.2f} > 0.5")
            
            # === CORRELATION FILTER (Advanced) ===
            if self.filters_enabled['correlation_filter']:
                # This would check correlation with existing positions
                # For now, always pass (implement later)
                filter_results['correlation_filter'] = True
            
            # Check if all enabled filters passed
            all_passed = all(
                result for filter_name, result in filter_results.items()
                if self.filters_enabled.get(filter_name, False)
            )
            
            return all_passed, filter_results
            
        except Exception as e:
            logger.error(f"Error applying strategic filters for {analysis.symbol}: {e}")
            return False, {'error': False}
    
    def _apply_portfolio_level_filters(
        self, 
        decisions: List[TradingDecision], 
        current_portfolio_risk: float
    ) -> List[TradingDecision]:
        """
        Apply portfolio-level filters that consider all decisions together
        
        Args:
            decisions: List of individual decisions
            current_portfolio_risk: Current portfolio risk
        
        Returns:
            List[TradingDecision]: Filtered decisions
        """
        try:
            # Count buy vs sell decisions
            buy_decisions = [d for d in decisions if d.final_action == 'buy']
            sell_decisions = [d for d in decisions if d.final_action == 'sell']
            
            # === POSITION CONCENTRATION FILTER ===
            # Limit number of new positions per cycle
            max_new_positions = 3  # Configurable
            
            if len(buy_decisions) > max_new_positions:
                # Sort by confidence and keep only top decisions
                buy_decisions.sort(key=lambda x: x.final_confidence, reverse=True)
                
                # Reject lower confidence buy decisions
                for decision in buy_decisions[max_new_positions:]:
                    decision.final_action = 'hold'
                    decision.decision_rationale += f" [REJECTED: Portfolio concentration limit]"
                    decision.risk_assessment = "PORTFOLIO_LIMIT"
                    decision.filters_applied.append('position_concentration_filter')
            
            # === RISK BUDGET FILTER ===
            # If portfolio risk is high, be more conservative
            if current_portfolio_risk > (MAX_PORTFOLIO_RISK * 0.8):  # 80% of max risk
                conservative_threshold = 0.8  # Higher confidence required
                
                for decision in decisions:
                    if decision.final_action in ['buy', 'sell'] and decision.final_confidence < conservative_threshold:
                        decision.final_action = 'hold'
                        decision.decision_rationale += f" [REJECTED: High portfolio risk requires confidence > {conservative_threshold}]"
                        decision.risk_assessment = "HIGH_PORTFOLIO_RISK"
                        decision.filters_applied.append('risk_budget_filter')
            
            logger.info(f"Portfolio-level filters applied to {len(decisions)} decisions")
            return decisions
            
        except Exception as e:
            logger.error(f"Error applying portfolio-level filters: {e}")
            return decisions
    
    def _generate_decision_rationale(
        self,
        analysis: AnalysisInput,
        final_confidence: float,
        filter_results: Dict[str, bool],
        approved: bool
    ) -> str:
        """Generate human-readable decision rationale"""
        
        rationale_parts = []
        
        # Base analysis
        rationale_parts.append(f"AI Analysis: {analysis.action.upper()} with {analysis.confidence:.1f} confidence")
        rationale_parts.append(f"Reasoning: {analysis.reasoning[:100]}...")
        
        # Confidence adjustments
        if abs(final_confidence - analysis.confidence) > 0.01:
            rationale_parts.append(f"Adjusted confidence: {analysis.confidence:.2f} → {final_confidence:.2f}")
        
        # Filter results
        passed_filters = [f for f, result in filter_results.items() if result]
        failed_filters = [f for f, result in filter_results.items() if not result]
        
        if passed_filters:
            rationale_parts.append(f"Passed filters: {', '.join(passed_filters)}")
        
        if failed_filters:
            rationale_parts.append(f"Failed filters: {', '.join(failed_filters)}")
        
        # Final decision
        if approved:
            rationale_parts.append(f"DECISION: Execute {analysis.action.upper()} order")
        else:
            rationale_parts.append(f"DECISION: HOLD (trade rejected)")
        
        return " | ".join(rationale_parts)
    
    def update_filter_settings(self, filter_name: str, enabled: bool):
        """Update filter settings dynamically"""
        if filter_name in self.filters_enabled:
            self.filters_enabled[filter_name] = enabled
            logger.info(f"Updated filter {filter_name}: {'enabled' if enabled else 'disabled'}")
        else:
            logger.warning(f"Unknown filter: {filter_name}")
    
    def get_filter_settings(self) -> Dict[str, bool]:
        """Get current filter settings"""
        return self.filters_enabled.copy()
    
    def get_decision_statistics(self, decisions: List[TradingDecision]) -> Dict[str, Any]:
        """Get statistics about a batch of decisions"""
        if not decisions:
            return {}
        
        stats = {
            'total_decisions': len(decisions),
            'buy_decisions': len([d for d in decisions if d.final_action == 'buy']),
            'sell_decisions': len([d for d in decisions if d.final_action == 'sell']),
            'hold_decisions': len([d for d in decisions if d.final_action == 'hold']),
            'avg_confidence': sum(d.final_confidence for d in decisions) / len(decisions),
            'approved_rate': len([d for d in decisions if d.final_action in ['buy', 'sell']]) / len(decisions),
        }
        
        # Filter failure statistics
        all_filters = set()
        for decision in decisions:
            all_filters.update(decision.filters_applied or [])
        
        stats['filters_applied'] = list(all_filters)
        
        return stats
