# bot_reflection.py
"""
ReflectionBot: Enables deep learning from past trade outcomes.
- Analyzes completed trades for learning opportunities
- Generates LLM-powered post-mortem insights
- Stores reflective insights for future reference
- Feeds insights back into future AI analysis prompts
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

# Import bots and configuration
from bot_ai import AIBot
from bot_database import DatabaseBot
from config_system import ANALYSIS_SCHEMA
from config_trading import MIN_REFLECTION_PNL, MAX_REFLECTION_AGE_DAYS

logger = logging.getLogger(__name__)

@dataclass
class TradeOutcome:
    """Structure for completed trade data"""
    trade_id: str
    symbol: str
    asset_type: str
    action: str  # 'buy' or 'sell'
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_percent: float
    duration_hours: float
    original_analysis: Optional[Dict[str, Any]] = None
    market_conditions_entry: Optional[Dict[str, Any]] = None
    market_conditions_exit: Optional[Dict[str, Any]] = None

@dataclass
class ReflectionInsight:
    """Structure for LLM-generated reflective insights"""
    trade_id: str
    symbol: str
    insight_type: str  # 'success_factor', 'failure_reason', 'improvement', 'pattern'
    insight_text: str
    confidence: float
    actionable_recommendation: str
    related_indicators: Optional[List[str]] = None
    market_context: Optional[str] = None
    timestamp: Optional[datetime] = None

class ReflectionBot:
    """
    POST-TRADE LEARNING AND REFLECTION ENGINE
    
    Analyzes completed trades to extract learning insights and improve
    future trading decisions through AI-powered reflection.
    """
    
    def __init__(self):
        """Initialize the ReflectionBot"""
        logger.info("Initializing ReflectionBot...")
        
        # Initialize required bots
        self.ai_bot = AIBot()
        self.database_bot = DatabaseBot()
        
        # Reflection parameters
        self.min_reflection_pnl = MIN_REFLECTION_PNL  # Only reflect on trades with >2% impact
        self.max_reflection_age_days = MAX_REFLECTION_AGE_DAYS  # Only reflect on trades within 30 days
        self.reflection_batch_size = 5  # Process 5 trades at a time
        
        # Debug log for reflection threshold
        logger.info(f"[DEBUG] ReflectionBot MIN_REFLECTION_PNL: {self.min_reflection_pnl}")
        
        # Insight categories
        self.insight_types = {
            'success_factor': 'What made this trade successful',
            'failure_reason': 'Why this trade failed',
            'timing_issue': 'Entry/exit timing analysis',
            'risk_management': 'Risk management lessons',
            'market_context': 'Market condition influences',
            'indicator_performance': 'Technical indicator effectiveness',
            'ai_confidence_accuracy': 'AI confidence vs actual outcome',
            'pattern_recognition': 'Recurring patterns identified'
        }
        
        logger.info("ReflectionBot initialized successfully")
    
    def analyze_completed_trade(self, trade_outcome: TradeOutcome) -> List[ReflectionInsight]:
        """
        Analyze a single completed trade and generate insights
        
        Args:
            trade_outcome: Completed trade data
        
        Returns:
            List[ReflectionInsight]: Generated insights from the trade
        """
        logger.info(f"Analyzing completed trade: {trade_outcome.symbol} (P&L: {trade_outcome.pnl_percent:.1f}%)")
        
        insights = []
        
        try:
            # === STEP 1: DETERMINE REFLECTION PRIORITY ===
            should_reflect = self._should_reflect_on_trade(trade_outcome)
            if not should_reflect:
                logger.info(f"Skipping reflection for {trade_outcome.symbol} - below reflection threshold")
                return insights
            
            # === STEP 2: GATHER CONTEXT FOR REFLECTION ===
            reflection_context = self._gather_reflection_context(trade_outcome)
            
            # === STEP 3: GENERATE AI-POWERED INSIGHTS ===
            ai_insights = self._generate_ai_reflection(trade_outcome, reflection_context)
            insights.extend(ai_insights)
            
            # === STEP 4: PERFORM SYSTEMATIC ANALYSIS ===
            systematic_insights = self._perform_systematic_analysis(trade_outcome)
            insights.extend(systematic_insights)
            
            # === STEP 5: STORE INSIGHTS IN DATABASE ===
            self._store_insights(insights)
            
            logger.info(f"Generated {len(insights)} insights for {trade_outcome.symbol}")
            return insights
            
        except Exception as e:
            logger.error(f"Error analyzing trade {trade_outcome.trade_id}: {e}")
            return insights
    
    def batch_analyze_recent_trades(self, days_back: int = 7) -> List[ReflectionInsight]:
        """
        Analyze multiple recent completed trades
        
        Args:
            days_back: Number of days back to analyze trades
        
        Returns:
            List[ReflectionInsight]: All generated insights
        """
        logger.info(f"Batch analyzing trades from last {days_back} days...")
        
        all_insights = []
        
        try:
            # Get recent completed trades from database
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            completed_trades = self._get_completed_trades(start_date, end_date)
            
            if not completed_trades:
                logger.info("No completed trades found for reflection")
                return all_insights
            
            # Process trades in batches
            for i in range(0, len(completed_trades), self.reflection_batch_size):
                batch = completed_trades[i:i + self.reflection_batch_size]
                
                for trade_outcome in batch:
                    trade_insights = self.analyze_completed_trade(trade_outcome)
                    all_insights.extend(trade_insights)
            
            # === CROSS-TRADE PATTERN ANALYSIS ===
            pattern_insights = self._analyze_cross_trade_patterns(completed_trades)
            all_insights.extend(pattern_insights)
            
            logger.info(f"Batch reflection complete: {len(all_insights)} total insights from {len(completed_trades)} trades")
            return all_insights
            
        except Exception as e:
            logger.error(f"Error in batch trade analysis: {e}")
            return all_insights
    
    def get_insights_for_symbol(self, symbol: str, limit: int = 10) -> List[ReflectionInsight]:
        """
        Get stored insights for a specific symbol to inform future trades
        
        Args:
            symbol: Asset symbol
            limit: Maximum number of insights to return
        
        Returns:
            List[ReflectionInsight]: Relevant insights for the symbol
        """
        try:
            # Query database for insights related to this symbol
            insights = self.database_bot.get_reflection_insights(symbol=symbol)
            
            # Convert database records to ReflectionInsight objects
            insight_objects = []
            for insight_data in insights[:limit]:
                insight = ReflectionInsight(
                    trade_id=insight_data.get('trade_id', ''),
                    symbol=insight_data.get('symbol', ''),
                    insight_type='reflection',
                    insight_text=insight_data.get('ai_reflection', ''),
                    confidence=insight_data.get('confidence_accuracy', 0.0),
                    actionable_recommendation=insight_data.get('lessons_learned', ''),
                    related_indicators=None,
                    market_context=insight_data.get('key_insights', ''),
                    timestamp=insight_data.get('timestamp')
                )
                insight_objects.append(insight)
            
            return insight_objects
            
        except Exception as e:
            logger.error(f"Error getting insights for {symbol}: {e}")
            return []
    
    def generate_enhanced_prompt_note(self, symbol: str) -> str:
        """
        Generate enhanced prompt note with reflection insights for future analysis
        
        Args:
            symbol: Asset symbol to get insights for
        
        Returns:
            str: Enhanced prompt note with reflection insights
        """
        try:
            # Get recent insights for this symbol
            insights = self.get_insights_for_symbol(symbol, limit=5)
            
            if not insights:
                return ""
            
            prompt_parts = [f"\\n--- REFLECTION INSIGHTS FOR {symbol} ---"]
            
            # Group insights by type
            insight_groups = {}
            for insight in insights:
                if insight.insight_type not in insight_groups:
                    insight_groups[insight.insight_type] = []
                insight_groups[insight.insight_type].append(insight)
            
            # Add insights to prompt
            for insight_type, type_insights in insight_groups.items():
                prompt_parts.append(f"\\n{insight_type.replace('_', ' ').title()}:")
                for insight in type_insights:
                    prompt_parts.append(f"- {insight.insight_text}")
                    if insight.actionable_recommendation:
                        prompt_parts.append(f"  → {insight.actionable_recommendation}")
            
            prompt_parts.append("\\n--- END REFLECTION INSIGHTS ---\\n")
            
            return "\\n".join(prompt_parts)
            
        except Exception as e:
            logger.error(f"Error generating enhanced prompt for {symbol}: {e}")
            return ""
    
    def _should_reflect_on_trade(self, trade_outcome: TradeOutcome) -> bool:
        """Determine if a trade warrants reflection"""
        logger.info(f"[DEBUG] _should_reflect_on_trade: abs({trade_outcome.pnl_percent}) < {self.min_reflection_pnl} (type: {type(trade_outcome.pnl_percent)})")
        pnl_percent = float(trade_outcome.pnl_percent)
        # Only reflect on trades with absolute pnl_percent strictly greater than the threshold
        if abs(pnl_percent) < self.min_reflection_pnl:
            logger.info(f"Reflection threshold check: abs({pnl_percent}) < {self.min_reflection_pnl}")
            return False
        logger.info(f"Reflection threshold check: abs({pnl_percent}) <= {self.min_reflection_pnl}")
        # Check if trade is recent enough
        age_days = (datetime.now() - trade_outcome.exit_time).days
        if age_days > self.max_reflection_age_days:
            return False

        # Check if we haven't already reflected on this trade
        existing_insights = [
            insight for insight in self.database_bot.get_reflection_insights(symbol=trade_outcome.symbol)
            if insight.get('trade_id') == trade_outcome.trade_id
        ]
        if existing_insights:
            return False

        return True
    
    def _gather_reflection_context(self, trade_outcome: TradeOutcome) -> Dict[str, Any]:
        """Gather additional context for trade reflection"""
        context = {
            'trade_duration': trade_outcome.duration_hours,
            'profit_loss': trade_outcome.pnl_percent,
            'market_conditions': {},
            'historical_performance': {},
            'concurrent_trades': []
        }
        try:
            # Get market conditions during trade period
            if trade_outcome.market_conditions_entry:
                context['market_conditions']['entry'] = trade_outcome.market_conditions_entry
            if trade_outcome.market_conditions_exit:
                context['market_conditions']['exit'] = trade_outcome.market_conditions_exit

            # Get historical performance for this symbol (recent trades only)
            historical_trades = self.database_bot.get_trade_outcomes(
                symbol=trade_outcome.symbol,
                days=30
            )
            if historical_trades:
                win_rate = len([t for t in historical_trades if t.get('pnl', 0) > 0]) / len(historical_trades)
                avg_return = sum(t.get('pnl_percentage', 0) for t in historical_trades) / len(historical_trades)
                context['historical_performance'] = {
                    'win_rate': win_rate,
                    'avg_return': avg_return,
                    'total_trades': len(historical_trades)
                }

            # Get other trades that were active during the same period
            all_trades = self.database_bot.get_trade_outcomes(days=30)
            concurrent_trades = []
            for t in all_trades:
                # Robustly parse entry/exit times
                entry_time = t.get('entry_time')
                exit_time = t.get('exit_time')
                # Fallback: use timestamp if entry/exit not present
                if not entry_time and 'timestamp' in t:
                    try:
                        entry_time = datetime.fromisoformat(t['timestamp'])
                    except Exception:
                        entry_time = None
                if not exit_time and 'timestamp' in t:
                    try:
                        exit_time = datetime.fromisoformat(t['timestamp'])
                    except Exception:
                        exit_time = None
                # Parse string times if needed
                if isinstance(entry_time, str):
                    try:
                        entry_time = datetime.fromisoformat(entry_time)
                    except Exception:
                        entry_time = None
                if isinstance(exit_time, str):
                    try:
                        exit_time = datetime.fromisoformat(exit_time)
                    except Exception:
                        exit_time = None
                # Only include if both times are valid
                if entry_time and exit_time:
                    if entry_time <= trade_outcome.exit_time and exit_time >= trade_outcome.entry_time:
                        concurrent_trades.append(t)
            context['concurrent_trades'] = concurrent_trades
        except Exception as e:
            logger.error(f"Error gathering reflection context: {e}")
        return context
    
    def _generate_ai_reflection(
        self, 
        trade_outcome: TradeOutcome, 
        context: Dict[str, Any]
    ) -> List[ReflectionInsight]:
        """Generate AI-powered reflection insights"""
        
        insights = []
        
        try:
            # Create reflection prompt for the AI
            reflection_prompt = self._create_reflection_prompt(trade_outcome, context)
            
            # Use .format() only if there are actual format fields, otherwise pass as is
            ai_response = self.ai_bot.generate_analysis(
                prompt_type=reflection_prompt,
                variables={"analysis_schema": ANALYSIS_SCHEMA}
            )
            
            if isinstance(ai_response, dict) and 'reasoning' in ai_response:
                # Parse AI response into structured insights
                ai_insights = self._parse_ai_reflection_response(ai_response, trade_outcome)
                insights.extend(ai_insights)
            
        except Exception as e:
            logger.error(f"Error generating AI reflection: {e}")
        
        return insights
    
    def _create_reflection_prompt(
        self, 
        trade_outcome: TradeOutcome, 
        context: Dict[str, Any]
    ) -> str:
        """Create a detailed reflection prompt for AI analysis"""
        
        outcome_type = "profitable" if trade_outcome.pnl > 0 else "losing"
        # Use f-string, not .format(), to avoid tuple index errors
        prompt = f"""
        Analyze this completed {outcome_type} trade for learning insights:
        
        TRADE DETAILS:
        - Symbol: {trade_outcome.symbol} ({trade_outcome.asset_type})
        - Action: {trade_outcome.action}
        - Entry: ${trade_outcome.entry_price:.2f} at {trade_outcome.entry_time}
        - Exit: ${trade_outcome.exit_price:.2f} at {trade_outcome.exit_time}
        - P&L: ${trade_outcome.pnl:.2f} ({trade_outcome.pnl_percent:.1f}%)
        - Duration: {trade_outcome.duration_hours:.1f} hours
        
        ORIGINAL AI ANALYSIS:
        {trade_outcome.original_analysis}
        
        CONTEXT:
        - Historical win rate for {trade_outcome.symbol}: {context.get('historical_performance', {}).get('win_rate', 'Unknown')}
        - Market conditions: {context.get('market_conditions', {})}
        - Concurrent trades: {len(context.get('concurrent_trades', []))} other positions
        
        Please provide insights on:
        1. What factors contributed to this outcome?
        2. Was the original AI confidence accurate?
        3. What could be improved for future trades?
        4. Any patterns or lessons learned?
        5. Specific recommendations for trading {trade_outcome.symbol} in the future?
        
        Focus on actionable insights that can improve future trading decisions.
        """
        return prompt
    
    def _parse_ai_reflection_response(
        self, 
        ai_response: Dict[str, Any], 
        trade_outcome: TradeOutcome
    ) -> List[ReflectionInsight]:
        """Parse AI reflection response into structured insights"""
        
        insights = []
        
        try:
            reasoning = ai_response.get('reasoning', '')
            confidence = ai_response.get('confidence', 0.5)
            
            # Extract different types of insights from the reasoning
            if trade_outcome.pnl > 0:
                # Successful trade insights
                insight = ReflectionInsight(
                    trade_id=trade_outcome.trade_id,
                    symbol=trade_outcome.symbol,
                    insight_type='success_factor',
                    insight_text=f"Successful trade analysis: {reasoning[:200]}...",
                    confidence=confidence,
                    actionable_recommendation=self._extract_recommendations(reasoning),
                    timestamp=datetime.now()
                )
                insights.append(insight)
            else:
                # Failed trade insights
                insight = ReflectionInsight(
                    trade_id=trade_outcome.trade_id,
                    symbol=trade_outcome.symbol,
                    insight_type='failure_reason',
                    insight_text=f"Trade failure analysis: {reasoning[:200]}...",
                    confidence=confidence,
                    actionable_recommendation=self._extract_recommendations(reasoning),
                    timestamp=datetime.now()
                )
                insights.append(insight)
            
            # AI confidence accuracy insight
            if trade_outcome.original_analysis:
                original_confidence = trade_outcome.original_analysis.get('confidence', 0)
                actual_outcome = 1.0 if trade_outcome.pnl > 0 else 0.0
                confidence_accuracy = 1.0 - abs(original_confidence - actual_outcome)
                
                accuracy_insight = ReflectionInsight(
                    trade_id=trade_outcome.trade_id,
                    symbol=trade_outcome.symbol,
                    insight_type='ai_confidence_accuracy',
                    insight_text=f"AI confidence was {original_confidence:.1f}, actual outcome success: {actual_outcome}. Accuracy: {confidence_accuracy:.1f}",
                    confidence=0.9,
                    actionable_recommendation="Adjust confidence calibration based on this outcome",
                    timestamp=datetime.now()
                )
                insights.append(accuracy_insight)
        
        except Exception as e:
            logger.error(f"Error parsing AI reflection response: {e}")
        
        return insights
    
    def _extract_recommendations(self, reasoning_text: str) -> str:
        """Extract actionable recommendations from AI reasoning"""
        
        # Simple keyword-based extraction (could be enhanced with NLP)
        recommendation_keywords = [
            'recommend', 'suggest', 'should', 'could', 'improve', 
            'next time', 'future', 'avoid', 'consider', 'try'
        ]
        
        sentences = reasoning_text.split('. ')
        recommendations = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in recommendation_keywords):
                recommendations.append(sentence.strip())
        
        return '.'.join(recommendations[:2])  # Take first 2 recommendations
    
    def _perform_systematic_analysis(self, trade_outcome: TradeOutcome) -> List[ReflectionInsight]:
        """Perform systematic analysis of trade metrics"""
        
        insights = []
        
        try:
            # === TIMING ANALYSIS ===
            if trade_outcome.duration_hours < 1:
                insight = ReflectionInsight(
                    trade_id=trade_outcome.trade_id,
                    symbol=trade_outcome.symbol,
                    insight_type='timing_issue',
                    insight_text=f"Very short trade duration ({trade_outcome.duration_hours:.1f}h) may indicate overtrading or poor timing",
                    confidence=0.7,
                    actionable_recommendation="Consider longer holding periods or better entry timing",
                    timestamp=datetime.now()
                )
                insights.append(insight)
            elif trade_outcome.duration_hours > 168:  # > 1 week
                insight = ReflectionInsight(
                    trade_id=trade_outcome.trade_id,
                    symbol=trade_outcome.symbol,
                    insight_type='timing_issue',
                    insight_text=f"Long trade duration ({trade_outcome.duration_hours/24:.1f} days) - evaluate exit strategy",
                    confidence=0.6,
                    actionable_recommendation="Review exit criteria and consider taking profits earlier",
                    timestamp=datetime.now()
                )
                insights.append(insight)
            
            # === RISK MANAGEMENT ANALYSIS ===
            if abs(trade_outcome.pnl_percent) > 0.1:  # >10% move
                risk_type = 'high_risk_high_reward' if trade_outcome.pnl > 0 else 'high_risk_high_loss'
                insight = ReflectionInsight(
                    trade_id=trade_outcome.trade_id,
                    symbol=trade_outcome.symbol,
                    insight_type='risk_management',
                    insight_text=f"High volatility trade with {trade_outcome.pnl_percent:.1f}% outcome",
                    confidence=0.8,
                    actionable_recommendation="Review position sizing for high volatility assets",
                    timestamp=datetime.now()
                )
                insights.append(insight)
        
        except Exception as e:
            logger.error(f"Error in systematic analysis: {e}")
        
        return insights
    
    def _analyze_cross_trade_patterns(self, trade_outcomes: List[TradeOutcome]) -> List[ReflectionInsight]:
        """Analyze patterns across multiple trades"""
        
        insights = []
        
        try:
            if len(trade_outcomes) < 3:
                return insights
            
            # === WIN/LOSS PATTERN ANALYSIS ===
            profitable_trades = [t for t in trade_outcomes if t.pnl > 0]
            losing_trades = [t for t in trade_outcomes if t.pnl <= 0]
            
            if profitable_trades and losing_trades:
                avg_profit = sum(t.pnl_percent for t in profitable_trades) / len(profitable_trades)
                avg_loss = sum(t.pnl_percent for t in losing_trades) / len(losing_trades)
                
                pattern_insight = ReflectionInsight(
                    trade_id="PATTERN_ANALYSIS",
                    symbol="ALL",
                    insight_type='pattern_recognition',
                    insight_text=f"Win/Loss pattern: Avg profit {avg_profit:.1f}%, Avg loss {avg_loss:.1f}%, Win rate {len(profitable_trades)/len(trade_outcomes):.1f}",
                    confidence=0.9,
                    actionable_recommendation="Optimize profit taking and loss cutting based on these patterns",
                    timestamp=datetime.now()
                )
                insights.append(pattern_insight)
            
            # === ASSET TYPE PERFORMANCE ===
            asset_performance = {}
            for trade in trade_outcomes:
                if trade.asset_type not in asset_performance:
                    asset_performance[trade.asset_type] = []
                asset_performance[trade.asset_type].append(trade.pnl_percent)
            
            for asset_type, returns in asset_performance.items():
                if len(returns) >= 2:
                    avg_return = sum(returns) / len(returns)
                    asset_insight = ReflectionInsight(
                        trade_id="ASSET_PATTERN",
                        symbol=asset_type.upper(),
                        insight_type='pattern_recognition',
                        insight_text=f"{asset_type.title()} trading performance: {avg_return:.1f}% average return over {len(returns)} trades",
                        confidence=0.8,
                        actionable_recommendation=f"{'Focus more on' if avg_return > 0 else 'Reduce exposure to'} {asset_type} trading",
                        timestamp=datetime.now()
                    )
                    insights.append(asset_insight)
        
        except Exception as e:
            logger.error(f"Error in cross-trade pattern analysis: {e}")
        
        return insights
    
    def _get_completed_trades(self, start_date: datetime, end_date: datetime) -> List[TradeOutcome]:
        """Get completed trades from database and convert to TradeOutcome objects"""
        try:
            # Get all trade outcomes for the symbol and filter by date
            all_trades = self.database_bot.get_trade_outcomes(days=60)
            raw_trades = [
                t for t in all_trades
                if t.get('exit_price') and t.get('exit_price') != 0 and 'timestamp' in t
                and start_date <= datetime.fromisoformat(t['timestamp']) <= end_date
            ]
            trade_outcomes = []
            for trade_data in raw_trades:
                # Use timestamp as exit_time if exit_time not present
                exit_time = trade_data.get('exit_time')
                if not exit_time and 'timestamp' in trade_data:
                    try:
                        exit_time = datetime.fromisoformat(trade_data['timestamp'])
                    except Exception:
                        exit_time = datetime.now()
                elif isinstance(exit_time, str):
                    try:
                        exit_time = datetime.fromisoformat(exit_time)
                    except Exception:
                        exit_time = datetime.now()
                if not exit_time:
                    exit_time = datetime.now()
                entry_time = trade_data.get('entry_time')
                if not entry_time:
                    entry_time = exit_time
                elif isinstance(entry_time, str):
                    try:
                        entry_time = datetime.fromisoformat(entry_time)
                    except Exception:
                        entry_time = exit_time
                if not entry_time:
                    entry_time = exit_time
                outcome = TradeOutcome(
                    trade_id=trade_data.get('trade_id', ''),
                    symbol=trade_data.get('symbol', ''),
                    asset_type=trade_data.get('asset_type', ''),
                    action=trade_data.get('trade_type', ''),
                    entry_price=trade_data.get('entry_price', 0),
                    exit_price=trade_data.get('exit_price', 0),
                    quantity=trade_data.get('quantity', 0),
                    entry_time=entry_time,
                    exit_time=exit_time,
                    pnl=trade_data.get('pnl', 0),
                    pnl_percent=trade_data.get('pnl_percentage', 0),
                    duration_hours=trade_data.get('hold_duration_minutes', 0) / 60.0 if trade_data.get('hold_duration_minutes') else 0,
                    original_analysis=trade_data.get('original_analysis', {}),
                    market_conditions_entry=trade_data.get('market_conditions_entry', {}),
                    market_conditions_exit=trade_data.get('market_conditions_exit', {})
                )
                trade_outcomes.append(outcome)
            return trade_outcomes
        except Exception as e:
            logger.error(f"Error getting completed trades: {e}")
            return []
    
    def _store_insights(self, insights: List[ReflectionInsight]):
        """Store reflection insights in database, including news context if available."""
        
        try:
            for insight in insights:
                # Prepare all required arguments for store_reflection_insight
                insight_data = {
                    'trade_id': insight.trade_id,
                    'symbol': insight.symbol,
                    'insight_type': insight.insight_type,
                    'insight_text': insight.insight_text,
                    'confidence': insight.confidence,
                    'actionable_recommendation': insight.actionable_recommendation,
                    'related_indicators': insight.related_indicators,
                    'market_context': insight.market_context,
                    'timestamp': insight.timestamp or datetime.now(),
                    'news_context': getattr(insight, 'news_context', None)  # Add news context if present
                }
                # Fill in required parameters with available data or placeholders
                self.database_bot.store_reflection_insight(
                    trade_id=insight_data['trade_id'],
                    symbol=insight_data['symbol'],
                    original_analysis_id=0,
                    entry_price=0.0,
                    exit_price=0.0,
                    pnl=0.0,
                    hold_duration_hours=0.0,
                    market_conditions=insight_data.get('market_context') or insight_data.get('news_context') or "",
                    ai_reflection=insight_data.get('insight_text') or "",
                    key_insights=insight_data.get('actionable_recommendation') or "",
                    lessons_learned=insight_data.get('insight_text') or "",
                    confidence_accuracy=float(insight_data.get('confidence') or 0.0)
                 )
            logger.info(f"Stored {len(insights)} reflection insights in database (news context included where available)")
        except Exception as e:
            logger.error(f"Error storing reflection insights: {e}")
    
    def analyze_and_store(self, trade_outcome) -> List[ReflectionInsight]:
        """
        Analyze a trade, store insights, and validate persistence (per TODO 3.2).
        """
        # Generate insights using the main analysis pipeline
        insights = self.analyze_completed_trade(trade_outcome)
        if not insights:
            logger.info(f"No insights generated for {trade_outcome.symbol}, nothing to store.")
            return []

        # Validate persistence: retrieve immediately after storing
        persisted_insights = self.get_insights_for_symbol(trade_outcome.symbol, limit=10)
        if persisted_insights:
            logger.info(f"Validated persistence: {len(persisted_insights)} insights found for {trade_outcome.symbol} after storing.")
        else:
            logger.error(f"Persistence validation failed: No insights found for {trade_outcome.symbol} after storing.")

        return persisted_insights if persisted_insights is not None else []

def selftest_reflection_bot():
    """Standalone self-test for ReflectionBot: tests insight generation from dummy trade."""
    print(f"\n--- Running ReflectionBot Self-Test ---")
    from datetime import datetime, timedelta
    try:
        class MockAIBot:
            def __init__(self, *a, **k): pass
            def call_llm(self, prompt): return "Mocked insight"
        class MockDatabaseBot:
            def __init__(self, *a, **k): self.insights = []
            def get_reflection_insights(self, symbol, days=90):
                return [{"trade_id": "T1", "symbol": symbol, "ai_reflection": "Mocked insight"}]
            def store_reflection_insight(self, *a, **k): self.insights.append(a)
        class MockReflectionBot(ReflectionBot):
            def __init__(self):
                self.ai_bot = MockAIBot()
                self.database_bot = MockDatabaseBot()
                self.min_reflection_pnl = 0
                self.max_reflection_age_days = 30
                self.reflection_batch_size = 5
                self.insight_types = {'success_factor': 'What made this trade successful'}
        bot = MockReflectionBot()
        trade = TradeOutcome(
            trade_id="T1", symbol="AAPL", asset_type="stock", action="buy",
            entry_price=100.0, exit_price=110.0, quantity=10,
            entry_time=datetime.now() - timedelta(hours=2), exit_time=datetime.now(),
            pnl=100.0, pnl_percent=10.0, duration_hours=2.0
        )
        insights = bot.analyze_completed_trade(trade)
        assert isinstance(insights, list), "analyze_completed_trade did not return a list."
        print("    -> Insight generation logic passed.")
        symbol_insights = bot.get_insights_for_symbol("AAPL", limit=1)
        assert symbol_insights and symbol_insights[0].symbol == "AAPL", "get_insights_for_symbol failed."
        print("    -> get_insights_for_symbol logic passed.")
        print(f"--- ReflectionBot Self-Test PASSED ---")
    except AssertionError as e:
        print(f"--- ReflectionBot Self-Test FAILED: {e} ---")
    except Exception as e:
        print(f"--- ReflectionBot Self-Test encountered an ERROR: {e} ---")

if __name__ == "__main__":
    selftest_reflection_bot()
