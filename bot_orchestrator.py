# bot_orchestrator.py
"""
OrchestratorBot: Central coordinator for the agentic trading system.
- Manages the main trading loop and workflow orchestration
- Handles inter-bot communication and data flow
- Implements error handling and recovery mechanisms
- Coordinates the complete trading lifecycle
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from colorama import Style

# Import all specialized bots
from bot_stock import StockBot
from bot_crypto import CryptoBot
from bot_portfolio import PortfolioBot
from bot_ai import AIBot
from bot_risk_manager import RiskManager, Position
from bot_trade_executor import TradeExecutorBot
from bot_position_sizer import PositionSizerBot
from bot_database import DatabaseBot
from bot_decision_maker import DecisionMakerBot, AnalysisInput

# Configuration imports
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
from config_trading_variables import (
    TRADING_ASSETS, MIN_CONFIDENCE, MAX_PORTFOLIO_RISK, 
    MAX_POSITION_RISK, TRADING_CYCLE_INTERVAL
)

logger = logging.getLogger(__name__)

@dataclass
class TradingDecision:
    """Data structure for trading decisions passed between bots"""
    symbol: str
    asset_type: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    reasoning: str
    current_price: float
    position_size: float = 0.0
    allocation_usd: float = 0.0
    risk_assessment: Optional[Dict[str, Any]] = None

class OrchestratorBot:
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
    
    def __init__(self):
        """Initialize all specialized bots and components"""
        logger.info("🤖 Initializing OrchestratorBot...")
        
        # Initialize all specialized bots
        self.stock_bot = StockBot()
        self.crypto_bot = CryptoBot()
        self.portfolio_bot = PortfolioBot()
        self.ai_bot = AIBot()
        self.position_sizer = PositionSizerBot()
        self.risk_manager = RiskManager()
        self.database_bot = DatabaseBot()
        self.decision_maker = DecisionMakerBot()
        
        # Initialize trade executor
        self.trade_executor = TradeExecutorBot(
            api_key=ALPACA_API_KEY,
            api_secret=ALPACA_SECRET_KEY,
            paper_trading=True  # Always start with paper trading for safety
        )
        
        # Trading configuration
        self.assets = TRADING_ASSETS
        self.cycle_interval = TRADING_CYCLE_INTERVAL
        self.running = False
        
        logger.info("✅ OrchestratorBot initialized successfully")
    
    def start_trading_loop(self):
        """Start the main agentic trading loop"""
        print(f"\n{Style.BRIGHT}🚀 Starting Full Agentic Trading Loop...{Style.RESET_ALL}")
        print(f"📋 Configuration: MIN_CONFIDENCE={MIN_CONFIDENCE}, MAX_PORTFOLIO_RISK={MAX_PORTFOLIO_RISK}")
        
        self.running = True
        
        try:
            while self.running:
                print(f"\n{Style.BRIGHT}=== New Trading Cycle ==={Style.RESET_ALL}")
                
                # Execute one complete trading cycle
                cycle_successful = self._execute_trading_cycle()
                
                if cycle_successful:
                    print(f"✅ Trading cycle completed successfully")
                else:
                    print(f"⚠️ Trading cycle completed with some issues")
                
                # Sleep until next cycle
                print(f"\n⏱️ Sleeping {self.cycle_interval} seconds until next cycle...")
                time.sleep(self.cycle_interval)
                
        except KeyboardInterrupt:
            logger.info("Trading loop stopped by user.")
            print(f"\n👋 Trading loop stopped by user.")
            self.running = False
        except Exception as e:
            logger.error(f"Critical error in trading loop: {e}")
            print(f"\n❌ Critical error in trading loop: {e}")
            self.running = False
            raise
    
    def stop_trading_loop(self):
        """Stop the trading loop gracefully"""
        self.running = False
        logger.info("Trading loop stop requested")
    
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
                print(f"🛑 Portfolio risk too high - halting new trades this cycle")
                return False
            
            # === STEP 1: AI LEARNING & ADAPTATION ===
            min_confidence, prompt_note = self._perform_ai_adaptation()
            
            # === STEP 2: ANALYZE ASSETS & MAKE DECISIONS ===
            trading_decisions = self._analyze_assets_and_decide(min_confidence, prompt_note)
            
            # === STEP 3: EXECUTE APPROVED TRADES ===
            execution_results = self._execute_approved_trades(trading_decisions)
            
            # === STEP 4: UPDATE PORTFOLIO STATE ===
            self._update_portfolio_state(execution_results)
            
            # === STEP 5: PORTFOLIO SUMMARY ===
            self._print_cycle_summary()
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            print(f"❌ Error in trading cycle: {e}")
            cycle_success = False
        
        return cycle_success
    
    def _assess_portfolio_risk(self) -> bool:
        """
        STEP 0: Portfolio-level risk assessment
        
        Returns:
            bool: True if portfolio risk is acceptable, False otherwise
        """
        print(f"\n🛡️ Portfolio Risk Assessment...")
        
        try:
            # Get current positions from portfolio bot
            open_positions = self.portfolio_bot.get_open_positions()

            # Convert to Position objects for risk manager
            position_objects = []
            for symbol, trade in open_positions.items():
                # Fetch live price for each open position (always use live price, fallback to 0.0 if unavailable)
                if trade.asset_type == 'stock':
                    current_price = self.stock_bot.get_current_price(symbol)
                    if current_price is None:
                        current_price = 0.0
                elif trade.asset_type == 'crypto':
                    data = self.crypto_bot.get_crypto_data(symbol)
                    current_price = data['current_price'] if data and data.get('current_price') is not None else 0.0
                else:
                    current_price = 0.0
                position_obj = Position(
                    symbol=symbol,
                    quantity=trade.quantity,
                    entry_price=trade.entry_price,
                    current_price=current_price,
                    asset_type=trade.asset_type
                )
                position_objects.append(position_obj)

            # Calculate portfolio-level risk
            portfolio_risk = self.risk_manager.calculate_portfolio_risk(position_objects)
            portfolio_risk_pct = portfolio_risk.get('portfolio_pnl_percent', 0)

            print(f"   📊 Portfolio Risk Level: {abs(portfolio_risk_pct)*100:.1f}%")

            # Check if we should halt trading due to high portfolio risk
            if abs(portfolio_risk_pct) > MAX_PORTFOLIO_RISK:
                print(f"   ⚠️ PORTFOLIO RISK TOO HIGH ({abs(portfolio_risk_pct)*100:.1f}% > {MAX_PORTFOLIO_RISK*100:.1f}%)")
                return False
            else:
                print(f"   ✅ Portfolio risk within acceptable limits")
                return True
                
        except Exception as e:
            logger.error(f"Error in portfolio risk assessment: {e}")
            print(f"   ⚠️ Portfolio risk assessment failed, proceeding with caution")
            return True  # Proceed with caution if assessment fails
    
    def _perform_ai_adaptation(self) -> Tuple[float, Optional[str]]:
        """
        STEP 1: AI Learning & Adaptation
        
        Returns:
            Tuple[float, Optional[str]]: (min_confidence, prompt_note)
        """
        print(f"\n🧠 AI Learning & Adaptation...")
        
        try:
            # Get trade history and performance metrics for AI learning
            trade_history = self.portfolio_bot.get_trade_history()
            metrics = self.portfolio_bot.get_portfolio_metrics()
            win_rate = metrics.get('win_rate', 0.0)
            
            # AI adapts its confidence threshold based on performance
            min_confidence, prompt_note = self.ai_bot.adapt_with_performance(trade_history, win_rate)
            print(f"   📈 Win Rate: {win_rate*100:.1f}% | Adapted Min Confidence: {min_confidence:.1f}")
            print(f"   💭 AI Prompt Note: {prompt_note[:100]}..." if prompt_note else "   💭 No specific prompt adaptations")
            
            return min_confidence, prompt_note
            
        except Exception as e:
            logger.error(f"Error in AI adaptation: {e}")
            return MIN_CONFIDENCE, None
    
    def _analyze_assets_and_decide(self, min_confidence: float, prompt_note: Optional[str]) -> List[TradingDecision]:
        """
        STEP 2: Analyze each asset and make trading decisions
        
        Args:
            min_confidence: Minimum confidence threshold from AI adaptation
            prompt_note: Enhanced prompt note from AI adaptation
        
        Returns:
            List[TradingDecision]: List of trading decisions for approved trades
        """
        print(f"\n📊 Asset Analysis & Trading Decisions...")
        
        trading_decisions = []
        
        for symbol, asset_type, allocation_usd in self.assets:
            try:
                print(f"\n   🔍 Analyzing {symbol} ({asset_type.upper()})...")
                
                # === GET AI ANALYSIS ===
                analysis = self._get_asset_analysis(symbol, asset_type, prompt_note)
                if not analysis or 'error' in analysis:
                    print(f"      ❌ Analysis failed for {symbol}")
                    continue
                
                # Fetch live price for each asset
                if asset_type == 'stock':
                    current_price = self.stock_bot.get_current_price(symbol)
                    if current_price is None:
                        current_price = 0.0
                elif asset_type == 'crypto':
                    data = self.crypto_bot.get_crypto_data(symbol)
                    current_price = data['current_price'] if data and data['current_price'] is not None else 0.0
                else:
                    current_price = 0.0
                # Prepare analysis input for DecisionMakerBot
                analysis_input = AnalysisInput(
                    symbol=symbol,
                    asset_type=asset_type,
                    action=analysis.get('action', '').lower(),
                    confidence=analysis.get('confidence', 0.0),
                    reasoning=analysis.get('reasoning', 'No reasoning provided'),
                    technical_indicators=analysis.get('indicators', {}),
                    risk_metrics=analysis.get('risk_metrics', {}),
                )
                # Get current portfolio risk for filters (build Position list)
                open_positions = self.portfolio_bot.get_open_positions()
                position_objects = []
                for sym, trade in open_positions.items():
                    # Use same price fetching logic as above
                    if trade.asset_type == 'stock':
                        pos_price = self.stock_bot.get_current_price(sym)
                        if pos_price is None:
                            pos_price = 0.0
                    elif trade.asset_type == 'crypto':
                        pos_data = self.crypto_bot.get_crypto_data(sym)
                        pos_price = pos_data['current_price'] if pos_data and pos_data['current_price'] is not None else 0.0
                    else:
                        pos_price = 0.0
                    position_obj = Position(
                        symbol=sym,
                        quantity=trade.quantity,
                        entry_price=trade.entry_price,
                        current_price=pos_price,
                        asset_type=trade.asset_type
                    )
                    position_objects.append(position_obj)
                portfolio_risk = self.risk_manager.calculate_portfolio_risk(position_objects)
                current_portfolio_risk = portfolio_risk.get('portfolio_pnl_percent', 0)
                
                # Use DecisionMakerBot for final decision
                decision_obj = self.decision_maker.make_trading_decision(
                    analysis_input,
                    min_confidence=min_confidence,
                    current_portfolio_risk=current_portfolio_risk
                )
                
                if decision_obj.final_action not in ['buy', 'sell']:
                    print(f"      ⏸️ DecisionMakerBot recommends HOLD - no action taken")
                    continue
                
                # Create trading decision for execution
                decision = TradingDecision(
                    symbol=symbol,
                    asset_type=asset_type,
                    action=decision_obj.final_action,
                    confidence=decision_obj.final_confidence,
                    reasoning=decision_obj.decision_rationale,
                    current_price=current_price,
                    allocation_usd=allocation_usd
                )
                
                trading_decisions.append(decision)
                print(f"      ✅ Trade decision approved for {symbol}")
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                print(f"   ❌ Error processing {symbol}: {e}")
                continue
        
        print(f"\n📋 Generated {len(trading_decisions)} trading decisions")
        return trading_decisions
    
    def _get_asset_analysis(self, symbol: str, asset_type: str, prompt_note: Optional[str]) -> Dict[str, Any]:
        """Get AI analysis for a specific asset"""
        try:
            # Get historical analysis context from database (RAG)
            historical_context = self._get_historical_analysis_context(symbol)
            
            # Enhanced prompt note with historical context
            enhanced_prompt = prompt_note or ""
            if historical_context:
                enhanced_prompt += f"\n\nHistorical Analysis Context:\n{historical_context}"
            
            # Get analysis from appropriate bot
            if asset_type == 'stock':
                analysis = self.stock_bot.analyze_stock(symbol, prompt_note=enhanced_prompt)
            elif asset_type == 'crypto':
                analysis = self.crypto_bot.analyze_crypto(symbol, prompt_note=enhanced_prompt)
            else:
                return {'error': f'Unknown asset type: {asset_type}'}
            
            # Ensure the analysis is always a dictionary
            if isinstance(analysis, dict):
                return analysis
            elif isinstance(analysis, str):
                return {'result': analysis}
            else:
                return {'error': 'Unknown analysis result type'}
            
        except Exception as e:
            logger.error(f"Error getting analysis for {symbol}: {e}")
            return {'error': str(e)}
    
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
                date = analysis.get('timestamp', 'Unknown')
                action = analysis.get('action', 'Unknown')
                confidence = analysis.get('confidence', 0.0)
                context_lines.append(f"- {date}: {action.upper()} (confidence: {confidence:.1f})")
            
            return f"Recent AI decisions for {symbol}:\n" + "\n".join(context_lines)
            
        except Exception as e:
            logger.error(f"Error getting historical context for {symbol}: {e}")
            return ""
    
    def _execute_approved_trades(self, trading_decisions: List[TradingDecision]) -> List[Dict[str, Any]]:
        """
        STEP 3: Execute approved trades with position sizing and risk checks
        
        Args:
            trading_decisions: List of approved trading decisions
        
        Returns:
            List[Dict[str, Any]]: Execution results
        """
        print(f"\n🚀 Executing Approved Trades...")
        
        execution_results = []
        
        for decision in trading_decisions:
            try:
                print(f"\n   💰 Processing {decision.symbol}...")
                
                # === POSITION SIZING ===
                shares = self._calculate_position_size(decision)
                if shares <= 0:
                    print(f"      ⏸️ Position size calculation resulted in 0 shares - skipping")
                    continue
                
                decision.position_size = shares
                print(f"      📏 Position Size: {shares} shares @ ${decision.current_price:.2f}")
                
                # === PRE-TRADE RISK CHECKS ===
                if not self._check_position_risk(decision):
                    print(f"      ❌ Position risk too high - skipping trade")
                    continue
                
                # === TRADE EXECUTION ===
                execution_result = self._execute_trade(decision)
                execution_results.append(execution_result)
                
            except Exception as e:
                logger.error(f"Error executing trade for {decision.symbol}: {e}")
                print(f"   ❌ Error executing {decision.symbol}: {e}")
                continue
        
        return execution_results
    
    def _calculate_position_size(self, decision: TradingDecision) -> float:
        """Calculate position size using PositionSizerBot"""
        try:
            available_cash = self.portfolio_bot.current_capital * 0.5  # Use 50% of capital max
            
            shares = self.position_sizer.calculate_position_size(
                price=decision.current_price,
                confidence=decision.confidence,
                volatility=None,  # Could add volatility calculation
                available_cash=decision.allocation_usd,
                min_shares=1,
                allow_fractional=(decision.asset_type == 'crypto')
            )
            
            return max(0, shares)
            
        except Exception as e:
            logger.error(f"Error calculating position size for {decision.symbol}: {e}")
            return 0
    
    def _check_position_risk(self, decision: TradingDecision) -> bool:
        """Check if position risk is acceptable"""
        try:
            # Create hypothetical position for risk calculation
            hypothetical_position = Position(
                symbol=decision.symbol,
                quantity=decision.position_size,
                entry_price=decision.current_price,
                current_price=decision.current_price,
                asset_type=decision.asset_type
            )
            
            # Calculate position risk
            position_risk = self.risk_manager.calculate_position_risk(hypothetical_position)
            position_investment = position_risk['investment']
            max_loss = position_risk['max_loss']
            
            # Check if position risk exceeds limits
            if position_investment > self.portfolio_bot.current_capital * MAX_POSITION_RISK:
                print(f"      ⚠️ Position risk too high: ${position_investment:.2f} > ${self.portfolio_bot.current_capital * MAX_POSITION_RISK:.2f}")
                return False
            
            print(f"      ✅ Risk acceptable - Max Loss: ${max_loss:.2f}")
            decision.risk_assessment = position_risk
            return True
            
        except Exception as e:
            logger.error(f"Error checking position risk for {decision.symbol}: {e}")
            return False
    
    def _execute_trade(self, decision: TradingDecision) -> Dict[str, Any]:
        """Execute a single trade"""
        try:
            print(f"      🚀 Executing {decision.action} order for {decision.symbol}...")
            
            success, order_response = self.trade_executor.execute_trade(
                symbol=decision.symbol,
                side=decision.action,
                quantity=decision.position_size,
                confidence=decision.confidence
            )
            
            execution_result = {
                'decision': decision,
                'success': success,
                'order_response': order_response,
                'timestamp': time.time()
            }
            
            if success:
                print(f"      ✅ Trade executed successfully!")
                print(f"         Order ID: {getattr(order_response, 'id', 'N/A')}")
            else:
                print(f"      ❌ Trade execution failed: {order_response}")
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Error executing trade for {decision.symbol}: {e}")
            return {
                'decision': decision,
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _update_portfolio_state(self, execution_results: List[Dict[str, Any]]):
        """
        STEP 4: Update portfolio state based on execution results
        """
        print(f"\n📊 Updating Portfolio State...")
        
        for result in execution_results:
            if not result['success']:
                continue
            
            try:
                decision = result['decision']
                
                if decision.action == 'buy':
                    self.portfolio_bot.add_or_update_position(
                        symbol=decision.symbol,
                        asset_type=decision.asset_type,
                        quantity=decision.position_size,
                        entry_price=decision.current_price
                    )
                    # Update capital (reduce by investment amount)
                    self.portfolio_bot.current_capital -= decision.position_size * decision.current_price
                    print(f"   📈 Added {decision.symbol} position to portfolio")
                    
                elif decision.action == 'sell':
                    self.portfolio_bot.close_position(decision.symbol, decision.current_price)
                    print(f"   📉 Closed {decision.symbol} position in portfolio")
                
            except Exception as e:
                logger.error(f"Error updating portfolio for {decision.symbol}: {e}")
                print(f"   ❌ Error updating portfolio for {decision.symbol}: {e}")
        
        print(f"   💰 Updated capital: ${self.portfolio_bot.current_capital:.2f}")
    
    def _print_cycle_summary(self):
        """Print summary of the trading cycle"""
        print(f"\n📊 Trading Cycle Summary...")
        self.portfolio_bot.print_portfolio_summary()
