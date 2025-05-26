# Core imports
import logging
import time
from colorama import init, Style

# Bot imports
from bot_stock import StockBot
from bot_crypto import CryptoBot
from bot_portfolio import PortfolioBot
from bot_indicators import IndicatorBot
from bot_risk_manager import RiskManager, Position
from bot_trade_executor import TradeExecutorBot
from bot_position_sizer import PositionSizerBot
from bot_ai import AIBot
from bot_reflection import ReflectionBot
from bot_asset_screener import AssetScreenerBot
from bot_backtester import BacktesterBot
from bot_database import DatabaseBot

# Configuration imports
from config import ENABLE_TRADING_BOT, ALPACA_API_KEY, ALPACA_SECRET_KEY
from config_trading_variables import TRADING_ASSETS, MIN_CONFIDENCE, MAX_PORTFOLIO_RISK, MAX_POSITION_RISK

def main():
    # Initialize colorama for Windows
    init()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    print(f"\n{Style.BRIGHT}🤖 Starting StockAgenticAI...{Style.RESET_ALL}")

    # Use assets from trading_variables.py
    assets = TRADING_ASSETS
    stock_bot = StockBot()
    crypto_bot = CryptoBot()
    portfolio_bot = PortfolioBot()
    reflection_bot = ReflectionBot()
    database_bot = DatabaseBot()
    ai_bot_instance = AIBot()
    asset_screener_bot = AssetScreenerBot(ai_bot=ai_bot_instance, database_bot=database_bot)
    backtester_bot = BacktesterBot()

    try:
        # Get win rate and trade history for learning
        trade_history = portfolio_bot.get_trade_history()
        metrics = portfolio_bot.get_portfolio_metrics()
        win_rate = metrics.get('win_rate', 0.0)
        min_confidence, prompt_note = ai_bot.adapt_with_performance(trade_history, win_rate)
        # Asset screening before analysis
        screened_assets = asset_screener_bot.screen_assets()
        for asset in screened_assets:
            symbol = asset.symbol
            asset_type = 'stock' if symbol in stock_bot.asset_type_map and stock_bot.asset_type_map[symbol] == 'stock' else 'crypto'
            print(f"\n📊 Analyzing {symbol} ({asset_type.upper()})...")
            if asset_type == 'stock':
                current_price = stock_bot.get_current_price(symbol)
                analysis = stock_bot.analyze_stock(symbol, prompt_note=prompt_note)
                stock_bot.print_performance_summary(symbol, asset_type, timeframe=None)
            elif asset_type == 'crypto':
                crypto_data = crypto_bot.get_crypto_data(symbol)
                current_price = crypto_data['current_price'] if crypto_data else None
                analysis = crypto_bot.analyze_crypto(symbol, prompt_note=prompt_note)
                crypto_bot.print_performance_summary(symbol, asset_type, timeframe=None)
            # Backtest the analysis (minimal config for demonstration)
            from datetime import datetime, timedelta
            from bot_backtester import BacktestConfig
            config = BacktestConfig(
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now(),
                initial_capital=10000,
                assets_to_test=[(symbol, asset_type, 1000)]
            )
            backtester_bot.run_backtest(config)
            # Reflection on recent trades
            reflection_bot.batch_analyze_recent_trades(days_back=7)
            # Optionally, update portfolio_bot with new analysis here
        portfolio_bot.print_portfolio_summary()
    except KeyboardInterrupt:
        logger.info("Stopping the trading bot...")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

TRADING_CYCLE_INTERVAL = 60  # seconds between trading cycles (adjust as needed)

def trading_loop():
    """
    FULL AGENTIC TRADING LIFECYCLE IMPLEMENTATION
    
    This is the main trading loop that implements the complete agentic trading workflow:
    1. AI Analysis & Decision Making
    2. Position Sizing based on confidence and risk
    3. Pre-trade Risk Checks
    4. Trade Execution
    5. Portfolio State Updates
    6. Continuous Learning & Adaptation
    """    # Initialize all bot components for the trading system
    stock_bot = StockBot()
    crypto_bot = CryptoBot()
    portfolio_bot = PortfolioBot()
    reflection_bot = ReflectionBot()
    database_bot = DatabaseBot()
    ai_bot_instance = AIBot()
    asset_screener_bot = AssetScreenerBot(ai_bot=ai_bot_instance, database_bot=database_bot)
    backtester_bot = BacktesterBot()
    assets = TRADING_ASSETS
    logger = logging.getLogger(__name__)
    
    # Initialize position sizer, risk manager, and trade executor
    position_sizer = PositionSizerBot()
    risk_manager = RiskManager()
    trade_executor = TradeExecutorBot(
        api_key=ALPACA_API_KEY,
        api_secret=ALPACA_SECRET_KEY,
        paper_trading=True  # Always start with paper trading for safety
    )
    
    print(f"\n{Style.BRIGHT}🚀 Starting Full Agentic Trading Loop...{Style.RESET_ALL}")
    print(f"📋 Configuration: MIN_CONFIDENCE={MIN_CONFIDENCE}, MAX_PORTFOLIO_RISK={MAX_PORTFOLIO_RISK}")
    
    try:
        while True:
            print(f"\n{Style.BRIGHT}=== New Trading Cycle ==={Style.RESET_ALL}")
            
            # === STEP 0: PORTFOLIO-LEVEL RISK ASSESSMENT ===
            print(f"\n🛡️ Portfolio Risk Assessment...")
            try:
                # Get current positions from portfolio bot
                open_positions = portfolio_bot.get_open_positions()
                  # Convert to Position objects for risk manager
                position_objects = []
                for symbol, trade in open_positions.items():
                    # Get current price (simplified - in production, fetch real current price)
                    current_price = trade.entry_price * 1.01  # Assume small gain for demo
                    position_obj = Position(
                        symbol=symbol,
                        quantity=trade.quantity,
                        entry_price=trade.entry_price,
                        current_price=current_price,
                        asset_type=trade.asset_type
                    )
                    position_objects.append(position_obj)
                
                # Calculate portfolio-level risk
                portfolio_risk = risk_manager.calculate_portfolio_risk(position_objects)
                portfolio_risk_pct = portfolio_risk.get('portfolio_pnl_percent', 0)
                
                print(f"   📊 Portfolio Risk Level: {abs(portfolio_risk_pct)*100:.1f}%")
                
                # Check if we should halt trading due to high portfolio risk
                if abs(portfolio_risk_pct) > MAX_PORTFOLIO_RISK:
                    print(f"   ⚠️ PORTFOLIO RISK TOO HIGH ({abs(portfolio_risk_pct)*100:.1f}% > {MAX_PORTFOLIO_RISK*100:.1f}%)")
                    print(f"   🛑 Halting new trades this cycle for risk management")
                    time.sleep(TRADING_CYCLE_INTERVAL)
                    continue
                else:
                    print(f"   ✅ Portfolio risk within acceptable limits")
                    
            except Exception as e:
                logger.error(f"Error in portfolio risk assessment: {e}")
                print(f"   ⚠️ Portfolio risk assessment failed, proceeding with caution")
            
            # === STEP 1: AI LEARNING & ADAPTATION ===
            print(f"\n🧠 AI Learning & Adaptation...")
            try:
                # Get trade history and performance metrics for AI learning
                trade_history = portfolio_bot.get_trade_history()
                metrics = portfolio_bot.get_portfolio_metrics()
                win_rate = metrics.get('win_rate', 0.0)
                
                # AI adapts its confidence threshold based on performance
                min_confidence, prompt_note = ai_bot.adapt_with_performance(trade_history, win_rate)
                print(f"   📈 Win Rate: {win_rate*100:.1f}% | Adapted Min Confidence: {min_confidence:.1f}")
                print(f"   💭 AI Prompt Note: {prompt_note[:100]}..." if prompt_note else "   💭 No specific prompt adaptations")
                
            except Exception as e:
                logger.error(f"Error in AI adaptation: {e}")
                min_confidence = MIN_CONFIDENCE
                prompt_note = None
            
            # === STEP 2: ANALYZE EACH ASSET & MAKE TRADING DECISIONS ===
            print(f"\n📊 Asset Analysis & Trading Decisions...")
            
            # Asset screening before analysis
            screened_assets = asset_screener_bot.screen_assets()
            for asset in screened_assets:
                symbol = asset.symbol
                asset_type = 'stock' if symbol in stock_bot.asset_type_map and stock_bot.asset_type_map[symbol] == 'stock' else 'crypto'
                allocation_usd = 1000  # Or derive from config if needed
                try:
                    print(f"\n   🔍 Analyzing {symbol} ({asset_type.upper()})...")
                    
                    # === STEP 2A: RAG - GET HISTORICAL ANALYSIS FOR CONTEXT ===
                    print(f"      📚 Retrieving historical analysis for context...")
                    try:
                        # Get database connection through portfolio bot
                        from bot_database import DatabaseBot
                        db_bot = DatabaseBot()
                        
                        # Get recent analysis history for this symbol
                        historical_analyses = db_bot.get_analysis_history(symbol, days=7)
                        
                        # Create context summary for the AI
                        context_summary = ""
                        if historical_analyses:
                            recent_count = len(historical_analyses)
                            recent_actions = [analysis[2] for analysis in historical_analyses[:3]]  # action is 3rd field
                            avg_confidence = sum(analysis[3] for analysis in historical_analyses[:5]) / min(5, len(historical_analyses))
                            
                            context_summary = f"Recent {recent_count} analyses: {', '.join(recent_actions)}. Avg confidence: {avg_confidence:.1f}"
                            print(f"         📈 Historical context: {context_summary}")
                        else:
                            context_summary = "No recent analysis history available"
                            print(f"         📊 No historical data for {symbol}")
                            
                        # Enhanced prompt note that includes historical context
                        enhanced_prompt_note = f"{prompt_note or ''}\nHistorical Context: {context_summary}".strip()
                        
                    except Exception as history_error:
                        logger.warning(f"Failed to retrieve historical analysis for {symbol}: {history_error}")
                        enhanced_prompt_note = prompt_note
                    
                    # === STEP 2B: GET AI ANALYSIS WITH ENHANCED CONTEXT ===                    
                    if asset_type == 'stock':
                        current_price = stock_bot.get_current_price(symbol)
                        analysis = stock_bot.analyze_stock(symbol, prompt_note=enhanced_prompt_note)
                    elif asset_type == 'crypto':
                        crypto_data = crypto_bot.get_crypto_data(symbol)
                        current_price = crypto_data['current_price'] if crypto_data else None
                        analysis = crypto_bot.analyze_crypto(symbol, prompt_note=enhanced_prompt_note)
                    else:
                        print(f"      ❌ Unknown asset type: {asset_type}")
                        continue
                      # Check if analysis was successful and is a dictionary
                    if not isinstance(analysis, dict):
                        print(f"      ❌ Analysis failed: Expected dictionary, got {type(analysis)}")
                        continue
                        
                    if 'error' in analysis:
                        print(f"      ❌ Analysis failed: {analysis['error']}")
                        continue
                      # Extract analysis results with type checking
                    action = analysis.get('action', '').lower()
                    confidence = analysis.get('confidence', 0.0)
                    reasoning = analysis.get('reasoning', 'No reasoning provided')
                    
                    # Ensure confidence is a float
                    try:
                        confidence = float(confidence)
                    except (ValueError, TypeError):
                        print(f"      ❌ Invalid confidence value: {confidence} - using default 0.0")
                        confidence = 0.0
                    
                    print(f"      🤖 AI Decision: {action.upper()} (Confidence: {confidence*100:.1f}%)")
                    print(f"      📝 Reasoning: {reasoning}")
                    
                    # === STEP 2B: CONFIDENCE CHECK ===
                    if confidence < min_confidence:
                        print(f"      ⏸️ Confidence too low ({confidence:.1f} < {min_confidence:.1f}) - HOLDING")
                        continue
                    
                    if action not in ['buy', 'sell']:
                        print(f"      ⏸️ AI recommends HOLD - no action taken")
                        continue
                    
                    # === STEP 3: POSITION SIZING ===
                    print(f"      💰 Calculating position size...")
                    
                    # Use live price for position sizing
                    available_cash = portfolio_bot.current_capital * 0.5  # Use 50% of capital max
                    shares = position_sizer.calculate_position_size(
                        price=current_price,
                        confidence=confidence,
                        volatility=None,
                        available_cash=allocation_usd,
                        min_shares=1,
                        allow_fractional=(asset_type == 'crypto')
                    )
                    
                    if shares <= 0:
                        print(f"      ⏸️ Position size calculation resulted in 0 shares - skipping")
                        continue
                    
                    print(f"      📏 Position Size: {shares} shares @ ${current_price:.2f} = ${shares * current_price:.2f}")
                    
                    # === STEP 4: PRE-TRADE RISK CHECKS ===
                    print(f"      🛡️ Pre-trade risk assessment...")
                      # Create a hypothetical position for risk calculation
                    hypothetical_position = Position(
                        symbol=symbol,
                        quantity=shares,
                        entry_price=current_price,
                        current_price=current_price,
                        asset_type=asset_type
                    )
                    
                    # Calculate position risk
                    position_risk = risk_manager.calculate_position_risk(hypothetical_position)
                    position_investment = position_risk['investment']
                    max_loss = position_risk['max_loss']
                    
                    # Check if position risk exceeds our limits
                    if position_investment > portfolio_bot.current_capital * MAX_POSITION_RISK:
                        print(f"      ⚠️ POSITION RISK TOO HIGH - reducing size")
                        # Reduce position size to fit within risk limits
                        max_investment = portfolio_bot.current_capital * MAX_POSITION_RISK
                        shares = max_investment / current_price
                        if asset_type != 'crypto':
                            shares = int(shares)
                        print(f"      📏 Reduced Position Size: {shares} shares")
                        
                        if shares <= 0:
                            print(f"      ❌ Even reduced position too risky - skipping trade")
                            continue
                    
                    print(f"      ✅ Risk acceptable - Max Loss: ${max_loss:.2f}")
                    
                    # === STEP 5: TRADE EXECUTION ===
                    print(f"      🚀 Executing trade...")
                    
                    try:
                        # Execute the trade using TradeExecutorBot
                        success, order_response = trade_executor.execute_trade(
                            symbol=symbol,
                            side=action,  # 'buy' or 'sell'
                            quantity=shares,
                            confidence=confidence
                        )
                        
                        if success:
                            print(f"      ✅ Trade executed successfully!")
                            print(f"         Order ID: {getattr(order_response, 'id', 'N/A')}")
                            
                            # === STEP 6: UPDATE PORTFOLIO STATE ===
                            if action == 'buy':
                                portfolio_bot.add_or_update_position(
                                    symbol=symbol,
                                    asset_type=asset_type,
                                    quantity=shares,
                                    entry_price=current_price
                                )
                                # Update capital (reduce by investment amount)
                                portfolio_bot.current_capital -= shares * current_price
                                print(f"         📈 Added position to portfolio")
                                
                            elif action == 'sell':
                                portfolio_bot.close_position(symbol, current_price)
                                print(f"         📉 Closed position in portfolio")
                            
                            print(f"         💰 Updated capital: ${portfolio_bot.current_capital:.2f}")
                            
                        else:
                            print(f"      ❌ Trade execution failed: {order_response}")
                            
                    except Exception as trade_error:
                        logger.error(f"Trade execution error for {symbol}: {trade_error}")
                        print(f"      ❌ Trade execution error: {trade_error}")
                    
                except Exception as asset_error:
                    logger.error(f"Error processing {symbol}: {asset_error}")
                    print(f"   ❌ Error processing {symbol}: {asset_error}")
                    continue
            
            # === STEP 7: PORTFOLIO SUMMARY & CYCLE COMPLETION ===
            print(f"\n📊 Portfolio Summary...")
            portfolio_bot.print_portfolio_summary()
            
            print(f"\n⏱️ Sleeping {TRADING_CYCLE_INTERVAL} seconds until next cycle...")
            time.sleep(TRADING_CYCLE_INTERVAL)
            
    except KeyboardInterrupt:
        logger.info("Trading loop stopped by user.")
        print(f"\n👋 Trading loop stopped by user.")
    except Exception as e:
        logger.error(f"Critical error in trading loop: {e}")
        print(f"\n❌ Critical error in trading loop: {e}")
        raise

if __name__ == "__main__":
    if ENABLE_TRADING_BOT:
        trading_loop()
    else:
        main()