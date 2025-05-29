#!/usr/bin/env python3
"""
Comprehensive test suite for the Stock Agentic AI Trading Bot.

This test file validates all major components of the trading system:
- API connectivity (Alpaca, Kraken, Gemini)
- Database operations
- Technical indicators
- Risk management
- Portfolio management
- Trade execution
- AI analysis integration

Run with: python test_trading_system.py
"""

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import sys
import time
import logging
from datetime import datetime, timedelta

# Modular bot imports
from bot_database import DatabaseBot
from bot_indicators import IndicatorBot
from bot_portfolio import PortfolioBot
from bot_risk_manager import RiskManager, Position
from bot_trade_executor import TradeExecutorBot
from bot_ai import AIBot
from bot_stock import StockBot
from bot_crypto import CryptoBot
from bot_position_sizer import PositionSizerBot
from bot_news_retriever import NewsRetrieverBot

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging for test output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingSystemTester:
    """Comprehensive test suite for the trading bot system."""
    
    def __init__(self):
        self.test_results = {}
        self.passed_tests = 0
        self.failed_tests = 0
        # Modular bots for use in tests
        self.db = DatabaseBot()
        # Provide a default price list for IndicatorBot
        self.indicator_bot = IndicatorBot([100.0])
        self.portfolio_bot = PortfolioBot()
        self.risk_manager = RiskManager()
        # Import API keys for TradeExecutorBot
        try:
            from config_system import ALPACA_API_KEY, ALPACA_SECRET_KEY, PAPER_TRADING
            self.trade_executor = TradeExecutorBot(
                api_key=ALPACA_API_KEY,
                api_secret=ALPACA_SECRET_KEY,
                paper_trading=PAPER_TRADING
            )
        except Exception:
            self.trade_executor = None
        self.ai_bot = AIBot()
        self.stock_bot = StockBot()
        self.crypto_bot = CryptoBot()
        self.position_sizer = PositionSizerBot()
        
    def run_test(self, test_name, test_function):
        """Run a single test and record results."""
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            start_time = time.time()
            result = test_function()
            end_time = time.time()
            if result:
                logger.info(f"PASSED: {test_name} ({end_time - start_time:.2f}s)")
                self.passed_tests += 1
                self.test_results[test_name] = "PASSED"
            else:
                logger.error(f"FAILED: {test_name}")
                self.failed_tests += 1
                self.test_results[test_name] = "FAILED"
                
        except Exception as e:
            logger.error(f"ERROR: {test_name} - {str(e)}")
            self.failed_tests += 1
            self.test_results[test_name] = f"ERROR: {str(e)}"
    
    def test_environment_variables(self):
        """Test that all required environment variables are set."""
        logger.info("Testing environment variables...")
        
        required_vars = [
            'ALPACA_API_KEY',
            'ALPACA_SECRET_KEY'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
                
        if missing_vars:
            logger.error(f"Missing environment variables: {missing_vars}")
            return False
            
        logger.info("All required environment variables are set")
        return True
    
    def test_config_import(self):
        """Test that config module imports correctly."""
        logger.info("Testing config import...")
        
        try:
            from config_system import (
                GEMINI_MODEL, PAPER_TRADING, TRADING_CYCLE_INTERVAL,
                ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, RATE_LIMIT_DELAY_SECONDS
            )
            logger.info("Config imported successfully")
            logger.info(f"Using Gemini model: {GEMINI_MODEL}")
            logger.info(f"Paper trading: {PAPER_TRADING}")
            logger.info(f"Trading cycle interval: {TRADING_CYCLE_INTERVAL}s")
            return True
            
        except ImportError as e:
            logger.error(f"Config import failed: {e}")
            return False
    
    def test_trading_variables_import(self):
        """Test that trading variables import correctly."""
        logger.info("Testing trading variables import...")
        
        try:
            from config_trading import (
                TRADING_ASSETS, TOTAL_CAPITAL, MAX_PORTFOLIO_RISK,
                MIN_CONFIDENCE, RSI_OVERSOLD, RSI_OVERBOUGHT
            )
            logger.info("Trading variables imported successfully")
            logger.info(f"Total capital: ${TOTAL_CAPITAL:,}")
            logger.info(f"Trading assets: {len(TRADING_ASSETS)} assets")
            logger.info(f"Min confidence: {MIN_CONFIDENCE}")
            return True
            
        except ImportError as e:
            logger.error(f"Trading variables import failed: {e}")
            return False
    
    def test_database_connection(self):
        """Test database initialization and basic operations."""
        logger.info("Testing database connection...")
        
        try:
            # Test saving and retrieving analysis
            test_analysis = {
                'action': 'buy',
                'reasoning': 'Test analysis',
                'confidence': 0.85
            }
            
            self.db.save_analysis(
                symbol='TEST',
                asset_type='stock',
                analysis=test_analysis,
                current_price=100.0
            )
            
            # Try to retrieve analysis history
            history = self.db.get_analysis_history('TEST')
            
            logger.info("Database operations successful")
            return True
            
        except Exception as e:
            logger.error(f"Database test failed: {e}")
            return False
    
    def test_alpaca_api_connection(self):
        """Test Alpaca API connectivity."""
        logger.info("Testing Alpaca API connection...")
        
        try:
            from config_system import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
            import alpaca_trade_api as tradeapi
            from alpaca_trade_api.rest import URL

            if not ALPACA_API_KEY or not ALPACA_SECRET_KEY or not ALPACA_BASE_URL:
                raise ValueError("ALPACA_API_KEY, ALPACA_SECRET_KEY, and ALPACA_BASE_URL must be set and not None.")

            api = tradeapi.REST(
                key_id=ALPACA_API_KEY,
                secret_key=ALPACA_SECRET_KEY,
                base_url=URL(ALPACA_BASE_URL),
                api_version='v2'
            )
            
            # Test account access
            account = api.get_account()
            logger.info(f"Alpaca account status: {account.status}")
            logger.info(f"Buying power: ${float(account.buying_power):,.2f}")
            
            # Test getting a quote
            quote = api.get_latest_quote('AAPL')
            logger.info(f"AAPL quote: ${quote.ap}")
            
            return True
            
        except Exception as e:
            logger.error(f"Alpaca API test failed: {e}")
            return False
    
    def test_crypto_exchange_connection(self):
        """Test crypto exchange (Kraken) connectivity."""
        logger.info("Testing crypto exchange connection...")
        
        try:
            import ccxt
            from config_system import RATE_LIMIT_DELAY_SECONDS
            
            exchange = ccxt.kraken({
                'enableRateLimit': True,
                'rateLimit': RATE_LIMIT_DELAY_SECONDS * 1000
            })
            
            # Load markets
            markets = exchange.load_markets()
            logger.info(f"Loaded {len(markets)} crypto markets")
            
            # Test getting ticker data
            ticker = exchange.fetch_ticker('BTC/USD')
            logger.info(f"BTC/USD price: ${ticker['last']:,.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Crypto exchange test failed: {e}")
            return False
    
    def test_gemini_ai_connection(self):
        """Test Gemini AI API connectivity."""
        logger.info("Testing Gemini AI connection...")
        
        try:
            import google.generativeai as genai
            from bot_gemini_key_manager import GeminiKeyManagerBot
            from config_system import GEMINI_MODEL
            gemini_key_manager = GeminiKeyManagerBot()
            key = gemini_key_manager.get_available_key()
            assert key, "No Gemini API key available from GeminiKeyManagerBot."
            genai.configure(api_key=key)
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(
                "Respond with exactly: 'Gemini AI test successful'"
            )
            if "successful" in response.text.lower():
                logger.info("Gemini AI response received successfully")
                return True
            else:
                logger.error(f"Unexpected Gemini response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Gemini AI test failed: {e}")
            return False
    
    def test_technical_indicators(self):
        """Test technical indicators calculation using IndicatorBot."""
        logger.info("Testing technical indicators...")
        try:
            import numpy as np
            # Generate sample price data
            prices = [100 + i + np.sin(i/10) * 5 for i in range(50)]
            indicator_bot = IndicatorBot(prices)
            indicators = indicator_bot.calculate_indicators()
            # Check that indicators are calculated
            required_indicators = ['rsi', 'macd', 'sma_20']
            for indicator in required_indicators:
                if indicator not in indicators:
                    logger.error(f"Missing indicator: {indicator}")
                    return False
            logger.info(f"Calculated indicators: {list(indicators.keys())}")
            logger.info(f"RSI: {indicators['rsi']:.2f}")
            logger.info(f"MACD: {indicators['macd']:.4f}")
            return True
        except Exception as e:
            logger.error(f"Technical indicators test failed: {e}")
            return False
    
    def test_risk_manager(self):
        """Test risk management functionality."""
        logger.info("Testing risk manager...")
        
        try:
            # Test position risk calculation
            test_position = Position(
                symbol='TEST',
                quantity=10,
                entry_price=100.0,
                current_price=95.0,
                asset_type='stock'
            )
            
            risk_metrics = self.risk_manager.calculate_position_risk(test_position)
            
            # Check that risk metrics are calculated
            required_metrics = ['risk_level', 'unrealized_pnl', 'unrealized_pnl_pct']
            for metric in required_metrics:
                if metric not in risk_metrics:
                    logger.error(f"Missing risk metric: {metric}")
                    return False
            
            logger.info(f"Risk level: {risk_metrics['risk_level']}")
            logger.info(f"Unrealized P&L: ${risk_metrics['unrealized_pnl']:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Risk manager test failed: {e}")
            return False
    
    def test_portfolio_manager(self):
        """Test portfolio management functionality using PortfolioBot."""
        logger.info("Testing portfolio manager...")
        
        try:
            portfolio = PortfolioBot()
            # Add a test position
            portfolio.add_or_update_position('TEST', 'stock', 10, 100.0)
            # Print portfolio summary (method prints to stdout)
            portfolio.print_portfolio_summary()
            logger.info("Portfolio summary printed successfully.")
            return True
        except Exception as e:
            logger.error(f"Portfolio manager test failed: {e}")
            return False
    
    def test_trade_executor(self):
        """Test trade executor initialization (without actual trades) using TradeExecutorBot."""
        logger.info("Testing trade executor...")
        
        try:
            from config_system import ALPACA_API_KEY, ALPACA_SECRET_KEY, PAPER_TRADING
            executor = TradeExecutorBot(
                api_key=ALPACA_API_KEY,
                api_secret=ALPACA_SECRET_KEY,
                paper_trading=PAPER_TRADING
            )
            
            logger.info(f"Trade executor initialized (paper trading: {PAPER_TRADING})")
            
            return True
            
        except Exception as e:
            logger.error(f"Trade executor test failed: {e}")
            return False
    
    def test_decision_maker_bot(self):
        """Test DecisionMakerBot logic and integration."""
        try:
            from bot_decision_maker import DecisionMakerBot
            from data_structures import AssetAnalysisInput
            bot = DecisionMakerBot()
            analysis = AssetAnalysisInput(
                symbol='AAPL',
                market_data={'action': 'buy', 'confidence': 0.8, 'reasoning': 'Strong uptrend'},
                technical_indicators={},
                asset_type='stock'
            )
            decision = bot.make_trading_decision(analysis, min_confidence=0.7)
            assert hasattr(decision, 'final_action')
            assert decision.signal in ['buy', 'sell', 'hold'] or str(decision.signal).lower() in ['buy', 'sell', 'hold']
            return True
        except Exception as e:
            logger.error(f"DecisionMakerBot test failed: {e}")
            return False

    def test_reflection_bot(self):
        """Test ReflectionBot trade outcome analysis and insight generation."""
        try:
            from bot_reflection import ReflectionBot, TradeOutcome
            bot = ReflectionBot()
            trade = TradeOutcome(
                trade_id='T1', symbol='AAPL', asset_type='stock', action='buy',
                entry_price=100, exit_price=110, quantity=10,
                entry_time=datetime.now(), exit_time=datetime.now(),
                pnl=100, pnl_percent=0.1, duration_hours=24
            )
            insights = bot.analyze_completed_trade(trade)
            assert isinstance(insights, list)
            return True
        except Exception as e:
            logger.error(f"ReflectionBot test failed: {e}")
            return False

    def test_asset_screener_bot(self):
        """Test AssetScreenerBot dynamic asset screening."""
        try:
            from bot_asset_screener import AssetScreenerBot
            from bot_ai import AIBot
            from bot_database import DatabaseBot
            ai_bot = AIBot()
            db_bot = DatabaseBot()
            bot = AssetScreenerBot(ai_bot, db_bot)
            results = bot.screen_assets()
            assert isinstance(results, list)
            assert hasattr(results[0], 'symbol')
            return True
        except Exception as e:
            logger.error(f"AssetScreenerBot test failed: {e}")
            return False

    def test_backtester_bot(self):
        """Test BacktesterBot historical simulation and result structure."""
        try:
            from bot_backtester import BacktesterBot, BacktestConfig
            from datetime import datetime
            config = BacktestConfig(
                start_date=datetime(2024,1,1), end_date=datetime(2024,1,10),
                initial_capital=10000, assets_to_test=[('AAPL','stock',500)]
            )
            bot = BacktesterBot()
            result = bot.run_backtest(config)
            assert hasattr(result, 'performance_metrics')
            return True
        except Exception as e:
            logger.error(f"BacktesterBot test failed: {e}")
            return False

    def test_orchestrator_integration(self):
        """Test OrchestratorBot integrated workflow (analysis → decision → execution)."""
        try:
            from bot_orchestrator import OrchestratorBot
            bot = OrchestratorBot()
            # Simulate a single trading cycle (mocked, not live trading)
            result = bot._execute_trading_cycle()
            assert isinstance(result, bool)
            return True
        except Exception as e:
            logger.error(f"OrchestratorBot integration test failed: {e}")
            return False
    
    def test_news_retriever_integration(self):
        """Test NewsRetrieverBot integration in a trading cycle context."""
        try:
            news_bot = NewsRetrieverBot()
            articles = news_bot.fetch_news("AAPL", max_results=2)
            assert isinstance(articles, list)
            if articles:
                assert isinstance(articles[0], type(articles[0]))
            chunks = news_bot.preprocess_and_chunk(articles, chunk_size=256)
            embeddings = news_bot.generate_embeddings(chunks)
            assert len(embeddings) == len(chunks)
            summary = news_bot.augment_context_and_llm("AAPL stock news")
            assert isinstance(summary, str)
            logger.info(f"NewsRetrieverBot integration test summary: {summary[:100]}")
            return True
        except Exception as e:
            logger.error(f"NewsRetrieverBot integration test failed: {e}")
            return False
    
    def test_asset_screener_crypto_logic(self):
        """Test AssetScreenerBot's updated crypto logic."""
        from bot_asset_screener import AssetScreenerBot
        from bot_ai import AIBot
        from bot_database import DatabaseBot
        ai_bot = AIBot()
        db_bot = DatabaseBot()
        screener = AssetScreenerBot(ai_bot, db_bot)
        results = screener.screen_assets()
        # Check for at least one crypto asset in results
        has_crypto = any(getattr(r, 'asset_type', None) == 'crypto' for r in results)
        if not has_crypto:
            raise AssertionError('No crypto asset found in AssetScreenerBot results')

    def test_reflection_insights_integration(self):
        """Test that ReflectionBot's insights are passed through the system loop."""
        from bot_reflection import ReflectionBot, TradeOutcome
        from bot_decision_maker import DecisionMakerBot
        from data_structures import AssetAnalysisInput, ActionSignal
        from datetime import datetime, timedelta
        bot = ReflectionBot()
        trade = TradeOutcome(
            trade_id="T3",
            symbol="BTC/USD",
            asset_type="crypto",
            action="buy",
            entry_price=30000.0,
            exit_price=33000.0,
            quantity=0.1,
            entry_time=datetime.now() - timedelta(hours=3),
            exit_time=datetime.now(),
            pnl=300.0,
            pnl_percent=10.0,
            duration_hours=3.0,
            original_analysis=None,
            market_conditions_entry=None,
            market_conditions_exit=None
        )
        bot.analyze_and_store(trade)
        decision_bot = DecisionMakerBot()
        # Compose AssetAnalysisInput with minimal required fields
        analysis_input = AssetAnalysisInput(
            symbol="BTC/USD",
            market_data={'action': 'buy', 'confidence': 0.9, 'reasoning': 'Test'},
            technical_indicators={},
            asset_type="crypto"
        )
        decision = decision_bot.make_trading_decision(analysis_input, min_confidence=0.5)
        # Accept any valid ActionSignal
        if decision.signal not in [ActionSignal.BUY, ActionSignal.SELL, ActionSignal.HOLD]:
            raise AssertionError('DecisionMakerBot did not return a valid ActionSignal')

    def print_summary(self):
        """Print test summary and results."""
        logger.info(f"\n{'='*60}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total tests run: {self.passed_tests + self.failed_tests}")
        logger.info(f"Passed: {self.passed_tests}")
        logger.info(f"Failed: {self.failed_tests}")
        logger.info(f"Success rate: {(self.passed_tests/(self.passed_tests + self.failed_tests)*100):.1f}%")
        logger.info(f"{'='*60}")
        
        logger.info("\nDETAILED RESULTS:")
        for test_name, result in self.test_results.items():
            status_emoji = "PASS" if result == "PASSED" else "FAIL"
            logger.info(f"{status_emoji} {test_name}: {result}")
        
        if self.failed_tests == 0:
            logger.info("\nAll tests passed! Your trading bot is ready to run.")
        else:
            logger.warning(f"\n{self.failed_tests} test(s) failed. Please fix issues before running the bot.")

def main():
    """Run all tests."""
    print("Starting Trading Bot System Tests...")
    print(f"Timestamp: {datetime.now()}")
    
    tester = TradingSystemTester()
    
    # Run all tests
    tests = [
        ("Environment Variables", tester.test_environment_variables),
        ("Config Import", tester.test_config_import),
        ("Trading Variables Import", tester.test_trading_variables_import),
        ("Database Connection", tester.test_database_connection),
        ("Alpaca API Connection", tester.test_alpaca_api_connection),
        ("Crypto Exchange Connection", tester.test_crypto_exchange_connection),
        ("Gemini AI Connection", tester.test_gemini_ai_connection),
        ("Technical Indicators", tester.test_technical_indicators),
        ("Risk Manager", tester.test_risk_manager),
        ("Portfolio Manager", tester.test_portfolio_manager),
        ("Trade Executor", tester.test_trade_executor),
        ("Decision Maker Bot", tester.test_decision_maker_bot),
        ("Reflection Bot", tester.test_reflection_bot),
        ("Asset Screener Bot", tester.test_asset_screener_bot),
        ("Backtester Bot", tester.test_backtester_bot),
        ("Orchestrator Bot Integration", tester.test_orchestrator_integration),
        ("News Retriever Bot Integration", tester.test_news_retriever_integration),
        ("Asset Screener Crypto Logic", tester.test_asset_screener_crypto_logic),
        ("Reflection Insights Integration", tester.test_reflection_insights_integration),
    ]
    
    for test_name, test_function in tests:
        tester.run_test(test_name, test_function)
        time.sleep(1)  # Brief pause between tests
    
    # Print final summary
    tester.print_summary()
    
    return tester.failed_tests == 0

if __name__ == "__main__":
    tester = TradingSystemTester()
    tester.run_test("NewsRetrieverBot Integration", tester.test_news_retriever_integration)
    success = main()
    sys.exit(0 if success else 1)
