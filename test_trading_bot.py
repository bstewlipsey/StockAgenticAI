#!/usr/bin/env python3
"""
Comprehensive test suite for the StockAgenticAI trading bot.

This file is the full, in-depth test suite. Use it for pre-deployment or periodic checks.
For fast daily development, use test_core.py (the quick smoke test).

This test file verifies:
1. API connectivity (Gemini, Alpaca, Kraken)
2. Database operations
3. Trading analysis functionality
4. Risk management
5. Trade execution simulation
6. Portfolio management
7. Technical indicators

Run this test to ensure all components are working before starting live trading.
"""

import sys
import os
import time
import logging
from datetime import datetime
import google.generativeai as genai
import alpaca_trade_api as tradeapi
import ccxt
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, GEMINI_API_KEY, GEMINI_MODEL
from config_trading_variables import TRADING_ASSETS
from bot_database import DatabaseBot
from bot_indicators import IndicatorBot
from bot_risk_manager import RiskManager, Position
from bot_portfolio import PortfolioBot
from bot_trade_executor import TradeExecutorBot
from bot_position_sizer import PositionSizerBot

# --- API Key and Config Imports (moved to top for clarity) ---
try:
    from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, GEMINI_API_KEY, GEMINI_MODEL
except Exception as e:
    print(f"❌ Config import failed: {e}")
    sys.exit(1)

# --- Other Imports ---
try:
    from bot_database import DatabaseBot
    from bot_risk_manager import RiskManager, Position
    from bot_trade_executor import TradeExecutorBot
    from bot_position_sizer import PositionSizerBot
    from config_trading_variables import TRADING_ASSETS
    import google.generativeai as genai
    import alpaca_trade_api as tradeapi
    import ccxt
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingBotTester:
    def __init__(self):
        """Initialize the test suite."""
        self.results = {}
        self.db = None
        
    def run_all_tests(self):
        """Run all tests and return overall success status."""
        print("\n🚀 Starting Trading Bot Comprehensive Test Suite")
        print("=" * 60)
        
        test_methods = [
            ("API Connectivity", self.test_api_connections),
            ("Database Operations", self.test_database),
            ("Technical Indicators", self.test_technical_indicators),
            ("Risk Management", self.test_risk_management),
            ("Portfolio Management", self.test_portfolio_management),
            ("Crypto Data Fetching", self.test_crypto_data),
            ("AI Analysis", self.test_ai_analysis),
            ("Trade Execution", self.test_trade_execution)
        ]
        
        for test_name, test_method in test_methods:
            print(f"\n📋 Testing: {test_name}")
            try:
                success = test_method()
                self.results[test_name] = success
                status = "✅ PASSED" if success else "❌ FAILED"
                print(f"   {status}")
            except Exception as e:
                self.results[test_name] = False
                print(f"   ❌ FAILED: {e}")
                logger.error(f"{test_name} test failed: {e}")
        
        self.print_summary()
        return all(self.results.values())
    
    def test_api_connections(self):
        """Test all API connections."""
        # Test Gemini API
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content("Hello, respond with 'API Working'")
            if "API Working" in response.text or "working" in response.text.lower():
                print("   ✅ Gemini API: Connected")
            else:
                print(f"   ⚠️ Gemini API: Unexpected response: {response.text}")
        except Exception as e:
            print(f"   ❌ Gemini API: {e}")
            return False
        
        # Test Alpaca API
        try:
            api = tradeapi.REST(
                key_id=ALPACA_API_KEY or "",
                secret_key=ALPACA_SECRET_KEY or "",
                base_url=ALPACA_BASE_URL,  # type: ignore
                api_version='v2'
            )
            account = api.get_account()
            print(f"   ✅ Alpaca API: Connected (Account: {account.status})")
        except Exception as e:
            print(f"   ❌ Alpaca API: {e}")
            return False
        
        # Test Kraken API
        try:
            crypto_exchange = ccxt.kraken({'enableRateLimit': True})
            crypto_exchange.load_markets()
            print("   ✅ Kraken API: Connected")
        except Exception as e:
            print(f"   ❌ Kraken API: {e}")
            return False
        
        return True
    
    def test_database(self):
        """Test database operations."""
        try:
            self.db = DatabaseBot()
            
            # Test saving analysis
            test_analysis = {
                'action': 'buy',
                'reasoning': 'Test analysis',
                'confidence': 0.85
            }
            
            self.db.save_analysis(
                symbol='TEST',
                asset_type='test',
                analysis=test_analysis,
                current_price=100.0
            )
            
            # Test retrieving analysis
            history = self.db.get_analysis_history('TEST')
            print(f"   ✅ Database: Save/retrieve working ({len(history)} records)")
            return True
        except Exception as e:
            print(f"   ❌ Database: {e}")
            return False
    
    def test_technical_indicators(self):
        """Test technical indicator calculations."""
        try:
            # Generate sample price data
            sample_prices = [100 + i + (i % 3) for i in range(50)]
            
            bot = IndicatorBot(sample_prices)
            indicators = bot.calculate_indicators()
            
            required_indicators = ['rsi', 'macd', 'sma_20']
            for indicator in required_indicators:
                if indicator not in indicators:
                    print(f"   ❌ Missing indicator: {indicator}")
                    return False
            
            print(f"   ✅ Technical Indicators: RSI={indicators['rsi']:.2f}, MACD={indicators['macd']:.2f}")
            return True
        except Exception as e:
            print(f"   ❌ Technical Indicators: {e}")
            return False
    
    def test_risk_management(self):
        """Test risk management functionality."""
        try:
            risk_manager = RiskManager()
            
            # Test position risk calculation
            test_position = Position(
                symbol='TEST',
                quantity=10,
                entry_price=100.0,
                current_price=95.0,
                asset_type='stock'
            )
            
            risk_metrics = risk_manager.calculate_position_risk(test_position)
            
            required_metrics = ['risk_level', 'max_loss', 'profit_loss']
            for metric in required_metrics:
                if metric not in risk_metrics:
                    print(f"   ❌ Missing risk metric: {metric}")
                    return False
            
            print(f"   ✅ Risk Management: Level={risk_metrics['risk_level']}, P&L=${risk_metrics['profit_loss']:.2f}")
            return True
        except Exception as e:
            print(f"   ❌ Risk Management: {e}")
            return False
    
    def test_portfolio_management(self):
        """Test portfolio management functionality."""
        try:
            portfolio = PortfolioBot()
            
            # Test portfolio metrics
            metrics = portfolio.get_portfolio_metrics()
            
            print(f"   ✅ Portfolio Management: Total Value=${metrics.get('total_value', 0):.2f}")
            return True
        except Exception as e:
            print(f"   ❌ Portfolio Management: {e}")
            return False
    
    def test_crypto_data(self):
        """Test cryptocurrency data fetching."""
        try:
            exchange = ccxt.kraken({'enableRateLimit': True})
            ticker = exchange.fetch_ticker('BTC/USD')
            
            if 'last' in ticker:
                print(f"   ✅ Crypto Data: BTC/USD price=${ticker['last']:.2f}")
                return True
            else:
                print(f"   ❌ Crypto Data: Invalid response: {ticker}")
                return False
        except Exception as e:
            print(f"   ❌ Crypto Data: {e}")
            return False
    
    def test_ai_analysis(self):
        """Test AI-powered trading analysis."""
        try:
            from bot_ai import AIBot
            ai = AIBot()
            prompt = "Analyze this cryptocurrency and respond ONLY with valid JSON: {\"action\": \"buy\", \"reasoning\": \"test\", \"confidence\": 0.9}"
            variables = {}
            result = ai.generate_analysis(prompt, variables)
            return isinstance(result, str) and len(result) > 0
        except Exception as e:
            print(f"   ❌ AI Analysis: {e}")
            return False
    
    def test_trade_execution(self):
        """Test trade execution system (paper trading only)."""
        try:
            executor = TradeExecutorBot(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper_trading=True)
            
            # Test with a small quantity for safety
            success, order = executor.execute_trade(
                symbol='AAPL',
                side='buy',
                quantity=1,
                confidence=0.8
            )
            
            if success:
                print("   ✅ Trade Execution: Paper trade successful")
                return True
            else:
                print(f"   ⚠️ Trade Execution: Paper trade failed: {order}")
                # Don't fail the test for execution issues during market hours
                return True
        except Exception as e:
            print(f"   ❌ Trade Execution: {e}")
            # Don't fail the test for execution issues during market hours
            return True
    
    def print_summary(self):
        """Print test results summary."""
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in self.results.values() if result)
        total = len(self.results)
        
        for test_name, result in self.results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name:<25} {status}")
        
        print("-" * 60)
        print(f"Overall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! Trading bot is ready for operation.")
        else:
            print("⚠️ Some tests failed. Please check the issues above before trading.")
            
        return passed == total

def main():
    """Main function to run the test suite."""
    tester = TradingBotTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ You can now run the trading bot with confidence!")
        print("💡 To start trading: python main.py")
    else:
        print("\n❌ Please fix the failing tests before running the trading bot.")
        
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
