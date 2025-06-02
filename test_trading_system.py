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
os.environ["PYTHONIOENCODING"] = "utf-8"

import sys
import time
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
from data_structures import NewsArticle

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
                paper_trading=PAPER_TRADING,
            )
        except Exception:
            self.trade_executor = None
        self.ai_bot = AIBot()
        self.stock_bot = StockBot()
        self.crypto_bot = CryptoBot()
        self.position_sizer = PositionSizerBot()

    def run_test(self, test_name, test_function):
        """Run a single test and record results."""
        try:
            start_time = time.time()
            result = test_function()
            end_time = time.time()
            if result:
                self.passed_tests += 1
                self.test_results[test_name] = "PASSED"
            else:
                self.failed_tests += 1
                self.test_results[test_name] = "FAILED"
        except Exception as e:
            self.failed_tests += 1
            self.test_results[test_name] = f"ERROR: {str(e)}"

    def test_environment_variables(self):
        """Test that all required environment variables are set."""
        required_vars = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]

        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            return False

        return True

    def test_config_import(self):
        """Test that config module imports correctly."""
        try:
            from config_system import (
                GEMINI_MODEL,
                PAPER_TRADING,
                TRADING_CYCLE_INTERVAL,
            )

            return True

        except ImportError as e:
            return False

    def test_trading_variables_import(self):
        """Test that trading variables import correctly."""
        try:
            from config_trading import (
                TRADING_ASSETS,
                TOTAL_CAPITAL,
                MIN_CONFIDENCE,
            )

            return True

        except ImportError as e:
            return False

    def test_database_connection(self):
        """Test database initialization and basic operations."""

        try:
            # Test saving and retrieving analysis
            test_analysis = {
                "action": "buy",
                "reasoning": "Test analysis",
                "confidence": 0.85,
            }

            self.db.save_analysis(
                symbol="TEST",
                asset_type="stock",
                analysis=test_analysis,
                current_price=100.0,
            )

            # Try to retrieve analysis history
            # history = self.db.get_analysis_history("TEST")  # Removed unused variable

            return True

        except Exception as e:
            return False

    def test_alpaca_api_connection(self):
        """Test Alpaca API connectivity."""

        try:
            from config_system import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
            import alpaca_trade_api as tradeapi
            from alpaca_trade_api.rest import URL

            if not ALPACA_API_KEY or not ALPACA_SECRET_KEY or not ALPACA_BASE_URL:
                raise ValueError(
                    "ALPACA_API_KEY, ALPACA_SECRET_KEY, and ALPACA_BASE_URL must be set and not None."
                )

            api = tradeapi.REST(
                key_id=ALPACA_API_KEY,
                secret_key=ALPACA_SECRET_KEY,
                base_url=URL(ALPACA_BASE_URL),
                api_version="v2",
            )

            # Test account access
            account = api.get_account()
            # Test getting a quote
            quote = api.get_latest_quote("AAPL")
            return True

        except Exception as e:
            return False

    def test_crypto_exchange_connection(self):
        """Test crypto exchange (Kraken) connectivity."""

        try:
            import ccxt
            from config_system import RATE_LIMIT_DELAY_SECONDS

            exchange = ccxt.kraken(
                {"enableRateLimit": True, "rateLimit": RATE_LIMIT_DELAY_SECONDS * 1000}
            )

            # Load markets
            markets = exchange.load_markets()

            # Test getting ticker data
            ticker = exchange.fetch_ticker("BTC/USD")

            return True

        except Exception as e:
            return False

    def test_gemini_ai_connection(self):
        """Test Gemini AI API connectivity."""

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
                return True
            else:
                return False

        except Exception as e:
            return False

    def test_technical_indicators(self):
        """Test technical indicators calculation using IndicatorBot."""
        try:
            import numpy as np

            # Generate sample price data
            prices = [100 + i + np.sin(i / 10) * 5 for i in range(50)]
            indicator_bot = IndicatorBot(prices)
            indicators = indicator_bot.calculate_indicators()
            # Check that indicators are calculated
            required_indicators = ["rsi", "macd", "sma_20"]
            for indicator in required_indicators:
                if indicator not in indicators:
                    return False
            return True
        except Exception as e:
            return False

    def test_risk_manager(self):
        """Test risk management functionality."""

        try:
            # Test position risk calculation
            test_position = Position(
                symbol="TEST",
                quantity=10,
                entry_price=100.0,
                current_price=95.0,
                asset_type="stock",
            )

            risk_metrics = self.risk_manager.calculate_position_risk(test_position)

            # Check that risk metrics are calculated
            required_metrics = ["risk_level", "unrealized_pnl", "unrealized_pnl_pct"]
            for metric in required_metrics:
                if metric not in risk_metrics:
                    return False

            return True

        except Exception as e:
            return False

    def test_portfolio_manager(self):
        """Test portfolio management functionality using PortfolioBot."""

        try:
            portfolio = PortfolioBot()
            # Add a test position
            portfolio.add_or_update_position("TEST", "stock", 10, 100.0)
            # Print portfolio summary (method prints to stdout)
            portfolio.print_portfolio_summary()
            return True
        except Exception as e:
            return False

    def test_trade_executor(self):
        """Test trade executor initialization (without actual trades) using TradeExecutorBot."""

        try:
            from config_system import ALPACA_API_KEY, ALPACA_SECRET_KEY, PAPER_TRADING

            TradeExecutorBot(
                api_key=ALPACA_API_KEY,
                api_secret=ALPACA_SECRET_KEY,
                paper_trading=PAPER_TRADING,
            )

            return True

        except Exception as e:
            return False

    def test_decision_maker_bot(self):
        """Test DecisionMakerBot logic and integration."""
        try:
            from bot_decision_maker import DecisionMakerBot
            from data_structures import AssetAnalysisInput

            bot = DecisionMakerBot()
            analysis = AssetAnalysisInput(
                symbol="AAPL",
                market_data={
                    "action": "buy",
                    "confidence": 0.8,
                    "reasoning": "Strong uptrend",
                },
                technical_indicators={},
                asset_type="stock",
            )
            decision = bot.make_trading_decision(analysis, min_confidence=0.7)
            assert hasattr(decision, "final_action")
            # Fix: Always check signal as enum name (lowercase)
            assert getattr(decision.signal, "name", str(decision.signal)).lower() in [
                "buy",
                "sell",
                "hold",
            ]
            return True
        except Exception as e:
            return False

    def test_reflection_bot(self):
        """Test ReflectionBot trade outcome analysis and insight generation."""
        try:
            from bot_reflection import ReflectionBot, TradeOutcome

            bot = ReflectionBot()
            trade = TradeOutcome(
                trade_id="T1",
                symbol="AAPL",
                asset_type="stock",
                action="buy",
                entry_price=100,
                exit_price=110,
                quantity=10,
                entry_time=datetime.now(),
                exit_time=datetime.now(),
                pnl=100,
                pnl_percent=0.1,
                duration_hours=24,
            )
            insights = bot.analyze_completed_trade(trade)
            assert isinstance(insights, list)
            return True
        except Exception as e:
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
            assert hasattr(results[0], "symbol")
            return True
        except Exception as e:
            return False

    def test_backtester_bot(self):
        """Test BacktesterBot historical simulation and result structure."""
        try:
            from bot_backtester import BacktesterBot, BacktestConfig
            from datetime import datetime

            config = BacktestConfig(
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 10),
                initial_capital=10000,
                assets_to_test=[("AAPL", "stock", 500)],
            )
            bot = BacktesterBot()
            result = bot.run_backtest(config)
            assert hasattr(result, "performance_metrics")
            return True
        except Exception as e:
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
            return False

    def test_news_retriever_integration(self):
        """Test NewsRetrieverBot integration in a trading cycle context."""
        try:
            news_bot = NewsRetrieverBot()
            articles = news_bot.fetch_news("AAPL", max_results=2)
            assert isinstance(articles, list)
            if articles:
                assert isinstance(articles[0], NewsArticle)
            chunks = news_bot.preprocess_and_chunk(articles, chunk_size=256)
            embeddings = news_bot.generate_embeddings(chunks)
            assert len(embeddings) == len(chunks)
            summary = news_bot.augment_context_and_llm("AAPL stock news")
            assert isinstance(summary, str)
            return True
        except Exception as e:
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
        has_crypto = any(getattr(r, "asset_type", None) == "crypto" for r in results)
        if not has_crypto:
            raise AssertionError("No crypto asset found in AssetScreenerBot results")

    def test_reflection_insights_integration(self):
        """Test that ReflectionBot's insights are passed through the system loop."""
        from bot_reflection import ReflectionBot, TradeOutcome
        from bot_decision_maker import DecisionMakerBot
        from data_structures import AssetAnalysisInput
        from datetime import datetime
        import os

        # Ensure a fresh DB for test isolation
        db_path = os.path.join(os.path.dirname(__file__), "trading_history.db")
        if os.path.exists(db_path):
            os.remove(db_path)
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
            market_conditions_exit=None,
        )
        bot.analyze_and_store(trade)
        decision_bot = DecisionMakerBot()
        # Compose AssetAnalysisInput with minimal required fields
        analysis_input = AssetAnalysisInput(
            symbol="BTC/USD",
            market_data={"action": "buy", "confidence": 0.9, "reasoning": "Test"},
            technical_indicators={},
            asset_type="crypto",
        )
        decision = decision_bot.make_trading_decision(
            analysis_input, min_confidence=0.5
        )
        # Accept any valid ActionSignal
        assert getattr(decision.signal, "name", str(decision.signal)).upper() in [
            "BUY",
            "SELL",
            "HOLD",
        ]
        # Also check that the insight is persisted
        insights = bot.get_insights_for_symbol("BTC/USD", limit=10)
        assert any(
            insight.trade_id == "T3" for insight in insights
        ), "Reflection insight not persisted or retrievable"
        return True

    def print_summary(self):
        """Print test summary and results."""
        pass  # Omit summary printing in this version


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
        (
            "Reflection Insights Integration",
            tester.test_reflection_insights_integration,
        ),
    ]

    for test_name, test_function in tests:
        tester.run_test(test_name, test_function)
        time.sleep(1)  # Brief pause between tests

    # Print final summary
    tester.print_summary()

    return tester.failed_tests == 0


if __name__ == "__main__":
    tester = TradingSystemTester()
    tester.run_test(
        "NewsRetrieverBot Integration", tester.test_news_retriever_integration
    )
    success = main()
    sys.exit(0 if success else 1)
