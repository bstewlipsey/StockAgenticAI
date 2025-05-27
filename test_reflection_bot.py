"""
Unit tests for ReflectionBot logic: post-mortem prompt generation, LLM insight storage, and RAG feedback.
"""
import unittest
from bot_reflection import ReflectionBot, TradeOutcome
from datetime import datetime, timedelta
import os

class TestReflectionBot(unittest.TestCase):
    def setUp(self):
        # Reset the database before each test for isolation
        db_path = os.path.join(os.path.dirname(__file__), 'trading_history.db')
        if os.path.exists(db_path):
            os.remove(db_path)
        self.bot = ReflectionBot()
        now = datetime.now()
        self.trade_outcome = TradeOutcome(
            trade_id="T1_unique",
            symbol="AAPL_TEST1",
            asset_type="stock",
            action="buy",
            entry_price=100.0,
            exit_price=115.0,
            quantity=10,
            entry_time=now - timedelta(hours=5),
            exit_time=now,
            pnl=150.0,
            pnl_percent=15.0,
            duration_hours=5.0,
            original_analysis=None,
            market_conditions_entry=None,
            market_conditions_exit=None
        )

    def test_analyze_and_store_persistence(self):
        """
        [TODO 3.2] Test that analyze_and_store persists insights and they are retrievable after 'restart'.
        """
        # Store insights
        persisted_insights = self.bot.analyze_and_store(self.trade_outcome)
        self.assertIsInstance(persisted_insights, list)
        self.assertGreaterEqual(len(persisted_insights), 1)

        # Simulate restart by re-instantiating ReflectionBot
        bot2 = ReflectionBot()
        retrieved = bot2.get_insights_for_symbol(self.trade_outcome.symbol, limit=10)
        self.assertIsInstance(retrieved, list)
        self.assertGreaterEqual(len(retrieved), 1)
        # Check that at least one insight matches the trade_id
        self.assertTrue(
            any(insight.trade_id == self.trade_outcome.trade_id for insight in retrieved),
            "No persisted insight matches the original trade_id"
        )

    def test_reflection_insights_persistence(self):
        """Test ReflectionBot's insights persist and are retrievable after restart."""
        bot = ReflectionBot()
        now = datetime.now()
        trade = TradeOutcome(
            trade_id="T2_unique",
            symbol="GOOG_TEST2",
            asset_type="stock",
            action="buy",
            entry_price=200.0,
            exit_price=222.0,
            quantity=5,
            entry_time=now - timedelta(hours=2),
            exit_time=now,
            pnl=110.0,
            pnl_percent=11.0,  # Above threshold
            duration_hours=2.0,
            original_analysis=None,
            market_conditions_entry=None,
            market_conditions_exit=None
        )
        print(f"[DEBUG] MIN_REFLECTION_PNL: 10.0, test pnl_percent: {trade.pnl_percent}")
        persisted = bot.analyze_and_store(trade)
        self.assertIsInstance(persisted, list)
        self.assertGreaterEqual(len(persisted), 1)
        bot2 = ReflectionBot()
        retrieved = bot2.get_insights_for_symbol(trade.symbol, limit=10)
        self.assertTrue(any(insight.trade_id == trade.trade_id for insight in retrieved))

if __name__ == "__main__":
    unittest.main()
