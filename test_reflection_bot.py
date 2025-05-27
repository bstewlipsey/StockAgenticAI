"""
Unit tests for ReflectionBot logic: post-mortem prompt generation, LLM insight storage, and RAG feedback.
"""
import unittest
from bot_reflection import ReflectionBot, TradeOutcome
from datetime import datetime, timedelta

class TestReflectionBot(unittest.TestCase):
    def setUp(self):
        self.bot = ReflectionBot()
        now = datetime.now()
        self.trade_outcome = TradeOutcome(
            trade_id="T1",
            symbol="AAPL",
            asset_type="stock",
            action="buy",
            entry_price=100.0,
            exit_price=110.0,
            quantity=10,
            entry_time=now - timedelta(hours=5),
            exit_time=now,
            pnl=100.0,
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

if __name__ == "__main__":
    unittest.main()
