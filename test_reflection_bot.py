"""
Unit tests for ReflectionBot logic: post-mortem prompt generation, LLM insight storage, and RAG feedback.
"""
import unittest
from bot_reflection import ReflectionBot, TradeOutcome
from data_structures import TradingDecision, ActionSignal
from datetime import datetime

class TestReflectionBot(unittest.TestCase):
    def setUp(self):
        self.bot = ReflectionBot()
        self.trade_outcome = TradeOutcome(
            trade_id="T1",
            symbol="AAPL",
            asset_type="stock",
            action="buy",
            entry_price=100.0,
            exit_price=110.0,
            quantity=10,
            entry_time=datetime(2024, 1, 1, 10, 0, 0),
            exit_time=datetime(2024, 1, 1, 15, 0, 0),
            pnl=100.0,
            pnl_percent=0.1,
            duration_hours=5.0,
            original_analysis=None,
            market_conditions_entry=None,
            market_conditions_exit=None
        )

    def test_should_reflect_on_trade(self):
        self.assertIsInstance(self.bot._should_reflect_on_trade(self.trade_outcome), bool)

    def test_gather_reflection_context(self):
        ctx = self.bot._gather_reflection_context(self.trade_outcome)
        self.assertIsInstance(ctx, dict)

    def test_generate_ai_reflection(self):
        ctx = self.bot._gather_reflection_context(self.trade_outcome)
        result = self.bot._generate_ai_reflection(self.trade_outcome, ctx)
        self.assertIsInstance(result, list)

    def test_analyze_completed_trade(self):
        result = self.bot.analyze_completed_trade(self.trade_outcome)
        self.assertIsInstance(result, list)

if __name__ == "__main__":
    unittest.main()
