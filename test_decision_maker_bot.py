import pytest
from bot_decision_maker import DecisionMakerBot
from data_structures import AssetAnalysisInput, ActionSignal

def test_decision_makerbot_output_fields():
    bot = DecisionMakerBot()
    # Minimal valid input for a BUY
    analysis_input = AssetAnalysisInput(
        symbol="AAPL",
        market_data={"action": "buy", "confidence": 0.9},
        technical_indicators={},
        news_sentiment=None,
        reflection_insights=None,
        historical_ai_context=None
    )
    decision = bot.make_trading_decision(analysis_input, min_confidence=0.5)
    assert hasattr(decision, 'final_action'), "Decision must have final_action field"
    assert hasattr(decision, 'signal'), "Decision must have signal field"
    assert decision.final_action == decision.signal.name.lower()
    # Minimal valid input for a HOLD
    analysis_input = AssetAnalysisInput(
        symbol="AAPL",
        market_data={"action": "hold", "confidence": 0.1},
        technical_indicators={},
        news_sentiment=None,
        reflection_insights=None,
        historical_ai_context=None
    )
    decision = bot.make_trading_decision(analysis_input, min_confidence=0.5)
    assert hasattr(decision, 'final_action'), "Decision must have final_action field"
    assert hasattr(decision, 'signal'), "Decision must have signal field"
    assert decision.final_action == decision.signal.name.lower()
    # Minimal valid input for a SELL
    analysis_input = AssetAnalysisInput(
        symbol="AAPL",
        market_data={"action": "sell", "confidence": 0.9},
        technical_indicators={},
        news_sentiment=None,
        reflection_insights=None,
        historical_ai_context=None
    )
    decision = bot.make_trading_decision(analysis_input, min_confidence=0.5)
    assert hasattr(decision, 'final_action'), "Decision must have final_action field"
    assert hasattr(decision, 'signal'), "Decision must have signal field"
    assert decision.final_action == decision.signal.name.lower()
