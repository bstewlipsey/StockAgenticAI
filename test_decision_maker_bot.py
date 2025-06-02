from bot_decision_maker import DecisionMakerBot
from data_structures import AssetAnalysisInput


def test_decision_makerbot_output_fields():
    bot = DecisionMakerBot()
    # Minimal valid input for a BUY
    analysis_input = AssetAnalysisInput(
        symbol="AAPL",
        market_data={"action": "buy", "confidence": 0.9},
        technical_indicators={},
        news_sentiment=None,
        reflection_insights=None,
        historical_ai_context=None,
    )
    decision = bot.make_trading_decision(analysis_input, min_confidence=0.5)
    assert hasattr(decision, "final_action"), "Decision must have final_action field"
    assert hasattr(decision, "signal"), "Decision must have signal field"
    assert decision.final_action == decision.signal.name.lower()
    # Minimal valid input for a HOLD
    analysis_input = AssetAnalysisInput(
        symbol="AAPL",
        market_data={"action": "hold", "confidence": 0.1},
        technical_indicators={},
        news_sentiment=None,
        reflection_insights=None,
        historical_ai_context=None,
    )
    decision = bot.make_trading_decision(analysis_input, min_confidence=0.5)
    assert hasattr(decision, "final_action"), "Decision must have final_action field"
    assert hasattr(decision, "signal"), "Decision must have signal field"
    assert decision.final_action == decision.signal.name.lower()
    # Minimal valid input for a SELL
    analysis_input = AssetAnalysisInput(
        symbol="AAPL",
        market_data={"action": "sell", "confidence": 0.9},
        technical_indicators={},
        news_sentiment=None,
        reflection_insights=None,
        historical_ai_context=None,
    )
    decision = bot.make_trading_decision(analysis_input, min_confidence=0.5)
    assert hasattr(decision, "final_action"), "Decision must have final_action field"
    assert hasattr(decision, "signal"), "Decision must have signal field"
    assert decision.final_action == decision.signal.name.lower()


def test_hold_override_logic():
    """Test that unjustified 'hold' is overridden to buy/sell when all indicators are present and not contradictory."""
    from data_structures import AssetAnalysisInput

    bot = DecisionMakerBot()
    # All indicators present, not contradictory, low confidence, should override to BUY
    analysis_input = AssetAnalysisInput(
        symbol="AAPL",
        market_data={
            "action": "hold",
            "confidence": 0.2,
            "reasoning": "No strong signal.",
        },
        technical_indicators={
            "rsi": 45,
            "macd": 0.1,
            "sma_20": 150,
            "price_vs_sma20": 0.5,
        },
        news_sentiment=None,
        reflection_insights=None,
        historical_ai_context=None,
    )
    decision = bot.make_trading_decision(analysis_input, min_confidence=0.5)
    assert decision.final_action in (
        "buy",
        "sell",
    ), f"Override failed, got: {decision.final_action}"
    assert decision.metadata.get("hold_override"), "Override flag not set in metadata"
    print("    -> HOLD override to actionable trade (BUY/SELL) logic passed.")

    # Contradictory indicators, should NOT override
    analysis_input = AssetAnalysisInput(
        symbol="TSLA",
        market_data={
            "action": "hold",
            "confidence": 0.2,
            "reasoning": "Mixed signals.",
        },
        technical_indicators={
            "rsi": 80,
            "macd": -1.0,
            "sma_20": 900,
            "price_vs_sma20": 2.0,
        },
        news_sentiment=None,
        reflection_insights=None,
        historical_ai_context=None,
    )
    decision = bot.make_trading_decision(analysis_input, min_confidence=0.5)
    assert (
        decision.final_action == "hold"
    ), f"Should not override, got: {decision.final_action}"
    assert not decision.metadata.get("hold_override"), "Override flag incorrectly set"
    print("    -> Contradictory indicators, no override logic passed.")

    # Missing indicators, should NOT override
    analysis_input = AssetAnalysisInput(
        symbol="GOOGL",
        market_data={
            "action": "hold",
            "confidence": 0.2,
            "reasoning": "Insufficient data.",
        },
        technical_indicators={"rsi": 50, "macd": 0.2},  # missing sma_20, price_vs_sma20
        news_sentiment=None,
        reflection_insights=None,
        historical_ai_context=None,
    )
    decision = bot.make_trading_decision(analysis_input, min_confidence=0.5)
    assert (
        decision.final_action == "hold"
    ), f"Should not override, got: {decision.final_action}"
    assert not decision.metadata.get("hold_override"), "Override flag incorrectly set"
    print("    -> Missing indicators, no override logic passed.")
