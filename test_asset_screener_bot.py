from bot_asset_screener import AssetScreenerBot
from bot_ai import AIBot
from bot_database import DatabaseBot
from data_structures import AssetScreeningResult


class DummyAIBot(AIBot):
    def __init__(self):
        pass


class DummyDatabaseBot(DatabaseBot):
    def __init__(self):
        self.db_file = ":memory:"  # Patch to avoid attribute errors
        super().__init__(db_file=":memory:")


def test_asset_screener_fallback_always_returns_crypto():
    screener = AssetScreenerBot(DummyAIBot(), DummyDatabaseBot())
    # Simulate error in screening to trigger fallback
    results = screener._get_fallback_assets()
    crypto_assets = [r for r in results if getattr(r, "asset_type", None) == "crypto"]
    assert (
        len(crypto_assets) >= 1
    ), "Fallback must always include at least one crypto asset"
    assert all(isinstance(r, AssetScreeningResult) for r in results)


def test_asset_screener_screen_assets_always_returns_crypto(monkeypatch):
    screener = AssetScreenerBot(DummyAIBot(), DummyDatabaseBot())
    # Monkeypatch _screen_stocks and _screen_crypto to return empty
    monkeypatch.setattr(screener, "_screen_stocks", lambda x: [])
    monkeypatch.setattr(screener, "_screen_crypto", lambda x: [])
    monkeypatch.setattr(
        screener,
        "_analyze_market_overview",
        lambda: type(
            "Dummy",
            (),
            {
                "market_sentiment": "neutral",
                "risk_environment": "medium",
                "top_sectors": [],
                "market_volatility": 20.0,
                "ai_insights": "",
            },
        )(),
    )
    results = screener.screen_assets()
    crypto_assets = [r for r in results if getattr(r, "asset_type", None) == "crypto"]
    assert (
        len(crypto_assets) >= 1
    ), "screen_assets must always return at least one crypto asset"
    assert all(isinstance(r, AssetScreeningResult) for r in results)
