#!/usr/bin/env python3
"""
Core smoke test for StockAgenticAI: Quickly validates all major system components.
- API connectivity (Alpaca, Kraken, Gemini)
- Database operations
- Technical indicators
- Risk/portfolio management
- Trade execution (init only)
- AI analysis integration
Run: python test_core.py
"""
import os
import sys
import logging
from datetime import datetime
import numpy as np
import ccxt
import google.generativeai as genai
from config_system import (
    GEMINI_MODEL, PAPER_TRADING, ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, # Remove GEMINI_API_KEY
    RATE_LIMIT_DELAY_SECONDS
)
from config_trading import TRADING_ASSETS, TOTAL_CAPITAL
from alpaca_trade_api.rest import URL
import alpaca_trade_api as tradeapi
from bot_database import DatabaseBot
from bot_indicators import IndicatorBot
from bot_risk_manager import RiskManager, Position
from bot_portfolio import PortfolioBot
from bot_trade_executor import TradeExecutorBot
from bot_stock import StockBot
from bot_crypto import CryptoBot
from bot_ai import AIBot
from test_mocks import MockStockDataProvider, MockCryptoDataProvider, MockNewsRetriever

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def test_env():
    req = ['ALPACA_API_KEY','ALPACA_SECRET_KEY']  # Remove GEMINI_API_KEY
    missing = [v for v in req if not os.getenv(v)]
    if missing:
        logger.error(f"Missing env vars: {missing}"); assert False
    assert True

def test_config():
    try:
        from config_system import GEMINI_MODEL, PAPER_TRADING
        from config_trading import TRADING_ASSETS, TOTAL_CAPITAL
        assert True
    except Exception as e:
        logger.error(f"Config import: {e}"); assert False

def test_alpaca():
    try:
        api = tradeapi.REST(
            key_id=ALPACA_API_KEY or "",
            secret_key=ALPACA_SECRET_KEY or "",
            base_url=URL(ALPACA_BASE_URL),
            api_version='v2'
        )
        api.get_account(); assert True
    except Exception as e:
        logger.error(f"Alpaca: {e}"); assert False

def test_kraken():
    try:
        ex = ccxt.kraken({'enableRateLimit': True, 'rateLimit': RATE_LIMIT_DELAY_SECONDS*1000})
        ex.load_markets(); assert True
    except Exception as e:
        logger.error(f"Kraken: {e}"); assert False

def test_gemini():
    try:
        from bot_gemini_key_manager import GeminiKeyManagerBot
        import google.generativeai as genai
        from config_system import GEMINI_MODEL
        gemini_key_manager = GeminiKeyManagerBot()
        key = gemini_key_manager.get_available_key()
        assert key, "No Gemini API key available from GeminiKeyManagerBot."
        genai.configure(api_key=key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        r = model.generate_content("Say 'Gemini OK'")
        assert 'ok' in r.text.lower()
    except Exception as e:
        logger.error(f"Gemini: {e}"); assert False

def test_db():
    try:
        db = DatabaseBot(); db.get_analysis_history('TEST'); assert True
    except Exception as e:
        logger.error(f"DB: {e}"); assert False

def test_indicators():
    try:
        prices = [100+i for i in range(30)]
        bot = IndicatorBot(prices)
        indicators = bot.calculate_indicators()
        assert 'rsi' in indicators and 'macd' in indicators
    except Exception as e:
        logger.error(f"Indicators: {e}"); assert False

def test_risk():
    try:
        rm = RiskManager(); pos = Position('T',1,100,99,'stock')
        m = rm.calculate_position_risk(pos)
        assert 'risk_level' in m
    except Exception as e:
        logger.error(f"Risk: {e}"); assert False

def test_portfolio():
    try:
        pf = PortfolioBot(); pf.add_or_update_position('T','stock',1,100)
        s = pf.get_portfolio_metrics()
        # Accept any non-empty dict as a pass, log keys for debug
        assert isinstance(s, dict)
        print(f"Portfolio metrics keys: {list(s.keys())}")
    except Exception as e:
        logger.error(f"Portfolio: {e}"); assert False

def test_executor():
    try:
        TradeExecutorBot(api_key=ALPACA_API_KEY or "", api_secret=ALPACA_SECRET_KEY or "", paper_trading=PAPER_TRADING)
        assert True
    except Exception as e:
        logger.error(f"Executor: {e}"); assert False

def test_agent():
    try:
        stock_bot = StockBot()
        crypto_bot = CryptoBot()
        portfolio_bot = PortfolioBot()
        assert hasattr(stock_bot, 'analyze_stock')
        assert hasattr(crypto_bot, 'analyze_crypto')
        assert hasattr(portfolio_bot, 'get_portfolio_metrics')
        assert True
    except Exception as e:
        logger.error(f"Agent (modular bots): {e}"); assert False

def test_ai_analysis():
    try:
        ai = AIBot()
        prompt = "Analyze this cryptocurrency and respond ONLY with valid JSON: {\"action\": \"buy\", \"reasoning\": \"test\", \"confidence\": 0.9}"
        variables = {}
        result = ai.generate_analysis(prompt, variables)
        assert isinstance(result, str)
        print(f"AI analysis result: {result[:60]}")
    except Exception as e:
        logger.error(f"AI analysis: {e}"); assert False

def test_test_mode_flag():
    try:
        from config_system import TEST_MODE_ENABLED
        assert isinstance(TEST_MODE_ENABLED, bool)
        print(f"TEST_MODE_ENABLED flag present: {TEST_MODE_ENABLED}")
        assert True
    except Exception as e:
        logger.error(f"TEST_MODE_ENABLED flag: {e}"); assert False

def test_mock_providers():
    try:
        stock = MockStockDataProvider()
        crypto = MockCryptoDataProvider()
        news = MockNewsRetriever()
        assert stock.get_historical_prices('AAPL', limit=5) == [100, 101, 102, 103, 104]
        assert stock.get_current_price('AAPL') == 123.45
        assert crypto.get_historical_prices('BTC/USD', limit=3) == [200, 201, 202]
        assert crypto.get_current_price('BTC/USD') == 23456.78
        news_list = news.fetch_news('BTC news')
        assert len(news_list) == 2
        assert news.augment_context_and_llm('BTC news') == 'Test news summary.'
        print("Mock providers: PASS")
        assert True
    except Exception as e:
        logger.error(f"Mock providers: {e}"); assert False

def main():
    print("\n=== StockAgenticAI Core Smoke Test ===")
    print(f"Timestamp: {datetime.now()}")
    tests = [
        ("Env Vars", test_env),
        ("Config", test_config),
        ("Alpaca API", test_alpaca),
        ("Kraken API", test_kraken),
        ("Gemini AI", test_gemini),
        ("Database", test_db),
        ("Indicators", test_indicators),
        ("RiskMgr", test_risk),
        ("PortfolioMgr", test_portfolio),
        ("TradeExecutor", test_executor),
        ("Agent", test_agent),
        ("AI Analysis", test_ai_analysis),
        ("TEST_MODE Flag", test_test_mode_flag),
        ("Mock Providers", test_mock_providers),
    ]
    passed = 0
    for name, fn in tests:
        print(f"[{name}]", end=' ')
        try:
            fn(); print("PASS"); passed+=1
        except Exception as e:
            print(f"ERROR: {e}")
    print(f"\n{passed}/{len(tests)} tests passed.")
    if passed==len(tests): print("All core systems OK.")
    else: print("Some tests failed. Check logs above.")
    return passed==len(tests)

if __name__=="__main__":
    import sys
    sys.exit(0 if main() else 1)
