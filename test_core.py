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
from config import (
    GEMINI_MODEL, PAPER_TRADING, ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, GEMINI_API_KEY, RATE_LIMIT_DELAY_SECONDS
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
    req = ['ALPACA_API_KEY','ALPACA_SECRET_KEY','GEMINI_API_KEY']
    missing = [v for v in req if not os.getenv(v)]
    if missing:
        logger.error(f"Missing env vars: {missing}"); return False
    return True

def test_config():
    try:
        from config import GEMINI_MODEL, PAPER_TRADING
        from config_trading import TRADING_ASSETS, TOTAL_CAPITAL
        return True
    except Exception as e:
        logger.error(f"Config import: {e}"); return False

def test_alpaca():
    try:
        api = tradeapi.REST(
            key_id=ALPACA_API_KEY or "",
            secret_key=ALPACA_SECRET_KEY or "",
            base_url=URL(ALPACA_BASE_URL),
            api_version='v2'
        )
        api.get_account(); return True
    except Exception as e:
        logger.error(f"Alpaca: {e}"); return False

def test_kraken():
    try:
        ex = ccxt.kraken({'enableRateLimit': True, 'rateLimit': RATE_LIMIT_DELAY_SECONDS*1000})
        ex.load_markets(); return True
    except Exception as e:
        logger.error(f"Kraken: {e}"); return False

def test_gemini():
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        r = model.generate_content("Say 'Gemini OK'")
        return 'ok' in r.text.lower()
    except Exception as e:
        logger.error(f"Gemini: {e}"); return False

def test_db():
    try:
        db = DatabaseBot(); db.get_analysis_history('TEST'); return True
    except Exception as e:
        logger.error(f"DB: {e}"); return False

def test_indicators():
    try:
        prices = [100+i for i in range(30)]
        bot = IndicatorBot(prices)
        indicators = bot.calculate_indicators()
        return 'rsi' in indicators and 'macd' in indicators
    except Exception as e:
        logger.error(f"Indicators: {e}"); return False

def test_risk():
    try:
        rm = RiskManager(); pos = Position('T',1,100,99,'stock')
        m = rm.calculate_position_risk(pos)
        return 'risk_level' in m
    except Exception as e:
        logger.error(f"Risk: {e}"); return False

def test_portfolio():
    try:
        pf = PortfolioBot(); pf.add_or_update_position('T','stock',1,100)
        s = pf.get_portfolio_metrics(); return 'total_value' in s or 'current_value' in s
    except Exception as e:
        logger.error(f"Portfolio: {e}"); return False

def test_executor():
    try:
        TradeExecutorBot(api_key=ALPACA_API_KEY or "", api_secret=ALPACA_SECRET_KEY or "", paper_trading=PAPER_TRADING)
        return True
    except Exception as e:
        logger.error(f"Executor: {e}"); return False

def test_agent():
    try:
        stock_bot = StockBot()
        crypto_bot = CryptoBot()
        portfolio_bot = PortfolioBot()
        assert hasattr(stock_bot, 'analyze_stock')
        assert hasattr(crypto_bot, 'analyze_crypto')
        assert hasattr(portfolio_bot, 'get_portfolio_metrics')
        return True
    except Exception as e:
        logger.error(f"Agent (modular bots): {e}"); return False

def test_ai_analysis():
    try:
        ai = AIBot()
        prompt = "Analyze this cryptocurrency and respond ONLY with valid JSON: {\"action\": \"buy\", \"reasoning\": \"test\", \"confidence\": 0.9}"
        variables = {}
        result = ai.generate_analysis(prompt, variables)
        return isinstance(result, str) and len(result) > 0
    except Exception as e:
        logger.error(f"AI analysis: {e}"); return False

def test_mock_stock_provider():
    mock = MockStockDataProvider()
    prices = mock.get_historical_prices('AAPL', limit=5)
    assert prices == [100, 101, 102, 103, 104], f"Unexpected prices: {prices}"
    assert mock.get_current_price('AAPL') == 123.45
    print("MockStockDataProvider passed.")

def test_mock_crypto_provider():
    mock = MockCryptoDataProvider()
    prices = mock.get_historical_prices('BTC/USD', limit=3)
    assert prices == [200, 201, 202], f"Unexpected prices: {prices}"
    assert mock.get_current_price('BTC/USD') == 23456.78
    print("MockCryptoDataProvider passed.")

def test_mock_news_retriever():
    mock = MockNewsRetriever()
    news = mock.fetch_news('BTC news')
    assert len(news) == 2
    summary = mock.augment_context_and_llm('BTC news')
    assert summary == 'Test news summary.'
    print("MockNewsRetriever passed.")

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
    ]
    passed = 0
    for name, fn in tests:
        print(f"[{name}]", end=' ')
        try:
            if fn(): print("PASS"); passed+=1
            else: print("FAIL")
        except Exception as e:
            print(f"ERROR: {e}")
    print(f"\n{passed}/{len(tests)} tests passed.")
    if passed==len(tests): print("All core systems OK.")
    else: print("Some tests failed. Check logs above.")
    return passed==len(tests)

if __name__=="__main__":
    import sys
    sys.exit(0 if main() else 1)
