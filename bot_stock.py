# bot_stock.py
"""
StockBot: Stock-specific analysis and trading logic for agentic trading systems.
- Fetches and analyzes stock market data using Alpaca
- Generates AI-driven trading signals and risk metrics
- Designed for modular integration with other trading bots
"""

# === STOCK BOT LOGIC ===
# Extracted from agent.py

import logging
import time
import re
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, PAPER_TRADING, DEFAULT_STOCK_TIMEFRAME
from alpaca_trade_api import REST
from alpaca_trade_api.rest import URL, TimeFrame, TimeFrameUnit
from bot_indicators import IndicatorBot
from bot_trade_executor import TradeExecutorBot
from bot_risk_manager import RiskManager, Position
from config_trading import TRADING_ASSETS, RSI_OVERSOLD, RSI_OVERBOUGHT, ANALYSIS_SCHEMA, STOCK_ANALYSIS_TEMPLATE
from bot_ai import generate_ai_analysis
from bot_database import DatabaseBot

            

logger = logging.getLogger(__name__)

class StockBot:
    def __init__(self):
        self.api = REST(
            key_id=ALPACA_API_KEY or "",
            secret_key=ALPACA_SECRET_KEY or "",
            base_url=URL(ALPACA_BASE_URL),
            api_version='v2'
        )
        self.executor = TradeExecutorBot(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper_trading=PAPER_TRADING)
        self.risk_manager = RiskManager()
        self.asset_type_map = {symbol: asset_type for symbol, asset_type, _ in TRADING_ASSETS}
        self.tf_value, self.tf_unit = DEFAULT_STOCK_TIMEFRAME  
        self.database_bot = DatabaseBot()
        
    def get_current_price(self, symbol):
        for _ in range(3):
            try:
                quote = self.api.get_latest_quote(symbol)
                if quote and hasattr(quote, 'ap'):
                    return float(quote.ap)
            except Exception as e:
                logger.error(f"Error getting price for {symbol}: {e}")
                time.sleep(2)
        return None

    def analyze_stock(self, symbol, prompt_note=None):
        """
        Analyze a stock using market data, technical analysis, and AI predictions.
        Optionally include a prompt_note for adaptive AI learning.
        """
        try:
            tf_value, tf_unit = self.tf_value, self.tf_unit
            timeframe = TimeFrame(tf_value, TimeFrameUnit[tf_unit])
            quote = self.api.get_latest_quote(symbol)
            bars = self.api.get_bars(symbol, timeframe, limit=50)
            if not bars:
                logger.warning(f"No bars returned for {symbol} with timeframe {DEFAULT_STOCK_TIMEFRAME}. Skipping analysis.")
                return {"error": f"No market data available for {symbol} ({DEFAULT_STOCK_TIMEFRAME})"}
            prices = [bar.c for bar in bars]  # Alpaca stock: bar.c = close
            volumes = [bar.v for bar in bars]  # Alpaca stock: bar.v = volume
            timestamps = [getattr(bar, 't', None) for bar in bars]  # Alpaca stock: bar.t = timestamp
            tech_analysis = IndicatorBot(prices)
            signals, indicators = tech_analysis.get_signals()
            signal_summary = "\n".join([f"- {signal}: {reason} (Confidence: {confidence*100:.0f}%)" for signal, reason, confidence in signals])
            # Market context (S&P 500)
            try:
                spy_timeframe = TimeFrame(tf_value, TimeFrameUnit[tf_unit])
                spy_bars = self.api.get_bars('SPY', spy_timeframe, limit=1)
                if not spy_bars:
                    logger.warning(f"No bars returned for SPY with timeframe {spy_timeframe}. Skipping market context.")
                    market_change = 0
                    market_context = "Neutral"
                else:
                    market_change = ((spy_bars[-1].c - spy_bars[-1].o) / spy_bars[-1].o) * 100
                    market_context = "Bullish" if market_change > 0 else "Bearish"
            except Exception as e:
                logger.warning(f"Could not fetch market context: {e}")
                market_change = 0
                market_context = "Neutral"
            # Position and risk
            position_qty = "0"
            avg_entry_price = "0.00"
            try:
                position = self.api.get_position(symbol)
                position_qty = str(position.qty)
                avg_entry_price = str(position.avg_entry_price)
            except Exception as e:
                pass
            current_price = float(quote.ap)
            position_obj = Position(
                symbol=symbol,
                quantity=float(position_qty),
                entry_price=float(avg_entry_price),
                current_price=current_price,
                asset_type='stock'
            )
            risk_metrics = self.risk_manager.calculate_position_risk(position_obj)
            price_change = ((current_price - prices[0]) / prices[0]) * 100
            avg_volume = sum(volumes) / len(volumes)
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            # Prepare template variables
            template_vars = {
                'schema': ANALYSIS_SCHEMA,
                'symbol': symbol,
                'current_price': quote.ap,
                'position_qty': position_qty,
                'entry_price': avg_entry_price,
                'investment': risk_metrics['investment'],
                'current_value': risk_metrics['current_value'],
                'pnl': risk_metrics['pnl_percent'] * 100,
                'risk_level': risk_metrics['risk_level'],
                'rsi': indicators['rsi'],
                'macd': indicators['macd'],
                'sma_20': indicators['sma_20'],
                'ema_20': indicators.get('ema_20', 0),
                'bb_high': indicators.get('bb_high', 0),
                'bb_low': indicators.get('bb_low', 0),
                'price_change': price_change,
                'volume_ratio': volume_ratio,
                'market_context': market_context,
                'market_change': market_change,
                'signal_summary': signal_summary
            }
            # === RAG: Retrieve historical AI context and reflection insights ===
            historical_context = self.database_bot.get_analysis_history(symbol)
            reflection_insights = self.database_bot.get_reflection_insights(symbol)
            rag_note = ''
            if historical_context:
                rag_note += '\nRecent AI decisions:'
                for row in historical_context[:5]:
                    rag_note += f"\n- {row[1]}: {row[3].upper()} (confidence: {row[4]:.2f})"
            if reflection_insights:
                rag_note += '\nRecent Reflection Insights:'
                for insight in reflection_insights[:3]:
                    rag_note += f"\n- {insight['timestamp']}: {insight['key_insights']}"
            # Combine prompt_note and RAG note
            full_prompt_note = (prompt_note or '') + rag_note
            # Generate prompt using template
            if full_prompt_note:
                prompt = STOCK_ANALYSIS_TEMPLATE + f"\n{full_prompt_note}"
            else:
                prompt = STOCK_ANALYSIS_TEMPLATE
            response = generate_ai_analysis(prompt, variables=template_vars)
            if 'error' in response:
                return response
            return response
        except Exception as e:
            return {"error": f"Error analyzing stock: {str(e)}"}
    
    # === Stock Trading Logic (migrated from agent.py) ===
    # Migrate TradingAgent methods related to stock trading here.

    def print_performance_summary(self, symbol, asset_type, timeframe, days_back=50):
        """
        Print a summary of performance for a given symbol and asset type.
        """
        # This would use DatabaseBot and VisualizerBot for reporting
        # Example stub:
        print(f"Performance summary for {symbol} ({asset_type}) over {days_back} days:")
        # ...fetch and print metrics...

# === Usage Example ===
if __name__ == "__main__":
    bot = StockBot()
    result = bot.analyze_stock("AAPL")
    print("AAPL analysis:", result)
