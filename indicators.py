import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

class TechnicalAnalysis:
    def __init__(self, prices, window=20):
        """Initialize with price data"""
        self.df = pd.DataFrame({
            'close': prices,
            'high': prices,
            'low': prices,
            'open': prices,
            'volume': [1] * len(prices)
        })
        self.window = window

    def calculate_indicators(self):
        """Calculate technical indicators"""
        # Initialize with default values
        indicators = {
            'rsi': 0.0,
            'macd': 0.0,
            'sma_20': 0.0,
            'ema_20': 0.0,
            'bb_high': 0.0,
            'bb_low': 0.0,
            'current_price': 0.0,
            'has_signals': False  # Flag to indicate valid signals
        }

        try:
            if len(self.df) > self.window:
                rsi = RSIIndicator(close=self.df['close'], window=14)
                macd = MACD(close=self.df['close'])
                sma = SMAIndicator(close=self.df['close'], window=self.window)
                ema = EMAIndicator(close=self.df['close'], window=self.window)
                bb = BollingerBands(close=self.df['close'], window=self.window)

                # Update with calculated values
                calculated = {
                    'rsi': rsi.rsi().iloc[-1],
                    'macd': macd.macd_diff().iloc[-1],
                    'sma_20': sma.sma_indicator().iloc[-1],
                    'ema_20': ema.ema_indicator().iloc[-1],
                    'bb_high': bb.bollinger_hband().iloc[-1],
                    'bb_low': bb.bollinger_lband().iloc[-1],
                    'current_price': self.df['close'].iloc[-1],
                    'has_signals': True
                }
                
                # Only update if values are not NaN
                indicators.update({k: v for k, v in calculated.items() if pd.notna(v)})

        except Exception as e:
            print(f"Error calculating indicators: {e}")

        return indicators

    def get_signals(self):
        """Generate trading signals based on indicators"""
        indicators = self.calculate_indicators()
        signals = []

        # Only generate signals if we have valid data
        if all(v is not None for v in [
            indicators['current_price'],
            indicators['sma_20'],
            indicators['rsi'],
            indicators['macd']
        ]):
            # RSI signals
            if indicators['rsi'] < 30:
                signals.append(('BUY', 'RSI oversold', 0.8))
            elif indicators['rsi'] > 70:
                signals.append(('SELL', 'RSI overbought', 0.8))

            # MACD signals
            if indicators['macd'] > 0:
                signals.append(('BUY', 'MACD positive', 0.6))
            else:
                signals.append(('SELL', 'MACD negative', 0.6))

            # Moving Average signals
            if indicators['current_price'] > indicators['sma_20']:
                signals.append(('BUY', 'Price above SMA20', 0.7))
            else:
                signals.append(('SELL', 'Price below SMA20', 0.7))

        return signals, indicators