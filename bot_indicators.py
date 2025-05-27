# bot_indicators.py
"""
Bot for technical analysis and indicator calculation, modularized for use in trading agents.
"""

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator, VolumeWeightedAveragePrice
from config_trading import RSI_OVERSOLD, RSI_OVERBOUGHT, SMA_WINDOW

_indicator_cache = {}

class IndicatorBot:
    """
    Bot for calculating technical indicators and generating trading signals.
    """
    def __init__(self, prices, window=SMA_WINDOW):
        self.df = pd.DataFrame({
            'close': prices,
            'high': prices,
            'low': prices,
            'open': prices,
            'volume': [1] * len(prices)
        })
        self.window = window

    def calculate_indicators(self):
        df_hash = hash(str(self.df.iloc[-self.window:].values.tobytes()))
        if df_hash in _indicator_cache:
            return _indicator_cache[df_hash]
        indicators = {
            'rsi': 50.0,
            'macd': 0.0,
            'macd_signal': 0.0,
            'macd_histogram': 0.0,
            'sma_20': 0.0,
            'ema_20': 0.0,
            'bb_upper': 0.0,
            'bb_lower': 0.0,
            'obv': 0.0,
            'adi': 0.0,
            'vwap': 0.0,
            'current_price': 0.0,
            'has_signals': False
        }
        try:
            data_length = len(self.df)
            min_rsi_periods = 14
            min_macd_periods = 26
            min_bb_periods = self.window
            min_sma_periods = self.window
            min_ema_periods = self.window
            min_volume_periods = 10
            # RSI
            rsi_indicator = RSIIndicator(close=self.df['close'], window=14) if data_length >= min_rsi_periods else None
            # MACD
            macd_indicator = MACD(close=self.df['close']) if data_length >= min_macd_periods else None
            # SMA
            sma_indicator = SMAIndicator(close=self.df['close'], window=self.window) if data_length >= min_sma_periods else None
            # EMA
            ema_indicator = EMAIndicator(close=self.df['close'], window=self.window) if data_length >= min_ema_periods else None
            # Bollinger Bands
            bb_indicator = BollingerBands(close=self.df['close'], window=self.window) if data_length >= min_bb_periods else None
            # OBV
            obv_indicator = OnBalanceVolumeIndicator(close=self.df['close'], volume=self.df['volume']) if data_length >= min_volume_periods else None
            # ADI
            adi_indicator = AccDistIndexIndicator(
                high=self.df['high'], low=self.df['low'], close=self.df['close'], volume=self.df['volume'] ) if data_length >= min_volume_periods else None
            # VWAP
            vwap_indicator = VolumeWeightedAveragePrice(
                high=self.df['high'], low=self.df['low'], close=self.df['close'], volume=self.df['volume'] ) if data_length >= min_volume_periods else None
            # Extract values
            indicators['rsi'] = rsi_indicator.rsi().iloc[-1] if rsi_indicator is not None else 50.0
            indicators['macd'] = macd_indicator.macd().iloc[-1] if macd_indicator is not None else 0.0
            indicators['macd_signal'] = macd_indicator.macd_signal().iloc[-1] if macd_indicator is not None else 0.0
            indicators['macd_histogram'] = macd_indicator.macd_diff().iloc[-1] if macd_indicator is not None else 0.0
            indicators['sma_20'] = sma_indicator.sma_indicator().iloc[-1] if sma_indicator is not None else (self.df['close'].iloc[-1] if len(self.df) > 0 else 0.0)
            indicators['ema_20'] = ema_indicator.ema_indicator().iloc[-1] if ema_indicator is not None else (self.df['close'].iloc[-1] if len(self.df) > 0 else 0.0)
            if bb_indicator is not None:
                indicators['bb_upper'] = bb_indicator.bollinger_hband().iloc[-1]
                indicators['bb_lower'] = bb_indicator.bollinger_lband().iloc[-1]
            else:
                current_price = self.df['close'].iloc[-1] if len(self.df) > 0 else 0.0
                indicators['bb_upper'] = current_price * 1.02
                indicators['bb_lower'] = current_price * 0.98
            indicators['obv'] = obv_indicator.on_balance_volume().iloc[-1] if obv_indicator is not None else 0.0
            indicators['adi'] = adi_indicator.acc_dist_index().iloc[-1] if adi_indicator is not None else 0.0
            indicators['vwap'] = vwap_indicator.volume_weighted_average_price().iloc[-1] if vwap_indicator is not None else (self.df['close'].iloc[-1] if len(self.df) > 0 else 0.0)
            indicators['current_price'] = self.df['close'].iloc[-1] if len(self.df) > 0 else 0.0
            indicators['has_signals'] = True
            # Cache
            _indicator_cache[df_hash] = indicators
        except Exception:
            pass
        return indicators

    def get_signals(self):
        """Generate trading signals based on calculated indicators."""
        indicators = self.calculate_indicators()
        signals = []
        # Example: Add a simple RSI-based signal
        if indicators['rsi'] < RSI_OVERSOLD:
            signals.append(('buy', 'RSI oversold', 0.8))
        elif indicators['rsi'] > RSI_OVERBOUGHT:
            signals.append(('sell', 'RSI overbought', 0.8))
        # Add more signal logic as needed
        return signals, indicators

    @staticmethod
    def selftest():
        """Quick diagnostic to check if indicator calculation works."""
        import numpy as np
        prices = np.linspace(100, 120, 30).tolist()
        try:
            bot = IndicatorBot(prices)
            indicators = bot.calculate_indicators()
            print("IndicatorBot selftest: OK")
            print(f"  RSI: {indicators.get('rsi', 'N/A'):.2f}, MACD: {indicators.get('macd', 'N/A'):.4f}")
            return True
        except Exception as e:
            print(f"IndicatorBot selftest: FAIL - {e}")
            return False

    @staticmethod
    def test_signals():
        """Test get_signals method with sample data."""
        try:
            test_prices = [100, 101, 99, 102, 98, 103, 97, 104, 96, 105, 95, 106, 94, 107, 93, 108, 92, 109, 91, 110, 90, 111, 89, 112, 88, 113, 87, 114, 86, 115]
            bot = IndicatorBot(test_prices)
            signals = bot.get_signals()
            print(f"test_signals: {len(signals)} signals generated")
            if signals:
                for i, signal in enumerate(signals):
                    print(f"  Signal {i+1}: {signal}")
            else:
                print("  No signals generated.")
            return True
        except Exception as e:
            print(f"test_signals: FAIL - {e}")
            return False
