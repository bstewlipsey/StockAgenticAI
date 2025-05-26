import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator, VolumeWeightedAveragePrice
from trading_variables import RSI_OVERSOLD, RSI_OVERBOUGHT, SMA_WINDOW

class TechnicalAnalysis:
    def __init__(self, prices, window=20):
        self.df = pd.DataFrame({
            'close': prices,
            'high': prices,
            'low': prices,
            'open': prices,
            'volume': [1] * len(prices)
        })
        self.window = window
    
    def test(self):
        return "TechnicalAnalysis class works"
