import numpy as np
from colorama import Fore, Style
import pandas as pd

class SignalVisualizer:
    def __init__(self, width=50, height=10):
        self.width = width
        self.height = height
        self.volume_height = 3  # Height for volume bars
        self.trend_symbols = {
            'up': '↗',
            'down': '↘',
            'sideways': '→'
        }
        self.patterns = {
            'double_top': '⋀⋀',
            'double_bottom': '⋁⋁',
            'head_shoulders': '⋀⋀⋀',
            'triangle': '◄►',
            'ascending_triangle': '▲',
            'descending_triangle': '▼',
            'flag': '⚑',
            'pennant': '⚐'
        }
        self.fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]

    def create_price_chart(self, prices, volumes=None):
        """Create ASCII price chart with volume profile"""
        if not prices:
            return []
            
        # Create price chart
        chart = []
        min_price = min(prices)
        max_price = max(prices)
        price_range = max_price - min_price
        
        if price_range == 0:
            return ['─' * self.width]
            
        normalized = [(p - min_price) / price_range * (self.height - 1) for p in prices]
        
        # Add trend direction
        trend = self.detect_trend(prices)
        trend_symbol = self.trend_symbols[trend]
        
        # Price lines with trend
        for y in range(self.height - 1, -1, -1):
            line = ''
            for x, price in enumerate(normalized):
                if abs(y - price) < 0.5:
                    if x == len(normalized) - 1:  # Last point
                        line += trend_symbol
                    else:
                        line += '●'
                else:
                    line += ' '
            chart.append(line)
        
        # Add volume bars if available
        if volumes:
            chart.append('─' * self.width)  # Separator
            max_volume = max(volumes)
            for i in range(self.volume_height):
                line = ''
                for vol in volumes:
                    norm_vol = (vol / max_volume) * self.volume_height
                    line += '█' if norm_vol > i else ' '
                chart.append(line)
                
        # Add price scale
        chart.append('─' * self.width)
        chart.append(f"Scale: {min_price:.2f} - {max_price:.2f}")
        
        return chart

    def detect_trend(self, prices, window=5):
        """Detect price trend direction"""
        if len(prices) < window:
            return 'sideways'
            
        recent_prices = prices[-window:]
        slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        
        if slope > 0.01:
            return 'up'
        elif slope < -0.01:
            return 'down'
        return 'sideways'

    def calculate_fibonacci_levels(self, prices):
        """Calculate Fibonacci retracement levels"""
        high = max(prices)
        low = min(prices)
        diff = high - low
        
        return {
            level: low + diff * level 
            for level in self.fib_levels
        }

    def detect_patterns(self, prices, window=20):
        """Detect common price patterns"""
        if len(prices) < window:
            return []
        
        patterns_found = []
        recent_prices = prices[-window:]
        
        # Detect Double Top
        if self._is_double_top(recent_prices):
            patterns_found.append(('double_top', 'Potential reversal'))
            
        # Detect Double Bottom
        if self._is_double_bottom(recent_prices):
            patterns_found.append(('double_bottom', 'Potential rally'))
            
        # Add triangle pattern detection
        if self._is_ascending_triangle(recent_prices):
            patterns_found.append(('ascending_triangle', 'Bullish continuation'))
            
        if self._is_descending_triangle(recent_prices):
            patterns_found.append(('descending_triangle', 'Bearish continuation'))
            
        if self._is_flag_pattern(recent_prices):
            patterns_found.append(('flag', 'Trend continuation likely'))
            
        return patterns_found

    def _is_double_top(self, prices, threshold=0.02):
        """Detect double top pattern"""
        peaks = [i for i in range(1, len(prices)-1) if 
                prices[i] > prices[i-1] and prices[i] > prices[i+1]]
        if len(peaks) >= 2:
            top1, top2 = prices[peaks[-2]], prices[peaks[-1]]
            return abs(top1 - top2) / top1 < threshold
        return False

    def _is_double_bottom(self, prices, threshold=0.02):
        """Detect double bottom pattern"""
        troughs = [i for i in range(1, len(prices)-1) if 
                  prices[i] < prices[i-1] and prices[i] < prices[i+1]]
        if len(troughs) >= 2:
            bottom1, bottom2 = prices[troughs[-2]], prices[troughs[-1]]
            return abs(bottom1 - bottom2) / bottom1 < threshold
        return False

    def _is_ascending_triangle(self, prices, threshold=0.02):
        """Detect ascending triangle pattern"""
        highs = [i for i in range(1, len(prices)-1) if 
                prices[i] > prices[i-1] and prices[i] > prices[i+1]]
        if len(highs) >= 2:
            # Check if highs are relatively equal
            high_prices = [prices[i] for i in highs]
            high_std = np.std(high_prices)
            high_mean = np.mean(high_prices)
            return high_std / high_mean < threshold
        return False

    def _is_descending_triangle(self, prices, threshold=0.02):
        """Detect descending triangle pattern"""
        lows = [i for i in range(1, len(prices)-1) if 
                prices[i] < prices[i-1] and prices[i] < prices[i+1]]
        if len(lows) >= 2:
            # Check if lows are trending lower
            low_prices = [prices[i] for i in lows]
            slope = np.polyfit(range(len(low_prices)), low_prices, 1)[0]
            return slope < -threshold
        return False

    def _is_flag_pattern(self, prices, window=10):
        """Detect flag pattern"""
        if len(prices) < window:
            return False
            
        # Check for strong trend before pattern
        pre_pattern = prices[:-window]
        if len(pre_pattern) < window:
            return False
            
        trend = self.detect_trend(pre_pattern)
        if trend == 'sideways':
            return False
            
        # Check for parallel channel in flag portion
        flag_portion = prices[-window:]
        highs = [i for i in range(1, len(flag_portion)-1) if 
                flag_portion[i] > flag_portion[i-1] and flag_portion[i] > flag_portion[i+1]]
        lows = [i for i in range(1, len(flag_portion)-1) if 
                flag_portion[i] < flag_portion[i-1] and flag_portion[i] < flag_portion[i+1]]
                
        if len(highs) >= 2 and len(lows) >= 2:
            high_slope = np.polyfit(highs, [flag_portion[i] for i in highs], 1)[0]
            low_slope = np.polyfit(lows, [flag_portion[i] for i in lows], 1)[0]
            # Check if slopes are roughly parallel
            return abs(high_slope - low_slope) < 0.01
            
        return False

    def visualize_signals(self, prices, signals, indicators, volumes=None):
        """Create visual representation of trading signals"""
        try:
            # Create basic chart
            chart = self.create_price_chart(prices, volumes)
            signal_summary = []

            # Only show significant indicators
            if indicators and indicators.get('has_signals', False):
                signal_summary.extend([
                    f"\n{Style.BRIGHT}Technical Analysis:{Style.RESET_ALL}",
                    f"└── Signal: {Fore.GREEN if indicators.get('macd', 0) > 0 else Fore.RED}"
                    f"{'BULLISH' if indicators.get('macd', 0) > 0 else 'BEARISH'}{Style.RESET_ALL}",
                    f"└── Trend: {self.trend_symbols[self.detect_trend(prices)]}",
                    f"└── RSI: {indicators.get('rsi', 0):.1f}",
                    f"└── Price vs SMA20: {'+' if prices[-1] > indicators.get('sma_20', 0) else '-'}"
                    f"${abs(prices[-1] - indicators.get('sma_20', 0)):.2f}"
                ])

            return chart, signal_summary
        except Exception as e:
            print(f"Visualization error: {e}")
            return [], ["Error generating visualization"]