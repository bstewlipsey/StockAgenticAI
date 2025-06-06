"""
VisualizerBot: Charting, trend, and pattern visualization for trading systems.
- For manual/diagnostic use and self-test only. Not used in production trading flow.
- Retained for developer diagnostics, standalone charting, and selftest_visualizer_bot().
"""

import numpy as np
from colorama import Fore, Style
import time
import os
import logging


class VisualizerBot:
    """
    VisualizerBot generates visual representations of trading signals, technical indicators, and price/volume charts for the agentic trading system.
    - Modular bot for ASCII/Unicode charting, trend detection, and pattern recognition.
    - Used by other bots and the orchestrator for reporting and diagnostics.
    """

    def __init__(self, width=50, height=10):
        """
        Initialize VisualizerBot with chart dimensions and symbols for trends/patterns.
        """
        # Chart dimensions
        self.width = width
        self.height = height
        self.volume_height = 3  # Height for volume bars below the price chart
        # ASCII-safe symbols for trend direction
        self.trend_symbols = {"up": "/^", "down": "\\v", "sideways": "->"}
        # ASCII-safe symbols for common chart patterns
        self.patterns = {
            "double_top": "^^",
            "double_bottom": "vv",
            "head_shoulders": "^^^",
            "triangle": "<>",
            "ascending_triangle": "/_",
            "descending_triangle": "\\_",
            "flag": "[F]",
            "pennant": "[P]",
        }
        # Standard Fibonacci retracement levels
        self.fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_price_chart(self, prices, volumes=None):
        """
        Create an ASCII/Unicode price chart with optional volume bars.
        - Normalizes prices to fit chart height.
        - Adds a trend symbol at the last price point.
        - Optionally draws volume bars below the chart.
        - Adds a price scale at the bottom.
        Returns a list of chart lines (strings).
        """
        if not prices:
            return []
        # Find min/max for normalization
        chart = []
        min_price = min(prices)
        max_price = max(prices)
        price_range = max_price - min_price
        if price_range == 0:
            # Flat price: just draw a line
            return ["-" * self.width]
        # Normalize prices to chart height
        normalized = [(p - min_price) / price_range * (self.height - 1) for p in prices]
        # Detect trend and get symbol
        trend = self.detect_trend(prices)
        trend_symbol = self.trend_symbols[trend]
        # Draw price lines, placing a dot for each price, and trend symbol at the end
        for y in range(self.height - 1, -1, -1):
            line = ""
            for x, price in enumerate(normalized):
                if abs(y - price) < 0.5:
                    if x == len(normalized) - 1:  # Last point
                        line += trend_symbol
                    else:
                        line += "o"
                else:
                    line += " "
            chart.append(line)
        # Draw volume bars if provided
        if volumes:
            chart.append("-" * self.width)  # Separator
            max_volume = max(volumes)
            for i in range(self.volume_height):
                line = ""
                for vol in volumes:
                    norm_vol = (vol / max_volume) * self.volume_height
                    line += "#" if norm_vol > i else " "
                chart.append(line)
        # Add price scale at the bottom
        chart.append("-" * self.width)
        chart.append(f"Scale: {min_price:.2f} - {max_price:.2f}")
        return chart

    def detect_trend(self, prices, window=5):
        """
        Detect price trend direction using linear regression on the last N prices.
        Returns 'up', 'down', or 'sideways'.
        """
        if len(prices) < window:
            return "sideways"
        recent_prices = prices[-window:]
        # Fit a line: slope > 0.01 is up, < -0.01 is down
        slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        if slope > 0.01:
            return "up"
        elif slope < -0.01:
            return "down"
        return "sideways"

    def calculate_fibonacci_levels(self, prices):
        """
        Calculate standard Fibonacci retracement levels between high and low.
        Returns a dict of {level: price}.
        """
        high = max(prices)
        low = min(prices)
        diff = high - low
        return {level: low + diff * level for level in self.fib_levels}

    def detect_patterns(self, prices, window=20):
        """
        Detect common price patterns (double top/bottom, triangles, flag) in the last N prices.
        Returns a list of (pattern, description) tuples.
        """
        if len(prices) < window:
            return []
        patterns_found = []
        recent_prices = prices[-window:]
        # Double Top
        if self._is_double_top(recent_prices):
            patterns_found.append(("double_top", "Potential reversal"))
        # Double Bottom
        if self._is_double_bottom(recent_prices):
            patterns_found.append(("double_bottom", "Potential rally"))
        # Ascending Triangle
        if self._is_ascending_triangle(recent_prices):
            patterns_found.append(("ascending_triangle", "Bullish continuation"))
        # Descending Triangle
        if self._is_descending_triangle(recent_prices):
            patterns_found.append(("descending_triangle", "Bearish continuation"))
        # Flag
        if self._is_flag_pattern(recent_prices):
            patterns_found.append(("flag", "Trend continuation likely"))
        return patterns_found

    def _is_double_top(self, prices, threshold=0.02):
        """
        Detect double top pattern by finding two similar peaks.
        """
        peaks = [
            i
            for i in range(1, len(prices) - 1)
            if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]
        ]
        if len(peaks) >= 2:
            top1, top2 = prices[peaks[-2]], prices[peaks[-1]]
            return abs(top1 - top2) / top1 < threshold
        return False

    def _is_double_bottom(self, prices, threshold=0.02):
        """
        Detect double bottom pattern by finding two similar troughs.
        """
        troughs = [
            i
            for i in range(1, len(prices) - 1)
            if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]
        ]
        if len(troughs) >= 2:
            bottom1, bottom2 = prices[troughs[-2]], prices[troughs[-1]]
            return abs(bottom1 - bottom2) / bottom1 < threshold
        return False

    def _is_ascending_triangle(self, prices, threshold=0.02):
        """
        Detect ascending triangle: highs are relatively equal (low stddev).
        """
        highs = [
            i
            for i in range(1, len(prices) - 1)
            if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]
        ]
        if len(highs) >= 2:
            high_prices = [prices[i] for i in highs]
            high_std = np.std(high_prices)
            high_mean = np.mean(high_prices)
            return high_std / high_mean < threshold
        return False

    def _is_descending_triangle(self, prices, threshold=0.02):
        """
        Detect descending triangle: lows are trending lower (negative slope).
        """
        lows = [
            i
            for i in range(1, len(prices) - 1)
            if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]
        ]
        if len(lows) >= 2:
            low_prices = [prices[i] for i in lows]
            slope = np.polyfit(range(len(low_prices)), low_prices, 1)[0]
            return slope < -threshold
        return False

    def _is_flag_pattern(self, prices, window=10):
        """
        Detect flag pattern: strong trend followed by a parallel channel.
        """
        if len(prices) < window:
            return False
        # Check for strong trend before the flag
        pre_pattern = prices[:-window]
        if len(pre_pattern) < window:
            return False
        trend = self.detect_trend(pre_pattern)
        if trend == "sideways":
            return False
        # Check for parallel channel in flag portion
        flag_portion = prices[-window:]
        highs = [
            i
            for i in range(1, len(flag_portion) - 1)
            if flag_portion[i] > flag_portion[i - 1]
            and flag_portion[i] > flag_portion[i + 1]
        ]
        lows = [
            i
            for i in range(1, len(flag_portion) - 1)
            if flag_portion[i] < flag_portion[i - 1]
            and flag_portion[i] < flag_portion[i + 1]
        ]
        if len(highs) >= 2 and len(lows) >= 2:
            high_slope = np.polyfit(highs, [flag_portion[i] for i in highs], 1)[0]
            low_slope = np.polyfit(lows, [flag_portion[i] for i in lows], 1)[0]
            # Slopes should be roughly parallel
            return abs(high_slope - low_slope) < 0.01
        return False

    def visualize_signals(self, prices, signals, indicators, volumes=None):
        """
        Create a visual representation of trading signals and technical indicators.
        - Generates a price/volume chart
        - Summarizes key technical indicators
        - Handles errors gracefully
        Returns:
            tuple: (chart_lines, signal_summary)
        """
        try:
            # Create the price/volume chart
            chart = self.create_price_chart(prices, volumes)
            signal_summary = []
            # Only show significant indicators if available
            if indicators and indicators.get("has_signals", False):
                signal_summary.extend(
                    [
                        f"\n{Style.BRIGHT}Technical Analysis:{Style.RESET_ALL}",
                        f"└── Signal: {Fore.GREEN if indicators.get('macd', 0) > 0 else Fore.RED}"
                        f"{'BULLISH' if indicators.get('macd', 0) > 0 else 'BEARISH'}{Style.RESET_ALL}",
                        f"└── Trend: {self.trend_symbols[self.detect_trend(prices)]}",
                        f"└── RSI: {indicators.get('rsi', 0):.1f}",
                        f"└── Price vs SMA20: {'+' if prices[-1] > indicators.get('sma_20', 0) else '-'}"
                        f"${abs(prices[-1] - indicators.get('sma_20', 0)):.2f}",
                    ]
                )
            return chart, signal_summary
        except Exception as e:
            print(f"Visualization error: {e}")
            return [], ["Error generating visualization"]

    def display_reflection_insights(self, symbol=None, limit=10):
        """
        Display a list of recent reflection insights generated by ReflectionBot from DatabaseBot.
        If symbol is provided, filter by symbol; otherwise, show cross-asset insights.
        """
        from bot_database import DatabaseBot

        db = DatabaseBot()
        if symbol:
            insights = db.get_reflection_insights(symbol, days=90)[:limit]
            print(f"\n=== Reflection Insights for {symbol} ===")
        else:
            insights = db.get_cross_asset_insights(days=30)
            print("\n=== Cross-Asset Reflection Insights ===")
        if not insights:
            print("No insights available.")
            return
        for i, insight in enumerate(insights, 1):
            if symbol:
                print(
                    f"[{i}] {insight['timestamp']} | PnL: {insight['pnl_percentage']:.2f}% | {insight['key_insights']} | Lessons: {insight['lessons_learned']}"
                )
            else:
                print(
                    f"[{i}] {insight['symbol']} | AvgPnL: {insight['avg_pnl']:.2f}% | Trades: {insight['trade_count']} | Insights: {insight['combined_insights']}"
                )

    def display_performance_trends(self, symbol=None, days=30):
        """
        Visualize key metrics (win rate, P&L %) from TradeOutcome records in DatabaseBot over time.
        """
        from bot_database import DatabaseBot

        db = DatabaseBot()
        perf = db.get_comprehensive_performance(symbol, days)
        if "error" in perf:
            print(perf["error"])
            return
        print(
            f"\n=== Performance Trends for {symbol or 'All Assets'} (last {days} days) ==="
        )
        print(f"Total Trades: {perf['total_trades']}")
        print(f"Profitable Trades: {perf['profitable_trades']}")
        print(f"Win Rate: {perf['win_rate']:.1f}%")
        print(f"Total PnL: {perf['total_pnl']:.2f}")
        print(f"Avg PnL %: {perf['avg_pnl_percentage']:.2f}%")
        print(f"Recent Reflection Insights: {perf['recent_insights_count']}")
        print(f"Avg Confidence Accuracy: {perf['avg_confidence_accuracy']:.2f}")
        if perf.get("key_lessons"):
            print("Key Lessons:")
            for lesson in perf["key_lessons"]:
                print(f"  - {lesson}")

    # Replace Unicode arrows and special characters in chart output
    def safe_print_chart(self, chart):
        ascii_chart = []
        for line in chart:
            # Replace common Unicode arrows with ASCII
            line = line.replace("\u2197", "/").replace("\u2198", "\\")
            # Add more replacements as needed
            ascii_chart.append(line)
        print("\n".join(ascii_chart))

    def plot(self, data):
        method = "plot"
        self.logger.debug(f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}(data_len={len(data)})] START")
        # ...existing code...
        self.logger.debug(f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}(data_len={len(data)})] END")


def tail_log(log_path, alert_config):
    print(f"[LogMonitor] Monitoring {log_path} for real-time events...")
    if not os.path.exists(log_path):
        print(f"[LogMonitor] Log file not found: {log_path}")
        return
    with open(log_path, "r") as f:
        f.seek(0, os.SEEK_END)
        no_trade_cycles = 0
        failure_count = 0
        while True:
            line = f.readline()
            if not line:
                time.sleep(1)
                continue
            print(line, end="")
            # --- Alerts ---
            if alert_config["no_trade_pattern"] in line:
                no_trade_cycles += 1
                if no_trade_cycles >= alert_config["no_trade_cycles"]:
                    print(
                        f"[ALERT] No trades for {no_trade_cycles} consecutive cycles!"
                    )
            else:
                no_trade_cycles = 0
            if alert_config["failure_pattern"] in line:
                failure_count += 1
                if failure_count >= alert_config["failure_threshold"]:
                    print(f"[ALERT] {failure_count} trade execution failures detected!")
            else:
                failure_count = 0
            if alert_config["rate_limit_pattern"] in line:
                print("[ALERT] API rate limit or data fetching error detected!")
            if alert_config["drawdown_pattern"] in line:
                print("[ALERT] Significant portfolio drawdown detected!")


def display_interactive_log_viewer(log_path="trading.log", filter_keywords=None):
    """
    Continuously tail and display trading.log, optionally filtering for message types (e.g., DECISION, TRADE_OUTCOME, ERROR).
    """
    import os
    import time

    print(f"[LogViewer] Tailing {log_path}... Press Ctrl+C to stop.")
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return
    with open(log_path, "r") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)
                continue
            if filter_keywords:
                if any(kw in line for kw in filter_keywords):
                    print(line, end="")
            else:
                print(line, end="")


# === Usage Example ===
if __name__ == "__main__":
    bot = VisualizerBot(width=40, height=8)
    prices = [100, 102, 105, 103, 108, 110, 112, 111, 115, 117, 120]
    volumes = [1000, 1200, 900, 1100, 1300, 1250, 1400, 1350, 1500, 1550, 1600]
    chart, summary = bot.visualize_signals(
        prices,
        signals=None,
        indicators={"macd": 1, "rsi": 65, "sma_20": 110, "has_signals": True},
        volumes=volumes,
    )
    print("\n".join(chart))
    # alert_config = {
    #     'no_trade_pattern': 'No trades executed',
    #     'no_trade_cycles': 3,
    #     'failure_pattern': 'Trade execution failed',
    #     'failure_threshold': 2,
    #     'rate_limit_pattern': 'rate limit',
    #     'drawdown_pattern': 'drawdown',
    # }
    # tail_log('trading.log', alert_config)  # Commented out for test
    # Uncomment to test interactive log viewer
    # display_interactive_log_viewer(filter_keywords=['DECISION', 'TRADE_OUTCOME', 'ERROR'])


def selftest_visualizer_bot():
    """Standalone self-test for VisualizerBot: tests display methods with dummy data."""
    print("\n--- Running VisualizerBot Self-Test ---")
    import sys
    import io

    try:
        bot = VisualizerBot(width=10, height=5)
        # Test 1: Price chart
        prices = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2]
        volumes = [10, 20, 30, 40, 30, 20, 10, 20, 30, 40]
        chart = bot.create_price_chart(prices, volumes)
        assert (
            isinstance(chart, list) and len(chart) > 0
        ), "create_price_chart did not return a chart."
        print("    -> Price chart logic passed.")
        # Test 2: Trend detection
        trend = bot.detect_trend(prices)
        assert trend in ["up", "down", "sideways"], f"Unexpected trend: {trend}"
        print("    -> Trend detection logic passed.")
        # Test 3: Display to console (capture stdout)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        for line in chart:
            print(line)
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        assert len(output) > 0, "No output printed to console."
        print("    -> Console output logic passed.")
        print("--- VisualizerBot Self-Test PASSED ---")
    except AssertionError as e:
        print(f"--- VisualizerBot Self-Test FAILED: {e} ---")
    except Exception as e:
        print(f"--- VisualizerBot Self-Test encountered an ERROR: {e} ---")


if __name__ == "__main__":
    selftest_visualizer_bot()
