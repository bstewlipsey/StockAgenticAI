# bot_portfolio.py
"""
PortfolioBot: Portfolio management, trade tracking, and performance metrics for trading systems.
- Tracks open positions and trade history
- Calculates portfolio-level metrics and win/loss rates
- Designed for modular agentic trading bots
"""

from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime
import pandas as pd
import logging
from config_trading import TOTAL_CAPITAL
from utils.logger_mixin import LoggerMixin


@dataclass
class Trade:
    symbol: str
    asset_type: str
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    entry_date: datetime
    exit_date: Optional[datetime]
    action: str
    confidence: float


class PortfolioBot(LoggerMixin):
    """
    Self-contained bot for portfolio management, tracking trades, positions, capital, and metrics.
    Provides methods for adding/updating/closing positions, calculating metrics, and reporting.
    """

    def __init__(self, initial_capital=TOTAL_CAPITAL):
        """
        Initialize PortfolioBot with starting capital and empty positions/trades.
        Sets up logger for portfolio operations.
        """
        super().__init__()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # symbol -> Trade
        self.trades = []  # List[Trade]
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_or_update_position(self, symbol, asset_type, quantity, entry_price):
        """
        Add a new position or update an existing one in the portfolio.
        Handles averaging entry price and updating position state.
        """
        method = "add_or_update_position"
        self.logger.debug(
            f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}(symbol='{symbol}', quantity={quantity}, entry_price={entry_price})] START"
        )
        if symbol in self.positions:
            pos = self.positions[symbol]
            total_qty = pos.quantity + quantity
            if total_qty > 0:
                avg_entry = (
                    (pos.quantity * pos.entry_price) + (quantity * entry_price)
                ) / total_qty
            else:
                avg_entry = entry_price
            pos.quantity = total_qty
            pos.entry_price = avg_entry
            pos.entry_date = datetime.now()
            self.logger.info(
                f"Updated position: {symbol}, Quantity: {total_qty}, Avg Entry Price: {avg_entry}"
            )
        else:
            self.positions[symbol] = Trade(
                symbol=symbol,
                asset_type=asset_type,
                entry_price=entry_price,
                exit_price=None,
                quantity=quantity,
                entry_date=datetime.now(),
                exit_date=None,
                action="buy",
                confidence=1.0,
            )
            self.logger.info(f"Added new position: {symbol}, Quantity: {quantity}")

        self.logger.debug(
            f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}(symbol='{symbol}', quantity={quantity}, entry_price={entry_price})] END"
        )

    def close_position(self, symbol, exit_price):
        """
        Close a position and record the trade.
        Removes from open positions and adds to trade history.
        """
        method = "close_position"
        self.logger.debug(
            f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}(symbol='{symbol}', exit_price={exit_price})] START"
        )
        if symbol in self.positions:
            pos = self.positions.pop(symbol)
            closed_trade = Trade(
                symbol=pos.symbol,
                asset_type=pos.asset_type,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                quantity=pos.quantity,
                entry_date=pos.entry_date,
                exit_date=datetime.now(),
                action="sell",
                confidence=pos.confidence,
            )
            self.trades.append(closed_trade)
            self.current_capital += (exit_price - pos.entry_price) * pos.quantity
            self.logger.info(
                f"Closed position: {symbol}, Quantity: {pos.quantity}, Exit Price: {exit_price}"
            )
        else:
            self.logger.warning(f"Attempted to close non-existent position: {symbol}")

        self.logger.debug(
            f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}(symbol='{symbol}', exit_price={exit_price})] END"
        )

    def calculate_metrics(self) -> Dict:
        """
        Calculate portfolio metrics such as return, win rate, and drawdown.
        Returns a dict with key metrics for performance tracking.
        """
        if not self.trades:
            return {
                "total_return": 0.0,
                "win_rate": 0.0,
                "avg_profit": 0.0,
                "avg_loss": 0.0,
                "largest_gain": 0.0,
                "largest_loss": 0.0,
            }
        df = pd.DataFrame([t.__dict__ for t in self.trades])
        profitable_trades = df[df["exit_price"] > df["entry_price"]]
        losing_trades = df[df["exit_price"] < df["entry_price"]]
        return {
            "total_return": (self.current_capital - self.initial_capital)
            / self.initial_capital,
            "win_rate": (
                float(len(profitable_trades)) / float(len(df)) if len(df) > 0 else 0.0
            ),
            "avg_profit": (
                profitable_trades["exit_price"].mean()
                if len(profitable_trades) > 0
                else 0.0
            ),
            "avg_loss": (
                losing_trades["exit_price"].mean() if len(losing_trades) > 0 else 0.0
            ),
            "largest_gain": (
                df["exit_price"].max() - df["entry_price"].min() if len(df) > 0 else 0.0
            ),
            "largest_loss": (
                df["exit_price"].min() - df["entry_price"].max() if len(df) > 0 else 0.0
            ),
        }

    def get_portfolio_metrics(self) -> Dict:
        """
        Return the latest portfolio metrics (calls calculate_metrics).
        """
        return self.calculate_metrics()

    def get_open_positions(self):
        """
        Return a dictionary of open positions (symbol -> Trade).
        """
        return self.positions

    def get_trade_history(self):
        """
        Return the trade history for the portfolio (list of Trade objects).
        """
        return self.trades

    def print_portfolio_summary(self):
        """
        Print a summary of the current portfolio state to the log or console.
        """
        metrics = self.get_portfolio_metrics()
        print("\n=== Portfolio Summary ===")
        for k, v in metrics.items():
            print(f"{k}: {v}")
        print("\nOpen positions:", self.get_open_positions())
        print("Trade history:", self.get_trade_history())

    @staticmethod
    def selftest() -> bool:
        """
        Run a self-test to verify portfolio logic. Returns True if self-test passes.
        """
        print("\n--- Running PortfolioBot Self-Test ---")
        try:
            bot = PortfolioBot(initial_capital=10000)
            # Test 1: Add position
            bot.add_or_update_position("AAPL", "stock", 10, 150)
            assert "AAPL" in bot.get_open_positions(), "AAPL should be in open positions."
            print("    -> Add position logic passed.")
            # Test 2: Update position (add more)
            bot.add_or_update_position("AAPL", "stock", 5, 160)
            pos = bot.get_open_positions()["AAPL"]
            assert pos.quantity == 15, f"Expected 15 shares, got {pos.quantity}"
            print("    -> Update position logic passed.")
            # Test 3: Close position
            bot.close_position("AAPL", 170)
            assert "AAPL" not in bot.get_open_positions(), "AAPL should be closed."
            assert len(bot.get_trade_history()) == 1, "Trade history should have 1 trade."
            print("    -> Close position logic passed.")
            # Test 4: Portfolio metrics
            metrics = bot.get_portfolio_metrics()
            assert (
                isinstance(metrics, dict) and "total_return" in metrics
            ), "Metrics missing 'total_return'."
            print("    -> Portfolio metrics logic passed.")
            print("--- PortfolioBot Self-Test PASSED ---")
            return True
        except AssertionError as e:
            print(f"--- PortfolioBot Self-Test FAILED: {e} ---")
        except Exception as e:
            print(f"--- PortfolioBot Self-Test encountered an ERROR: {e} ---")
        return False


# === Usage Example ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    portfolio = PortfolioBot(initial_capital=10000)
    # Add a position
    portfolio.add_or_update_position("AAPL", "stock", 10, 150)
    # Close the position
    portfolio.close_position("AAPL", 155)
    # Print metrics
    print("Portfolio metrics:", portfolio.get_portfolio_metrics())
    print("Open positions:", portfolio.get_open_positions())
    print("Trade history:", portfolio.get_trade_history())
    # Print portfolio summary
    portfolio.print_portfolio_summary()

    PortfolioBot.selftest()
