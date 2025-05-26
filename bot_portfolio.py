# bot_portfolio.py
"""
PortfolioBot: Portfolio management, trade tracking, and performance metrics for trading systems.
- Tracks open positions and trade history
- Calculates portfolio-level metrics and win/loss rates
- Designed for modular agentic trading bots
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from config_trading_variables import TOTAL_CAPITAL

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

class PortfolioBot:
    """
    Self-contained bot for portfolio management, tracking trades, positions, capital, and metrics.
    """
    def __init__(self, initial_capital=TOTAL_CAPITAL):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # symbol -> Trade
        self.trades = []     # List[Trade]

    def add_or_update_position(self, symbol, asset_type, quantity, entry_price):
        """Add a new position or update an existing one in the portfolio."""
        if symbol in self.positions:
            pos = self.positions[symbol]
            total_qty = pos.quantity + quantity
            if total_qty > 0:
                avg_entry = ((pos.quantity * pos.entry_price) + (quantity * entry_price)) / total_qty
            else:
                avg_entry = entry_price
            pos.quantity = total_qty
            pos.entry_price = avg_entry
            pos.entry_date = datetime.now()
        else:
            self.positions[symbol] = Trade(
                symbol=symbol,
                asset_type=asset_type,
                entry_price=entry_price,
                exit_price=None,
                quantity=quantity,
                entry_date=datetime.now(),
                exit_date=None,
                action='buy',
                confidence=1.0
            )

    def close_position(self, symbol, exit_price):
        """Close a position and record the trade."""
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
                action='sell',
                confidence=pos.confidence
            )
            self.trades.append(closed_trade)
            self.current_capital += (exit_price - pos.entry_price) * pos.quantity

    def calculate_metrics(self) -> Dict:
        """Calculate portfolio performance metrics based on trade history."""
        if not self.trades:
            return {
                'total_return': 0.0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'largest_gain': 0.0,
                'largest_loss': 0.0
            }
        df = pd.DataFrame([t.__dict__ for t in self.trades])
        profitable_trades = df[df['exit_price'] > df['entry_price']]
        losing_trades = df[df['exit_price'] < df['entry_price']]
        return {
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'win_rate': len(profitable_trades) / len(df) if len(df) > 0 else 0,
            'avg_profit': profitable_trades['exit_price'].mean() if len(profitable_trades) > 0 else 0,
            'avg_loss': losing_trades['exit_price'].mean() if len(losing_trades) > 0 else 0,
            'largest_gain': df['exit_price'].max() - df['entry_price'].min() if len(df) > 0 else 0,
            'largest_loss': df['exit_price'].min() - df['entry_price'].max() if len(df) > 0 else 0
        }

    def get_portfolio_metrics(self) -> Dict:
        """Alias for calculate_metrics."""
        return self.calculate_metrics()

    def get_open_positions(self):
        """Return all open positions."""
        return self.positions

    def get_trade_history(self):
        """Return the trade history."""
        return self.trades

    def print_portfolio_summary(self):
        """Prints a summary of portfolio metrics, open positions, and trade history."""
        metrics = self.get_portfolio_metrics()
        print("\n=== Portfolio Summary ===")
        for k, v in metrics.items():
            print(f"{k}: {v}")
        print("\nOpen positions:", self.get_open_positions())
        print("Trade history:", self.get_trade_history())

# === Usage Example ===
if __name__ == "__main__":
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
