from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
from trading_variables import TOTAL_CAPITAL

@dataclass
class Trade:
    """
    Represents a completed or open trade in the portfolio.
    
    Attributes:
        symbol (str): Ticker symbol of the asset
        asset_type (str): Type of asset (e.g., 'stock')
        entry_price (float): Price at which the trade was entered
        exit_price (Optional[float]): Price at which the trade was exited (None if open)
        quantity (float): Number of shares/contracts
        entry_date (datetime): Date/time of entry
        exit_date (Optional[datetime]): Date/time of exit (None if open)
        action (str): 'buy' or 'sell'
        confidence (float): AI confidence score for the trade
    """
    symbol: str
    asset_type: str
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    entry_date: datetime
    exit_date: Optional[datetime]
    action: str
    confidence: float

class PortfolioManager:
    """
    The PortfolioManager tracks all trades and calculates portfolio-level performance metrics.
    
    Key Responsibilities:
    - Maintain a record of all trades (open and closed)
    - Track current and initial capital
    - Calculate key performance metrics for the portfolio
    - Provide summary statistics for reporting and analysis
    
    Safety Features:
    - Handles empty trade lists gracefully
    - Uses Pandas for robust data analysis
    - Returns zeroed metrics if no trades are present
    """
    def __init__(self, initial_capital=TOTAL_CAPITAL):
        self.initial_capital = initial_capital  # Starting capital for the portfolio
        self.current_capital = initial_capital  # Updated as trades are closed
        self.positions = {}                     # Open positions (symbol -> Trade)
        self.trades = []                        # List of all completed trades (dicts or Trade objects)
        
    def calculate_metrics(self) -> Dict:
        """
        Calculate portfolio performance metrics based on trade history.
        Returns:
            dict: Contains total return, win rate, average profit/loss, and largest gain/loss
        Logic:
        - If no trades, returns all metrics as zero
        - Uses Pandas DataFrame for flexible analysis
        - Win rate: % of profitable trades
        - Average profit: Mean exit price of profitable trades
        - Average loss: Mean exit price of losing trades (or 0 if none)
        - Largest gain/loss: Difference between best/worst exit and entry prices
        """
        if not self.trades:
            # No trades yet: return all metrics as zero
            return {
                'total_return': 0.0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'largest_gain': 0.0,
                'largest_loss': 0.0
            }
        # Convert trade records to DataFrame for analysis
        df = pd.DataFrame(self.trades)
        # Identify profitable trades (exit price > entry price)
        profitable_trades = df[df['exit_price'] > df['entry_price']]
        # Identify losing trades (exit price < entry price)
        losing_trades = df[df['exit_price'] < df['entry_price']]
        return {
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'win_rate': len(profitable_trades) / len(df) if len(df) > 0 else 0,
            'avg_profit': profitable_trades['exit_price'].mean() if len(profitable_trades) > 0 else 0,
            'avg_loss': losing_trades['exit_price'].mean() if len(losing_trades) > 0 else 0,
            'largest_gain': df['exit_price'].max() - df['entry_price'].min() if len(df) > 0 else 0,
            'largest_loss': df['exit_price'].min() - df['entry_price'].max() if len(df) > 0 else 0
        }