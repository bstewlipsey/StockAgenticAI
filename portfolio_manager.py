from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

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

class PortfolioManager:
    def __init__(self, initial_capital=100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        
    def calculate_metrics(self) -> Dict:
        """Calculate portfolio performance metrics"""
        if not self.trades:
            return {
                'total_return': 0.0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'largest_gain': 0.0,
                'largest_loss': 0.0
            }
            
        df = pd.DataFrame(self.trades)
        profitable_trades = df[df['exit_price'] > df['entry_price']]
        
        return {
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'win_rate': len(profitable_trades) / len(df) if len(df) > 0 else 0,
            'avg_profit': profitable_trades['exit_price'].mean() if len(profitable_trades) > 0 else 0,
            'avg_loss': (df[df['exit_price'] < df['entry_price']]['exit_price'].mean() or 0),
            'largest_gain': df['exit_price'].max() - df['entry_price'].min() if len(df) > 0 else 0,
            'largest_loss': df['exit_price'].min() - df['entry_price'].max() if len(df) > 0 else 0
        }