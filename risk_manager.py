from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    asset_type: str

class RiskManager:
    def __init__(self, max_portfolio_risk=0.02, max_position_risk=0.01):
        self.max_portfolio_risk = max_portfolio_risk  # 2% max portfolio risk
        self.max_position_risk = max_position_risk    # 1% max position risk
        
    def calculate_position_risk(self, position: Position) -> Dict:
        """Calculate risk metrics for a position"""
        investment = position.quantity * position.entry_price
        current_value = position.quantity * position.current_price
        pnl = current_value - investment
        pnl_percent = (pnl / investment) if investment != 0 else 0
        
        return {
            'investment': investment,
            'current_value': current_value,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'risk_level': 'HIGH' if abs(pnl_percent) > self.max_position_risk else 'LOW'
        }

    def get_position_size(self, capital: float, price: float, risk_per_share: float) -> float:
        """Calculate recommended position size based on risk"""
        max_loss = capital * self.max_position_risk
        shares = max_loss / risk_per_share
        return min(shares, capital / price)