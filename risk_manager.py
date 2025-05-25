from dataclasses import dataclass
from typing import Dict, List
from trading_variables import MAX_PORTFOLIO_RISK, MAX_POSITION_RISK

@dataclass
class Position:
    """
    Represents a single open trading position.
    
    Attributes:
        symbol (str): The stock or asset ticker symbol (e.g., 'AAPL')
        quantity (float): Number of shares/contracts held
        entry_price (float): Price at which the position was opened
        current_price (float): Current market price of the asset
        asset_type (str): Type of asset (e.g., 'stock', 'option', 'crypto')
    """
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    asset_type: str

class RiskManager:
    """
    The RiskManager class provides methods to assess and control trading risk at both the position and portfolio level.
    
    Key Responsibilities:
    - Calculate risk metrics for individual positions (P&L, risk level)
    - Enforce maximum risk per position and for the overall portfolio
    - Recommend position sizes based on risk tolerance and stop-loss distance
    - Help prevent catastrophic losses by capping risk exposure
    
    Safety Features:
    - Uses configurable risk limits from trading_variables.py
    - Handles zero-investment edge cases to avoid division errors
    - Returns clear risk levels for use in trade management logic
    """
    def __init__(self, max_portfolio_risk=MAX_PORTFOLIO_RISK, max_position_risk=MAX_POSITION_RISK):
        """
        Initialize the RiskManager with portfolio and position risk limits.
        Args:
            max_portfolio_risk (float): Max % of portfolio to risk at once (e.g., 0.02 for 2%)
            max_position_risk (float): Max % of portfolio to risk on a single trade (e.g., 0.01 for 1%)
        """
        self.max_portfolio_risk = max_portfolio_risk  # From variables file
        self.max_position_risk = max_position_risk    # From variables file
        
    def calculate_position_risk(self, position: Position) -> Dict:
        """
        Calculate risk metrics for a single open position.
        Args:
            position (Position): The position to analyze
        Returns:
            dict: Contains investment, current value, P&L, P&L %, and risk level
        Logic:
        - Computes total invested and current value
        - Calculates profit/loss (P&L) and percent return
        - Flags as 'HIGH' risk if P&L % exceeds max_position_risk
        - Handles zero or negative quantity/price gracefully
        """
        # Defensive: Handle zero or negative quantity/price
        if position.quantity <= 0 or position.entry_price <= 0:
            return {
                'investment': 0.0,
                'current_value': 0.0,
                'pnl': 0.0,
                'pnl_percent': 0.0,
                'risk_level': 'NONE'  # Not a valid position
            }
        investment = position.quantity * position.entry_price
        current_value = position.quantity * position.current_price
        pnl = current_value - investment
        pnl_percent = (pnl / investment) if investment != 0 else 0
        return {
            'investment': investment,           # Total amount invested in this position
            'current_value': current_value,     # Current market value of the position
            'pnl': pnl,                        # Profit or loss in dollars
            'pnl_percent': pnl_percent,         # Profit or loss as a percent of investment
            'risk_level': 'HIGH' if abs(pnl_percent) > self.max_position_risk else 'LOW'  # Risk flag
        }

    def calculate_portfolio_risk(self, positions: List[Position]) -> Dict:
        """
        Calculate total portfolio risk and exposure across all open positions.
        Args:
            positions (List[Position]): List of all open positions
        Returns:
            dict: Contains total investment, current value, total P&L, portfolio P&L %, and risk flag
        """
        total_investment = 0.0
        total_current_value = 0.0
        for pos in positions:
            if pos.quantity > 0 and pos.entry_price > 0:
                total_investment += pos.quantity * pos.entry_price
                total_current_value += pos.quantity * pos.current_price
        total_pnl = total_current_value - total_investment
        portfolio_pnl_percent = (total_pnl / total_investment) if total_investment != 0 else 0
        risk_level = 'HIGH' if abs(portfolio_pnl_percent) > self.max_portfolio_risk else 'LOW'
        return {
            'total_investment': total_investment,
            'total_current_value': total_current_value,
            'total_pnl': total_pnl,
            'portfolio_pnl_percent': portfolio_pnl_percent,
            'risk_level': risk_level
        }

    def get_position_size(self, capital: float, price: float, risk_per_share: float, min_shares: int = 1, allow_fractional: bool = False) -> float:
        """
        Calculate the recommended position size (number of shares) based on risk per trade.
        Args:
            capital (float): Total capital available for this trade
            price (float): Current price per share
            risk_per_share (float): Dollar risk per share (e.g., entry - stop loss)
            min_shares (int): Minimum number of shares to consider (prevents too-small positions)
            allow_fractional (bool): Whether to allow fractional shares in the calculation
        Returns:
            float: Number of shares to buy/sell (not rounded)
        Logic:
        - Caps max loss per trade to max_position_risk * capital
        - Divides max loss by risk per share to get share count
        - Ensures you never buy more than you can afford (capital/price)
        - Enforces minimum shares and supports fractional shares
        - Handles zero/negative price or risk_per_share safely
        """
        if price <= 0 or risk_per_share <= 0 or capital <= 0:
            return 0
        max_loss = capital * self.max_position_risk
        shares = max_loss / risk_per_share
        # Never exceed what you can afford with available capital
        max_affordable = capital / price
        shares = min(shares, max_affordable)
        # Enforce minimum shares if calculation is positive
        if shares > 0 and shares < min_shares:
            shares = min_shares if not allow_fractional else float(min_shares)
        if shares <= 0:
            shares = 0
        return shares