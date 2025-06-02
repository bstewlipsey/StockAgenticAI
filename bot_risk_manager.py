"""
RiskManager & RiskBot: Position and portfolio risk analysis for trading systems.
- Provides risk metrics, sizing, and risk flagging for positions and portfolios
- Designed for integration with agentic and modular trading bots
"""

from dataclasses import dataclass
from typing import Dict, List
from config_trading import MAX_PORTFOLIO_RISK, MAX_POSITION_RISK
from data_structures import AgenticBotError
from utils.logger_mixin import LoggerMixin
from utils.logging_decorators import log_method_calls
import logging


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


class RiskManager(LoggerMixin):
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

    @log_method_calls
    def __init__(
        self, max_portfolio_risk=MAX_PORTFOLIO_RISK, max_position_risk=MAX_POSITION_RISK
    ):
        """
        Initialize the RiskManager with portfolio and position risk limits.
        Args:
            max_portfolio_risk (float): Max % of portfolio to risk at once (e.g., 0.02 for 2%)
            max_position_risk (float): Max % of portfolio to risk on a single trade (e.g., 0.01 for 1%)
        """
        super().__init__()
        self.max_portfolio_risk = max_portfolio_risk  # From variables file
        self.max_position_risk = max_position_risk  # From variables file

    @log_method_calls
    def calculate_position_risk(self, position: Position) -> Dict:
        """
        Analyze risk for a single position and return risk metrics (e.g., exposure, stop loss, risk %).
        Handles edge cases for zero/negative quantity or price.
        Returns a dict with risk metrics and status.
        """
        method = "calculate_position_risk"
        self.logger.debug(
            f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}(symbol='{getattr(position, 'symbol', None)}')] START"
        )
        logger = logging.getLogger(__name__)
        logger.info(
            f"[RiskManager] Calculating risk for {position.symbol} ({position.asset_type}): qty={position.quantity}, entry={position.entry_price}, current={position.current_price}"
        )
        # Defensive: Handle zero or negative quantity/price
        if position.quantity <= 0 or position.entry_price <= 0:
            logger.warning(
                f"[RiskManager] Position {position.symbol} rejected: zero or negative quantity/entry price."
            )
            return {
                "error": f"Position {position.symbol} rejected: zero or negative quantity/entry price.",
                "status": "failed",
                "investment": 0.0,
                "current_value": 0.0,
                "pnl": 0.0,
                "unrealized_pnl": 0.0,
                "pnl_percent": 0.0,
                "unrealized_pnl_pct": 0.0,
                "risk_level": "NONE",
                "max_loss": 0.0,
                "profit_loss": 0.0,
                "rejection_reason": "Zero or negative quantity/entry price.",
            }
        investment = position.quantity * position.entry_price
        current_value = position.quantity * position.current_price
        pnl = current_value - investment
        pnl_percent = (pnl / investment) if investment != 0 else 0
        max_loss = abs(investment * self.max_position_risk)
        risk_level = "HIGH" if abs(pnl_percent) > self.max_position_risk else "LOW"
        result = {
            "investment": investment,
            "current_value": current_value,
            "pnl": pnl,
            "unrealized_pnl": pnl,  # Alias for test compatibility
            "pnl_percent": pnl_percent,
            "unrealized_pnl_pct": pnl_percent,  # Alias for test compatibility
            "risk_level": risk_level,
            "max_loss": max_loss,
            "profit_loss": pnl,  # Alias for P&L for test compatibility
        }
        self.logger.debug(
            f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}(symbol='{getattr(position, 'symbol', None)}')] END"
        )
        return result

    @log_method_calls
    def calculate_portfolio_risk(self, positions: List[Position]) -> Dict:
        """
        Analyze risk for the entire portfolio and return aggregate risk metrics (e.g., total exposure, VaR).
        Returns a dict with portfolio risk metrics and status.
        """
        logger = logging.getLogger(__name__)
        logger.info(
            f"[RiskManager] Calculating portfolio risk for {len(positions)} positions."
        )
        total_investment = 0.0
        total_current_value = 0.0
        for pos in positions:
            if pos.quantity > 0 and pos.entry_price > 0:
                total_investment += pos.quantity * pos.entry_price
                total_current_value += pos.quantity * pos.current_price
        total_pnl = total_current_value - total_investment
        portfolio_pnl_percent = (
            (total_pnl / total_investment) if total_investment != 0 else 0
        )
        risk_level = (
            "HIGH" if abs(portfolio_pnl_percent) > self.max_portfolio_risk else "LOW"
        )
        return {
            "total_investment": total_investment,
            "total_current_value": total_current_value,
            "total_pnl": total_pnl,
            "unrealized_pnl": total_pnl,  # Alias for test compatibility
            "portfolio_pnl_percent": portfolio_pnl_percent,
            "risk_level": risk_level,
        }

    @log_method_calls
    def get_position_size(
        self,
        capital: float,
        price: float,
        risk_per_share: float,
        min_shares: int = 1,
        allow_fractional: bool = False,
    ) -> float:
        """
        Recommend a position size based on risk per share, capital, and constraints.
        Uses risk limits to cap position size and supports fractional shares if allowed.
        Returns the recommended position size as float.
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

    @log_method_calls
    def analyze_position(self, position: Position) -> dict:
        # Wrapper for compatibility with function reference
        return self.calculate_position_risk(position)

    @log_method_calls
    def analyze_portfolio(self, positions: list[Position]) -> dict:
        # Wrapper for compatibility with function reference
        return self.calculate_portfolio_risk(positions)

    @log_method_calls
    def recommend_position_size(
        self,
        capital: float,
        price: float,
        risk_per_share: float,
        min_shares: int = 1,
        allow_fractional: bool = False,
    ) -> float:
        # Wrapper for compatibility with function reference
        return self.get_position_size(capital, price, risk_per_share, min_shares, allow_fractional)

    @staticmethod
    @log_method_calls
    def selftest_risk_manager_bot() -> None:
        """
        Run a self-test to verify risk management logic and print results.
        """
        print("\n--- Running RiskManagerBot Self-Test ---")
        try:
            risk_bot = RiskBot(max_portfolio_risk=0.05, max_position_risk=0.02)
            # Test 1: Normal position (profit)
            pos1 = Position(
                symbol="AAPL",
                quantity=10,
                entry_price=100,
                current_price=110,
                asset_type="stock",
            )
            result1 = risk_bot.analyze_position(pos1)
            assert (
                abs(result1["unrealized_pnl"] - 100) < 1e-6
            ), f"Expected PnL 100, got {result1['unrealized_pnl']}"
            assert (
                result1["risk_level"] == "HIGH"
            ), f"Expected HIGH risk, got {result1['risk_level']}"
            print("    -> Normal position (profit) logic passed.")
            # Test 2: Normal position (loss)
            pos2 = Position(
                symbol="TSLA",
                quantity=5,
                entry_price=200,
                current_price=190,
                asset_type="stock",
            )
            result2 = risk_bot.analyze_position(pos2)
            assert (
                abs(result2["unrealized_pnl"] + 50) < 1e-6
            ), f"Expected PnL -50, got {result2['unrealized_pnl']}"
            assert (
                result2["risk_level"] == "HIGH"
            ), f"Expected HIGH risk, got {result2['risk_level']}"
            print("    -> Normal position (loss) logic passed.")
            # Test 3: Edge case (zero quantity)
            pos3 = Position(
                symbol="GOOGL",
                quantity=0,
                entry_price=100,
                current_price=120,
                asset_type="stock",
            )
            result3 = risk_bot.analyze_position(pos3)
            assert result3["investment"] == 0, "Expected zero investment for zero quantity."
            assert (
                result3["risk_level"] == "NONE"
            ), f"Expected NONE risk, got {result3['risk_level']}"
            print("    -> Zero quantity edge case logic passed.")
            # Test 4: Portfolio risk
            portfolio = [pos1, pos2]
            port_result = risk_bot.analyze_portfolio(portfolio)
            expected_pnl = result1["unrealized_pnl"] + result2["unrealized_pnl"]
            assert (
                abs(port_result["unrealized_pnl"] - expected_pnl) < 1e-6
            ), "Portfolio PnL calculation error."
            print("    -> Portfolio risk logic passed.")
            print("--- RiskManagerBot Self-Test PASSED ---")
        except AssertionError as e:
            print(f"--- RiskManagerBot Self-Test FAILED: {e} ---")
        except Exception as e:
            print(f"--- RiskManagerBot Self-Test encountered an ERROR: {e} ---")


# Alias for compatibility with function reference and legacy code
RiskManagerBot = RiskManager
RiskBot = RiskManagerBot

# === Usage Example ===
if __name__ == "__main__":
    RiskManager.selftest_risk_manager_bot()
    # Example position
    pos = Position(
        symbol="AAPL",
        quantity=10,
        entry_price=150,
        current_price=145,
        asset_type="stock",
    )
    risk_bot = RiskBot()
    print("Single position risk:", risk_bot.analyze_position(pos))
    print("Portfolio risk:", risk_bot.analyze_portfolio([pos]))
    print("Recommended position size:", risk_bot.recommend_position_size(10000, 150, 5))
