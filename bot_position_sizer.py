"""
PositionSizerBot: Position sizing logic for agentic trading systems.
- Calculates optimal trade size based on capital, risk, confidence, and volatility
- Supports both integer and fractional share brokers
- Designed for modular integration with trading bots
"""

from config_trading import TOTAL_CAPITAL, MAX_POSITION_SIZE
from utils.logger_mixin import LoggerMixin


class PositionSizerBot(LoggerMixin):
    """
    PositionSizerBot determines the optimal number of shares to buy or sell for each trade in a modular, bot-style architecture.

    Responsibilities:
    - Calculate position size based on total capital and max position size constraints
    - Adjust position size according to AI confidence in the trade signal
    - Optionally reduce position size for high-volatility stocks (risk parity logic)
    - Enforce minimum shares per trade and support fractional shares if needed
    - Cap position size by available cash to prevent over-allocation
    - Provide clear, auditable sizing logic for risk management and compliance
    - Prevent negative or zero share trades
    - Avoid over-allocation by checking available cash
    - Support both integer and fractional share brokers
    """

    def __init__(
        self, total_capital=TOTAL_CAPITAL, max_position_size=MAX_POSITION_SIZE
    ):
        """
        Initialize the PositionSizer with portfolio capital and risk settings.

        Args:
            total_capital (float): Total portfolio capital available for trading
            max_position_size (float): Maximum fraction of capital to allocate to a single trade
        """
        super().__init__()
        self.total_capital = total_capital
        self.max_position_size = max_position_size

    def calculate_position_size(
        self,
        price,
        confidence,
        volatility=None,
        available_cash=None,
        min_shares=1,
        allow_fractional=False,
        asset_type=None,
    ):
        """
        Calculate the number of shares to trade based on capital, confidence, volatility, and cash constraints.

        Sizing Logic:
        1. Base position size is set by max_position_size * total_capital
        2. Position is scaled by AI confidence (higher confidence = larger size)
        3. If volatility is provided, use risk parity logic to reduce size for high-volatility stocks
        4. Cap position size by available cash if provided
        5. Calculate shares (integer or fractional)
        6. Enforce a minimum number of shares if calculation is positive
        7. Never return negative or zero shares for valid trades

        Args:
            price (float): Current price per share
            confidence (float): AI confidence score (0.0 to 1.0)
            volatility (float, optional): Volatility metric (e.g., ATR or stddev). If provided, use risk parity sizing.
            available_cash (float, optional): Max cash available for this trade. If provided, cap position size.
            min_shares (int, optional): Minimum shares to buy if calculation is positive (default 1)
            allow_fractional (bool, optional): If True, allow fractional shares (default False)
        Returns:
            shares (int or float): Number of shares to buy (int unless allow_fractional=True)
        """
        method = "calculate_position_size"
        self.logger.debug(
            f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}(price={price}, confidence={confidence}, available_cash={available_cash}, min_shares={min_shares})] START"
        )
        # === BASE POSITION SIZE ===
        # Calculate the maximum dollar amount to risk per position
        base_size = self.total_capital * self.max_position_size

        # === CONFIDENCE ADJUSTMENT ===
        # Scale position size by AI confidence (0.0 to 1.0)
        adjusted_size = base_size * confidence

        # === VOLATILITY ADJUSTMENT (RISK PARITY) ===
        if volatility is not None and volatility > 0:
            # Lower volatility = larger position, higher volatility = smaller position
            # This is a simple risk parity formula: position_size ~ 1/volatility
            risk_parity_factor = (
                0.02  # Target risk per trade (2% of capital, can be tuned)
            )
            adjusted_size = min(
                adjusted_size,
                (self.total_capital * risk_parity_factor) / volatility * price,
            )
            # This ensures that for high-volatility stocks, we take smaller positions

        # === AVAILABLE CASH CHECK ===
        if available_cash is not None:
            adjusted_size = min(adjusted_size, available_cash)
            # Prevents over-allocating if cash is limited

        # === SHARE CALCULATION ===
        if allow_fractional:
            shares = adjusted_size / price
        else:
            shares = int(adjusted_size / price)

        # === MINIMUM SHARES ENFORCEMENT ===
        if shares > 0 and shares < min_shares:
            shares = min_shares if not allow_fractional else float(min_shares)

        # === FINAL SAFETY CHECK ===
        if shares <= 0:
            self.logger.warning(
                f"[PositionSizerBot] Position size is 0 for {asset_type or 'unknown'} (reason: insufficient capital/confidence/price too high). Inputs: price={price}, confidence={confidence}, available_cash={available_cash}"
            )
            shares = 0
        else:
            self.logger.info(
                f"[PositionSizerBot] Calculated position size for {asset_type or 'unknown'}: {shares}"
            )
        self.logger.debug(
            f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}(price={price}, confidence={confidence}, available_cash={available_cash}, min_shares={min_shares})] END"
        )
        return shares

    @staticmethod
    def selftest_position_sizer_bot():
        """
        Run a self-test to verify position sizing logic. Prints test results.
        """
        print("\n--- Running PositionSizerBot Self-Test ---")
        try:
            bot = PositionSizerBot(
                total_capital=10000, max_position_size=0.1
            )  # $10,000, max 10% per trade
            # Test 1: Normal case
            shares = bot.calculate_position_size(price=100, confidence=1.0)
            expected = int(10000 * 0.1 / 100)
            assert shares == expected, f"Expected {expected} shares, got {shares}"
            print(f"    -> Normal sizing logic passed: {shares} shares.")
            # Test 2: Low confidence
            shares = bot.calculate_position_size(price=100, confidence=0.2)
            expected = int(10000 * 0.1 * 0.2 / 100)
            assert shares == expected, f"Expected {expected} shares, got {shares}"
            print(f"    -> Low confidence sizing logic passed: {shares} shares.")
            # Test 3: High price, insufficient capital
            shares = bot.calculate_position_size(price=20000, confidence=1.0)
            assert shares == 0, f"Expected 0 shares, got {shares}"
            print("    -> High price/insufficient capital logic passed.")
            # Test 4: allow_fractional=True
            shares = bot.calculate_position_size(
                price=333, confidence=0.5, allow_fractional=True
            )
            expected = (10000 * 0.1 * 0.5) / 333
            assert (
                abs(shares - expected) < 1e-6
            ), f"Expected {expected} shares, got {shares}"
            print(f"    -> Fractional share sizing logic passed: {shares:.4f} shares.")
            # Test 5: Zero confidence
            shares = bot.calculate_position_size(price=100, confidence=0.0)
            assert shares == 0, f"Expected 0 shares for zero confidence, got {shares}"
            print("    -> Zero confidence logic passed.")
            print("--- PositionSizerBot Self-Test PASSED ---")
        except AssertionError as e:
            print(f"--- PositionSizerBot Self-Test FAILED: {e} ---")
        except Exception as e:
            print(f"--- PositionSizerBot Self-Test encountered an ERROR: {e} ---")


# === Usage Example ===
if __name__ == "__main__":
    sizer = PositionSizerBot(total_capital=10000, max_position_size=0.02)
    # Calculate position size for a $100 stock, 90% confidence, low volatility
    print(
        "Shares to buy:",
        sizer.calculate_position_size(
            price=100,
            confidence=0.9,
            volatility=0.05,
            available_cash=500,
            min_shares=1,
            allow_fractional=False,
        ),
    )
    PositionSizerBot.selftest_position_sizer_bot()
