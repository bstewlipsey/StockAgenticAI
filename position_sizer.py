from trading_variables import TOTAL_CAPITAL, MAX_POSITION_SIZE

class PositionSizer:
    """
    The PositionSizer determines the optimal number of shares to buy or sell for each trade.

    This class is responsible for translating AI trading signals and risk parameters into concrete
    position sizes, ensuring that each trade is appropriately scaled for both opportunity and risk.
    It supports advanced sizing logic including confidence weighting, volatility adjustment, and
    available capital checks, making it a robust component for real-world trading systems.

    Key Responsibilities:
    - Calculate position size based on total capital and max position size constraints
    - Adjust position size according to AI confidence in the trade signal
    - Optionally reduce position size for high-volatility stocks (risk parity logic)
    - Enforce minimum shares per trade and support fractional shares if needed
    - Cap position size by available cash to prevent over-allocation
    - Provide clear, auditable sizing logic for risk management and compliance

    Safety Features:
    - Prevents negative or zero share trades
    - Avoids over-allocation by checking available cash
    - Supports both integer and fractional share brokers
    - All logic is fully documented for transparency
    """
    def __init__(self, total_capital=TOTAL_CAPITAL, max_position_size=MAX_POSITION_SIZE):
        """
        Initialize the PositionSizer with portfolio capital and risk settings.

        Args:
            total_capital (float): Total portfolio capital available for trading
            max_position_size (float): Maximum fraction of capital to allocate to a single trade
        """
        self.total_capital = total_capital
        self.max_position_size = max_position_size

    def calculate_position_size(self, price, confidence, volatility=None, available_cash=None, min_shares=1, allow_fractional=False):
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
            risk_parity_factor = 0.02  # Target risk per trade (2% of capital, can be tuned)
            adjusted_size = min(adjusted_size, (self.total_capital * risk_parity_factor) / volatility * price)
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
            shares = 0
        return shares