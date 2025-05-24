class PositionSizer:
    def __init__(self, total_capital, max_position_size=0.02):
        self.total_capital = total_capital
        self.max_position_size = max_position_size

    def calculate_position_size(self, price, confidence, volatility=None):
        """Calculate position size based on capital and confidence"""
        base_size = self.total_capital * self.max_position_size
        # Adjust size based on confidence
        adjusted_size = base_size * confidence
        # Calculate number of shares
        shares = int(adjusted_size / price)
        return shares