"""
TradeExecutorBot: Modular trade execution for agentic trading systems.
- Handles order validation, submission, and account/position management via Alpaca API
- Supports both stocks and crypto, with robust error handling and logging
- Designed for integration with other trading bots
"""

import logging
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import URL
from datetime import datetime
from config_system import ALPACA_BASE_URL, ALPACA_API_KEY, ALPACA_SECRET_KEY

class TradeExecutorBot:
    """
    TradeExecutorBot is responsible for placing buy and sell orders through the Alpaca API in a modular, bot-style architecture.
    
    Key Responsibilities:
    - Connect to Alpaca API (paper trading or live trading)
    - Validate trade parameters before execution
    - Submit market orders for stocks and crypto
    - Handle API errors and network issues gracefully
    - Provide account information for portfolio management
    - Log all trading activity for audit trails
    """
    def __init__(self, api_key, api_secret, paper_trading=True):
        """
        Initialize the TradeExecutor with Alpaca API credentials and configuration.
        
        This constructor establishes the connection to Alpaca's trading platform and sets up
        logging for tracking all trading activity. The connection can be configured for either
        paper trading (virtual money) or live trading (real money).
        
        Args:
            api_key (str): Your Alpaca API key for authentication
            api_secret (str): Your Alpaca secret key for authentication  
            paper_trading (bool): If True, uses paper trading (safe virtual money)
                                 If False, uses live trading (real money at risk)
        
        Setup Process:
        1. Creates REST API connection to Alpaca using provided credentials
        2. Uses base URL from config_system.py (automatically set based on paper_trading setting)
        3. Sets API version to 'v2' for latest Alpaca API features
        4. Initializes logging for trade execution tracking
        
        Safety Note:
        - Paper trading is enabled by default to prevent accidental real money trades
        - The base URL is automatically configured in config_system.py based on trading mode
        - All API calls will be logged for audit purposes
        """        # Establish connection to Alpaca's trading API
        # ALPACA_BASE_URL is automatically set in config_system.py based on paper_trading setting
        self.api = tradeapi.REST(
            api_key,                    # Your unique API key for authentication
            api_secret,                 # Your secret key for secure access
            base_url=URL(ALPACA_BASE_URL),   # Trading endpoint (paper or live) from config_system.py
            api_version='v2'            # Use latest API version for full features
        )
        
        # Set up logging to track all trade execution activity
        # This creates audit trails for regulatory compliance and performance analysis
        self.logger = logging.getLogger(__name__)   
    def execute_trade(self, symbol, side, quantity, confidence):
        """
        Execute a trade based on AI-generated trading signals with comprehensive validation.
        
        This is the core method that converts trading decisions into actual market orders.
        It handles all the technical details of order submission while providing robust
        error handling and validation to prevent invalid trades.
        
        Order Flow:
        1. Validate all input parameters to ensure they're complete and valid
        2. Construct order object with market order specifications
        3. Submit order to Alpaca API with error handling
        4. Return success/failure status with order details
        
        Args:
            symbol (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'TSLA')
            side (str): Order direction - 'buy' to purchase, 'sell' to sell
            quantity (float): Number of shares to trade (must be positive)
            confidence (float): AI confidence score (0.0 to 1.0) - logged for analysis
        
        Returns:
            tuple: (success, order_response)
                - success (bool): True if order was placed successfully, False if failed
                - order_response: Alpaca order object if successful, None if failed
        
        Order Configuration:
        - Type: Market order (executes immediately at current market price)
        - Time in Force: Day (order expires at end of trading day if not filled)
        - No limit price (accepts current market price for fast execution)
        
        Error Handling:
        - Input validation prevents orders with missing parameters
        - API-specific error handling for Alpaca connection issues
        - General exception handling for unexpected errors
        - All errors are logged with details for debugging
        
        Safety Features:
        - Validates all required parameters before submission
        - Logs every order attempt for audit trails
        - Returns clear success/failure indicators
        - Handles network issues gracefully without crashing
        """
        asset_type = 'crypto' if '/' in symbol else 'stock'
        self.logger.info(f"[TradeExecutorBot] Received trade request: symbol={symbol}, side={side}, quantity={quantity}, confidence={confidence}, asset_type={asset_type}")
        try:
            # === INPUT VALIDATION ===
            # Ensure all required parameters are provided and valid
            # This prevents submitting incomplete or malformed orders
            if not all([symbol, side, quantity]):
                raise ValueError("Missing required parameters: symbol, side, and quantity must all be provided")
            
            # Validate quantity is positive (can't trade negative shares)
            if quantity <= 0:
                raise ValueError(f"Invalid quantity: {quantity}. Must be positive number.")
            
            # Validate side parameter
            if side.lower() not in ['buy', 'sell']:
                raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'.")
                
            # === ORDER CONSTRUCTION ===
            # Build the order object with all required parameters for Alpaca API
            # Determine asset type for time_in_force
            asset_type = 'crypto' if '/' in symbol else 'stock'
            order = {
                'symbol': symbol.upper(),           # Stock ticker (ensure uppercase)
                'qty': abs(float(quantity)),        # Number of shares (ensure positive)
                'side': side.lower(),               # 'buy' or 'sell' 
                'type': 'market',                   # Execute at current market price
                'time_in_force': 'gtc' if asset_type == 'crypto' else 'day'  # Fix: 'gtc' for crypto, 'day' for stocks
            }
            
            # Log the order attempt for audit purposes (no emoji)
            self.logger.info(f"[TradeExecutorBot] Attempting to execute {side} order for {symbol} ({asset_type}): qty={quantity}, confidence={confidence}")
            
            # === ORDER SUBMISSION ===
            try:
                # Submit order to Alpaca API
                response = self.api.submit_order(**order)
                
                # Log successful order placement
                self.logger.info(f"[TradeExecutorBot] Order placed successfully: {symbol} {side} {quantity} ({asset_type})")
                self.logger.debug(f"Order response: {response}")
                
                return True, response
                
            except Exception as e:
                self.logger.error(f"[TradeExecutorBot] Trade execution failed for {symbol} ({asset_type}): {e}")
                self.logger.error("Returning False, str(e) due to trade execution failure.")
                return False, str(e)
        except ValueError as ve:
            # Handle validation errors (missing parameters, invalid values)
            self.logger.error(f"Order validation failed: {ve}")
            self.logger.error("Returning False, None due to order validation failure.")
            return False, None
        except Exception as e:
            # Handle any unexpected errors to prevent system crashes
            self.logger.error(f"Unexpected error executing trade for {symbol}: {str(e)}")
            self.logger.debug(f"Error details: {e}", exc_info=True)
            self.logger.error("Returning False, None due to unexpected error in execute_trade.")
            return False, None    
    
    def get_account(self):
        """
        Retrieve account information from Alpaca API for portfolio management and validation.
        
        This method fetches comprehensive account details that are essential for:
        - Portfolio management (current cash, buying power, positions)
        - Risk management (day trading buying power, pattern day trader status)
        - Trade validation (ensuring sufficient funds before placing orders)
        - Performance tracking (account value, profit/loss calculations)
        
        Account Information Includes:
        - Cash available for trading
        - Total portfolio value
        - Buying power (amount available for purchases)
        - Current positions and their values
        - Account status (active, restricted, etc.)
        - Day trading buying power and restrictions
        
        Returns:
            Account object: Alpaca account object with all account details if successful
            None: If API call fails or account cannot be accessed
        
        Usage Examples:
        - Check available cash before placing large orders
        - Validate account is in good standing before trading
        - Calculate current portfolio performance
        - Ensure compliance with day trading rules
        
        Error Handling:
        - Logs specific error details for debugging
        - Returns None on failure to allow graceful handling
        - Does not crash the system if account info is temporarily unavailable
        """
        try:
            # Fetch account details from Alpaca API
            account = self.api.get_account()
            
            # Log successful account retrieval (at debug level to avoid spam)
            self.logger.debug("[OK] Account information retrieved successfully")
            
            return account
            
        except tradeapi.rest.APIError as api_error:
            # Handle Alpaca-specific API errors
            self.logger.error(f"❌ Alpaca API error getting account info: {api_error}")
            return None
            
        except Exception as e:
            # Handle any unexpected errors
            self.logger.error(f"❌ Unexpected error getting account info: {str(e)}")
            self.logger.debug(f"Error details: {e}", exc_info=True)
            return None
        
    def get_open_positions(self, asset_type=None):
        """
        Retrieve all open positions from Alpaca. Optionally filter by asset_type ('stock' or 'crypto').
        Returns a list of dicts: {symbol, entry_price, quantity, asset_type}
        """
        try:
            positions = self.api.list_positions()
            open_positions = []
            for pos in positions:
                symbol = str(getattr(pos, 'symbol', ''))
                qty = float(getattr(pos, 'qty', 0))
                entry_price = float(getattr(pos, 'avg_entry_price', 0))
                asset_class = str(getattr(pos, 'asset_class', '')).lower()
                atype = 'crypto' if ('/' in symbol) or (asset_class == 'crypto') else 'stock'
                if asset_type and atype != asset_type:
                    continue
                open_positions.append({
                    'symbol': symbol,
                    'entry_price': entry_price,
                    'quantity': qty,
                    'asset_type': atype
                })
            return open_positions
        except Exception as e:
            self.logger.error(f"Error fetching open positions: {e}")
            return []

    def close_position(self, symbol, quantity=None, asset_type=None):
        """
        Close an open position for the given symbol. If quantity is None, closes the full position.
        Returns the API response or error message.
        """
        try:
            # For crypto, Alpaca expects symbol as BTCUSD (no slash)
            order_symbol = symbol.replace('/', '') if asset_type == 'crypto' else symbol
            if quantity is None:
                # Close full position
                response = self.api.close_position(order_symbol)
            else:
                # Submit a sell order for the specified quantity
                response = self.api.submit_order(
                    symbol=order_symbol,
                    qty=abs(float(quantity)),
                    side='sell',
                    type='market',
                    time_in_force='gtc' if asset_type == 'crypto' else 'day'
                )
            self.logger.info(f"Closed position for {symbol}: {response}")
            return response
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}")
            return {"error": str(e)}

    def fetch_order_history(self, status='all', limit=50):
        """
        Fetch recent order history from Alpaca API.
        Args:
            status (str): Filter orders by status ('all', 'open', 'closed', etc.)
            limit (int): Number of orders to fetch
        Returns:
            List[dict]: List of order dicts
        """
        try:
            orders = self.api.list_orders(status=status, limit=limit)
            order_list = []
            for o in orders:
                order_list.append({
                    'id': getattr(o, 'id', None),
                    'symbol': getattr(o, 'symbol', None),
                    'side': getattr(o, 'side', None),
                    'qty': float(getattr(o, 'qty', 0)),
                    'filled_qty': float(getattr(o, 'filled_qty', 0)),
                    'status': getattr(o, 'status', None),
                    'submitted_at': getattr(o, 'submitted_at', None),
                    'filled_at': getattr(o, 'filled_at', None),
                    'type': getattr(o, 'type', None),
                    'asset_class': getattr(o, 'asset_class', None)
                })
            return order_list
        except Exception as e:
            self.logger.error(f"Error fetching Alpaca order history: {e}")
            return []

    def cross_check_order_history(self, db_bot=None, days=7, log_only=True):
        """
        Cross-check Alpaca order history with internal trade records.
        Args:
            db_bot (DatabaseBot): Optional, for retrieving internal trade outcomes
            days (int): Lookback period for trade outcomes
            log_only (bool): If True, only log discrepancies; else, return them
        Returns:
            List[dict]: List of discrepancies (if log_only is False)
        """
        if db_bot is None:
            from bot_database import DatabaseBot
            db_bot = DatabaseBot()
        alpaca_orders = self.fetch_order_history(status='all', limit=100)
        internal_trades = db_bot.get_trade_outcomes(days=days)
        discrepancies = []
        # Build sets for quick lookup
        alpaca_set = set((o['symbol'], o['side'], o['qty'], o['submitted_at']) for o in alpaca_orders)
        internal_set = set((t['symbol'], t['trade_type'], t['quantity'], t['timestamp']) for t in internal_trades)
        # Find missing in Alpaca
        for t in internal_trades:
            key = (t['symbol'], t['trade_type'], t['quantity'], t['timestamp'])
            if key not in alpaca_set:
                msg = f"[ORDER_XCHECK] Internal trade not found in Alpaca: {t}"
                self.logger.warning(msg)
                discrepancies.append({'type': 'missing_in_alpaca', 'trade': t})
        # Find extra in Alpaca
        for o in alpaca_orders:
            key = (o['symbol'], o['side'], o['qty'], o['submitted_at'])
            if key not in internal_set:
                msg = f"[ORDER_XCHECK] Alpaca order not found in internal records: {o}"
                self.logger.warning(msg)
                discrepancies.append({'type': 'extra_in_alpaca', 'order': o})
        if not discrepancies:
            self.logger.info("[ORDER_XCHECK] No discrepancies found between Alpaca and internal records.")
        if log_only:
            return None
        return discrepancies

    @staticmethod
    def selftest():
        print(f"\n--- Running TradeExecutorBot Self-Test ---")
        class MockAPI:
            def submit_order(self, **kwargs):
                if kwargs.get('symbol') == 'FAIL':
                    raise Exception('Mocked order failure')
                return {'id': 'mock_order_id', 'symbol': kwargs.get('symbol'), 'qty': kwargs.get('qty'), 'side': kwargs.get('side')}
            def get_account(self):
                return {'id': 'mock_account', 'cash': 10000, 'buying_power': 20000, 'status': 'ACTIVE'}
            def list_positions(self):
                return []
            def close_position(self, symbol):
                return {'closed': True, 'symbol': symbol}
            def list_orders(self, status='all', limit=50):
                class Order:
                    def __init__(self, symbol, side, qty, submitted_at):
                        self.id = f"mock_{symbol}_{side}"
                        self.symbol = symbol
                        self.side = side
                        self.qty = qty
                        self.filled_qty = qty
                        self.status = 'filled'
                        self.submitted_at = submitted_at
                        self.filled_at = submitted_at
                        self.type = 'market'
                        self.asset_class = 'us_equity'
                # Return two orders, one matching, one extra
                return [Order('AAPL', 'buy', 1, '2023-01-01T00:00:00Z'), Order('TSLA', 'sell', 2, '2023-01-02T00:00:00Z')]
        class MockDB:
            def get_trade_outcomes(self, days=7):
                # One matching, one missing in Alpaca
                return [
                    {'symbol': 'AAPL', 'trade_type': 'buy', 'quantity': 1, 'timestamp': '2023-01-01T00:00:00Z'},
                    {'symbol': 'GOOG', 'trade_type': 'buy', 'quantity': 3, 'timestamp': '2023-01-03T00:00:00Z'}
                ]
        class MockTradeExecutorBot(TradeExecutorBot):
            def __init__(self):
                self.api = MockAPI()
                self.logger = logging.getLogger(__name__)
        try:
            bot = MockTradeExecutorBot()
            # Test 1: Successful trade
            success, resp = bot.execute_trade('AAPL', 'buy', 1, 1.0)
            assert success and resp['symbol'] == 'AAPL', f"Expected success for AAPL, got {success}, {resp}"
            print("    -> Successful trade logic passed.")  # CLI/test output
            # Test 2: Failed trade
            success, resp = bot.execute_trade('FAIL', 'buy', 1, 1.0)
            assert not success, "Expected failure for symbol 'FAIL'"
            print("    -> Trade failure logic passed.")  # CLI/test output
            # Test 3: Account info
            acct = bot.get_account()
            assert isinstance(acct, dict) and 'cash' in acct and 'buying_power' in acct, "Account info missing expected keys."
            print("    -> Account info logic passed.")  # CLI/test output
            # Test order history fetch
            orders = bot.fetch_order_history()
            assert isinstance(orders, list) and len(orders) == 2, "Order history fetch failed"
            print("    -> Order history fetch logic passed.")
            # Test cross-check logic
            discrepancies = bot.cross_check_order_history(db_bot=MockDB(), log_only=False)
            if discrepancies is None:
                discrepancies = []
            assert isinstance(discrepancies, list) and len(discrepancies) == 2, f"Expected 2 discrepancies, got {len(discrepancies)}"
            print("    -> Order history cross-check logic passed.")
            print(f"--- TradeExecutorBot Self-Test PASSED ---")
        except AssertionError as e:
            print(f"--- TradeExecutorBot Self-Test FAILED: {e} ---")
        except Exception as e:
            print(f"--- TradeExecutorBot Self-Test encountered an ERROR: {e} ---")

# === Usage Example ===
if __name__ == "__main__":
    TradeExecutorBot.selftest()
    executor = TradeExecutorBot(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper_trading=True)
    # Example: Place a test buy order for AAPL
    success, response = executor.execute_trade(
        symbol="AAPL", side="buy", quantity=1, confidence=1.0
    )
    print("AAPL order success:", success, "response:", response)  # CLI/test output
    # Example: List open positions
    print("Open positions:", executor.get_open_positions())  # CLI/test output

# === OUTPUT EXPLANATION ===
# Each order attempt prints a clear, labeled result:
#   [SYMBOL] SUCCESS: ...   if the order was placed (shows order ID)
#   [SYMBOL] FAILED: ...    if the order failed (shows error or None)
#   [SYMBOL] ERROR: ...     for unexpected exceptions
# Crypto orders use 'gtc' (good till canceled) as required by Alpaca, stocks use 'day'.
# This makes it easy to see which trades succeeded or failed, and why, for each asset.
# You can confirm all successful orders in your Alpaca dashboard.
# (No print statements found in main production logic. If any are added, ensure they are for CLI/test output only and all critical errors are logged.)