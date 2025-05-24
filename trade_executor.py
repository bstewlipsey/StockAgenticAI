import logging
import alpaca_trade_api as tradeapi
from datetime import datetime

class TradeExecutor:
    def __init__(self, api_key, api_secret, paper_trading=True):
        self.api = tradeapi.REST(
            api_key, 
            api_secret, 
            base_url='https://paper-api.alpaca.markets' if paper_trading else 'https://api.alpaca.markets',
            api_version='v2'
        )
        self.logger = logging.getLogger(__name__)

    def execute_trade(self, symbol, side, quantity, confidence):
        """Execute a trade based on signal"""
        try:
            # Validate inputs
            if not all([symbol, side, quantity]):
                raise ValueError("Missing required parameters")
                
            order = {
                'symbol': symbol,
                'qty': quantity,
                'side': side.lower(),
                'type': 'market',
                'time_in_force': 'day'
            }
            
            # Submit order with error handling
            try:
                response = self.api.submit_order(**order)
                self.logger.info(f"Order placed successfully: {order}")
                return True, response
            except tradeapi.rest.APIError as api_error:
                self.logger.error(f"Alpaca API error: {api_error}")
                return False, None
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            return False, None

    def get_account(self):
        """Get account information"""
        try:
            return self.api.get_account()
        except Exception as e:
            self.logger.error(f"Error getting account info: {str(e)}")
            return None