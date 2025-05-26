# === Standard Library Imports ===
import json # For handling JSON data
import logging
import time
import random # For jitter in retry strategy
import gc # For garbage collection
from datetime import datetime, timedelta

# === Third-Party Library Imports ===
import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation
from colorama import init, Fore, Style  # Colored terminal output

import google.generativeai as genai  # Gemini AI API
from google.generativeai.types import GenerationConfig # Configuration for AI generation
import alpaca_trade_api as tradeapi  # Alpaca trading API
from alpaca_trade_api import REST
import ccxt  # Crypto exchange API

# === Project Module Imports ===
from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, GEMINI_API_KEY, PAPER_TRADING, # API Keys & Settings
    GEMINI_MODEL, MAX_TOKENS, TEMPERATURE,                               # LLM settings
    DEFAULT_TIMEFRAME, LOOKBACK_PERIOD,                  # Market data settings
    MAX_RETRIES, RETRY_DELAY, RATE_LIMIT_DELAY_SECONDS,  # System settings
    ERROR_RETRY_DELAY, BASE_RETRY_DELAY, MAX_RETRY_BACKOFF_DELAY, # Retry strategy settings
    JITTER_DELAY,
    TRADING_CYCLE_INTERVAL                              # Add this line to import the interval
)
from database import Database
from indicators import TechnicalAnalysis
from risk_manager import RiskManager, Position
from portfolio_manager import PortfolioManager
from visualizer import SignalVisualizer
from trade_executor import TradeExecutor
from position_sizer import PositionSizer
from trading_variables import (
    TOTAL_CAPITAL,
    MAX_PORTFOLIO_RISK,
    MAX_POSITION_SIZE,
    MAX_POSITION_RISK,
    STOCK_STOP_LOSS_PCT,      # Import new stock stop loss
    CRYPTO_STOP_LOSS_PCT,     # Import new crypto stop loss
    MIN_CONFIDENCE,
    TRADING_ASSETS,
    RSI_OVERSOLD,
    RSI_OVERBOUGHT,
    SMA_WINDOW,
    # Analysis templates
    ANALYSIS_SCHEMA,
    CRYPTO_ANALYSIS_TEMPLATE,
    STOCK_ANALYSIS_TEMPLATE
)

# Initialize colorama for colored output and configure basic logging
init()
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__) # General logger

# === Metrics Logger Setup ===
metrics_logger = logging.getLogger('metrics')
metrics_logger.setLevel(logging.INFO)
metrics_file_handler = logging.FileHandler('portfolio_metrics.log', mode='a') # Append mode
metrics_file_handler.setFormatter(logging.Formatter('%(asctime)s,%(message)s')) # CSV-like
metrics_logger.addHandler(metrics_file_handler)
metrics_logger.propagate = False # Prevent metrics from going to root logger's console output




# === API & Service Initialization Section ===

# 1. Initialize Gemini AI API (for AI-powered analysis)
try:
    # Configure Gemini API with your API key
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini API configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    raise

# 2. Initialize Alpaca API (for stock trading)
try:
    # Use Alpaca's paper trading endpoint for safety
    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)
    logger.info("Alpaca API initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Alpaca API: {e}")
    raise

# 3. Initialize Crypto Exchange API (using ccxt for Kraken)
try:
    # Enable rate limiting to avoid hitting API limits
    crypto_exchange = ccxt.kraken({
        'enableRateLimit': True,
        'rateLimit': RATE_LIMIT_DELAY_SECONDS * 1000  # Convert seconds to milliseconds for ccxt
    })
    logger.info("Kraken (ccxt) crypto exchange initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize crypto exchange: {e}")
    raise

# 4. Initialize Gemini AI Model (for generating trading analysis)
try:
    # Use the configured Gemini model
    model = genai.GenerativeModel(GEMINI_MODEL)
    logger.info("Gemini model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Gemini model: {e}")
    raise

# 5. Initialize Database (for storing analysis and trade history)
try:
    db = Database()
    logger.info("Database initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")
    raise

# 6. Initialize Risk Manager (for position risk assessment)
try:
    risk_manager = RiskManager()
    logger.info("Risk manager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize risk manager: {e}")
    raise

# 7. Initialize Portfolio Manager (for tracking overall portfolio performance)
try:
    portfolio = PortfolioManager()
    logger.info("Portfolio manager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize portfolio manager: {e}")
    raise




class TradingAgent:
    def __init__(self):
        """
        Initialize the TradingAgent with configuration, risk, execution, and database components.
        """
        # === Trading Parameters ===
        # self.stop_loss_pct = STOP_LOSS_PCT       # Replaced by asset-specific stops
        self.stock_stop_loss_pct = STOCK_STOP_LOSS_PCT
        self.crypto_stop_loss_pct = CRYPTO_STOP_LOSS_PCT
        self.min_confidence = MIN_CONFIDENCE      # Minimum confidence from variables

        # === Core Trading Components ===
        self.executor = TradeExecutor(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper_trading=PAPER_TRADING)  # Handles trade execution
        self.sizer = PositionSizer(total_capital=TOTAL_CAPITAL)           # Determines position sizes
        self.db = Database()                                              # Handles analysis and trade history storage
        # Create a lookup map for asset types for quick access in monitor_positions
        self.asset_type_map = {symbol: asset_type for symbol, asset_type in TRADING_ASSETS}
        
    def get_current_price(self, symbol, asset_type):
        """
        Get the current price for a given asset (stock or crypto) with retry logic and validation.

        Args:
            symbol (str): The asset symbol (e.g., 'AAPL' or 'XXBTZUSD').
            asset_type (str): The type of asset ('stock' or 'crypto').

        Returns:
            float or None: The current price if successful, otherwise None.
        """
        max_retries = MAX_RETRIES      # Maximum number of retry attempts (from config.py)
        retry_delay = RETRY_DELAY      # Delay (in seconds) between retries (from config.py)

        for attempt in range(max_retries):
            try:
                if asset_type == 'crypto':
                    # Fetch current price for cryptocurrency
                    data = get_crypto_data(symbol)
                    if data and 'current_price' in data:
                        return data['current_price']
                else:
                    # Fetch current price for stock
                    quote = api.get_last_quote(symbol)
                    if quote and hasattr(quote, 'ap'):
                        return float(quote.ap)

                    logger.warning(f"Attempt {attempt + 1}: Invalid response for {symbol}")
                    time.sleep(retry_delay)
            except Exception as e:
                logger.error(f"Error getting price for {symbol} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    
        # Return None if all attempts fail
        return None
    
    def process_trading_signals(self, symbol, signals, indicators, current_price):
        """
        Process trading signals and execute trades based on risk assessment and per-asset USD allocation.

        Args:
            symbol (str): The trading symbol (e.g., 'AAPL', 'XXBTZUSD')
            signals (List[Tuple[str, str, float]]): List of (signal, reason, confidence) tuples
            indicators (dict): Technical indicators for the symbol
            current_price (float): Current market price of the asset

        Returns:
            bool: True if any trades were executed, False otherwise

        Strategy:
        1. For each signal:
            - Check if risk limits allow for a new position
            - Calculate position size based on per-asset allocation_usd (from TRADING_ASSETS)
            - Execute trade if conditions are met
        2. Uses risk management to protect capital
        3. Implements stop-loss and risk controls
        """
        if not signals:
            logger.debug(f"No signals to process for {symbol}")
            return False

        trades_executed = False
        significant_indicators = {
            'RSI': indicators.get('rsi'),
            'MACD': indicators.get('macd'),
            'SMA20': indicators.get('sma_20')
        }

        # === Get allocation_usd for this symbol from TRADING_ASSETS ===
        allocation_usd = None
        asset_type = None
        for asset in TRADING_ASSETS:
            if asset[0] == symbol:
                asset_type = asset[1]
                allocation_usd = asset[2] if len(asset) > 2 else None
                break
        if allocation_usd is None or allocation_usd == 0:
            from trading_variables import DEFAULT_TRADE_AMOUNT_USD
            allocation_usd = DEFAULT_TRADE_AMOUNT_USD

        for signal, reason, confidence in signals:
            # Skip if confidence is below minimum threshold
            if confidence < self.min_confidence:
                logger.debug(f"Signal confidence {confidence} below minimum threshold for {symbol}")
                continue

            # Validate signal with technical indicators
            signal_confirmed = False
            if signal.upper() == 'BUY':
                # Confirm buy signals:
                # - RSI below 70 (not overbought)
                # - MACD positive or crossing up
                # - Price above SMA20 (uptrend)
                if (significant_indicators['RSI'] < RSI_OVERBOUGHT and 
                    significant_indicators['MACD'] > 0 and 
                    current_price > significant_indicators['SMA20']):
                    signal_confirmed = True
            elif signal.upper() == 'SELL':
                # Confirm sell signals:
                # - RSI above 30 (not oversold)
                # - MACD negative or crossing down
                # - Price below SMA20 (downtrend)
                if (significant_indicators['RSI'] > RSI_OVERSOLD and 
                    significant_indicators['MACD'] < 0 and 
                    current_price < significant_indicators['SMA20']):
                    signal_confirmed = True

            if not signal_confirmed:
                logger.debug(f"Signal {signal} not confirmed by technical indicators for {symbol}")
                continue

            # Check risk before executing trade
            risk_metrics = risk_manager.calculate_position_risk(Position(
                symbol=symbol,
                quantity=0,  # Hypothetical new position for risk check
                entry_price=current_price,
                current_price=current_price,
                asset_type=asset_type if asset_type else ('stock' if 'ZUSD' not in symbol else 'crypto')
            ))

            # Skip trade if risk is too high
            if risk_metrics['risk_level'] == 'HIGH':
                logger.warning(f"Risk too high for {signal} on {symbol} - skipping trade")
                continue

            # === Calculate quantity based on allocation_usd and current price ===
            qty = allocation_usd / current_price if current_price > 0 else 0
            if asset_type == 'crypto':
                qty = round(qty, 6)  # Allow fractional for crypto
            else:
                qty = int(qty)       # Whole shares for stocks
            if qty <= 0:
                logger.debug(f"Allocation (${allocation_usd}) too small for current price (${current_price:.2f}) on {symbol}")
                continue

            # Execute the trade
            success, order = self.executor.execute_trade(
                symbol=symbol,
                side=signal,
                quantity=qty,
                confidence=confidence
            )
            if success:
                trades_executed = True
                # Log basic trade info at INFO level
                logger.info(f"Trade executed for {symbol}: {signal} {qty} @ ${current_price:.2f}")
                # Log detailed analysis at DEBUG level
                logger.debug(f"Trade details - Confidence: {confidence:.2f}, Indicators: {significant_indicators}, Allocation: ${allocation_usd}")
                print(f"Trade executed: {signal} {qty} {symbol} (USD Alloc: ${allocation_usd})")

        return trades_executed  # Return after processing all signals    
    
    def run_trading_cycle(self, assets):
        """
        Execute a single complete trading cycle for a list of assets.

        Args:
            assets (List[Tuple[str, str]]): List of tuples containing (symbol, asset_type)
                Example: [("AAPL", "stock"), ("XXBTZUSD", "crypto")]
                - symbol: The trading symbol (e.g., 'AAPL' or 'XXBTZUSD')
                - asset_type: Type of asset ('stock' or 'crypto')

        Returns:
            List[Tuple[str, dict]]: List of tuples containing (symbol, analysis)
                - symbol: The analyzed symbol
                - analysis: Dictionary containing:
                    - action: 'buy', 'sell', or 'hold'
                    - reasoning: Brief explanation
                    - confidence: Float between 0 and 1

        Strategy:
        1. For each asset in the list:
            - Print analysis status
            - Get asset analysis using AI and technical indicators
            - Get current market price
            - If analysis is valid and price is available:
                - Save analysis to database
                - Add to results list
        2. Skip assets where:
            - Analysis returns error
            - Unable to get current price
        3. Uses retry and error handling from get_current_price

        Note: This is a high-level coordination method that delegates the actual
        analysis and trading logic to specialized components.
        """
        results = []
        
        for symbol, asset_type in assets:
            print(f"\n📊 Analyzing {symbol} ({asset_type.upper()})...")
            # Analyze the asset to get trading recommendations
            analysis = analyze_asset(symbol, asset_type)
            
            if isinstance(analysis, dict) and "error" not in analysis:
                current_price = self.get_current_price(symbol, asset_type)
                if current_price:
                    self.db.save_analysis(symbol, asset_type, analysis, current_price)
                    results.append((symbol, analysis))
                    
        return results

    def monitor_positions(self):
        """
        Monitor all open positions and enforce stop loss rules to manage risk.
        
        This method:
        1. Fetches all current open positions from Alpaca
        2. For each position, compares current price to entry price
        3. If price drops below stop loss threshold, triggers a sell order
        4. Helps protect capital by automatically closing losing positions
        
        The stop loss percentage is defined in the class initialization 
        and can be configured through STOP_LOSS_PCT in trading_variables.py
        
        This is a critical risk management function that should run regularly 
        as part of the trading cycle to protect against significant losses.
        """
        try:
            # List all open positions from Alpaca
            positions = api.list_positions()
            for position in positions:
                entry_price = float(position.avg_entry_price)
                current_price = float(position.current_price)
                
                # Determine which stop-loss percentage to use
                asset_type = self.asset_type_map.get(str(position.symbol))
                
                stop_loss_pct_to_use = self.stock_stop_loss_pct # Default to stock
                if asset_type == 'crypto':
                    stop_loss_pct_to_use = self.crypto_stop_loss_pct
                elif asset_type == 'stock':
                    stop_loss_pct_to_use = self.stock_stop_loss_pct
                else:
                    logger.warning(f"Unknown asset type for symbol {position.symbol} in monitor_positions. Defaulting to stock stop-loss.")

                # Check stop loss
                # If current price drops below the stop loss percentage from entry price
                if current_price < entry_price * (1 - stop_loss_pct_to_use):
                    self.executor.execute_trade(
                        symbol=position.symbol,
                        side='sell',
                        quantity=position.qty,
                        confidence=1.0  # High confidence for stop loss
                    )
                    logger.warning(f"Stop loss triggered for {position.symbol} ({asset_type}) at {stop_loss_pct_to_use*100}%")
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")

    def save_analysis(self, symbol, asset_type, analysis, current_price):
        """
        Persist trading analysis results to the database with error handling.
        
        This method is crucial for:
        1. Historical record keeping - stores all analysis results for future reference
        2. Performance tracking - enables calculation of strategy effectiveness
        3. Audit trail - maintains a log of all trading decisions and their rationale
        4. Machine learning - provides data for training and improving the trading model
        
        Args:
            symbol (str): The trading symbol (e.g., 'AAPL', 'XXBTZUSD')
            asset_type (str): Type of asset ('stock' or 'crypto')
            analysis (dict): Trading analysis results containing:
                - action: 'buy', 'sell', or 'hold'
                - reasoning: Brief explanation
                - confidence: Float between 0 and 1
            current_price (float): Current market price when analysis was made
        
        Returns:
            bool: True if save successful, False if error occurred
        
        Note: This data is used by print_performance_summary() to show trading history
        and by the portfolio manager to track overall performance metrics.
        """        
        try:
            timestamp = datetime.now().isoformat()
            analysis_data = {
                'timestamp': timestamp,
                'symbol': symbol,
                'asset_type': asset_type,
                'analysis': analysis,
                'price': current_price
            }
            # Save the properly structured analysis data
            self.db.save_analysis(**analysis_data)
            return True
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")
            # Log additional context if available
            if 'analysis_data' in locals():
                logger.debug(f"Failed analysis data: {analysis_data}")
            return False

    def check_connections(self):
        """
        Verify that all critical external services and APIs are accessible and working.
        
        This method is a crucial system health check that:
        1. Validates Alpaca API connectivity by attempting to fetch account details
        2. Verifies crypto exchange connection by loading market data
        3. Tests database accessibility by attempting to read historical data
        
        The check is performed:
        - At system startup before beginning trading operations
        - After any connection errors to verify system recovery
        - Periodically during the trading cycle
        
        Returns:
            bool: True if all connections are working, False if any connection fails
        
        Note:
            Includes rate limiting and timeout protection to prevent API abuse
        """
        try:
            # Add rate limiting delay to prevent API throttling
            time.sleep(RATE_LIMIT_DELAY_SECONDS)
            
            # Check Alpaca connection by fetching account details
            account = api.get_account()
            if not account:
                raise Exception("Could not connect to Alpaca")
            logger.debug("Alpaca API connection verified")
                
            # Check crypto exchange connection with rate limiting
            time.sleep(crypto_exchange.rateLimit / 1000)
            crypto_exchange.load_markets()
            logger.debug("Crypto exchange connection verified")
            
            # Check database connection by fetching recent analysis
            try:
                # Use the first asset from TRADING_ASSETS as test symbol
                test_symbol = TRADING_ASSETS[0][0] if TRADING_ASSETS else "AAPL"
                self.db.get_analysis_history(test_symbol)
                logger.debug("Database connection verified")
            except Exception as db_exc:
                raise Exception(f"Database connection failed: {db_exc}")
            
            return True
            
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False

def get_crypto_data(symbol):
    """
    Fetch cryptocurrency market data with built-in rate limiting and error handling.
    
    This function is critical for:
    1. Real-time price monitoring - Gets current market prices for trading decisions
    2. Historical analysis - Fetches OHLCV data for technical indicators
    3. Market metrics - Retrieves volume and price changes for trade analysis
    
    The function includes:
    - Rate limiting to prevent API throttling
    - Symbol format conversion for Kraken exchange
    - Automatic retry on rate limit errors
    - Comprehensive error handling
    
    Args:
        symbol (str): Trading symbol (e.g., 'BTC/USD', 'ETH/USD')
                     Will be converted to exchange format (e.g., 'XXBTZUSD')
    
    Returns:
        dict: Market data including:
            - current_price: Latest traded price
            - volume: 24h trading volume
            - price_change: 24h price percentage change
            - high_24h: 24h highest price
            - low_24h: 24h lowest price
            - yesterday_close: Previous day's closing price
        None: If data fetch fails after retries
    """
    try:
        # Add rate limiting delay to prevent API throttling
        time.sleep(crypto_exchange.rateLimit / 1000)  # Convert ms to seconds
        
        # Validate and convert symbol format for Kraken
        if '/' not in symbol:
            logger.warning(f"Invalid symbol format: {symbol}. Expected format: BASE/QUOTE (e.g., BTC/USD)")
            return None
            
        base, quote = symbol.split('/')
        
        # Special cases for currency codes
        currency_mapping = {
            'BTC': 'XBT',  # Bitcoin is XBT on Kraken
            'DOGE': 'XDG', # Dogecoin
            'LUNA': 'LUNA2'  # Terra 2.0
        }
        base = currency_mapping.get(base, base)
        
        # Prefixing rules for Kraken:
        major_cryptos = ['XBT', 'ETH', 'XRP', 'XDG', 'XLM', 'ETC', 'LTC']  # Major cryptos
        fiat_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF']  # Major fiat
        
        # Determine prefixes:
        # 1. Major cryptos and 3/4 letter standard tokens get 'X' prefix
        # 2. Fiat currencies get 'Z' prefix
        # 3. New alt-coins typically don't get a prefix
        base_prefix = 'X' if (base is not None and (base in major_cryptos or (len(base) <= 4 and base.isalpha()))) else ''
        quote_prefix = 'Z' if quote in fiat_currencies else ''
        
        # Construct the final symbol
        symbol = f"{base_prefix}{base}{quote_prefix}{quote}"
        
        # Log the conversion for debugging
        logger.debug(f"Converted {base}/{quote} to Kraken format: {symbol}")
        
        # Fetch current market data
        ticker = crypto_exchange.fetch_ticker(symbol)
        # Get yesterday's closing price from OHLCV data
        ohlcv = crypto_exchange.fetch_ohlcv(symbol, '1d', limit=2)
        
        return {
            'current_price': ticker['last'],  # Most recent trade price
            'volume': ticker['quoteVolume'] if 'quoteVolume' in ticker else ticker['baseVolume'],
            'price_change': ticker['percentage'],  # 24h price change %
            'high_24h': ticker['high'],
            'low_24h': ticker['low'],
            'yesterday_close': ohlcv[0][4] if len(ohlcv) > 1 else None  # Yesterday's close or None
        }
    except ccxt.RateLimitExceeded:
        # Handle rate limiting with exponential backoff
        logger.warning(f"Rate limit hit for {symbol}, waiting 30 seconds...")
        time.sleep(30)
        return get_crypto_data(symbol)  # Recursive retry
    except Exception as e:
        logger.error(f"Error fetching crypto data: {e}")
        return None

def clean_json_response(text):
    """Clean and extract valid JSON from an AI model's response text.
    
    This function is crucial when working with AI models like Gemini that may return JSON
    wrapped in markdown code blocks or with additional context. It ensures we extract only
    the valid JSON portion of the response for further processing.
    
    The function handles these cases:
    1. JSON wrapped in markdown code blocks (```)
    2. JSON mixed with explanatory text
    3. Raw JSON responses
    
    Args:
        text (str): The raw response text from the AI model
        
    Returns:
        str: Cleaned text containing only the JSON portion (everything between { and })
             If no JSON is found, returns the original text
             
    Example:
        >>> response = ```json
        ... {
        ...     "action": "buy",
        ...     "confidence": 0.85
        ... }
        ... ```
        >>> clean_json_response(response)
        '{"action": "buy", "confidence": 0.85}'
    """
    # If response is wrapped in markdown code blocks, extract the content
    if text.startswith('```'):
        # Find the first { and last } to extract just the JSON object
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return text[start:end + 1]
    return text

def validate_analysis(analysis_data):
    """Validate that an AI-generated trading analysis meets all required criteria.
    
    This function is crucial for maintaining data integrity and preventing invalid
    trading decisions. It ensures that the AI's output follows our strict schema
    before any trading actions are taken.
    
    Validation checks:
    1. Required fields presence ('action', 'reasoning', 'confidence')
    2. Action value must be one of: 'buy', 'sell', 'hold'
    3. Confidence must be a number between 0 and 1
    
    Args:
        analysis_data (dict): The cleaned and parsed JSON data from the AI model
            Expected format:
            {
                "action": "buy"|"sell"|"hold",
                "reasoning": "Brief explanation",
                "confidence": 0.85  # float between 0 and 1
            }
    
    Returns:
        tuple: (is_valid, error_message)
            - is_valid (bool): True if all validation checks pass
            - error_message (str): Empty if valid, otherwise describes the error
    
    Usage:
        >>> data = {"action": "buy", "reasoning": "Strong uptrend", "confidence": 0.85}
        >>> is_valid, error = validate_analysis(data)
        >>> if is_valid:
        >>>     execute_trade(data)
        >>> else:
        >>>     log_error(error)
    """
    # 1. Check for required fields
    required_keys = ['action', 'reasoning', 'confidence']
    if not all(key in analysis_data for key in required_keys):
        return False, "Missing required fields"
    
    # 2. Validate action value
    valid_actions = ['buy', 'sell', 'hold']
    if analysis_data['action'] not in valid_actions:
        return False, f"Invalid action value: {analysis_data['action']}"
    
    # 3. Validate confidence value
    if not isinstance(analysis_data['confidence'], (int, float)) or \
       not 0 <= float(analysis_data['confidence']) <= 1:
        return False, "Confidence must be between 0.0 and 1.0"
    
    # All validation checks passed
    return True, ""

def analyze_asset(symbol, asset_type='stock'):
    """Analyze a financial asset (stock or cryptocurrency) using market data and AI predictions.
    
    This function is the core analysis engine of the trading bot that:
    1. Fetches current market data for the asset
    2. Calculates technical indicators (RSI, MACD, SMA)
    3. Uses the Gemini AI model to analyze the data
    4. Validates and returns trading recommendations
    
    The analysis process:
    1. For cryptocurrencies:
        - Gets current price, volume, and 24h changes
        - Calculates technical indicators from historical data
        - Generates AI prompt with comprehensive market context
    2. For stocks:
        - Delegates to analyze_stock() for stock-specific analysis
        - Includes position data and risk metrics
    
    Args:
        symbol (str): Trading symbol (e.g., 'BTC/USD' for crypto, 'AAPL' for stocks)
        asset_type (str, optional): Type of asset to analyze. Defaults to 'stock'.
                                  Valid values: 'stock' or 'crypto'
    
    Returns:
        dict: Analysis results containing:
            - On success: {
                "action": "buy"|"sell"|"hold",
                "reasoning": "Brief explanation",
                "confidence": float (0.0 to 1.0)
            }
            - On failure: {
                "error": "Error message",
                "details": "Additional error context" (optional)
            }
    
    Integration points:
    - Uses get_crypto_data() for cryptocurrency market data
    - Uses TechnicalAnalysis for indicator calculations
    - Uses Gemini AI model for market analysis
    - Uses clean_json_response() and validate_analysis() for response processing
    """    
    try:
        if asset_type == 'crypto':
            # Fetch current market data for cryptocurrency
            data = get_crypto_data(symbol)
            if not data:
                return {"error": "Could not fetch crypto data"}
            
            # Get historical prices for technical analysis
            ohlcv = crypto_exchange.fetch_ohlcv(symbol, '1d', limit=50)
            prices = [candle[4] for candle in ohlcv]  # Using closing prices
            
            # Calculate technical indicators
            tech_analysis = TechnicalAnalysis(prices)
            signals, indicators = tech_analysis.get_signals()
            
            # Format trading signals into readable text
            signal_summary = "\n".join([f"- {signal}: {reason} (Confidence: {confidence*100:.0f}%)" 
                                      for signal, reason, confidence in signals])
            
            # Prepare template variables
            template_vars = {
                'schema': ANALYSIS_SCHEMA,
                'symbol': symbol,
                'current_price': data['current_price'],
                'volume': data['volume'],
                'price_change': data['price_change'],
                'high_24h': data['high_24h'],
                'low_24h': data['low_24h'],
                'rsi': indicators['rsi'],
                'macd': indicators['macd'],
                'sma_20': indicators['sma_20'],
                'sma_diff': abs(data['current_price'] - indicators['sma_20']),
                'sma_diff_prefix': '+' if data['current_price'] > indicators['sma_20'] else '-',
                'signal_summary': signal_summary
            }
            
            # Generate prompt using template
            prompt = CRYPTO_ANALYSIS_TEMPLATE.format(**template_vars)
            # Generate content using the Gemini model

        else:
            return analyze_stock(symbol, asset_type)  # Pass asset_type parameter

        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=TEMPERATURE,
                top_p=0.8,
                top_k=MAX_TOKENS
            )
        )
        

        # Check if the AI's response was blocked
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            return {
                "error": "Content generation blocked by AI",
                "reason": str(response.prompt_feedback.block_reason)
            }

        # Parse and validate response
        try:
            cleaned_response = clean_json_response(response.text)
            analysis_data = json.loads(cleaned_response)
            is_valid, error_msg = validate_analysis(analysis_data)
            
            if not is_valid:
                return {
                    "error": "Invalid analysis format",
                    "details": error_msg
                }
            
            return analysis_data
            
        except json.JSONDecodeError as e:
            return {
                "error": "Failed to parse AI's JSON response",
                "details": str(e)
            }
        
    except Exception as e:
        return {"error": f"Error analyzing asset: {str(e)}"}

def analyze_stock(symbol, asset_type='stock'):
    """
    Analyze a stock using comprehensive market data, technical analysis, and AI predictions.
    
    This function serves as the stock-specific analysis engine that:
    1. Retrieves current and historical market data from Alpaca
    2. Calculates technical indicators (RSI, MACD, SMA, etc.)
    3. Gets current position information and risk metrics
    4. Analyzes broader market context (market sentiment, sector performance)
    5. Generates AI prompt with comprehensive market context
    6. Gets and validates AI-generated trading recommendations
    
    Integration Points:
    - Alpaca API: For real-time quotes and position data
    - Risk Manager: For position risk assessment
    - Gemini AI: For market analysis and trading decisions
    - Template System: Uses predefined templates from trading_variables.py
    
    Args:
        symbol (str): Stock symbol to analyze (e.g., 'AAPL', 'MSFT')
        asset_type (str): Type of asset, defaults to 'stock'. Used in risk calculations
                         and position management
    
    Returns:
        dict: Analysis results containing:
            On success: {
                "action": "buy"|"sell"|"hold",
                "reasoning": "Brief explanation",
                "confidence": float (0.0 to 1.0)
            }
            On failure: {
                "error": "Error message",
                "details": "Additional error context"
            }
    
    Note:
        This function is specifically designed for stock analysis and includes
        stock-specific metrics like position size and risk levels that might
        not be relevant for other asset types.
    """
    # Function specifically for analyzing stock assets    
    
    
    try:
        # Get current stock data
        quote = api.get_last_quote(symbol)
        
        # Get historical data for technical analysis
        historical_bars = api.get_barset(symbol, 'day', limit=50)[symbol]
        prices = [bar.c for bar in historical_bars]  # Close prices
        volumes = [bar.v for bar in historical_bars]  # Volume data
        
        # Calculate technical indicators
        tech_analysis = TechnicalAnalysis(prices)
        signals, indicators = tech_analysis.get_signals()
        
        # Format trading signals into readable text
        signal_summary = "\n".join([f"- {signal}: {reason} (Confidence: {confidence*100:.0f}%)" 
                                  for signal, reason, confidence in signals])
        
        # Get market context
        try:
            # Get S&P 500 data for market context
            spy_bars = api.get_barset('SPY', 'day', limit=1)['SPY']
            market_change = ((spy_bars[-1].c - spy_bars[-1].o) / spy_bars[-1].o) * 100
            market_context = "Bullish" if market_change > 0 else "Bearish"
        except Exception as e:
            logger.warning(f"Could not fetch market context: {e}")
            market_change = 0
            market_context = "Neutral"
        
        # Get current position details for the stock
        position_qty = "0"
        avg_entry_price = "0.00"
        try:
            position = api.get_position(symbol)
            position_qty = str(position.qty)
            avg_entry_price = str(position.avg_entry_price)
        except tradeapi.rest.APIError as e:
            if e.status_code != 404:  # If error is not "position not found"
                raise   

        current_price = float(quote.ap)
        position = Position(
            symbol=symbol,
            quantity=float(position_qty),
            entry_price=float(avg_entry_price),
            current_price=current_price,
            asset_type=asset_type
        )
        
        risk_metrics = risk_manager.calculate_position_risk(position)
        # Calculate additional metrics
        price_change = ((current_price - historical_bars[0].c) / historical_bars[0].c) * 100
        avg_volume = sum(volumes) / len(volumes)
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # Prepare template variables
        template_vars = {
            'schema': ANALYSIS_SCHEMA,
            'symbol': symbol,
            'current_price': quote.ap,
            'position_qty': position_qty,
            'entry_price': avg_entry_price,
            
            # Position and Risk Metrics
            'investment': risk_metrics['investment'],
            'current_value': risk_metrics['current_value'],
            'pnl': risk_metrics['pnl_percent'] * 100,
            'risk_level': risk_metrics['risk_level'],
            
            # Technical Indicators
            'rsi': indicators['rsi'],
            'macd': indicators['macd'],
            'sma_20': indicators['sma_20'],
            'ema_20': indicators['ema_20'],
            'bb_high': indicators['bb_high'],
            'bb_low': indicators['bb_low'],
            
            # Market Data
            'price_change': price_change,
            'volume_ratio': volume_ratio,
            'market_context': market_context,
            'market_change': market_change,
            
            # Technical Signals
            'signal_summary': signal_summary
        }
        
        # Generate prompt using template
        prompt = STOCK_ANALYSIS_TEMPLATE.format(**template_vars)

        # Get AI analysis based on the constructed prompt
        # Get AI analysis
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                top_k=40
            )
        )

        # Parse and validate response
        try:
            cleaned_response = clean_json_response(response.text)
            analysis_data = json.loads(cleaned_response)
            is_valid, error_msg = validate_analysis(analysis_data)
            
            if not is_valid:
                return {
                    "error": "Invalid analysis format",
                    "details": error_msg
                }
            
            return analysis_data
            
        except json.JSONDecodeError as e:
            return {
                "error": "Failed to parse AI's JSON response",
                "details": str(e)
            }
        
    except Exception as e:
        return {"error": f"Error analyzing stock: {str(e)}"}

def print_performance_summary(symbol, asset_type, timeframe='1d', days_back=50):
    """
    Generate and display a comprehensive performance summary for a trading symbol.
    
    This function serves several key purposes:
    1. Historical Analysis:
       - Retrieves past trading analyses and decisions
       - Shows performance metrics over time
       - Displays win/loss ratios and confidence levels
    
    2. Technical Analysis:
       - Calculates and displays current technical indicators
       - Shows trading signals and their strength
       - Provides price range and movement data
    
    3. Visualization:
       - Generates ASCII-based price charts
       - Highlights key signals and trends
       - Shows volume analysis
    
    4. Performance Benchmarking:
       - Compares performance against market indices (S&P 500 for stocks, BTC for crypto)
       - Calculates alpha and beta metrics
       - Shows relative strength against the market
    
    Args:
        symbol (str): The trading symbol to analyze (e.g., 'AAPL' for stocks, 'BTC/USD' for crypto)
        asset_type (str): Type of asset ('stock' or 'crypto') - determines data source and calculations
        timeframe (str): Data timeframe ('1d', '1h', etc.). Defaults to '1d'
        days_back (int): Number of days of historical data to analyze. Defaults to 50
    
    Integration Points:
    - Database: Retrieves historical analysis and metrics
    - Technical Analysis: Calculates current market indicators
    - Visualization: Generates ASCII charts and signal representations
    - Alpaca/Crypto APIs: Fetches current market data
    
    Note: This function requires active connections to:
    - Database (for historical data)
    - Market APIs (for current prices)
    - Technical Analysis module (for indicators)
    """
    try:
        # Initialize cache key for this analysis
        cache_key = f"{symbol}_{asset_type}_{timeframe}_{days_back}"
        
        # Try to get cached data (if within last 5 minutes)
        cached_data = _get_cached_data(cache_key)
        if cached_data:
            history, metrics = cached_data
        else:
            # Retrieve and display historical analysis and performance metrics from the database
            history = db.get_analysis_history(symbol)
            metrics = db.get_performance_metrics(symbol)
            
            # Cache the retrieved data
            _cache_data(cache_key, (history, metrics))
        
        # Exit if no historical data is available
        if not history or not metrics:
            print(f"No historical data available for {symbol}")
            return

        # === Section 1: Basic Performance Metrics ===
        print(f"\n{Style.BRIGHT}📈 Performance Summary for {symbol}{Style.RESET_ALL}")
        print(f"└── Total Analyses: {metrics[0]}")
        print(f"└── Buy Signals: {metrics[1]} ({metrics[1]/metrics[0]*100:.1f}%)")
        print(f"└── Sell Signals: {metrics[2]} ({metrics[2]/metrics[0]*100:.1f}%)")
        print(f"└── Average Confidence: {metrics[3]*100:.1f}%")
        print(f"└── Price Range: ${metrics[4]:.2f} - ${metrics[5]:.2f}")
        
        # === Section 2: Recent Trading History ===
        print("\nRecent Recommendations:")
        for timestamp, action, confidence, price in history[:5]:  # Show last 5 recommendations
            action_color = {
                'buy': Fore.GREEN,
                'sell': Fore.RED,
                'hold': Fore.YELLOW
            }.get(action.lower(), '')
            print(f"└── {datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M')}: "
                  f"{action_color}{action.upper()}{Style.RESET_ALL} "
                  f"(${price:.2f}, {confidence*100:.0f}% confidence)")

        # === Section 3: Technical Analysis ===
        # Fetch current market data based on asset type with error handling and retries
        prices, volumes = _get_market_data(symbol, asset_type, timeframe, days_back)
        
        if not prices or not volumes:
            print("Error: Could not fetch market data")
            return
        
        # Calculate technical indicators with error handling
        try:
            tech_analysis = TechnicalAnalysis(prices)
            signals, indicators = tech_analysis.get_signals()
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            return
        
        # Display current technical indicators
        print(f"\n{Style.BRIGHT}📊 Technical Indicators{Style.RESET_ALL}")
        print(f"└── RSI (14): {indicators.get('rsi', 'nan')}")
        print(f"└── MACD: {indicators.get('macd', 'nan')}")
        print(f"└── SMA20: ${indicators.get('sma_20', 'nan')}")
        
        # === Section 4: Technical Signals ===
        print("\nTechnical Signals:")
        for signal, reason, strength in signals:
            color = Fore.GREEN if signal == 'BUY' else Fore.RED
            print(f"└── {color}{signal}{Style.RESET_ALL}: {reason} ({strength*100:.0f}% confidence)")
        
        # === Section 5: Market Comparison ===
        benchmark_data = _get_benchmark_data(asset_type, timeframe, days_back)
        if benchmark_data:
            alpha, beta, relative_strength = _calculate_market_metrics(prices, benchmark_data)
            print(f"\n{Style.BRIGHT}📊 Market Comparison{Style.RESET_ALL}")
            print(f"└── Alpha: {alpha:.2f}")
            print(f"└── Beta: {beta:.2f}")
            print(f"└── Relative Strength: {relative_strength:.2f}")
        
        # === Section 6: Visual Analysis ===
        # Generate and display ASCII chart with signals
        try:
            visualizer = SignalVisualizer()
            chart, signal_summary = visualizer.visualize_signals(
                prices, signals, indicators, volumes=volumes
            )
            
            print(f"\n{Style.BRIGHT}📊 Price Chart{Style.RESET_ALL}")
            for line in chart:
                print(f"  {line}")
            
            print("\n".join(signal_summary))
        except Exception as e:
            print(f"Error generating visualization: {e}")
        
    except Exception as e:
        print(f"Error generating performance summary: {e}")
        logger.error(f"Performance summary error for {symbol}: {e}")

def _get_cached_data(cache_key, max_age_minutes=5):
    """Helper function to retrieve cached data if it exists and is not expired"""
    if not hasattr(_get_cached_data, 'cache'):
        _get_cached_data.cache = {}
    
    if cache_key in _get_cached_data.cache:
        timestamp, data = _get_cached_data.cache[cache_key]
        if datetime.now() - timestamp < timedelta(minutes=max_age_minutes):
            return data
    return None

def _cache_data(cache_key, data):
    """Helper function to cache data with timestamp"""
    if not hasattr(_get_cached_data, 'cache'):
        _get_cached_data.cache = {}
    _get_cached_data.cache[cache_key] = (datetime.now(), data)

def _get_market_data(symbol, asset_type, timeframe, days_back, max_retries=3):
    """Helper function to fetch market data with retries"""
    for attempt in range(max_retries):
        try:
            if asset_type == "crypto":
                ohlcv = crypto_exchange.fetch_ohlcv(symbol, timeframe, limit=days_back)
                return [candle[4] for candle in ohlcv], [candle[5] for candle in ohlcv]
            else:  # Stock data from Alpaca
                bars = api.get_barset(symbol, timeframe, limit=days_back)[symbol]
                return [bar.c for bar in bars], [bar.v for bar in bars]
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to fetch market data after {max_retries} attempts: {e}")
                return None, None
            time.sleep(RETRY_DELAY)
    return None, None

def _get_benchmark_data(asset_type, timeframe, days_back):
    """Helper function to fetch benchmark data (S&P 500 for stocks, BTC for crypto)"""
    try:
        if asset_type == "crypto":
            ohlcv = crypto_exchange.fetch_ohlcv("BTC/USD", timeframe, limit=days_back)
            return [candle[4] for candle in ohlcv]
        else:
            bars = api.get_barset("SPY", timeframe, limit=days_back)["SPY"]
            return [bar.c for bar in bars]
    except Exception as e:
        logger.error(f"Failed to fetch benchmark data: {e}")
        return None

def _calculate_market_metrics(prices, benchmark_prices):
    """Calculate alpha, beta, and relative strength compared to benchmark"""
    try:
        if len(prices) != len(benchmark_prices):
            return 0, 0, 0
            
        # Calculate returns
        price_returns = [(prices[i] - prices[i-1])/prices[i-1] for i in range(1, len(prices))]
        benchmark_returns = [(benchmark_prices[i] - benchmark_prices[i-1])/benchmark_prices[i-1] 
                           for i in range(1, len(benchmark_prices))]
        
        # Calculate beta (market sensitivity)
        covariance = np.cov(price_returns, benchmark_returns)[0][1]
        variance = np.var(benchmark_returns)
        beta = covariance / variance if variance != 0 else 0
        
        # Calculate alpha (excess return)
        avg_price_return = np.mean(price_returns)
        avg_benchmark_return = np.mean(benchmark_returns)
        alpha = avg_price_return - (beta * avg_benchmark_return)
        
        # Calculate relative strength
        relative_strength = (prices[-1] / prices[0]) / (benchmark_prices[-1] / benchmark_prices[0])
        
        return alpha, beta, relative_strength
    except Exception as e:
        logger.error(f"Error calculating market metrics: {e}")
        return 0, 0, 0
    
def run_trading_bot():
    """
    Main execution loop for the automated trading system. This function orchestrates the entire
    trading operation with comprehensive error handling and recovery mechanisms.
    
    Key Components:
    1. Initialization:
       - Creates TradingAgent instance to handle trading operations
       - Loads trading assets from configuration
       - Sets up retry mechanism for error recovery
    
    2. Connection Management:
       - Validates all critical API connections before trading
       - Includes Alpaca API, crypto exchange, and database
       - Implements automatic reconnection attempts
    
    3. Trading Cycle:
       - Monitors existing positions for risk management
       - Analyzes trading opportunities for configured assets
       - Executes trades based on analysis results
       - Updates performance metrics and generates summaries
    
    4. Error Handling:
       - Implements multiple retry layers for different error types
       - Handles both fatal and non-fatal errors appropriately
       - Includes graceful shutdown on critical failures
    
    5. Performance Monitoring:
       - Generates performance summaries for each asset
       - Tracks overall portfolio performance
       - Logs all trading activities and errors
    
    Execution Flow:
    1. Start with fresh TradingAgent instance
    2. Load assets from TRADING_ASSETS configuration
    3. Begin main trading loop:
       a. Verify all connections
       b. Monitor existing positions
       c. Analyze and trade assets
       d. Generate performance reports
       e. Sleep between cycles
    4. Handle any errors with appropriate retries
    5. Allow graceful shutdown on keyboard interrupt
    
    Configuration Dependencies:
    - MAX_RETRIES: Maximum number of restart attempts
    - TRADING_CYCLE_INTERVAL: Time between trading cycles
    - ERROR_RETRY_DELAY: Delay between error retries
    - TRADING_ASSETS: List of assets to trade
    
    Returns:
        None: Function runs indefinitely until interrupted or max retries exceeded
    """
    # Initialize trading components
    agent = TradingAgent()
    assets = TRADING_ASSETS
    retry_count = 0
    max_retries = MAX_RETRIES
    
    while retry_count < max_retries:
        try:
            # Verify all API and database connections before starting
            if not agent.check_connections():
                raise Exception("Failed to establish required connections")
                
            while True:
                try:
                    # Step 1: Risk Management - Monitor existing positions
                    agent.monitor_positions()
                    
                    # Step 2: Trading Analysis and Execution
                    results = agent.run_trading_cycle(assets)
                    
                    # Step 3: Performance Reporting
                    for symbol, analysis in results:
                        if analysis and "error" not in analysis:
                            # Generate performance summary for successful analyses
                            print_performance_summary(symbol, 
                                "crypto" if "ZUSD" in symbol else "stock")
                        else:
                            # Log failed analyses for monitoring
                            logger.warning(f"Skipping analysis for {symbol}: {analysis.get('error', 'Unknown error')}")
                    
                    # Step 4: Portfolio Overview
                    print_portfolio_summary()
                    try:
                        portfolio_metrics = portfolio.calculate_metrics()
                        metrics_log_entry = (
                            f"total_capital={portfolio.current_capital:.2f},"
                            f"total_return={portfolio_metrics.get('total_return', 0.0):.4f},"
                            f"win_rate={portfolio_metrics.get('win_rate', 0.0):.4f},"
                            f"avg_profit={portfolio_metrics.get('avg_profit', 0.0):.2f},"
                            f"avg_loss={portfolio_metrics.get('avg_loss', 0.0):.2f}"
                        )
                        metrics_logger.info(metrics_log_entry)
                    except Exception as metrics_err:
                        logger.error(f"Error logging portfolio metrics: {metrics_err}")
                    
                    # Reset retry counter after successful cycle
                    retry_count = 0
                    
                    # Wait for next trading cycle
                    time.sleep(TRADING_CYCLE_INTERVAL)

                    # Explicitly run garbage collection
                    gc.collect()
                    
                except Exception as e:
                    # Handle non-fatal errors within the trading cycle
                    logger.error(f"Error in trading cycle: {e}")
                    time.sleep(ERROR_RETRY_DELAY)
                    
        except KeyboardInterrupt:
            # Handle manual interruption (Ctrl+C)
            logger.info("Trading bot shutdown requested by user...")
            break
            
        except Exception as e:
            # Handle fatal errors that require full restart
            retry_count += 1
            logger.error(f"Fatal error (attempt {retry_count}/{max_retries}): {e}")
            if retry_count < max_retries:
                # Exponential backoff with jitter
                delay = min(MAX_RETRY_BACKOFF_DELAY, BASE_RETRY_DELAY * (2 ** (retry_count -1))) + random.uniform(0, JITTER_DELAY)
                logger.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                logger.critical("Maximum retry attempts reached, shutting down trading bot")
                break

if __name__ == "__main__":
    run_trading_bot()

def print_portfolio_summary():
    """
    Generate and display a comprehensive portfolio summary showing overall performance,
    positions, risk metrics, and allocation across all trading assets.
    
    This function serves several key purposes:
    1. Portfolio Overview:
       - Shows total portfolio value and performance
       - Displays overall return percentage and P&L
       - Shows available capital and investment allocation
    
    2. Position Analysis:
       - Lists all current positions with their performance
       - Shows individual position risk levels
       - Displays position sizes and allocation percentages
    
    3. Risk Assessment:
       - Calculates overall portfolio risk exposure
       - Shows concentration risk across assets
       - Displays stop-loss triggers and risk levels
    
    4. Performance Metrics:
       - Shows win/loss ratios across all trades
       - Displays average profit/loss metrics
       - Compares performance to configured benchmarks
    
    5. Asset Allocation:
       - Shows allocation across stocks vs crypto
       - Displays individual asset weights
       - Shows diversification metrics
    
    6. Portfolio Health:
       - Calculates overall portfolio health score
       - Shows key recommendations for improvement
       - Displays portfolio status and warnings
    
    Integration Points:
    - PortfolioManager: For calculating overall portfolio metrics
    - Risk Manager: For assessing portfolio and position risks
    - Alpaca API: For current position and account information
    - Database: For historical performance data
    - Trading Variables: For portfolio limits and risk thresholds
    
    Note: This function requires access to the global 'portfolio' object
    and active API connections for real-time position data.
    """
    try:
        print(f"\n{Style.BRIGHT}💼 Portfolio Summary{Style.RESET_ALL}")
        print("=" * 60)
        
        # Initialize default values
        portfolio_metrics = {}
        current_value = 0.0
        initial_value = 0.0
        total_return = 0.0
        total_pnl = 0.0
        win_rate = 0.0
        avg_profit = 0.0
        avg_loss = 0.0
        total_trades = 0
        total_position_value = 0.0
        total_risk_exposure = 0.0
        max_allowed_risk = float(MAX_PORTFOLIO_RISK)
        
        # === Section 1: Overall Portfolio Performance ===
        try:
            portfolio_metrics = portfolio.calculate_metrics()
            current_value = float(portfolio.current_capital)
            initial_value = float(portfolio.initial_capital)
            total_return = float(portfolio_metrics.get('total_return', 0.0))
            total_pnl = current_value - initial_value
            
            print(f"\n{Style.BRIGHT}📊 Overall Performance{Style.RESET_ALL}")
            print(f"└── Initial Capital: ${initial_value:,.2f}")
            print(f"└── Current Value: ${current_value:,.2f}")
            
            # Color-code P&L based on performance
            pnl_color = Fore.GREEN if total_pnl >= 0 else Fore.RED
            return_color = Fore.GREEN if total_return >= 0 else Fore.RED
            
            print(f"└── Total P&L: {pnl_color}${total_pnl:+,.2f}{Style.RESET_ALL}")
            print(f"└── Total Return: {return_color}{total_return:+.2%}{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"Error calculating portfolio metrics: {e}")
            logger.error(f"Portfolio metrics calculation failed: {e}")
        
        # === Section 2: Trading Performance ===
        try:
            print(f"\n{Style.BRIGHT}🎯 Trading Performance{Style.RESET_ALL}")
            win_rate = float(portfolio_metrics.get('win_rate', 0.0))
            avg_profit = float(portfolio_metrics.get('avg_profit', 0.0))
            avg_loss = float(portfolio_metrics.get('avg_loss', 0.0))
            total_trades = len(portfolio.trades)
            
            win_rate_color = Fore.GREEN if win_rate >= 0.5 else Fore.YELLOW if win_rate >= 0.3 else Fore.RED
            
            print(f"└── Total Trades: {total_trades}")
            print(f"└── Win Rate: {win_rate_color}{win_rate:.1%}{Style.RESET_ALL}")
            print(f"└── Average Profit: {Fore.GREEN}${avg_profit:.2f}{Style.RESET_ALL}")
            print(f"└── Average Loss: {Fore.RED}${avg_loss:.2f}{Style.RESET_ALL}")
            
            # Calculate risk-adjusted return if we have trade data
            if total_trades > 0:
                profit_factor = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
                print(f"└── Profit Factor: {profit_factor:.2f}")
                
        except Exception as e:
            print(f"Error calculating trading performance: {e}")
            logger.error(f"Trading performance calculation failed: {e}")
        
        # === Section 3: Current Positions ===
        try:
            print(f"\n{Style.BRIGHT}📋 Current Positions{Style.RESET_ALL}")
            
            # Get current positions from Alpaca for stocks
            stock_positions = []
            total_position_value = 0.0
            
            try:
                account = api.get_account()
                positions = api.list_positions()
                
                if positions:
                    for position in positions:
                        try:
                            qty = float(position.qty)
                            market_value = float(position.market_value)
                            unrealized_pl = float(position.unrealized_pl)
                            unrealized_plpc = float(position.unrealized_plpc)
                            
                            total_position_value += market_value
                            
                            # Color-code based on P&L
                            pnl_color = Fore.GREEN if unrealized_pl >= 0 else Fore.RED
                            
                            print(f"└── {position.symbol}: {qty} shares")
                            print(f"    ├── Value: ${market_value:,.2f}")
                            print(f"    ├── P&L: {pnl_color}${unrealized_pl:+,.2f} ({unrealized_plpc:+.2%}){Style.RESET_ALL}")
                            
                            # Calculate position risk
                            current_price = market_value / qty if qty != 0 else 0
                            pos_obj = Position(
                                symbol=str(position.symbol),
                                quantity=qty,
                                entry_price=float(position.avg_entry_price),
                                current_price=current_price,
                                asset_type='stock'
                            )
                            risk_metrics = risk_manager.calculate_position_risk(pos_obj)
                            risk_color = Fore.RED if risk_metrics['risk_level'] == 'HIGH' else Fore.GREEN
                            print(f"    └── Risk: {risk_color}{risk_metrics['risk_level']}{Style.RESET_ALL}")
                            
                        except Exception as pos_error:
                            print(f"    └── Error processing position: {pos_error}")
                            
                else:
                    print("└── No current stock positions")
                    
            except Exception as api_error:
                print(f"└── Error fetching positions: {api_error}")
                logger.error(f"Failed to fetch Alpaca positions: {api_error}")
            
            # Show portfolio allocation
            if total_position_value > 0:
                cash_balance = current_value - total_position_value
                investment_ratio = total_position_value / current_value
                
                print(f"\n{Style.BRIGHT}💰 Portfolio Allocation{Style.RESET_ALL}")
                print(f"└── Invested: ${total_position_value:,.2f} ({investment_ratio:.1%})")
                print(f"└── Cash: ${cash_balance:,.2f} ({(1-investment_ratio):.1%})")
                
        except Exception as e:
            print(f"Error analyzing positions: {e}")
            logger.error(f"Position analysis failed: {e}")
        
        # === Section 4: Risk Assessment ===
        try:
            print(f"\n{Style.BRIGHT}⚠️ Risk Assessment{Style.RESET_ALL}")
            
            # Calculate portfolio risk exposure
            total_risk_exposure = float(total_position_value) / float(current_value) if current_value > 0 else 0.0
            max_allowed_risk = float(MAX_PORTFOLIO_RISK)
            
            risk_status_color = (Fore.RED if total_risk_exposure > max_allowed_risk 
                               else Fore.YELLOW if total_risk_exposure > max_allowed_risk * 0.8 
                               else Fore.GREEN)
            
            print(f"└── Current Risk Exposure: {risk_status_color}{total_risk_exposure:.1%}{Style.RESET_ALL}")
            print(f"└── Maximum Allowed: {max_allowed_risk:.1%}")
            print(f"└── Risk Level: {risk_status_color}{'HIGH' if total_risk_exposure > max_allowed_risk else 'MODERATE' if total_risk_exposure > max_allowed_risk * 0.5 else 'LOW'}{Style.RESET_ALL}")
            
            # Show configured risk limits
            print(f"└── Max Position Size: {float(MAX_POSITION_SIZE):.1%}")
            print(f"└── Stock Stop Loss: {float(STOCK_STOP_LOSS_PCT):.1%}")
            print(f"└── Crypto Stop Loss: {float(CRYPTO_STOP_LOSS_PCT):.1%}")
            
        except Exception as e:
            print(f"Error calculating risk assessment: {e}")
            logger.error(f"Risk assessment calculation failed: {e}")
        
        # === Section 5: Asset Coverage ===
        try:
            print(f"\n{Style.BRIGHT}🎯 Asset Coverage{Style.RESET_ALL}")
            
            # Show which assets from TRADING_ASSETS we're tracking
            print("└── Configured Assets:")
            for symbol, asset_type in TRADING_ASSETS:
                # Check if we have any historical data for this asset
                try:
                    history = db.get_analysis_history(symbol)
                    analysis_count = len(history) if history else 0
                    status_color = Fore.GREEN if analysis_count > 0 else Fore.YELLOW
                    
                    print(f"    ├── {symbol} ({asset_type}): {status_color}{analysis_count} analyses{Style.RESET_ALL}")
                    
                except Exception as asset_error:
                    print(f"    ├── {symbol} ({asset_type}): {Fore.RED}Error fetching data{Style.RESET_ALL}")
            
            # Show diversification
            stock_count = sum(1 for _, asset_type in TRADING_ASSETS if asset_type == 'stock')
            crypto_count = sum(1 for _, asset_type in TRADING_ASSETS if asset_type == 'crypto')
            
            print(f"└── Diversification:")
            print(f"    ├── Stocks: {stock_count} assets")
            print(f"    └── Crypto: {crypto_count} assets")
            
        except Exception as e:
            print(f"Error analyzing asset coverage: {e}")
            logger.error(f"Asset coverage analysis failed: {e}")
        
        # === Section 6: Portfolio Health Summary ===
        try:
            print(f"\n{Style.BRIGHT}📈 Portfolio Health Summary{Style.RESET_ALL}")
            
            # Overall portfolio health score
            health_factors = []
            
            # Factor 1: Positive returns
            if total_return >= 0:
                health_factors.append("positive_return")
            
            # Factor 2: Reasonable risk exposure
            if total_risk_exposure <= max_allowed_risk:
                health_factors.append("risk_controlled")
            
            # Factor 3: Decent win rate
            if win_rate >= 0.5:
                health_factors.append("good_win_rate")
            
            # Factor 4: Active trading
            if total_trades >= 1:
                health_factors.append("active_trading")
            
            health_score = len(health_factors) / 4.0  # Out of 4 possible factors
            health_color = (Fore.GREEN if health_score >= 0.75 
                          else Fore.YELLOW if health_score >= 0.5 
                          else Fore.RED)
            
            print(f"└── Portfolio Health: {health_color}{health_score:.0%}{Style.RESET_ALL}")
            
            # Show key recommendations
            recommendations = []
            if total_risk_exposure > max_allowed_risk:
                recommendations.append("🔴 Reduce position sizes")
            if win_rate < 0.3:
                recommendations.append("🟡 Review trading strategy")
            if total_trades == 0:
                recommendations.append("🟡 Consider initiating positions")
            if len(TRADING_ASSETS) < 3:
                recommendations.append("🟡 Consider more diversification")
            
            if recommendations:
                print("└── Recommendations:")
                for rec in recommendations:
                    print(f"    └── {rec}")
            else:
                print(f"└── Status: {Fore.GREEN}Portfolio performing well{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"Error generating portfolio health summary: {e}")
            logger.error(f"Portfolio health summary generation failed: {e}")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"Error generating portfolio summary: {e}")
        logger.error(f"Portfolio summary error: {e}")