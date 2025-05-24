# === Standard Library Imports ===
import json # For handling JSON data
import logging
import time
from datetime import datetime

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
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, GEMINI_API_KEY # Configuration settings
from database import Database
from indicators import TechnicalAnalysis
from risk_manager import RiskManager, Position
from portfolio_manager import PortfolioManager
from visualizer import SignalVisualizer
from trade_executor import TradeExecutor
from position_sizer import PositionSizer

# Initialize colorama for colored output and configure basic logging
init()
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)



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
    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, "https://paper-api.alpaca.markets")
    logger.info("Alpaca API initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Alpaca API: {e}")
    raise

# 3. Initialize Crypto Exchange API (using ccxt for Kraken)
try:
    # Enable rate limiting to avoid hitting API limits
    crypto_exchange = ccxt.kraken({
        'enableRateLimit': True,
        'rateLimit': 3000  # milliseconds between requests
    })
    logger.info("Kraken (ccxt) crypto exchange initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize crypto exchange: {e}")
    raise

# 4. Initialize Gemini AI Model (for generating trading analysis)
try:
    # Use the Gemini 1.5 Flash model for fast, cost-effective analysis
    model = genai.GenerativeModel('models/gemini-1.5-flash')
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
        self.stop_loss_pct = 0.02           # 2% stop loss threshold
        self.min_confidence = 0.7           # Minimum confidence required to execute a trade

        # === Core Trading Components ===
        self.executor = TradeExecutor(ALPACA_API_KEY, ALPACA_SECRET_KEY)  # Handles trade execution
        self.sizer = PositionSizer(total_capital=100000)                  # Determines position sizes
        self.db = Database()                                              # Handles analysis and trade history storage
        
    def get_current_price(self, symbol, asset_type):
        """Get current price for an asset with validation"""
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                if asset_type == 'crypto':
                    data = get_crypto_data(symbol)
                    if data and 'current_price' in data:
                        return data['current_price']
                else:
                    quote = api.get_last_quote(symbol)
                    if quote and hasattr(quote, 'ap'):
                        return float(quote.ap)
                
                logger.warning(f"Attempt {attempt + 1}: Invalid response for {symbol}")
                time.sleep(retry_delay)
                
            except Exception as e:
                logger.error(f"Error getting price for {symbol} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
            
        return None

    def process_signals(self, symbol, signals, indicators, current_price):
        """Process signals with risk checks"""
        # Get portfolio value
        account = self.executor.get_account() # Fetch account information to get portfolio value
        if not account:
            return
            
        portfolio_value = float(account.portfolio_value)
        
        for signal, reason, confidence in signals:
            # Check if we're already at max risk
            risk_metrics = risk_manager.calculate_position_risk(Position(
                symbol=symbol,
                quantity=0,
                entry_price=current_price,
                current_price=current_price,
                asset_type='stock'
            ))
            
            if risk_metrics['risk_level'] == 'HIGH':
                logger.warning(f"Skipping {signal} for {symbol} - risk too high")
                continue
                
            # Calculate position size
            quantity = self.sizer.calculate_position_size(
                price=current_price,
                confidence=confidence
            )
            
            if quantity > 0:
                # Execute trade using the TradeExecutor
                success, order = self.executor.execute_trade(
                    symbol=symbol,
                    side=signal,
                    quantity=quantity,
                    confidence=confidence
                )
                
                if success:
                    print(f"Trade executed: {signal} {quantity} {symbol}")

    def run_trading_cycle(self, assets):
        """Run a single trading cycle for all assets"""
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
        """Monitor open positions for stop loss"""
        try:
            # List all open positions from Alpaca
            positions = api.list_positions()
            for position in positions:
                entry_price = float(position.avg_entry_price)
                current_price = float(position.current_price)
                
                # Check stop loss
                # If current price drops below the stop loss percentage from entry price
                if current_price < entry_price * (1 - self.stop_loss_pct):
                    self.executor.execute_trade(
                        symbol=position.symbol,
                        side='sell',
                        quantity=position.qty,
                        confidence=1.0  # High confidence for stop loss
                    )
                    logger.warning(f"Stop loss triggered for {position.symbol}")
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")

    def save_analysis(self, symbol, asset_type, analysis, current_price):
        """Save analysis with error handling"""
        # Save the analysis result to the database
        try:
            timestamp = datetime.now().isoformat()
            analysis_data = {
                'timestamp': timestamp,
                'symbol': symbol,
                'asset_type': asset_type,
                'analysis': analysis,
                'price': current_price
            }
            self.db.save_analysis(symbol, asset_type, analysis, current_price)
            return True
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")
            return False

    def check_connections(self):
        """Verify all required connections are working"""
        try:
            # Check Alpaca connection by fetching account details
            account = api.get_account()
            if not account:
                raise Exception("Could not connect to Alpaca")
                
            # Check crypto exchange connection
            # Load markets to verify connection and credentials

            crypto_exchange.load_markets()
            
            # Optionally check database connection by fetching analysis history for a known symbol
            try:
                self.db.get_analysis_history("AAPL")
            except Exception as db_exc:
                raise Exception(f"Database connection failed: {db_exc}")
            
            return True
            
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False

def get_crypto_data(symbol):
    """Get cryptocurrency market data with rate limiting"""
    # Fetch real-time and historical data for cryptocurrencies using ccxt
    try:
        # Add rate limiting delay
        time.sleep(crypto_exchange.rateLimit / 1000)  # Convert ms to seconds
        
        # Convert symbol format for Kraken (BTC/USD -> XXBTZUSD)
        if 'XBT/USD' in symbol:
            symbol = 'XXBTZUSD'
        elif 'ETH/USD' in symbol:
            symbol = 'XETHZUSD'
        
        ticker = crypto_exchange.fetch_ticker(symbol)
        ohlcv = crypto_exchange.fetch_ohlcv(symbol, '1d', limit=2)
        
        return {
            'current_price': ticker['last'],
            'volume': ticker['quoteVolume'] if 'quoteVolume' in ticker else ticker['baseVolume'],
            'price_change': ticker['percentage'],
            'high_24h': ticker['high'],
            'low_24h': ticker['low'],
            'yesterday_close': ohlcv[0][4] if len(ohlcv) > 1 else None
        }
    except ccxt.RateLimitExceeded:
        logger.warning(f"Rate limit hit for {symbol}, waiting 30 seconds...")
        time.sleep(30)
        return get_crypto_data(symbol)  # Retry
    except Exception as e:
        logger.error(f"Error fetching crypto data: {e}")
        return None

def clean_json_response(text):
    """Clean the response text to get valid JSON"""
    # Remove markdown code block formatting if present
    if text.startswith('```'):
        # Find the first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return text[start:end + 1]
    return text

def validate_analysis(analysis_data):
    """Validate the AI's analysis response"""
    # Ensure the AI's response conforms to the expected JSON schema
    required_keys = ['action', 'reasoning', 'confidence']
    if not all(key in analysis_data for key in required_keys):
        return False, "Missing required fields"
    
    if analysis_data['action'] not in ['buy', 'sell', 'hold']:
        return False, f"Invalid action value: {analysis_data['action']}"
    
    if not isinstance(analysis_data['confidence'], (int, float)) or \
       not 0 <= float(analysis_data['confidence']) <= 1:
        return False, "Confidence must be between 0.0 and 1.0"
    
    return True, ""

def analyze_asset(symbol, asset_type='stock'):
    """Analyze stocks or crypto assets"""
    # Main function to analyze either stocks or cryptocurrencies
    try:
        if asset_type == 'crypto':
            data = get_crypto_data(symbol)
            if not data:
                return {"error": "Could not fetch crypto data"}
            
            # Get historical prices for technical analysis
            ohlcv = crypto_exchange.fetch_ohlcv(symbol, '1d', limit=50)
            prices = [candle[4] for candle in ohlcv]  # Using closing prices
            
            # Calculate technical indicators
            tech_analysis = TechnicalAnalysis(prices)
            signals, indicators = tech_analysis.get_signals()
            
            # Add technical analysis to prompt
            prompt = f"""Analyze this cryptocurrency and respond ONLY with valid JSON matching this schema exactly:
{{
    "action": "buy"|"sell"|"hold",
    "reasoning": "<1 brief sentence>",
    "confidence": <number between 0.0 and 1.0>
}}

Crypto Data:
Symbol: {symbol}
Current Price: ${data['current_price']}
24h Volume: {data['volume']}
24h Change: {data['price_change']}%
24h High: ${data['high_24h']}
24h Low: ${data['low_24h']}

Technical Indicators:
RSI (14): {indicators['rsi']:.2f}
MACD: {indicators['macd']:.2f}
SMA20: ${indicators['sma_20']:.2f}
Price vs SMA20: {'+' if data['current_price'] > indicators['sma_20'] else '-'}{abs(data['current_price'] - indicators['sma_20']):.2f}
"""
            # Generate content using the Gemini model

        else:
            return analyze_stock(symbol, asset_type)  # Pass asset_type parameter

        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                top_k=40
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
    """Analyze a stock using market data and Gemini's analysis"""
    # Function specifically for analyzing stock assets
    try:
        # Get stock data
        quote = api.get_last_quote(symbol)
        
        # Get current position details for the stock
        # Get position data
        position_qty = "0"
        avg_entry_price = "0.00"
        try:
            position = api.get_position(symbol)
            position_qty = str(position.qty)
            avg_entry_price = str(position.avg_entry_price)
        except tradeapi.rest.APIError as e:
            if e.status_code != 404:  # If error is not "position not found"
                raise

        # Create analysis prompt
        prompt = f"""Analyze this stock data and respond ONLY with valid JSON matching this schema exactly:
{{
    "action": "buy"|"sell"|"hold",
    "reasoning": "<1 brief sentence>",
    "confidence": <number between 0.0 and 1.0>
}}

Stock Data:
Symbol: {symbol}
Current Price: ${quote.ap}
Position: {position_qty} shares
Entry Price: ${avg_entry_price}"""

        # Add risk analysis
        # Calculate risk metrics for the current position
        current_price = float(quote.ap)  # Simplified this line
        position = Position(
            symbol=symbol,
            quantity=float(position_qty),
            entry_price=float(avg_entry_price),
            current_price=current_price,
            asset_type=asset_type  # Pass the asset_type parameter
        )
        
        risk_metrics = risk_manager.calculate_position_risk(position)
        
        # Add risk metrics to prompt
        prompt += f"""
Risk Metrics:
Investment: ${risk_metrics['investment']:.2f}
Current Value: ${risk_metrics['current_value']:.2f}
P&L: {risk_metrics['pnl_percent']*100:+.2f}%
Risk Level: {risk_metrics['risk_level']}
"""

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

def print_performance_summary(symbol, asset_type):
    """Print performance summary for a symbol"""
    # Retrieve and display historical analysis and performance metrics from the database
    history = db.get_analysis_history(symbol)
    metrics = db.get_performance_metrics(symbol)
    
    if not history or not metrics:
        print(f"No historical data available for {symbol}")
        return

    print(f"\n{Style.BRIGHT}📈 Performance Summary for {symbol}{Style.RESET_ALL}")
    print(f"└── Total Analyses: {metrics[0]}")
    print(f"└── Buy Signals: {metrics[1]} ({metrics[1]/metrics[0]*100:.1f}%)")
    print(f"└── Sell Signals: {metrics[2]} ({metrics[2]/metrics[0]*100:.1f}%)")
    print(f"└── Average Confidence: {metrics[3]*100:.1f}%")
    print(f"└── Price Range: ${metrics[4]:.2f} - ${metrics[5]:.2f}")
    
    # Show last 5 recommendations
    # Display recent trading recommendations from the analysis history
    print("\nRecent Recommendations:")
    for timestamp, action, confidence, price in history[:5]:
        action_color = {
            'buy': Fore.GREEN,
            'sell': Fore.RED,
            'hold': Fore.YELLOW
        }.get(action.lower(), '')
        print(f"└── {datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M')}: "
              f"{action_color}{action.upper()}{Style.RESET_ALL} "
              f"(${price:.2f}, {confidence*100:.0f}% confidence)")

    # Add technical analysis summary
    # Fetch historical price data to calculate technical indicators
    if asset_type == "crypto":
        ohlcv = crypto_exchange.fetch_ohlcv(symbol, '1d', limit=50)
        prices = [candle[4] for candle in ohlcv]
    else:
        bars = api.get_barset(symbol, 'day', limit=50)[symbol]
        prices = [bar.c for bar in bars]
    
    tech_analysis = TechnicalAnalysis(prices)
    # Get technical signals and indicator values
    signals, indicators = tech_analysis.get_signals()
    
    print(f"\n{Style.BRIGHT}📊 Technical Indicators{Style.RESET_ALL}")
    print(f"└── RSI (14): {indicators.get('rsi', 'nan')}")
    print(f"└── MACD: {indicators.get('macd', 'nan')}")
    print(f"└── SMA20: ${indicators.get('sma_20', 'nan')}")
    
    print("\nTechnical Signals:")
    # Print the technical signals generated
    for signal, reason, strength in signals:
        color = Fore.GREEN if signal == 'BUY' else Fore.RED
        print(f"└── {color}{signal}{Style.RESET_ALL}: {reason} ({strength*100:.0f}% confidence)")
    
    # Get price and volume data
    if asset_type == "crypto":
        ohlcv = crypto_exchange.fetch_ohlcv(symbol, '1d', limit=50)
        prices = [candle[4] for candle in ohlcv]  # Close prices
        volumes = [candle[5] for candle in ohlcv]  # Volume data
    else:
        bars = api.get_barset(symbol, 'day', limit=50)[symbol]
        prices = [bar.c for bar in bars]
        volumes = [bar.v for bar in bars]
    
    # Update visualization call
    # Generate and print a visual representation of price and signals
    visualizer = SignalVisualizer()
    chart, signal_summary = visualizer.visualize_signals(
        prices, signals, indicators, volumes=volumes
    )
    
    print(f"\n{Style.BRIGHT}📊 Price Chart{Style.RESET_ALL}")
    for line in chart:
        print(f"  {line}")
    
    print("\n".join(signal_summary))

def analyze_trading_signals(symbol, metrics, signal_data):
    """Analyze trading signals and generate recommendations"""
    # Analyze the strength and direction of trading signals based on historical data
    signal_strength, price_change = signal_data
    
    # Calculate trend strength (between -1 and 1)
    trend = min(max(signal_strength, -1), 1)
    
    # Determine signal confidence
    confidence = abs(trend)
    
    # Generate signal message
    if abs(trend) < 0.3:
        signal = "NEUTRAL"
        color = Fore.YELLOW
    else:
        signal = "STRONG BUY" if trend > 0 else "STRONG SELL"
        color = Fore.GREEN if trend > 0 else Fore.RED
    
    print(f"\n{Style.BRIGHT}🎯 Trading Signals Analysis{Style.RESET_ALL}")
    print(f"└── Signal: {color}{signal}{Style.RESET_ALL}")
    print(f"└── Strength: {abs(trend)*100:.1f}%")
    print(f"└── Price Trend: {Fore.GREEN if price_change > 0 else Fore.RED}{price_change*100:+.1f}%{Style.RESET_ALL}")

def print_portfolio_summary():
    """Print portfolio performance summary"""
    # Display the overall performance metrics of the trading portfolio
    metrics = portfolio.calculate_metrics()
    
    print(f"\n{Style.BRIGHT}📊 Portfolio Summary{Style.RESET_ALL}")
    print(f"└── Total Return: {Fore.GREEN if metrics['total_return'] > 0 else Fore.RED}"
          f"{metrics['total_return']*100:+.2f}%{Style.RESET_ALL}")
    print(f"└── Win Rate: {metrics['win_rate']*100:.1f}%")
    print(f"└── Average Profit: ${metrics['avg_profit']:.2f}")
    print(f"└── Average Loss: ${metrics['avg_loss']:.2f}")
    print(f"└── Largest Gain: ${metrics['largest_gain']:.2f}")
    print(f"└── Largest Loss: ${metrics['largest_loss']:.2f}")

def run_trading_bot():
    """Main trading bot loop with improved error handling"""
    # The main execution loop for the trading bot
    agent = TradingAgent()
    assets = [
        ("AAPL", "stock"),
        ("MSFT", "stock"),
        ("XXBTZUSD", "crypto")
    ]
    
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            # Check API and database connections before starting the trading cycle
            # Check connections before starting
            if not agent.check_connections():
                raise Exception("Failed connection check")
                
            while True:
                try:
                    # Monitor existing positions
                    agent.monitor_positions()
                    # Run the trading cycle to analyze assets and potentially execute trades
                    
                    # Analyze and trade assets
                    results = agent.run_trading_cycle(assets)
                    
                    # Print updates
                    for symbol, analysis in results:
                        if analysis and "error" not in analysis:
                            print_performance_summary(symbol, 
                                "crypto" if "ZUSD" in symbol else "stock")
                        else:
                            logger.warning(f"Skipping analysis for {symbol}: {analysis.get('error', 'Unknown error')}")
                    
                    print_portfolio_summary()
                    # Print the overall portfolio summary after each cycle
                    
                    # Reset retry count on successful cycle
                    retry_count = 0
                    
                    # Sleep between cycles
                    time.sleep(300)
                    
                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}")
                    time.sleep(60)  # Wait before retrying cycle
                    
        except KeyboardInterrupt:
            # Handle manual interruption (e.g., Ctrl+C)
            logger.info("Stopping trading bot...")
            break
        except Exception as e:
            retry_count += 1
            logger.error(f"Fatal error (attempt {retry_count}/{max_retries}): {e}")
            if retry_count < max_retries:
                time.sleep(300)  # Wait 5 minutes before retrying
            else:
                logger.critical("Max retries reached, shutting down")
                break

if __name__ == "__main__":
    run_trading_bot()

# Import necessary libraries
import numpy as np
from alpaca_trade_api import REST
from datetime import datetime, timedelta
import pandas as pd
import time

class SimpleTradingAgent:
    def __init__(self, api_key, api_secret, base_url, symbols):
        """
        Initialize the trading agent with API credentials and trading parameters
        """
        self.api = REST(api_key, api_secret, base_url)
        self.symbols = symbols
        self.positions = {}
        self.data = {}
        self.stop_losses = {}
        self.take_profits = {}

    def fetch_data(self, symbol, timeframe='1D', limit=100):
        """
        Fetch historical market data for analysis
        """
        try:
            bars = self.api.get_barset(symbol, timeframe, limit=limit)[symbol]
            df = pd.DataFrame({
                'open': [bar.o for bar in bars],
                'high': [bar.h for bar in bars],
                'low': [bar.l for bar in bars],
                'close': [bar.c for bar in bars],
                'volume': [bar.v for bar in bars]
            })
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def calculate_indicators(self, df):
        """
        Calculate technical indicators for trading decisions
        """
        # Simple Moving Average
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        
        # Relative Strength Index
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df

    def generate_signals(self, df):
        """
        Generate trading signals based on technical indicators
        """
        signals = []
        
        # Example strategy: Buy when 20 SMA crosses above 50 SMA and RSI < 70
        if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] and df['RSI'].iloc[-1] < 70:
            signals.append('BUY')
        # Sell when 20 SMA crosses below 50 SMA or RSI > 30
        elif df['SMA_20'].iloc[-1] < df['SMA_50'].iloc[-1] or df['RSI'].iloc[-1] > 30:
            signals.append('SELL')
        else:
            signals.append('HOLD')
            
        return signals

    def execute_trade(self, symbol, signal):
        """
        Execute trades based on generated signals
        """
        try:
            if signal == 'BUY' and symbol not in self.positions:
                # Implement buy logic
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=1,  # Adjust position size based on your risk management
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
                self.positions[symbol] = order
                print(f"Bought {symbol}")
                
            elif signal == 'SELL' and symbol in self.positions:
                # Implement sell logic
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=1,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                del self.positions[symbol]
                print(f"Sold {symbol}")
                
        except Exception as e:
            print(f"Error executing trade for {symbol}: {str(e)}")

    def run(self):
        """
        Main trading loop
        """
        while True:
            for symbol in self.symbols:
                try:
                    # Fetch and analyze data
                    df = self.fetch_data(symbol)
                    if df is not None:
                        df = self.calculate_indicators(df)
                        signals = self.generate_signals(df)
                        self.execute_trade(symbol, signals[-1])
                    
                    # Add delay to avoid API rate limits
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"Error processing {symbol}: {str(e)}")
                    continue
            
            # Add delay between iterations
            time.sleep(60)  # Adjust based on your trading frequency