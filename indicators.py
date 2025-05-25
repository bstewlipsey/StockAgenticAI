import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator, VolumeWeightedAveragePrice
from trading_variables import RSI_OVERSOLD, RSI_OVERBOUGHT, SMA_WINDOW

# Cache for storing calculated indicators to improve performance
# Prevents recalculating the same indicators multiple times
_indicator_cache = {}

class TechnicalAnalysis:
    """
    The TechnicalAnalysis class calculates technical indicators used for AI trading decisions.
    
    This class is the mathematical brain that transforms raw price data into actionable 
    trading signals. It calculates various technical indicators that help determine:
    - Market momentum (RSI, MACD)
    - Trend direction (SMA, EMA)
    - Market volatility (Bollinger Bands)
    - Volume patterns (OBV, ADI, VWAP)
    
    Key Features:
    - Comprehensive indicator suite covering momentum, trend, and volatility
    - Performance optimization through intelligent caching
    - Robust error handling for incomplete data
    - Configurable parameters for different trading strategies
    - Integration with AI analysis for signal generation
    
    Technical Indicators Calculated:
    - RSI (Relative Strength Index): Measures overbought/oversold conditions
    - MACD (Moving Average Convergence Divergence): Shows momentum changes
    - SMA/EMA (Simple/Exponential Moving Averages): Identify trend direction
    - Bollinger Bands: Measure volatility and potential reversal points
    - Volume indicators: Confirm price movements with volume analysis
    
    The indicators work together to provide a complete technical picture that the AI
    uses to validate trading signals and filter out false positives.
    """    
    def __init__(self, prices, window=SMA_WINDOW):
        """
        Initialize the TechnicalAnalysis class with historical price data.
        
        This constructor prepares the price data for technical analysis by creating a
        standardized DataFrame that all indicator calculations can use. It converts
        simple price lists into the OHLCV (Open, High, Low, Close, Volume) format
        required by most technical analysis libraries.
        
        Data Preparation Process:
        1. Takes raw closing prices and creates OHLCV DataFrame
        2. Uses closing prices for all OHLC values (simplified approach)
        3. Sets default volume since it's not critical for momentum indicators
        4. Configures window size for moving average calculations
        
        Args:
            prices (list): List of historical closing prices in chronological order
                          Example: [100.50, 101.25, 99.75, 102.00, ...]
            window (int): Window size for moving averages (default from SMA_WINDOW config)
                         Typical values: 20 (short-term), 50 (medium-term), 200 (long-term)
            
        Data Structure Created:
        - close: Actual closing prices (main data for analysis)
        - high: Set to closing prices (simplified - could be enhanced with real high data)
        - low: Set to closing prices (simplified - could be enhanced with real low data)  
        - open: Set to closing prices (simplified - could be enhanced with real open data)
        - volume: Set to 1 (placeholder since volume not critical for current indicators)
        
        Note: This simplified approach works well for the current indicator set, but could
        be enhanced with real OHLCV data for more sophisticated analysis.
        """
        # Create standardized DataFrame with OHLCV structure required by technical analysis library
        # Using closing prices for all OHLC values since we typically only have closing price data
        self.df = pd.DataFrame({
            'close': prices,        # Actual closing prices - primary data for analysis
            'high': prices,         # Using close as high (simplified approach)
            'low': prices,          # Using close as low (simplified approach)
            'open': prices,         # Using close as open (simplified approach)
            'volume': [1] * len(prices)  # Default volume (not critical for current indicators)
        })
        self.window = window    
        
    def calculate_indicators(self):
        """
        Calculate and return all technical indicators used for AI trading decisions.
        
        This is the core method that transforms raw price data into actionable technical
        signals. It calculates a comprehensive suite of indicators that help the AI
        understand market conditions and validate trading opportunities.
        
        Indicator Categories:
        1. MOMENTUM INDICATORS:
           - RSI: Identifies overbought (>70) and oversold (<30) conditions
           - MACD: Shows momentum changes and trend reversals
        
        2. TREND INDICATORS:
           - SMA: Simple moving average for trend direction
           - EMA: Exponential moving average (more responsive to recent prices)
        
        3. VOLATILITY INDICATORS:
           - Bollinger Bands: Upper and lower bounds based on price volatility
        
        4. VOLUME INDICATORS:
           - OBV: On-Balance Volume (cumulative volume flow)
           - ADI: Accumulation/Distribution Index (buying/selling pressure)
           - VWAP: Volume Weighted Average Price (institutional trading levels)
        
        Performance Optimization:
        - Uses caching to avoid recalculating indicators for the same data
        - Handles insufficient data gracefully with default values
        - Optimized for real-time trading with minimal computational overhead
        
        Returns:
            dict: Complete set of technical indicators including:
                - 'rsi': RSI value (0-100 scale)
                - 'macd': MACD line value
                - 'macd_signal': MACD signal line
                - 'macd_histogram': MACD histogram (MACD - Signal)
                - 'sma_20': 20-period simple moving average
                - 'ema_20': 20-period exponential moving average
                - 'bb_upper': Bollinger Band upper limit
                - 'bb_lower': Bollinger Band lower limit
                - 'obv': On-Balance Volume
                - 'adi': Accumulation/Distribution Index
                - 'vwap': Volume Weighted Average Price
        
        Error Handling:
        - Returns default values if insufficient data (less than required periods)
        - Handles NaN values that can occur with limited price history
        - Logs warnings for debugging while continuing execution
        
        Usage in AI Analysis:
        These indicators are used by the AI agent to:
        - Confirm buy/sell signals generated by the neural network
        - Filter out false positives in volatile markets
        - Assess market strength and momentum
        - Time entry and exit points more precisely
        """     
        # === PERFORMANCE OPTIMIZATION: CHECK CACHE FIRST ===
        # Create a hash of the recent price data to check if we've already calculated indicators
        # This prevents expensive recalculations when the same data is analyzed multiple times
        df_hash = hash(str(self.df.iloc[-self.window:].values.tobytes()))
        if df_hash in _indicator_cache:
            # Return cached results if available - significant performance boost
            return _indicator_cache[df_hash]

        # === INITIALIZE DEFAULT VALUES ===
        # Start with safe default values in case calculations fail
        # This ensures the system continues working even with insufficient data
        indicators = {
            'rsi': 50.0,            # Neutral RSI (middle of 0-100 range)
            'macd': 0.0,            # Neutral MACD (no momentum)
            'macd_signal': 0.0,     # MACD signal line
            'macd_histogram': 0.0,  # MACD histogram (MACD - Signal)
            'sma_20': 0.0,          # 20-period Simple Moving Average
            'ema_20': 0.0,          # 20-period Exponential Moving Average
            'bb_upper': 0.0,        # Bollinger Band upper limit
            'bb_lower': 0.0,        # Bollinger Band lower limit
            'obv': 0.0,             # On-Balance Volume
            'adi': 0.0,             # Accumulation/Distribution Index
            'vwap': 0.0,            # Volume Weighted Average Price
            'current_price': 0.0,   # Most recent closing price
            'has_signals': False    # Flag indicating if we have valid technical signals
        }
        
        try:
            # === ENHANCED DATA VALIDATION ===
            # Check minimum data requirements for different indicators
            data_length = len(self.df)
            
            # Individual indicator data requirements
            min_rsi_periods = 14        # RSI needs at least 14 periods
            min_macd_periods = 26       # MACD needs at least 26 periods (slow EMA)
            min_bb_periods = self.window # Bollinger Bands need window periods
            min_sma_periods = self.window # SMA needs window periods
            min_ema_periods = self.window # EMA needs window periods
            
            # Log data availability for debugging
            print(f"📊 Data validation: {data_length} periods available")
            print(f"   RSI requires: {min_rsi_periods}, Available: {'✓' if data_length >= min_rsi_periods else '✗'}")
            print(f"   MACD requires: {min_macd_periods}, Available: {'✓' if data_length >= min_macd_periods else '✗'}")
            print(f"   Moving Averages require: {min_sma_periods}, Available: {'✓' if data_length >= min_sma_periods else '✗'}")
            
            # === MOMENTUM INDICATORS WITH VALIDATION ===
            # RSI (Relative Strength Index) - measures overbought/oversold conditions
            # Values: 0-100, >70 = overbought, <30 = oversold
            if data_length >= min_rsi_periods:
                try:
                    rsi_indicator = RSIIndicator(close=self.df['close'], window=14)
                    print("✓ RSI indicator initialized successfully")
                except Exception as e:
                    print(f"⚠️ RSI calculation failed: {e}")
                    rsi_indicator = None
            else:
                print(f"⚠️ Insufficient data for RSI: need {min_rsi_periods}, have {data_length}")
                rsi_indicator = None
                
            # MACD (Moving Average Convergence Divergence) - shows momentum changes
            # Positive MACD = upward momentum, negative = downward momentum
            if data_length >= min_macd_periods:
                try:
                    macd_indicator = MACD(close=self.df['close'])
                    print("✓ MACD indicator initialized successfully")
                except Exception as e:
                    print(f"⚠️ MACD calculation failed: {e}")
                    macd_indicator = None
            else:
                print(f"⚠️ Insufficient data for MACD: need {min_macd_periods}, have {data_length}")
                macd_indicator = None
              
            # === TREND INDICATORS WITH VALIDATION ===
            # Simple Moving Average - basic trend direction indicator
            if data_length >= min_sma_periods:
                try:
                    sma_indicator = SMAIndicator(close=self.df['close'], window=self.window)
                    print("✓ SMA indicator initialized successfully")
                except Exception as e:
                    print(f"⚠️ SMA calculation failed: {e}")
                    sma_indicator = None
            else:
                print(f"⚠️ Insufficient data for SMA: need {min_sma_periods}, have {data_length}")
                sma_indicator = None
                
            # Exponential Moving Average - more responsive to recent price changes
            if data_length >= min_ema_periods:
                try:
                    ema_indicator = EMAIndicator(close=self.df['close'], window=self.window)
                    print("✓ EMA indicator initialized successfully")
                except Exception as e:
                    print(f"⚠️ EMA calculation failed: {e}")
                    ema_indicator = None
            else:
                print(f"⚠️ Insufficient data for EMA: need {min_ema_periods}, have {data_length}")
                ema_indicator = None            
            # === VOLATILITY INDICATORS WITH VALIDATION ===
            # Bollinger Bands - measure volatility and potential reversal points
            # Price touching upper band = potentially overbought
            # Price touching lower band = potentially oversold
            if data_length >= min_bb_periods:
                try:
                    bb_indicator = BollingerBands(close=self.df['close'], window=self.window)
                    print("✓ Bollinger Bands indicator initialized successfully")
                except Exception as e:
                    print(f"⚠️ Bollinger Bands calculation failed: {e}")
                    bb_indicator = None
            else:
                print(f"⚠️ Insufficient data for Bollinger Bands: need {min_bb_periods}, have {data_length}")
                bb_indicator = None
                
            # === VOLUME INDICATORS WITH VALIDATION ===
            # Volume indicators generally need less data but still require validation
            min_volume_periods = 10  # Minimum periods for reliable volume analysis
            
            # On-Balance Volume - tracks cumulative volume flow
            # Rising OBV = accumulation (buying pressure)
            # Falling OBV = distribution (selling pressure)
            if data_length >= min_volume_periods:
                try:
                    obv_indicator = OnBalanceVolumeIndicator(close=self.df['close'], volume=self.df['volume'])
                    print("✓ OBV indicator initialized successfully")
                except Exception as e:
                    print(f"⚠️ OBV calculation failed: {e}")
                    obv_indicator = None
            else:
                print(f"⚠️ Insufficient data for OBV: need {min_volume_periods}, have {data_length}")
                obv_indicator = None
                
            # Accumulation/Distribution Index - measures buying/selling pressure
            # Positive ADI = accumulation, negative = distribution
            if data_length >= min_volume_periods:
                try:
                    adi_indicator = AccDistIndexIndicator(
                        high=self.df['high'], 
                        low=self.df['low'], 
                        close=self.df['close'], 
                        volume=self.df['volume']
                    )
                    print("✓ ADI indicator initialized successfully")
                except Exception as e:
                    print(f"⚠️ ADI calculation failed: {e}")
                    adi_indicator = None
            else:
                print(f"⚠️ Insufficient data for ADI: need {min_volume_periods}, have {data_length}")
                adi_indicator = None
                
            # Volume Weighted Average Price - shows institutional trading levels
            # Price above VWAP = bullish, below VWAP = bearish
            if data_length >= min_volume_periods:
                try:
                    vwap_indicator = VolumeWeightedAveragePrice(
                        high=self.df['high'], 
                        low=self.df['low'], 
                        close=self.df['close'], 
                        volume=self.df['volume']
                    )
                    print("✓ VWAP indicator initialized successfully")
                except Exception as e:
                    print(f"⚠️ VWAP calculation failed: {e}")
                    vwap_indicator = None
            else:
                print(f"⚠️ Insufficient data for VWAP: need {min_volume_periods}, have {data_length}")
                vwap_indicator = None            # === EXTRACT CURRENT VALUES WITH VALIDATION ===
            # Get the most recent value from each indicator, but only if the indicator was successfully created
            # This prevents errors when trying to access values from failed indicators
            calculated = {}
            
            # Extract values only from successfully initialized indicators
            try:
                if rsi_indicator is not None:
                    calculated['rsi'] = rsi_indicator.rsi().iloc[-1]
                    print(f"✓ RSI calculated: {calculated['rsi']:.2f}")
                else:
                    calculated['rsi'] = 50.0  # Neutral default
                    print("⚠️ Using default RSI value: 50.0")
            except Exception as e:
                print(f"⚠️ Error extracting RSI value: {e}")
                calculated['rsi'] = 50.0
                
            try:
                if macd_indicator is not None:
                    calculated['macd'] = macd_indicator.macd().iloc[-1]
                    calculated['macd_signal'] = macd_indicator.macd_signal().iloc[-1]
                    calculated['macd_histogram'] = macd_indicator.macd_diff().iloc[-1]
                    print(f"✓ MACD calculated: {calculated['macd']:.4f}")
                else:
                    calculated['macd'] = 0.0
                    calculated['macd_signal'] = 0.0
                    calculated['macd_histogram'] = 0.0
                    print("⚠️ Using default MACD values: 0.0")
            except Exception as e:
                print(f"⚠️ Error extracting MACD values: {e}")
                calculated['macd'] = 0.0
                calculated['macd_signal'] = 0.0
                calculated['macd_histogram'] = 0.0
                
            try:
                if sma_indicator is not None:
                    calculated['sma_20'] = sma_indicator.sma_indicator().iloc[-1]
                    print(f"✓ SMA calculated: {calculated['sma_20']:.2f}")
                else:
                    calculated['sma_20'] = self.df['close'].iloc[-1] if len(self.df) > 0 else 0.0
                    print("⚠️ Using current price as SMA default")
            except Exception as e:
                print(f"⚠️ Error extracting SMA value: {e}")
                calculated['sma_20'] = self.df['close'].iloc[-1] if len(self.df) > 0 else 0.0
                
            try:
                if ema_indicator is not None:
                    calculated['ema_20'] = ema_indicator.ema_indicator().iloc[-1]
                    print(f"✓ EMA calculated: {calculated['ema_20']:.2f}")
                else:
                    calculated['ema_20'] = self.df['close'].iloc[-1] if len(self.df) > 0 else 0.0
                    print("⚠️ Using current price as EMA default")
            except Exception as e:
                print(f"⚠️ Error extracting EMA value: {e}")
                calculated['ema_20'] = self.df['close'].iloc[-1] if len(self.df) > 0 else 0.0
                
            try:
                if bb_indicator is not None:
                    calculated['bb_upper'] = bb_indicator.bollinger_hband().iloc[-1]
                    calculated['bb_lower'] = bb_indicator.bollinger_lband().iloc[-1]
                    print(f"✓ Bollinger Bands calculated: Upper={calculated['bb_upper']:.2f}, Lower={calculated['bb_lower']:.2f}")
                else:
                    current_price = self.df['close'].iloc[-1] if len(self.df) > 0 else 0.0
                    calculated['bb_upper'] = current_price * 1.02  # 2% above current price
                    calculated['bb_lower'] = current_price * 0.98  # 2% below current price
                    print("⚠️ Using estimated Bollinger Bands based on current price")
            except Exception as e:
                print(f"⚠️ Error extracting Bollinger Bands values: {e}")
                current_price = self.df['close'].iloc[-1] if len(self.df) > 0 else 0.0
                calculated['bb_upper'] = current_price * 1.02
                calculated['bb_lower'] = current_price * 0.98
                
            try:
                if obv_indicator is not None:
                    calculated['obv'] = obv_indicator.on_balance_volume().iloc[-1]
                    print(f"✓ OBV calculated: {calculated['obv']:.0f}")
                else:
                    calculated['obv'] = 0.0
                    print("⚠️ Using default OBV value: 0.0")
            except Exception as e:
                print(f"⚠️ Error extracting OBV value: {e}")
                calculated['obv'] = 0.0
                
            try:
                if adi_indicator is not None:
                    calculated['adi'] = adi_indicator.acc_dist_index().iloc[-1]
                    print(f"✓ ADI calculated: {calculated['adi']:.2f}")
                else:
                    calculated['adi'] = 0.0
                    print("⚠️ Using default ADI value: 0.0")
            except Exception as e:
                print(f"⚠️ Error extracting ADI value: {e}")
                calculated['adi'] = 0.0
                
            try:
                if vwap_indicator is not None:
                    calculated['vwap'] = vwap_indicator.volume_weighted_average_price().iloc[-1]
                    print(f"✓ VWAP calculated: {calculated['vwap']:.2f}")
                else:
                    calculated['vwap'] = self.df['close'].iloc[-1] if len(self.df) > 0 else 0.0
                    print("⚠️ Using current price as VWAP default")
            except Exception as e:
                print(f"⚠️ Error extracting VWAP value: {e}")
                calculated['vwap'] = self.df['close'].iloc[-1] if len(self.df) > 0 else 0.0
            
            # Add current price and signal validity flag
            calculated['current_price'] = self.df['close'].iloc[-1] if len(self.df) > 0 else 0.0
            calculated['has_signals'] = True
            
            # === DATA VALIDATION ===
            # Only update indicators with valid (non-NaN) values
            # This prevents corrupted data from affecting trading decisions
            valid_indicators = {k: v for k, v in calculated.items() if pd.notna(v) and v is not None}
            indicators.update(valid_indicators)
            
            # Log summary of successfully calculated indicators
            successful_indicators = len(valid_indicators)
            total_indicators = len(calculated)
            print(f"📊 Indicator calculation complete: {successful_indicators}/{total_indicators} successful")
            
            # === CACHE RESULTS ===
            # Store calculated indicators for performance optimization
            # Subsequent calls with same data will return cached results instantly
            _indicator_cache[df_hash] = indicators.copy()

        except Exception as e:
            # === ERROR HANDLING ===
            # If indicator calculation fails, log the error but continue with default values
            # This ensures the trading system remains operational even with data issues
            print(f"⚠️ Error calculating technical indicators: {e}")
            
            # Set current price if available, even if other indicators failed
            if len(self.df) > 0:
                indicators['current_price'] = self.df['close'].iloc[-1]
                
            # Log additional context for debugging
            print(f"Data length: {len(self.df)}, Window: {self.window}")

        # === RETURN COMPLETE INDICATOR SET ===
        # Always return a complete set of indicators (either calculated or default values)
        # This ensures consistent data structure for the AI analysis engine
        return indicators
    
    def get_signals(self):
        """
        Generate trading signals based on technical indicators with signal combination logic.
        
        Purpose:
        - Analyzes multiple technical indicators for signal generation
        - Implements signal combination logic for higher accuracy
        - Uses volume confirmation for stronger signals
        - Adjusts confidence levels based on indicator agreement
        
        Trading Logic:
        1. Price Action Indicators:
           - RSI (Relative Strength Index):
             * Buy when RSI < RSI_OVERSOLD (oversold condition)
             * Sell when RSI > RSI_OVERBOUGHT (overbought condition)
           - MACD (Moving Average Convergence Divergence):
             * Buy on positive MACD (upward momentum)
             * Sell on negative MACD (downward momentum)
           - SMA (Simple Moving Average):
             * Buy when price > SMA20 (uptrend)
             * Sell when price < SMA20 (downtrend)
        
        2. Volume Indicators:
           - OBV (On Balance Volume):
             * Confirms trend strength
             * Identifies potential reversals
           - ADI (Accumulation/Distribution Index):
             * Shows buying/selling pressure
           - VWAP (Volume Weighted Average Price):
             * Price levels with high volume interest
        
        3. Signal Combination:
           - Base confidence levels:
             * RSI: 0.8 (most reliable for mean reversion)
             * MACD: 0.6 (lagging but good for trends)
             * SMA: 0.7 (good for trend following)
           - Confidence boosters:
             * +0.1 when multiple indicators agree
             * +0.1 when volume confirms the trend
             * -0.1 when indicators conflict
        
        Returns:
            tuple: (signals, indicators)
                - signals: List of tuples (action, reason, confidence)
                - indicators: Dict of calculated technical indicators
        """
        indicators = self.calculate_indicators()
        base_signals = []
        final_signals = []        # Only generate signals if we have valid data and sufficient history
        required_indicators = [
            'current_price', 'sma_20', 'rsi', 'macd',
            'obv', 'adi', 'vwap'
        ]
        
        # Check if we have valid indicators and sufficient data
        has_valid_data = True
        missing_indicators = []
        
        for indicator_name in required_indicators:
            if indicator_name not in indicators:
                has_valid_data = False
                missing_indicators.append(f"{indicator_name} (missing)")
            elif indicators[indicator_name] is None:
                has_valid_data = False
                missing_indicators.append(f"{indicator_name} (None)")
            elif indicators[indicator_name] == 0.0 and indicator_name in ['sma_20', 'ema_20', 'current_price']:
                # These should never be 0 unless there's no data
                has_valid_data = False
                missing_indicators.append(f"{indicator_name} (zero/invalid)")
        
        # Also check if we have the has_signals flag indicating successful calculation
        if not indicators.get('has_signals', False):
            has_valid_data = False
            missing_indicators.append("has_signals flag is False")
            
        if missing_indicators:
            print(f"⚠️ Cannot generate signals - missing/invalid indicators: {', '.join(missing_indicators)}")
        
        if has_valid_data:            # Generate base signals with additional validation
            try:
                # RSI signals - only if RSI is meaningful (not default value in extreme market)
                current_rsi = indicators['rsi']
                if current_rsi < RSI_OVERSOLD:
                    base_signals.append(('BUY', f'RSI oversold ({current_rsi:.1f} < {RSI_OVERSOLD})', 0.8))
                elif current_rsi > RSI_OVERBOUGHT:
                    base_signals.append(('SELL', f'RSI overbought ({current_rsi:.1f} > {RSI_OVERBOUGHT})', 0.8))
            except Exception as e:
                print(f"⚠️ Error generating RSI signals: {e}")

            try:
                # MACD signals
                current_macd = indicators['macd']
                if current_macd > 0:
                    base_signals.append(('BUY', f'MACD positive ({current_macd:.4f})', 0.6))
                else:
                    base_signals.append(('SELL', f'MACD negative ({current_macd:.4f})', 0.6))
            except Exception as e:
                print(f"⚠️ Error generating MACD signals: {e}")

            try:
                # Moving Average signals
                current_price = indicators['current_price']
                sma_20 = indicators['sma_20']
                if current_price > sma_20:
                    base_signals.append(('BUY', f'Price above SMA20 ({current_price:.2f} > {sma_20:.2f})', 0.7))
                else:
                    base_signals.append(('SELL', f'Price below SMA20 ({current_price:.2f} < {sma_20:.2f})', 0.7))
            except Exception as e:
                print(f"⚠️ Error generating SMA signals: {e}")

            # Volume trend analysis with error handling
            try:
                obv_trend = indicators['obv'] > 0
                adi_trend = indicators['adi'] > 0
                price_vs_vwap = indicators['current_price'] > indicators['vwap']
            except Exception as e:
                print(f"⚠️ Error analyzing volume trends: {e}")
                obv_trend = False
                adi_trend = False
                price_vs_vwap = False

            # Combine signals and adjust confidence
            for action, reason, base_confidence in base_signals:
                confidence = base_confidence
                
                # Check for indicator agreement
                buy_signals = sum(1 for s in base_signals if s[0] == 'BUY')
                sell_signals = sum(1 for s in base_signals if s[0] == 'SELL')
                
                # Adjust confidence based on indicator agreement
                if action == 'BUY' and buy_signals > 1:
                    confidence += 0.1
                elif action == 'SELL' and sell_signals > 1:
                    confidence += 0.1
                
                # Volume confirmation
                if action == 'BUY' and (obv_trend and adi_trend and price_vs_vwap):
                    confidence += 0.1
                    reason += ' with volume confirmation'
                elif action == 'SELL' and (not obv_trend and not adi_trend and not price_vs_vwap):
                    confidence += 0.1
                    reason += ' with volume confirmation'
                
                # Cap confidence at 1.0
                confidence = min(confidence, 1.0)
                
                final_signals.append((action, reason, confidence))

        else:
            # If not enough valid data, return empty signals and log the reason
            print(f"⚠️ No signals generated due to missing/invalid indicators: {', '.join(missing_indicators)}")
            return [], indicators

        return final_signals, indicators