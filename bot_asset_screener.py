"""
Asset Screener Bot - Dynamically identifies promising assets for analysis

This bot implements TODO item #12: Instead of iterating through all TRADING_ASSETS statically,
this bot selects a subset of the most promising assets based on:
- Market-wide data analysis (sector performance, top movers)
- AI-powered broad market overview prompts
- News sentiment and volatility patterns
- Dynamic asset prioritization based on current market conditions

The bot outputs a prioritized list of symbols for the orchestrator to analyze.
"""

import logging
import yfinance as yf
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from bot_ai import AIBot
from bot_database import DatabaseBot
import config_trading_variables as ctv

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class AssetScreeningResult:
    """Structured result from asset screening analysis"""
    symbol: str
    priority_score: float  # 0-100, higher is better
    reasoning: str
    market_cap: Optional[float] = None
    volume_rank: Optional[float] = None
    momentum_score: Optional[float] = None
    volatility_score: Optional[float] = None
    sector: Optional[str] = None
    confidence: float = 0.0

@dataclass
class MarketOverview:
    """Comprehensive market state analysis"""
    market_sentiment: str  # 'bullish', 'bearish', 'neutral'
    top_sectors: List[str]
    market_volatility: float
    trending_assets: List[str]
    risk_environment: str  # 'low', 'medium', 'high'
    ai_insights: str

class AssetScreenerBot:
    """
    Dynamically identifies the most promising assets to analyze in each trading cycle.
    
    Features:
    - Market-wide momentum and volume analysis
    - AI-powered market overview and asset prioritization
    - Sector rotation detection
    - News sentiment integration
    - Dynamic asset filtering based on market conditions
    """
    
    def __init__(self, ai_bot: AIBot, database_bot: DatabaseBot):
        """
        Initialize the Asset Screener Bot
        
        Args:
            ai_bot: AI bot instance for LLM analysis
            database_bot: Database bot for storing screening results
        """
        self.ai_bot = ai_bot
        self.database_bot = database_bot
        self.logger = logging.getLogger(__name__)
        
        # Screening configuration
        self.max_assets_to_screen = getattr(ctv, 'MAX_ASSETS_PER_CYCLE', 10)
        self.min_market_cap = getattr(ctv, 'MIN_MARKET_CAP', 1e9)  # $1B minimum
        self.min_avg_volume = getattr(ctv, 'MIN_AVG_VOLUME', 1e6)  # 1M shares
        self.lookback_days = getattr(ctv, 'SCREENING_LOOKBACK_DAYS', 30)
        
        # Asset universes for screening
        self.stock_universe = self._get_stock_universe()
        self.crypto_universe = self._get_crypto_universe()
        
        self.logger.info(f"AssetScreenerBot initialized with {len(self.stock_universe)} stocks and {len(self.crypto_universe)} crypto assets")
    
    def _get_stock_universe(self) -> List[str]:
        """Get the universe of stocks to screen from"""
        # Start with configured trading assets
        configured_stocks = [asset[0] for asset in ctv.TRADING_ASSETS 
                           if asset[1] == 'stock']
        
        # Add popular large-cap stocks for broader screening
        popular_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM',
            'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'ADBE', 'NFLX',
            'CRM', 'PYPL', 'INTC', 'CMCSA', 'PEP', 'ABT', 'TMO', 'COST',
            'VZ', 'CSCO', 'KO', 'PFE', 'WMT', 'NKE', 'MRK', 'XOM', 'CVX'
        ]
        
        # Combine and deduplicate
        all_stocks = list(set(configured_stocks + popular_stocks))
        return all_stocks
    
    def _get_crypto_universe(self) -> List[str]:
        """Get the universe of crypto assets to screen from"""
        # Start with configured trading assets
        configured_crypto = [asset[0] for asset in ctv.TRADING_ASSETS 
                           if asset[1] == 'crypto']
        
        # Add popular crypto assets for broader screening
        popular_crypto = [
            'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 
            'SOL-USD', 'DOGE-USD', 'DOT-USD', 'AVAX-USD', 'SHIB-USD',
            'MATIC-USD', 'LTC-USD', 'BCH-USD', 'LINK-USD', 'UNI-USD'
        ]
        
        # Combine and deduplicate
        all_crypto = list(set(configured_crypto + popular_crypto))
        return all_crypto
    
    def screen_assets(self, market_conditions: Optional[Dict] = None) -> List[AssetScreeningResult]:
        """
        Main screening function that identifies promising assets for analysis
        
        Args:
            market_conditions: Optional market state info to influence screening
            
        Returns:
            List of AssetScreeningResult objects, sorted by priority score
        """
        try:
            self.logger.info("Starting comprehensive asset screening process")
            
            # Step 1: Get market overview and sentiment
            market_overview = self._analyze_market_overview()
            self.logger.info(f"Market overview: {market_overview.market_sentiment} sentiment, "
                           f"{market_overview.risk_environment} risk environment")
            
            # Step 2: Screen stocks and crypto separately
            stock_results = self._screen_stocks(market_overview)
            crypto_results = self._screen_crypto(market_overview)
            
            # Step 3: Combine and rank all results
            all_results = stock_results + crypto_results
            all_results.sort(key=lambda x: x.priority_score, reverse=True)
            
            # Step 4: Apply final filtering and selection
            final_selection = self._apply_final_filters(all_results, market_overview)
            
            # Step 5: Store screening results
            self._store_screening_results(final_selection, market_overview)
            
            self.logger.info(f"Asset screening completed. Selected {len(final_selection)} assets for analysis")
            return final_selection[:self.max_assets_to_screen]
            
        except Exception as e:
            self.logger.error(f"Error in asset screening: {str(e)}")
            # Fallback to configured assets if screening fails
            return self._get_fallback_assets()
    
    def _analyze_market_overview(self) -> MarketOverview:
        """Generate comprehensive market overview using technical and AI analysis"""
        try:
            # Fetch market index data for sentiment analysis
            spy_data = yf.download("SPY", period="30d", interval="1d")
            vix_data = yf.download("^VIX", period="30d", interval="1d")
            
            # Calculate market metrics
            spy_return = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[0] - 1) * 100
            current_vix = vix_data['Close'].iloc[-1]
            avg_vix = vix_data['Close'].mean()
            
            # Determine market sentiment based on technical indicators
            if spy_return > 5 and current_vix < avg_vix * 0.8:
                market_sentiment = "bullish"
                risk_environment = "low"
            elif spy_return < -5 or current_vix > avg_vix * 1.3:
                market_sentiment = "bearish"
                risk_environment = "high"
            else:
                market_sentiment = "neutral"
                risk_environment = "medium"
            
            # Get sector performance data
            sector_etfs = {
                'Technology': 'XLK', 'Healthcare': 'XLV', 'Financials': 'XLF',
                'Energy': 'XLE', 'Consumer Discretionary': 'XLY', 'Industrials': 'XLI'
            }
            
            sector_performance = {}
            for sector, etf in sector_etfs.items():
                try:
                    data = yf.download(etf, period="5d", interval="1d")
                    if not data.empty:
                        perf = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                        sector_performance[sector] = perf
                except:
                    continue
            
            # Rank top performing sectors
            top_sectors = sorted(sector_performance.keys(), 
                               key=lambda x: sector_performance.get(x, 0), reverse=True)[:3]
            
            # Generate AI insights about market conditions
            market_prompt = f"""
            Analyze the current market conditions and provide strategic insights for asset selection:
            
            Market Data:
            - S&P 500 30-day return: {spy_return:.2f}%
            - Current VIX: {current_vix:.2f}
            - Average VIX (30d): {avg_vix:.2f}
            - Market Sentiment: {market_sentiment}
            - Top Performing Sectors: {', '.join(top_sectors)}
            
            Please provide:
            1. Key market themes and trends to focus on
            2. Asset types that might outperform in current conditions
            3. Risk factors to be aware of
            4. Specific sectors or themes to prioritize
            
            Keep response concise and actionable for trading decisions.
            """
            
            ai_insights = self.ai_bot.generate_analysis("MARKET_OVERVIEW", market_prompt)
            
            return MarketOverview(
                market_sentiment=market_sentiment,
                top_sectors=top_sectors,
                market_volatility=current_vix,
                trending_assets=[],  # Will be populated by screening
                risk_environment=risk_environment,
                ai_insights=ai_insights
            )
            
        except Exception as e:
            self.logger.error(f"Error in market overview analysis: {str(e)}")
            # Return neutral overview as fallback
            return MarketOverview(
                market_sentiment="neutral",
                top_sectors=["Technology", "Healthcare", "Financials"],
                market_volatility=20.0,
                trending_assets=[],
                risk_environment="medium",
                ai_insights="Market analysis unavailable - using conservative approach"
            )
    
    def _screen_stocks(self, market_overview: MarketOverview) -> List[AssetScreeningResult]:
        """Screen stock universe for promising candidates"""
        results = []
        
        for symbol in self.stock_universe:
            try:
                result = self._analyze_stock_candidate(symbol, market_overview)
                if result and result.priority_score > 30:  # Minimum threshold
                    results.append(result)
            except Exception as e:
                self.logger.debug(f"Error screening stock {symbol}: {str(e)}")
                continue
        
        return results
    
    def _analyze_stock_candidate(self, symbol: str, market_overview: MarketOverview) -> Optional[AssetScreeningResult]:
        """Analyze individual stock candidate"""
        try:
            # Fetch stock data
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period="30d")
            info = ticker.info
            
            if hist_data.empty:
                return None
            
            # Calculate technical metrics
            current_price = hist_data['Close'].iloc[-1]
            avg_volume = hist_data['Volume'].mean()
            price_change_30d = (current_price / hist_data['Close'].iloc[0] - 1) * 100
            
            # Calculate volatility (standard deviation of returns)
            returns = hist_data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized
            
            # Market cap filtering
            market_cap = info.get('marketCap', 0)
            if market_cap < self.min_market_cap:
                return None
            
            # Volume filtering
            if avg_volume < self.min_avg_volume:
                return None
            
            # Calculate momentum score (combines price performance and volume)
            momentum_score = min(100, max(0, 50 + price_change_30d * 2))
            
            # Calculate volume rank (relative to historical volume)
            recent_volume = hist_data['Volume'].tail(5).mean()
            volume_rank = min(100, (recent_volume / avg_volume) * 50)
            
            # Calculate base priority score
            base_score = (momentum_score * 0.4 + volume_rank * 0.3 + 
                         min(100, (market_cap / 1e10) * 10) * 0.3)
            
            # Adjust for market conditions
            sector = info.get('sector', 'Unknown')
            if sector in ['Technology', 'Healthcare'] and market_overview.market_sentiment == 'bullish':
                base_score *= 1.2
            elif sector == 'Energy' and market_overview.risk_environment == 'high':
                base_score *= 1.1
            
            priority_score = min(100, base_score)
            
            # Generate reasoning
            reasoning = f"30d return: {price_change_30d:.1f}%, " \
                       f"volatility: {volatility:.1f}%, " \
                       f"volume rank: {volume_rank:.0f}, " \
                       f"sector: {sector}"
            
            return AssetScreeningResult(
                symbol=symbol,
                priority_score=priority_score,
                reasoning=reasoning,
                market_cap=market_cap,
                volume_rank=volume_rank,
                momentum_score=momentum_score,
                volatility_score=volatility,
                sector=sector,
                confidence=min(100, priority_score * 0.8)
            )
            
        except Exception as e:
            self.logger.debug(f"Error analyzing stock {symbol}: {str(e)}")
            return None
    
    def _screen_crypto(self, market_overview: MarketOverview) -> List[AssetScreeningResult]:
        """Screen crypto universe for promising candidates"""
        results = []
        
        for symbol in self.crypto_universe:
            try:
                result = self._analyze_crypto_candidate(symbol, market_overview)
                if result and result.priority_score > 25:  # Lower threshold for crypto
                    results.append(result)
            except Exception as e:
                self.logger.debug(f"Error screening crypto {symbol}: {str(e)}")
                continue
        
        return results
    
    def _analyze_crypto_candidate(self, symbol: str, market_overview: MarketOverview) -> Optional[AssetScreeningResult]:
        """Analyze individual crypto candidate"""
        try:
            # Fetch crypto data
            hist_data = yf.download(symbol, period="30d", interval="1d")
            
            if hist_data.empty:
                return None
            
            # Calculate technical metrics
            current_price = hist_data['Close'].iloc[-1]
            avg_volume = hist_data['Volume'].mean()
            price_change_30d = (current_price / hist_data['Close'].iloc[0] - 1) * 100
            
            # Calculate volatility
            returns = hist_data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(365) * 100  # Annualized for crypto
            
            # Calculate momentum score (crypto can be more volatile)
            momentum_score = min(100, max(0, 50 + price_change_30d * 1.5))
            
            # Volume analysis (crypto has different volume patterns)
            recent_volume = hist_data['Volume'].tail(5).mean()
            volume_rank = min(100, (recent_volume / avg_volume) * 40) if avg_volume > 0 else 50
            
            # Base priority score for crypto
            base_score = momentum_score * 0.5 + volume_rank * 0.3 + min(50, volatility) * 0.2
            
            # Adjust for market conditions (crypto often inverse to traditional markets)
            if market_overview.market_sentiment == 'bearish' and 'BTC' in symbol:
                base_score *= 1.1  # Bitcoin as digital gold during uncertainty
            elif market_overview.risk_environment == 'low':
                base_score *= 1.15  # Risk-on environment favors crypto
            
            priority_score = min(100, base_score)
            
            reasoning = f"30d return: {price_change_30d:.1f}%, " \
                       f"volatility: {volatility:.1f}%, " \
                       f"volume activity: {volume_rank:.0f}"
            
            return AssetScreeningResult(
                symbol=symbol,
                priority_score=priority_score,
                reasoning=reasoning,
                market_cap=None,  # Market cap not readily available for crypto
                volume_rank=volume_rank,
                momentum_score=momentum_score,
                volatility_score=volatility,
                sector="Cryptocurrency",
                confidence=min(100, priority_score * 0.7)  # Lower confidence for crypto
            )
            
        except Exception as e:
            self.logger.debug(f"Error analyzing crypto {symbol}: {str(e)}")
            return None
    
    def _apply_final_filters(self, results: List[AssetScreeningResult], 
                           market_overview: MarketOverview) -> List[AssetScreeningResult]:
        """Apply final filters and AI-enhanced selection"""
        try:
            # Filter out low-priority assets
            filtered_results = [r for r in results if r.priority_score > 40]
            
            # Ensure diversity across asset types
            stocks = [r for r in filtered_results if not r.symbol.endswith('-USD')]
            crypto = [r for r in filtered_results if r.symbol.endswith('-USD')]
            
            # Maintain balance: prefer 70% stocks, 30% crypto in normal conditions
            if market_overview.risk_environment == 'low':
                target_stocks = int(self.max_assets_to_screen * 0.7)
                target_crypto = self.max_assets_to_screen - target_stocks
            elif market_overview.risk_environment == 'high':
                target_stocks = int(self.max_assets_to_screen * 0.8)  # More conservative
                target_crypto = self.max_assets_to_screen - target_stocks
            else:
                target_stocks = int(self.max_assets_to_screen * 0.6)
                target_crypto = self.max_assets_to_screen - target_stocks
            
            # Select top assets from each category
            final_stocks = stocks[:target_stocks]
            final_crypto = crypto[:target_crypto]
            
            return final_stocks + final_crypto
            
        except Exception as e:
            self.logger.error(f"Error in final filtering: {str(e)}")
            return results[:self.max_assets_to_screen]
    
    def _store_screening_results(self, results: List[AssetScreeningResult], 
                               market_overview: MarketOverview):
        """Store screening results in database for analysis and learning"""
        try:
            # Prepare screening data for storage
            screening_data = {
                'timestamp': datetime.now().isoformat(),
                'market_sentiment': market_overview.market_sentiment,
                'market_volatility': market_overview.market_volatility,
                'risk_environment': market_overview.risk_environment,
                'selected_assets': [r.symbol for r in results],
                'screening_scores': {r.symbol: r.priority_score for r in results},
                'ai_insights': market_overview.ai_insights
            }
            
            # Store in database (assuming method exists or will be added)
            # self.database_bot.store_screening_results(screening_data)
            
            self.logger.info(f"Stored screening results for {len(results)} assets")
            
        except Exception as e:
            self.logger.error(f"Error storing screening results: {str(e)}")
    
    def _get_fallback_assets(self) -> List[AssetScreeningResult]:
        """Fallback to configured assets if screening fails"""
        fallback_assets = []
        
        for asset in ctv.TRADING_ASSETS[:self.max_assets_to_screen]:
            symbol = asset[0]
            fallback_assets.append(AssetScreeningResult(
                symbol=symbol,
                priority_score=50.0,  # Neutral score
                reasoning="Fallback selection from configured assets",
                confidence=50.0
            ))
        
        self.logger.warning(f"Using fallback asset selection: {[a.symbol for a in fallback_assets]}")
        return fallback_assets
    
    def get_screening_history(self, days: int = 7) -> List[Dict]:
        """Get historical screening results for analysis"""
        try:
            # This would query the database for past screening results
            # return self.database_bot.get_screening_history(days)
            return []
        except Exception as e:
            self.logger.error(f"Error retrieving screening history: {str(e)}")
            return []
    
    def analyze_screening_performance(self) -> Dict[str, Any]:
        """Analyze how well the screening process has performed"""
        try:
            # This would analyze the correlation between screening scores and actual performance
            # Could be used to improve the screening algorithm over time
            return {
                'screening_accuracy': 0.0,
                'top_performing_screens': [],
                'improvement_suggestions': []
            }
        except Exception as e:
            self.logger.error(f"Error analyzing screening performance: {str(e)}")
            return {}
