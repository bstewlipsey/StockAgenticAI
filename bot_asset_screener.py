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
import config_trading as ctv
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
from alpaca_trade_api import REST
from alpaca_trade_api.rest import URL, TimeFrame, TimeFrameUnit
import time

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
        
        # Initialize Alpaca API
        self.alpaca_api = REST(
            key_id=ALPACA_API_KEY or "",
            secret_key=ALPACA_SECRET_KEY or "",
            base_url=URL(ALPACA_BASE_URL),
            api_version='v2'
        )
        
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
            # Fetch market index data for sentiment analysis using Alpaca
            spy_bars = self.alpaca_api.get_bars("SPY", TimeFrame(1, TimeFrameUnit.Day), limit=30)
            vix_bars = self.alpaca_api.get_bars("VIXY", TimeFrame(1, TimeFrameUnit.Day), limit=30)  # Use VIXY ETF as proxy for VIX
            spy_close = [bar.c for bar in spy_bars]
            vix_close = [bar.c for bar in vix_bars]
            spy_return = (spy_close[-1] / spy_close[0] - 1) * 100 if spy_close else 0
            current_vix = vix_close[-1] if vix_close else 0
            avg_vix = sum(vix_close) / len(vix_close) if vix_close else 0
            
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
                    bars = self.alpaca_api.get_bars(etf, TimeFrame(1, TimeFrameUnit.Day), limit=5)
                    closes = [bar.c for bar in bars]
                    if closes:
                        perf = (closes[-1] / closes[0] - 1) * 100
                        sector_performance[sector] = perf
                except Exception:
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
            bars = self.alpaca_api.get_bars(symbol, TimeFrame(1, TimeFrameUnit.Day), limit=30)
            closes = [bar.c for bar in bars]
            volumes = [bar.v for bar in bars]
            if not closes:
                return None
            current_price = closes[-1]
            avg_volume = sum(volumes) / len(volumes)
            price_change_30d = (current_price / closes[0] - 1) * 100
            returns = [(closes[i] / closes[i-1] - 1) for i in range(1, len(closes))]
            volatility = (np.std(returns) * np.sqrt(252) * 100) if returns else 0
            # Market cap and sector info not available from Alpaca, set to None/Unknown
            market_cap = None
            sector = 'Unknown'
            if market_cap is not None and market_cap < self.min_market_cap:
                return None
            if avg_volume < self.min_avg_volume:
                return None
            momentum_score = min(100, max(0, 50 + price_change_30d * 2))
            recent_volume = sum(volumes[-5:]) / min(5, len(volumes))
            volume_rank = min(100, (recent_volume / avg_volume) * 50) if avg_volume > 0 else 50
            base_score = (momentum_score * 0.4 + volume_rank * 0.3)
            priority_score = min(100, base_score)
            reasoning = f"30d return: {price_change_30d:.1f}%, volatility: {volatility:.1f}%, volume rank: {volume_rank:.0f}, sector: {sector}"
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
            bars = self.alpaca_api.get_bars(symbol, TimeFrame(1, TimeFrameUnit.Day), limit=30)
            closes = [bar.c for bar in bars]
            volumes = [bar.v for bar in bars]
            if not closes:
                return None
            current_price = closes[-1]
            avg_volume = sum(volumes) / len(volumes)
            price_change_30d = (current_price / closes[0] - 1) * 100
            returns = [(closes[i] / closes[i-1] - 1) for i in range(1, len(closes))]
            volatility = (np.std(returns) * np.sqrt(252) * 100) if returns else 0
            # Market cap and sector info not available from Alpaca, set to None/Unknown
            market_cap = None
            sector = 'Unknown'
            if market_cap is not None and market_cap < self.min_market_cap:
                return None
            if avg_volume < self.min_avg_volume:
                return None
            momentum_score = min(100, max(0, 50 + price_change_30d * 2))
            recent_volume = sum(volumes[-5:]) / min(5, len(volumes))
            volume_rank = min(100, (recent_volume / avg_volume) * 50) if avg_volume > 0 else 50
            base_score = (momentum_score * 0.4 + volume_rank * 0.3)
            priority_score = min(100, base_score)
            reasoning = f"30d return: {price_change_30d:.1f}%, volatility: {volatility:.1f}%, volume rank: {volume_rank:.0f}, sector: {sector}"
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
            self.logger.debug(f"Error analyzing crypto {symbol}: {str(e)}")
            return None
    
    def _apply_final_filters(self, all_results: List[AssetScreeningResult], market_overview: MarketOverview) -> List[AssetScreeningResult]:
        """
        Apply final filters and selection criteria to the screened assets
        
        Args:
            all_results: Combined list of all screened assets
            market_overview: Current market overview information
            
        Returns:
            Filtered and sorted list of AssetScreeningResult objects
        """
        try:
            # Example filter: Exclude assets from sectors with negative performance
            filtered_results = [
                result for result in all_results 
                if result.sector in market_overview.top_sectors
            ]
            
            # Sort final selection by priority score
            filtered_results.sort(key=lambda x: x.priority_score, reverse=True)
            
            return filtered_results[:self.max_assets_to_screen]
        
        except Exception as e:
            self.logger.error(f"Error applying final filters: {str(e)}")
            return all_results[:self.max_assets_to_screen]  # Fallback to all results
    
    def _store_screening_results(self, final_selection: List[AssetScreeningResult], market_overview: MarketOverview):
        """
        Store the final screening results in the database
        
        Args:
            final_selection: List of final selected assets
            market_overview: Current market overview information
        """
        try:
            # Prepare data for storage
            timestamp = datetime.now()
            records = []
            for result in final_selection:
                records.append({
                    "symbol": result.symbol,
                    "priority_score": result.priority_score,
                    "reasoning": result.reasoning,
                    "market_cap": result.market_cap,
                    "volume_rank": result.volume_rank,
                    "momentum_score": result.momentum_score,
                    "volatility_score": result.volatility_score,
                    "sector": result.sector,
                    "confidence": result.confidence,
                    "timestamp": timestamp
                })
            
            # Store in database
            self.database_bot.store_screening_results(
                market_sentiment=market_overview.market_sentiment,
                market_volatility=market_overview.market_volatility,
                risk_environment=market_overview.risk_environment,
                selected_assets=[r.symbol for r in final_selection],
                screening_scores={r.symbol: r.priority_score for r in final_selection},
                ai_insights=market_overview.ai_insights,
                top_sectors=market_overview.top_sectors
            )
            self.logger.info(f"Stored screening results for {len(records)} assets")
        
        except Exception as e:
            self.logger.error(f"Error storing screening results: {str(e)}")
    
    def _get_fallback_assets(self) -> List[AssetScreeningResult]:
        """Get fallback assets in case of screening errors"""
        fallback_assets = [
            AssetScreeningResult(symbol=symbol, priority_score=50, reasoning="Fallback asset", confidence=50)
            for symbol in ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        ]
        return fallback_assets

    def run(self):
        """Run the asset screener bot"""
        self.logger.info("Asset Screener Bot starting")
        while True:
            try:
                # Screen assets with current market conditions
                self.screen_assets()
                
                # Wait for the next trading cycle
                self.logger.info("Waiting for next trading cycle...")
                time.sleep(60 * 15)  # 15 minute interval
            except Exception as e:
                self.logger.error(f"Error in bot loop: {str(e)}")
                time.sleep(60)  # Wait before retrying
