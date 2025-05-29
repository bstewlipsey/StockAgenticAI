"""
data_structures.py

This module defines common data structures used across various bots in the
Agentic Stock AI system. Centralizing these structures promotes consistency,
reusability, and easier maintenance.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

class ActionSignal(Enum):
    """Represents the trading action signal."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    NO_SIGNAL = "NO_SIGNAL"

@dataclass(frozen=True)
class TradingDecision:
    """
    Represents a trading decision made by the DecisionMakerBot.
    This structure is intended to be immutable once created.
    """
    symbol: str
    signal: ActionSignal
    confidence: float  # e.g., 0.0 to 1.0
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    rationale: Optional[str] = "No rationale provided."
    # Optional field for any additional metadata from the AI or analysis
    metadata: Dict[str, Any] = field(default_factory=dict)
    final_action: Optional[str] = None  # Always set to signal.value (for test compatibility)

@dataclass
class AssetAnalysisInput:
    """
    Represents the input data required for an AIBot to perform analysis on an asset.
    """
    symbol: str
    market_data: Dict[str, Any] # e.g., current price, volume, historical data
    technical_indicators: Dict[str, Any] # e.g., RSI, MACD values
    news_sentiment: Optional[Dict[str, Any]] = None # e.g., {'overall_sentiment': 'positive', 'key_articles': [...]}
    reflection_insights: Optional[List[str]] = None # Relevant past reflections
    historical_ai_context: Optional[List[Dict[str, Any]]] = None # Past AI decisions/analyses for this symbol
    asset_type: Optional[str] = None  # Added for orchestrator/test compatibility

@dataclass(frozen=True)
class TradeOutcome:
    """
    Represents the outcome of a completed trade for reflection.
    """
    trade_id: str
    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    profit_loss: float
    profit_loss_percent: float
    initial_decision: TradingDecision # The decision that initiated this trade
    # Add any other relevant details like entry/exit timestamps, fees, etc.

@dataclass
class NewsArticle:
    """Represents a news article and its embedding for RAG."""
    title: str
    url: str
    date: str
    source: str
    full_text: str
    embedding: Optional[List[float]] = None

@dataclass(frozen=True)
class TradingSignal:
    """Represents a trading signal output by DecisionMakerBot."""
    symbol: str
    signal: str  # 'buy', 'sell', 'hold'
    confidence: float
    rationale: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MarketSnapshot:
    """Represents a snapshot of current market data for context sharing."""
    timestamp: str
    market_sentiment: str
    volatility: float
    top_sectors: List[str]
    trending_assets: List[str]
    additional_data: Dict[str, Any] = field(default_factory=dict)

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
    asset_type: str = "stock"  # Default to stock, set to 'crypto' for crypto assets
    allocation_usd: Optional[float] = None  # USD allocation for this asset

@dataclass
class LLMAnalysisMemory:
    """Stores a past LLM analysis for memory-based decision making."""
    symbol: str
    action: str
    confidence: float
    reasoning: str
    timestamp: str
    news_sentiment: Optional[Dict[str, Any]] = None
    technical_indicators: Optional[Dict[str, Any]] = None
    market_data: Optional[Dict[str, Any]] = None
    reflection_insights: Optional[List[str]] = None
    ai_model: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)