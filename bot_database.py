"""
DatabaseBot: Persistent storage and analytics for trading signals and performance.
- Uses SQLite for lightweight, file-based storage
- Methods for saving, retrieving, and analyzing trading analyses
- Enhanced support for reflection insights, screening results, and RAG functionality
- Designed for integration with agentic trading systems
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from data_structures import AgenticBotError
import logging
from utils.logger_mixin import LoggerMixin


class DatabaseBot(LoggerMixin):
    """
    DatabaseBot manages persistent storage and retrieval of trading analyses and performance metrics for the agentic trading system.

    Enhanced Responsibilities:
    - Initialize and manage the SQLite database schema with multiple tables
    - Save new trading analyses (signals, reasoning, confidence, price, etc.)
    - Store and retrieve reflection insights from post-trade analysis
    - Store asset screening results and market overviews
    - Retrieve historical analyses for backtesting and RAG functionality
    - Calculate performance metrics and analytics across all stored data
    - Support for trade outcomes and learning feedback loops
    """

    def __init__(self, db_file="trading_history.db"):
        """
        Initialize the DatabaseBot with the specified database file.
        Calls init_db() to ensure all tables exist.
        """
        super().__init__()
        self.db_file = db_file
        self.init_db()

    def init_db(self):
        """
        Create all necessary tables if they do not exist.
        Handles schema for analyses, reflection insights, screening results, and trade outcomes.
        """
        with sqlite3.connect(self.db_file) as conn:
            # Original analyses table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    asset_type TEXT,
                    action TEXT,
                    reasoning TEXT,
                    confidence REAL,
                    price REAL
                )
            """
            )

            # New table for reflection insights (TODO item #10)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reflection_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    trade_id TEXT,
                    original_analysis_id INTEGER,
                    entry_price REAL,
                    exit_price REAL,
                    pnl REAL,
                    pnl_percentage REAL,
                    hold_duration_hours REAL,
                    market_conditions TEXT,
                    ai_reflection TEXT,
                    key_insights TEXT,
                    lessons_learned TEXT,
                    confidence_accuracy REAL,
                    FOREIGN KEY (original_analysis_id) REFERENCES analyses (id)
                )
            """
            )

            # New table for asset screening results (TODO item #12)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS screening_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    market_sentiment TEXT,
                    market_volatility REAL,
                    risk_environment TEXT,
                    selected_assets TEXT,  -- JSON string of selected symbols
                    screening_scores TEXT,  -- JSON string of {symbol: score}
                    ai_insights TEXT,
                    top_sectors TEXT  -- JSON string of top performing sectors
                )
            """
            )

            # Enhanced trade outcomes table for comprehensive tracking
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    asset_type TEXT,
                    trade_type TEXT,  -- 'buy' or 'sell'
                    entry_price REAL,
                    exit_price REAL,
                    quantity REAL,
                    pnl REAL,
                    pnl_percentage REAL,
                    hold_duration_minutes INTEGER,
                    original_confidence REAL,
                    analysis_id INTEGER,
                    execution_status TEXT,
                    fees REAL,
                    FOREIGN KEY (analysis_id) REFERENCES analyses (id)
                )
            """
            )

            # Table for storing analysis context (for RAG functionality)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_context (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    context_type TEXT,  -- 'technical', 'fundamental', 'sentiment', 'reflection'
                    context_data TEXT,  -- JSON string of context information
                    relevance_score REAL,
                    expires_at TEXT  -- Optional expiration for time-sensitive context
                )
            """
            )

    # === Original Methods (Enhanced) ===

    def save_analysis(
        self, symbol, asset_type, analysis, current_price, context_data=None
    ):
        """Save a new trading analysis (signal) to the database with optional context."""
        method = "save_analysis"
        self.logger.debug(
            f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}(symbol='{symbol}')] START"
        )
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.execute(
                """
                INSERT INTO analyses (timestamp, symbol, asset_type, action, reasoning, confidence, price)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    datetime.now().isoformat(),
                    symbol,
                    asset_type,
                    analysis["action"],
                    analysis["reasoning"],
                    analysis["confidence"],
                    current_price,
                ),
            )
            analysis_id = cursor.lastrowid
            # Store context data if provided (for RAG functionality)
            if context_data:
                self.store_analysis_context(symbol, "analysis_input", context_data)

            self.logger.debug(
                f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}(symbol='{symbol}')] END"
            )
            return analysis_id or 0

    def get_analysis_history(self, symbol, days=30):
        """Retrieve historical analyses for a given symbol within the last N days."""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.execute(
                """
                SELECT id, timestamp, action, confidence, price, reasoning
                FROM analyses
                WHERE symbol = ?
                AND timestamp > ?
                ORDER BY timestamp DESC
            """,
                (symbol, (datetime.now() - timedelta(days=days)).isoformat()),
            )
            return cursor.fetchall()

    # === New Methods for Reflection Bot Support (TODO item #10) ===
    def store_reflection_insight(
        self,
        symbol: str,
        trade_id: str,
        original_analysis_id: int,
        entry_price: float,
        exit_price: float,
        pnl: float,
        hold_duration_hours: float,
        market_conditions: str,
        ai_reflection: str,
        key_insights: str,
        lessons_learned: str,
        confidence_accuracy: float,
    ) -> int:
        """Store reflection insights from post-trade analysis."""
        pnl_percentage = (
            ((exit_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
        )

        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.execute(
                """
                INSERT INTO reflection_insights 
                (timestamp, symbol, trade_id, original_analysis_id, entry_price, exit_price, 
                 pnl, pnl_percentage, hold_duration_hours, market_conditions, ai_reflection, 
                 key_insights, lessons_learned, confidence_accuracy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    datetime.now().isoformat(),
                    symbol,
                    trade_id,
                    original_analysis_id,
                    entry_price,
                    exit_price,
                    pnl,
                    pnl_percentage,
                    hold_duration_hours,
                    market_conditions,
                    ai_reflection,
                    key_insights,
                    lessons_learned,
                    confidence_accuracy,
                ),
            )
            return cursor.lastrowid or 0

    def get_reflection_insights(self, symbol: str, days: int = 90) -> List[Dict]:
        """Retrieve reflection insights for a symbol to inform future analysis."""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.execute(
                """
                SELECT timestamp, trade_id, pnl_percentage, hold_duration_hours,
                       ai_reflection, key_insights, lessons_learned, confidence_accuracy
                FROM reflection_insights
                WHERE symbol = ?
                AND timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 10
            """,
                (symbol, (datetime.now() - timedelta(days=days)).isoformat()),
            )

            results = []
            for row in cursor.fetchall():
                results.append(
                    {
                        "timestamp": row[0],
                        "trade_id": row[1],
                        "pnl_percentage": row[2],
                        "hold_duration_hours": row[3],
                        "ai_reflection": row[4],
                        "key_insights": row[5],
                        "lessons_learned": row[6],
                        "confidence_accuracy": row[7],
                    }
                )
            return results

    def get_cross_asset_insights(
        self, asset_type: Optional[str] = None, days: int = 30
    ) -> List[Dict]:
        """Get insights across multiple assets for pattern recognition."""
        base_query = """
            SELECT symbol, AVG(pnl_percentage) as avg_pnl, COUNT(*) as trade_count,
                   AVG(confidence_accuracy) as avg_confidence_accuracy,
                   GROUP_CONCAT(key_insights, ' | ') as combined_insights
            FROM reflection_insights
            WHERE timestamp > ?
        """

        params = [(datetime.now() - timedelta(days=days)).isoformat()]

        if asset_type:
            # Join with analyses table to filter by asset_type
            base_query = """
                SELECT ri.symbol, AVG(ri.pnl_percentage) as avg_pnl, COUNT(*) as trade_count,
                       AVG(ri.confidence_accuracy) as avg_confidence_accuracy,
                       GROUP_CONCAT(ri.key_insights, ' | ') as combined_insights
                FROM reflection_insights ri
                JOIN analyses a ON ri.original_analysis_id = a.id
                WHERE ri.timestamp > ? AND a.asset_type = ?
            """
            params.append(asset_type)

        base_query += " GROUP BY symbol ORDER BY avg_pnl DESC"

        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.execute(base_query, params)

            results = []
            for row in cursor.fetchall():
                results.append(
                    {
                        "symbol": row[0],
                        "avg_pnl": row[1],
                        "trade_count": row[2],
                        "avg_confidence_accuracy": row[3],
                        "combined_insights": row[4],
                    }
                )
            return results

    # === Methods for Asset Screening Support (TODO item #12) ===

    def store_screening_results(
        self,
        market_sentiment: str,
        market_volatility: float,
        risk_environment: str,
        selected_assets: List[str],
        screening_scores: Dict[str, float],
        ai_insights: str,
        top_sectors: List[str],
    ) -> int:
        """Store asset screening results for analysis and learning."""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.execute(
                """
                INSERT INTO screening_results 
                (timestamp, market_sentiment, market_volatility, risk_environment,
                 selected_assets, screening_scores, ai_insights, top_sectors)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    datetime.now().isoformat(),
                    market_sentiment,
                    market_volatility,
                    risk_environment,
                    json.dumps(selected_assets),
                    json.dumps(screening_scores),
                    ai_insights,
                    json.dumps(top_sectors),
                ),
            )
            return cursor.lastrowid or 0

    def get_screening_history(self, days: int = 7) -> List[Dict]:
        """Get historical screening results for analysis."""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.execute(
                """
                SELECT timestamp, market_sentiment, market_volatility, risk_environment,
                       selected_assets, screening_scores, ai_insights, top_sectors
                FROM screening_results
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """,
                ((datetime.now() - timedelta(days=days)).isoformat(),),
            )

            results = []
            for row in cursor.fetchall():
                results.append(
                    {
                        "timestamp": row[0],
                        "market_sentiment": row[1],
                        "market_volatility": row[2],
                        "risk_environment": row[3],
                        "selected_assets": json.loads(row[4]),
                        "screening_scores": json.loads(row[5]),
                        "ai_insights": row[6],
                        "top_sectors": json.loads(row[7]),
                    }
                )
            return results

    # === Enhanced Analysis Context Methods for RAG ===
    def store_analysis_context(
        self,
        symbol: str,
        context_type: str,
        context_data: Dict,
        relevance_score: float = 1.0,
        expires_hours: Optional[int] = None,
    ) -> int:
        """Store analysis context for RAG functionality."""
        expires_at = None
        if expires_hours:
            expires_at = (datetime.now() + timedelta(hours=expires_hours)).isoformat()

        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.execute(
                """
                INSERT INTO analysis_context 
                (timestamp, symbol, context_type, context_data, relevance_score, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    datetime.now().isoformat(),
                    symbol,
                    context_type,
                    json.dumps(context_data),
                    relevance_score,
                    expires_at,
                ),
            )
            return cursor.lastrowid or 0

    def get_analysis_context(
        self, symbol: str, context_types: Optional[List[str]] = None, days=30, limit=20
    ) -> List[Dict]:
        """Retrieve analysis context for RAG functionality."""
        base_query = """
            SELECT timestamp, context_type, context_data, relevance_score
            FROM analysis_context
            WHERE symbol = ?
            AND timestamp > ?
            AND (expires_at IS NULL OR expires_at > ?)
        """

        params = [
            symbol,
            (datetime.now() - timedelta(days=days)).isoformat(),
            datetime.now().isoformat(),
        ]

        if context_types:
            placeholders = ",".join(["?"] * len(context_types))
            base_query += f" AND context_type IN ({placeholders})"
            params.extend(context_types)

        base_query += " ORDER BY relevance_score DESC, timestamp DESC LIMIT ?"
        params.append(str(limit))

        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.execute(base_query, params)

            results = []
            for row in cursor.fetchall():
                results.append(
                    {
                        "timestamp": row[0],
                        "context_type": row[1],
                        "context_data": json.loads(row[2]),
                        "relevance_score": row[3],
                    }
                )
            return results

    # === Trade Outcome Tracking ===

    def store_trade_outcome(
        self,
        symbol: str,
        asset_type: str,
        trade_type: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        hold_duration_minutes: int,
        original_confidence: float,
        analysis_id: int,
        execution_status: str,
        fees: float = 0.0,
    ) -> int:
        """Store complete trade outcome for comprehensive tracking."""
        pnl = (
            (exit_price - entry_price) * quantity
            if trade_type == "buy"
            else (entry_price - exit_price) * quantity
        )
        pnl_percentage = (
            ((exit_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
        )

        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.execute(
                """
                INSERT INTO trade_outcomes 
                (timestamp, symbol, asset_type, trade_type, entry_price, exit_price, quantity,
                 pnl, pnl_percentage, hold_duration_minutes, original_confidence, analysis_id,
                 execution_status, fees)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    datetime.now().isoformat(),
                    symbol,
                    asset_type,
                    trade_type,
                    entry_price,
                    exit_price,
                    quantity,
                    pnl,
                    pnl_percentage,
                    hold_duration_minutes,
                    original_confidence,
                    analysis_id,
                    execution_status,
                    fees,
                ),
            )
            return cursor.lastrowid or 0

    def get_trade_outcomes(
        self, symbol: Optional[str] = None, days: int = 30
    ) -> List[Dict]:
        """Retrieve trade outcomes for performance analysis."""
        base_query = """
            SELECT timestamp, symbol, asset_type, trade_type, entry_price, exit_price,
                   quantity, pnl, pnl_percentage, hold_duration_minutes, original_confidence,
                   execution_status, fees
            FROM trade_outcomes
            WHERE timestamp > ?
        """

        params = [(datetime.now() - timedelta(days=days)).isoformat()]

        if symbol:
            base_query += " AND symbol = ?"
            params.append(symbol)

        base_query += " ORDER BY timestamp DESC"

        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.execute(base_query, params)

            rows = cursor.fetchall()
            if not rows:
                return []

            results = []
            for row in rows:
                results.append(
                    {
                        "timestamp": row[0],
                        "symbol": row[1],
                        "asset_type": row[2],
                        "trade_type": row[3],
                        "entry_price": row[4],
                        "exit_price": row[5],
                        "quantity": row[6],
                        "pnl": row[7],
                        "pnl_percentage": row[8],
                        "hold_duration_minutes": row[9],
                        "original_confidence": row[10],
                        "execution_status": row[11],
                        "fees": row[12],
                    }
                )
            return results

    # === Enhanced Performance Analytics ===

    def get_performance_metrics(self, symbol):
        """Calculate performance metrics for a symbol over the last 30 days."""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.execute(
                """
                SELECT 
                    COUNT(*) as total_analyses,
                    SUM(CASE WHEN action = 'buy' THEN 1 ELSE 0 END) as buy_signals,
                    SUM(CASE WHEN action = 'sell' THEN 1 ELSE 0 END) as sell_signals,
                    AVG(confidence) as avg_confidence,
                    MIN(price) as min_price,
                    MAX(price) as max_price,
                    AVG(price) as avg_price
                FROM analyses
                WHERE symbol = ?
                AND timestamp > ?
            """,
                (symbol, (datetime.now() - timedelta(days=30)).isoformat()),
            )
            return cursor.fetchone()

    def get_comprehensive_performance(
        self, symbol: Optional[str] = None, days: int = 30
    ) -> Dict[str, Any]:
        """Get comprehensive performance metrics including reflection insights."""
        # Get basic trade metrics
        trade_outcomes = self.get_trade_outcomes(symbol, days)

        if not trade_outcomes:
            return {"error": "No trade data available", "status": "failed"}

        # Calculate metrics
        total_trades = len(trade_outcomes)
        profitable_trades = len([t for t in trade_outcomes if t["pnl"] > 0])
        win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
        total_pnl = sum(t["pnl"] for t in trade_outcomes)
        avg_pnl_percentage = (
            sum(t["pnl_percentage"] for t in trade_outcomes) / total_trades
            if total_trades > 0
            else 0
        )

        # Get reflection insights for learning
        insights = self.get_reflection_insights(symbol, days) if symbol else []

        return {
            "total_trades": total_trades,
            "profitable_trades": profitable_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl_percentage": avg_pnl_percentage,
            "recent_insights_count": len(insights),
            "avg_confidence_accuracy": (
                sum(i["confidence_accuracy"] for i in insights) / len(insights)
                if insights
                else 0
            ),
            "key_lessons": [
                i["lessons_learned"] for i in insights[:3]
            ],  # Top 3 recent lessons
        }

    # === Existing Methods (Updated) ===

    def get_signal_strength(self, symbol, lookback_days=7):
        """Calculate trading signal strength and average price change for a symbol over recent analyses."""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.execute(
                """
                WITH recent_analyses AS (
                    SELECT 
                        action,
                        confidence,
                        price,
                        LAG(price) OVER (ORDER BY timestamp) as prev_price
                    FROM analyses
                    WHERE symbol = ?
                    AND timestamp > ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                )
                SELECT 
                    SUM(CASE 
                        WHEN action = 'buy' THEN confidence
                        WHEN action = 'sell' THEN -confidence
                        ELSE 0
                    END) as signal_strength,
                    AVG(CASE WHEN prev_price IS NOT NULL 
                        THEN (price - prev_price) / prev_price 
                        ELSE 0 
                    END) as avg_price_change
                FROM recent_analyses
            """,
                (
                    symbol,
                    (datetime.now() - timedelta(days=lookback_days)).isoformat(),
                    lookback_days,
                ),
            )
            return cursor.fetchone()

    # === Cleanup and Maintenance ===

    def cleanup_expired_context(self):
        """Remove expired analysis context entries."""
        with sqlite3.connect(self.db_file) as conn:
            conn.execute(
                """
                DELETE FROM analysis_context
                WHERE expires_at IS NOT NULL AND expires_at < ?
            """,
                (datetime.now().isoformat(),),
            )

    # === Database Logic (migrated from agent.py) ===
    # Migrate TradingAgent methods related to caching and database here.

    def _get_cached_data(self, cache_key, max_age_minutes=5):
        """
        Retrieve cached data if not expired.
        """
        # Example stub: implement caching logic if needed
        return None

    def _cache_data(self, cache_key, data):
        """
        Cache data for later retrieval.
        """
        # Example stub: implement caching logic if needed
        pass


# === Usage Example ===
if __name__ == "__main__":
    db = DatabaseBot(db_file="test_trading_history.db")
    # Save a sample analysis
    analysis_id = db.save_analysis(
        symbol="BTC/USD",
        asset_type="crypto",
        analysis={
            "action": "buy",
            "reasoning": "AI detected bullish momentum",
            "confidence": 0.92,
        },
        current_price=68000.0,
    )

    # Store reflection insight
    db.store_reflection_insight(
        symbol="BTC/USD",
        trade_id="trade_001",
        original_analysis_id=analysis_id,
        entry_price=68000.0,
        exit_price=70000.0,
        pnl=2000.0,
        hold_duration_hours=24.0,
        market_conditions="bullish trend",
        ai_reflection="Trade executed well, momentum continued as predicted",
        key_insights="Strong volume confirmation was key indicator",
        lessons_learned="Trust high-confidence signals in trending markets",
        confidence_accuracy=0.95,
    )

    # Retrieve and print recent analyses
    print("Recent analyses:", db.get_analysis_history("BTC/USD", days=7))
    # Print performance metrics
    print("Performance metrics:", db.get_performance_metrics("BTC/USD"))
    # Print comprehensive performance
    print("Comprehensive performance:", db.get_comprehensive_performance("BTC/USD"))


def selftest_database_bot():
    """Standalone self-test for DatabaseBot: tests CRUD for TradeOutcome and ReflectionInsight."""
    print("\n--- Running DatabaseBot Self-Test ---")
    import os

    test_db = "test_trading_history.db"
    try:
        # Remove test DB if exists
        if os.path.exists(test_db):
            os.remove(test_db)
        db = DatabaseBot(db_file=test_db)
        # Test 1: Save and retrieve analysis
        analysis_id = db.save_analysis(
            symbol="AAPL",
            asset_type="stock",
            analysis={"action": "buy", "reasoning": "Test reason", "confidence": 0.8},
            current_price=100.0,
        )
        history = db.get_analysis_history("AAPL", days=7)
        assert any(
            str(analysis_id) in str(row) for row in history
        ), "Analysis not found in history."
        print("    -> Save/retrieve analysis logic passed.")
        # Test 2: Save and retrieve reflection insight
        db.store_reflection_insight(
            symbol="AAPL",
            trade_id="T1",
            original_analysis_id=analysis_id,
            entry_price=100.0,
            exit_price=110.0,
            pnl=10.0,
            hold_duration_hours=1.0,
            market_conditions="test",
            ai_reflection="insight",
            key_insights="key",
            lessons_learned="lesson",
            confidence_accuracy=1.0,
        )
        insights = db.get_reflection_insights("AAPL", days=7)
        assert any(
            "insight" in str(row) for row in insights
        ), "Reflection insight not found."
        print("    -> Save/retrieve reflection insight logic passed.")
        # Test 3: Save and retrieve screening result
        db.store_screening_results(
            market_sentiment="bullish",
            market_volatility=10.0,
            risk_environment="low",
            selected_assets=["AAPL"],
            screening_scores={"AAPL": 90},
            ai_insights="insight",
            top_sectors=["Tech"],
        )
        print("    -> Save screening results logic passed.")
        print("--- DatabaseBot Self-Test PASSED ---")
    except AssertionError as e:
        print(f"--- DatabaseBot Self-Test FAILED: {e} ---")
    except Exception as e:
        print(f"--- DatabaseBot Self-Test encountered an ERROR: {e} ---")
    finally:
        if os.path.exists(test_db):
            os.remove(test_db)


if __name__ == "__main__":
    selftest_database_bot()
