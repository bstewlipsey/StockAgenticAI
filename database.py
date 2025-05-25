import sqlite3
from datetime import datetime, timedelta

class Database:
    """
    The Database class manages persistent storage and retrieval of trading analyses and performance metrics.

    This class provides a simple interface to a local SQLite database for logging AI trading decisions,
    tracking historical signals, and calculating performance statistics. It is designed for reliability,
    transparency, and easy integration with the rest of the trading system.

    Key Responsibilities:
    - Initialize and manage the SQLite database schema
    - Save new trading analyses (signals, reasoning, confidence, price, etc.)
    - Retrieve historical analyses for backtesting and review
    - Calculate performance metrics (signal counts, average confidence, price stats)
    - Compute signal strength and price change patterns for advanced analytics

    Safety Features:
    - Uses parameterized queries to prevent SQL injection
    - Ensures the database schema exists before any operation
    - Handles date filtering for time-based queries
    - Returns results in a format suitable for further analysis or reporting
    """
    def __init__(self, db_file='trading_history.db'):
        """
        Initialize the Database object and ensure the schema exists.
        Args:
            db_file (str): Path to the SQLite database file
        """
        self.db_file = db_file
        self.init_db()
    
    def init_db(self):
        """
        Create the analyses table if it does not exist.
        This ensures the database is ready for use on first run.
        """
        with sqlite3.connect(self.db_file) as conn:
            conn.execute("""
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
            """)
    
    def save_analysis(self, symbol, asset_type, analysis, current_price):
        """
        Save a new trading analysis (signal) to the database.
        Args:
            symbol (str): Ticker symbol
            asset_type (str): Type of asset (e.g., 'stock')
            analysis (dict): Contains 'action', 'reasoning', 'confidence'
            current_price (float): Price at time of analysis
        """
        with sqlite3.connect(self.db_file) as conn:
            conn.execute("""
                INSERT INTO analyses (timestamp, symbol, asset_type, action, reasoning, confidence, price)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),  # Store ISO timestamp for easy sorting/filtering
                symbol,
                asset_type,
                analysis['action'],
                analysis['reasoning'],
                analysis['confidence'],
                current_price
            ))
    
    def get_analysis_history(self, symbol, days=30):
        """
        Retrieve historical analyses for a given symbol within the last N days.
        Args:
            symbol (str): Ticker symbol
            days (int): Number of days to look back
        Returns:
            list of tuples: (timestamp, action, confidence, price)
        """
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.execute("""
                SELECT timestamp, action, confidence, price
                FROM analyses
                WHERE symbol = ?
                AND timestamp > ?
                ORDER BY timestamp DESC
            """, (
                symbol,
                (datetime.now() - timedelta(days=days)).isoformat()
            ))
            return cursor.fetchall()

    def get_performance_metrics(self, symbol):
        """
        Calculate performance metrics for a symbol over the last 30 days.
        Args:
            symbol (str): Ticker symbol
        Returns:
            tuple: (total_analyses, buy_signals, sell_signals, avg_confidence, min_price, max_price, avg_price)
        """
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.execute("""
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
            """, (
                symbol,
                (datetime.now() - timedelta(days=30)).isoformat()
            ))
            return cursor.fetchone()

    def get_signal_strength(self, symbol, lookback_days=7):
        """
        Calculate trading signal strength and average price change for a symbol over recent analyses.
        Args:
            symbol (str): Ticker symbol
            lookback_days (int): Number of days to look back
        Returns:
            tuple: (signal_strength, avg_price_change)
                - signal_strength: Sum of confidence for buys minus sells
                - avg_price_change: Average price change between consecutive signals
        """
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.execute("""
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
            """, (
                symbol,
                (datetime.now() - timedelta(days=lookback_days)).isoformat(),
                lookback_days
            ))
            return cursor.fetchone()