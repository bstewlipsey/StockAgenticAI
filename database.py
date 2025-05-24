import sqlite3
from datetime import datetime, timedelta

class Database:
    def __init__(self, db_file='trading_history.db'):
        self.db_file = db_file
        self.init_db()
    
    def init_db(self):
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
        with sqlite3.connect(self.db_file) as conn:
            conn.execute("""
                INSERT INTO analyses (timestamp, symbol, asset_type, action, reasoning, confidence, price)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                symbol,
                asset_type,
                analysis['action'],
                analysis['reasoning'],
                analysis['confidence'],
                current_price
            ))
    
    def get_analysis_history(self, symbol, days=30):
        """Get historical analyses for a symbol"""
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
        """Calculate performance metrics for a symbol"""
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
        """Calculate trading signal strength based on historical patterns"""
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