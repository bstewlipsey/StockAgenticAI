"""
Mock data providers for StockAgenticAI unit/integration tests.
- Use these mocks to simulate data fetching and trading for deterministic, fast tests.
"""

class MockStockDataProvider:
    def get_historical_prices(self, symbol, timeframe='1D', limit=30):
        # Return a fixed price series for testing
        return [100 + i for i in range(limit)]
    def get_current_price(self, symbol):
        return 123.45

class MockCryptoDataProvider:
    def get_historical_prices(self, symbol, timeframe='1h', limit=30):
        # Return a fixed price series for testing
        return [200 + i for i in range(limit)]
    def get_current_price(self, symbol):
        return 23456.78

class MockNewsRetriever:
    def fetch_news(self, query, max_results=5):
        return [
            {'title': 'Test News 1', 'content': 'Market is bullish.'},
            {'title': 'Test News 2', 'content': 'Market is bearish.'}
        ]
    def preprocess_and_chunk(self, articles):
        return [a['content'] for a in articles]
    def generate_embeddings(self, chunks):
        return [0.1] * len(chunks)
    def augment_context_and_llm(self, query):
        return 'Test news summary.'

# Example usage in a test:
if __name__ == "__main__":
    stock = MockStockDataProvider()
    print(stock.get_historical_prices('AAPL'))
    print(stock.get_current_price('AAPL'))
    crypto = MockCryptoDataProvider()
    print(crypto.get_historical_prices('BTC/USD'))
    print(crypto.get_current_price('BTC/USD'))
    news = MockNewsRetriever()
    print(news.fetch_news('BTC news'))
