"""
Unit tests for NewsRetrieverBot.
"""

import unittest
from bot_news_retriever import NewsRetrieverBot
from data_structures import NewsArticle


class TestNewsRetrieverBot(unittest.TestCase):
    def setUp(self):
        self.bot = NewsRetrieverBot()

    def test_fetch_news(self):
        articles = self.bot.fetch_news("AAPL", max_results=2)
        self.assertIsInstance(articles, list)
        if articles:
            self.assertIsInstance(articles[0], NewsArticle)

    def test_preprocess_and_chunk(self):
        articles = [
            NewsArticle(title="Test", url="", date="", source="", full_text="A" * 1024)
        ]
        chunks = self.bot.preprocess_and_chunk(articles, chunk_size=256)
        self.assertGreaterEqual(len(chunks), 4)
        self.assertIsInstance(chunks[0], NewsArticle)

    def test_generate_embeddings(self):
        articles = [
            NewsArticle(title="Test", url="", date="", source="", full_text="Test text")
        ]
        embeddings = self.bot.generate_embeddings(articles)
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(len(embeddings[0]), 384)

    def test_similarity_search(self):
        articles = [
            NewsArticle(title="Test", url="", date="", source="", full_text="Test text")
        ]
        self.bot.generate_embeddings(articles)
        results = self.bot.similarity_search("Test text", top_k=1)
        self.assertIsInstance(results, list)
        if results:
            self.assertIsInstance(results[0], NewsArticle)

    def test_augment_context_and_llm(self):
        # Patch ai_bot with a dummy to avoid real LLM call
        class DummyAIBot:
            def generate_analysis(self, prompt, variables=None):
                return "Dummy analysis result."
            
            def generate_embedding(self, text):
                return [0.1] * 384

        self.bot.ai_bot = DummyAIBot()  # type: ignore
        articles = [
            NewsArticle(title="Test", url="", date="", source="", full_text="Test text")
        ]
        self.bot.generate_embeddings(articles)
        result = self.bot.augment_context_and_llm("Test text")
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
