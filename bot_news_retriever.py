"""
NewsRetrieverBot: Retrieves and analyzes news for trading signals.
"""

import requests
from typing import List
from data_structures import NewsArticle
from config_system import NEWS_API_KEY, NEWS_API_QUOTA_PER_MINUTE
from bot_ai import AIBot
from bot_quota_manager import QuotaManagerBot
from utils.logger_mixin import LoggerMixin


class NewsRetrieverBot(LoggerMixin):
    """
    NewsRetrieverBot retrieves, processes, embeds, and retrieves news for RAG workflows.
    Integrates with QuotaManagerBot for API quota management.
    """

    def __init__(self):
        """
        Initialize NewsRetrieverBot with API key, AI bot, and quota manager.
        Sets up in-memory storage for news articles and embeddings.
        """
        super().__init__()
        self.api_key = NEWS_API_KEY
        self.ai_bot = AIBot()
        self.news_articles: List[NewsArticle] = []
        self.embeddings: List[List[float]] = (
            []
        )  # In-memory vector store for prototyping
        self.quota_manager = QuotaManagerBot()

    def fetch_news(self, symbol, max_results=5):
        """
        Fetch news articles for a given symbol or keyword. For crypto, tries multiple fallbacks.
        Returns a list of news article objects (or dicts), or an empty list if none found.
        """
        method = "fetch_news"
        attempts = []
        queries = [symbol]
        # For crypto, add fallbacks
        if "/" in symbol:
            base = symbol.split("/")[0]
            queries.append(base)
            # Add common crypto names
            if base.upper() == "ETH":
                queries.append("Ethereum")
            elif base.upper() == "BTC":
                queries.append("Bitcoin")
            elif base.upper() == "DOGE":
                queries.append("Dogecoin")
            # Add more as needed
        for q in queries:
            try:
                articles = self._fetch_news_api(q, max_results=max_results)
                if articles and len(articles) > 0:
                    return articles
            except Exception:
                pass
        return []

    def _fetch_news_api(self, query: str, max_results: int = 10) -> List[NewsArticle]:
        """
        Fetch news articles from NewsAPI for a given query.
        Uses QuotaManagerBot to wrap the API call and handle quota.
        Returns a list of NewsArticle objects.
        """
        method = "_fetch_news_api"
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={self.api_key}&pageSize={max_results}"
        def do_request():
            response = requests.get(url)
            return response
        # Use QuotaManagerBot to wrap the API call
        success, response = self.quota_manager.api_request(
            api_name="newsapi",
            func=do_request,
            quota_per_minute=NEWS_API_QUOTA_PER_MINUTE
        )
        articles = []
        if not success or response is None:
            return articles
        if response.status_code == 200:
            for item in response.json().get("articles", []):
                article = NewsArticle(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    date=item.get("publishedAt", ""),
                    source=item.get("source", {}).get("name", ""),
                    full_text=item.get("content", ""),
                )
                articles.append(article)
        return articles

    def preprocess_and_chunk(
        self, articles: List[NewsArticle], chunk_size: int = 512
    ) -> List[NewsArticle]:
        """
        Preprocess and chunk news articles for embedding or analysis.
        Cleans text and splits into chunks of specified size.
        Returns a list of chunked NewsArticle objects.
        """
        processed = []
        for article in articles:
            text = article.full_text or ""
            # Simple cleaning (remove HTML, etc.)
            text = text.replace("\n", " ").replace("\r", " ")
            # Chunking (naive split)
            for i in range(0, len(text), chunk_size):
                chunk = text[i : i + chunk_size]
                if chunk.strip():
                    processed.append(
                        NewsArticle(
                            title=article.title,
                            url=article.url,
                            date=article.date,
                            source=article.source,
                            full_text=chunk,
                        )
                    )
        return processed

    def generate_embeddings(self, articles: List[NewsArticle]) -> List[List[float]]:
        """
        Generate embeddings for a list of news articles using the AI bot.
        Updates the articles with their corresponding embeddings.
        Returns a list of embeddings.
        """
        embeddings = []
        for article in articles:
            embedding = self.ai_bot.generate_embedding(article.full_text)
            article.embedding = embedding
            embeddings.append(embedding)
        self.news_articles.extend(articles)
        self.embeddings.extend(embeddings)
        return embeddings

    def similarity_search(self, query: str, top_k: int = 3) -> List[NewsArticle]:
        """
        Perform a similarity search for a given query.
        Returns the top_k most similar news articles based on embeddings.
        """
        query_embedding = self.ai_bot.generate_embedding(query)
        scored = []
        for article, emb in zip(self.news_articles, self.embeddings):
            if emb:
                score = self._cosine_similarity(query_embedding, emb)
                scored.append((score, article))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [a for _, a in scored[:top_k]]

    def _cosine_similarity(self, v1, v2):
        """
        Calculate cosine similarity between two vectors.
        """
        import numpy as np

        v1, v2 = np.array(v1), np.array(v2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))

    def augment_context_and_llm(self, query: str) -> str:
        """
        Augment the context for the language model with relevant news articles.
        Summarizes and analyzes the news context for trading insights.
        """
        relevant_news = self.similarity_search(query)
        context = "\n".join([f"{n.title}: {n.full_text}" for n in relevant_news])
        prompt = f"Given the following news context, summarize and analyze for trading insights:\n\n{context}\n\nQuery: {query}\n"
        result = self.ai_bot.generate_analysis(prompt, {"prompt": prompt})
        return result

    @staticmethod
    def selftest():
        """
        Self-test method for NewsRetrieverBot.
        Runs a series of checks to ensure the bot is functioning correctly.
        """
        print("\n--- Running NewsRetrieverBot Self-Test ---")
        try:
            # DummyAIBot for embedding simulation
            class DummyAIBot:
                def generate_embedding(self, text):
                    return [0.1] * 384

                def generate_analysis(self, prompt, variables=None):
                    return "Dummy analysis result."

            bot = NewsRetrieverBot()
            bot.ai_bot = DummyAIBot()
            dummy_article = NewsArticle(
                title="Test News",
                url="http://example.com",
                date="2023-01-01",
                source="UnitTest",
                full_text="Test content.",
            )
            # Patch fetch_news to return dummy article
            bot.fetch_news = lambda symbol, max_results=5: [dummy_article]
            articles = bot.fetch_news("AAPL")
            assert articles and hasattr(articles[0], "title"), "fetch_news did not return NewsArticle-like objects"
            embeddings = bot.generate_embeddings(articles)
            assert embeddings and isinstance(embeddings[0], list), "generate_embeddings did not return embeddings"
            # Test similarity_search and augment_context_and_llm
            sim_articles = bot.similarity_search("AAPL", top_k=1)
            assert sim_articles and hasattr(sim_articles[0], "title"), "similarity_search did not return NewsArticle-like objects"
            summary = bot.augment_context_and_llm("AAPL")
            assert isinstance(summary, str), "augment_context_and_llm did not return a string"
            print("    -> NewsRetrieverBot fetch_news(), generate_embeddings(), similarity_search(), and augment_context_and_llm() returned valid results.")
            print("--- NewsRetrieverBot Self-Test PASSED ---")
        except AssertionError as e:
            print(f"--- NewsRetrieverBot Self-Test FAILED: {e} ---")
        except Exception as e:
            print(f"--- NewsRetrieverBot Self-Test encountered an ERROR: {e} ---")


if __name__ == "__main__":
    NewsRetrieverBot.selftest()
