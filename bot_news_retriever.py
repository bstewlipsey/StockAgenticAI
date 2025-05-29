"""
NewsRetrieverBot: Retrieves and analyzes news for trading signals.
"""
import requests
from typing import List, Optional, Dict, Any
from data_structures import NewsArticle
from config_system import NEWS_API_KEY
from bot_ai import AIBot

class NewsRetrieverBot:
    """
    NewsRetrieverBot retrieves, processes, embeds, and retrieves news for RAG workflows.
    """
    def __init__(self):
        self.api_key = NEWS_API_KEY
        self.ai_bot = AIBot()
        self.news_articles: List[NewsArticle] = []
        self.embeddings: List[List[float]] = []  # In-memory vector store for prototyping

    def fetch_news(self, query: str, max_results: int = 10) -> List[NewsArticle]:
        """Fetch news articles from NewsAPI."""
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={self.api_key}&pageSize={max_results}"
        response = requests.get(url)
        articles = []
        if response.status_code == 200:
            for item in response.json().get('articles', []):
                article = NewsArticle(
                    title=item.get('title', ''),
                    url=item.get('url', ''),
                    date=item.get('publishedAt', ''),
                    source=item.get('source', {}).get('name', ''),
                    full_text=item.get('content', '')
                )
                articles.append(article)
        return articles

    def preprocess_and_chunk(self, articles: List[NewsArticle], chunk_size: int = 512) -> List[NewsArticle]:
        """Clean and chunk news articles for embedding."""
        processed = []
        for article in articles:
            text = article.full_text or ''
            # Simple cleaning (remove HTML, etc.)
            text = text.replace('\n', ' ').replace('\r', ' ')
            # Chunking (naive split)
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                if chunk.strip():
                    processed.append(NewsArticle(
                        title=article.title,
                        url=article.url,
                        date=article.date,
                        source=article.source,
                        full_text=chunk
                    ))
        return processed

    def generate_embeddings(self, articles: List[NewsArticle]) -> List[List[float]]:
        """Generate embeddings for each news chunk using AIBot."""
        embeddings = []
        for article in articles:
            embedding = self.ai_bot.generate_embedding(article.full_text)
            article.embedding = embedding
            embeddings.append(embedding)
        self.news_articles.extend(articles)
        self.embeddings.extend(embeddings)
        return embeddings

    def similarity_search(self, query: str, top_k: int = 3) -> List[NewsArticle]:
        """Retrieve most relevant news chunks for a query using cosine similarity."""
        query_embedding = self.ai_bot.generate_embedding(query)
        scored = []
        for article, emb in zip(self.news_articles, self.embeddings):
            if emb:
                score = self._cosine_similarity(query_embedding, emb)
                scored.append((score, article))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [a for _, a in scored[:top_k]]

    def _cosine_similarity(self, v1, v2):
        import numpy as np
        v1, v2 = np.array(v1), np.array(v2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))

    def augment_context_and_llm(self, query: str) -> str:
        """Augment context with relevant news and call LLM for summary/analysis."""
        relevant_news = self.similarity_search(query)
        context = '\n'.join([f"{n.title}: {n.full_text}" for n in relevant_news])
        prompt = f"Given the following news context, summarize and analyze for trading insights:\n\n{context}\n\nQuery: {query}\n"
        return self.ai_bot.generate_analysis("{prompt}", {"prompt": prompt})

    @staticmethod
    def selftest():
        print(f"\n--- Running NewsRetrieverBot Self-Test ---")
        try:
            class DummyAIBot:
                def generate_embedding(self, text):
                    return [0.1] * 384
            bot = NewsRetrieverBot()
            bot.ai_bot = DummyAIBot()
            # Use NewsArticle for type compatibility
            dummy_article = NewsArticle(
                title="Test News",
                url="http://example.com",
                date="2023-01-01",
                source="UnitTest",
                full_text="Test content."
            )
            bot.fetch_news = lambda query, max_results=10: [dummy_article]
            articles = bot.fetch_news("AAPL")
            assert articles and hasattr(articles[0], 'title'), "fetch_news did not return NewsArticle-like objects"
            embeddings = bot.generate_embeddings(articles)
            assert embeddings and isinstance(embeddings[0], list), "generate_embeddings did not return embeddings"
            print("    -> NewsRetrieverBot fetch_news() and generate_embeddings() returned valid results.")
            print(f"--- NewsRetrieverBot Self-Test PASSED ---")
        except AssertionError as e:
            print(f"--- NewsRetrieverBot Self-Test FAILED: {e} ---")
        except Exception as e:
            print(f"--- NewsRetrieverBot Self-Test encountered an ERROR: {e} ---")

if __name__ == "__main__":
    NewsRetrieverBot.selftest()
