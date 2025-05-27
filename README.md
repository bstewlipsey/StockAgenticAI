# StockAgenticAI - Setup & Installation Guide

## Project Status

**As of May 2025, StockAgenticAI is fully refactored, integrated, and tested.**
- All configuration is centralized in `config.py` (system/API) and `config_trading.py` (strategy/bot).
- Core RAG (Retrieval-Augmented Generation) and inter-agent data structures are defined in `data_structures.py`.
- NewsRetrieverBot is fully implemented and integrated, providing news context to the trading decision process.
- All bots dynamically load their configuration from config files (no hardcoded strategy parameters).
- All unit, integration, and regression tests pass.
- Logging is ASCII-safe and all performance metrics are floats.
- The system is ready for further refinement or deployment.

## Prerequisites
- **Python 3.10** (recommended for compatibility)
- **Windows users:** [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (one-time install)

## Setup Steps

1. **Clone or download this repository.**

2. **Install Python dependencies:**
   - Open PowerShell in the project directory.
   - (Optional but recommended) Create and activate a virtual environment:
     ```powershell
     py -3.10 -m venv .venv
     .\.venv\Scripts\Activate
     ```
   - Install all required packages:
     ```powershell
     pip install --upgrade pip setuptools wheel
     pip install -r requirements.txt
     ```

3. **Configure your API keys and settings:**
   - Edit `config.py` and set your Alpaca, Gemini, NewsAPI, and other API keys as needed.
   - Ensure your `.env` file contains `NEWS_API_KEY` for news retrieval features.

4. **Run the trading bot:**
   ```powershell
   python main.py
   ```

## Troubleshooting
- If you see errors about missing C headers (like `longintrepr.h` or `aiohttp` build failures), make sure you have installed the Microsoft C++ Build Tools.
- If you use a different Python version, some dependencies may not work. Python 3.10 is recommended.

## Testing & Monitoring Methods

### Core Test Files

- **test_core.py**: Unit/integration tests for core system logic and configuration flags (e.g., `TEST_MODE_ENABLED`).
- **test_mocks.py**: Tests using mock data providers for deterministic, API-independent testing.
- **test_news_retriever_bot.py**: Tests for NewsRetrieverBot, including data fetching and context integration.
- **test_reflection_bot.py**: Tests for ReflectionBot, including storage and retrieval of insights from DatabaseBot.
- **test_regression.py**: Regression tests for end-to-end flow with diversified asset types and quick testing modes.
- **test_trading_bot.py**: Tests for PositionSizerBot, RiskManagerBot, and TradeExecutorBot logic for both crypto and stock.
- **test_trading_system.py**: Integration tests for the full system loop, including AssetScreenerBot's crypto logic and passing of reflection insights.

### Monitoring & Visualization Methods

- **bot_visualizer.py**
  - `display_reflection_insights(symbol=None, limit=10)`: View the latest AI-generated reflection insights.
  - `display_performance_trends(symbol=None, days=30)`: View win rates, average P&L, and other performance metrics.
  - `display_interactive_log_viewer(log_path='trading.log', filter_keywords=None)`: Watch the trading log in real time, with optional keyword filtering (e.g., "TRADE_OUTCOME", "ERROR").

### How to Run Tests

In the VS Code terminal, run any of the following:

```
python -m unittest test_core.py
python -m unittest test_mocks.py
python -m unittest test_news_retriever_bot.py
python -m unittest test_reflection_bot.py
python -m unittest test_regression.py
python -m unittest test_trading_bot.py
python -m unittest test_trading_system.py
```

### How to Use the Visualizer

In the VS Code terminal, start a Python shell:

```
python
```
Then run:

```python
from bot_visualizer import VisualizerBot
visualizer = VisualizerBot()
visualizer.display_reflection_insights()
visualizer.display_performance_trends()
visualizer.display_interactive_log_viewer('trading.log')
```

### Additional Notes
- All logs are written to `trading.log` in the project root.
- Trading history is stored in `trading_history.db`.
- For more details, see the docstrings in each bot and test file.

---

**This setup ensures anyone can replicate your environment with minimal manual steps.**
