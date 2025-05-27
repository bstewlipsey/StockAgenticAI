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

## Configuration Files Overview

StockAgenticAI uses two main configuration files:

- `config.py`: Controls system-level and API settings (API keys, trading mode, logging, timeframes, etc.).
- `config_trading.py`: Controls trading strategy, asset selection, risk management, and AI prompt templates.

**Key Settings:**
- `TEST_MODE_ENABLED` (in `config.py`):
  - `True` for rapid iteration and safe testing (paper trading, fast cycles).
  - `False` for realistic simulation or live trading.
- `TRADING_CYCLE_INTERVAL` (in both configs):
  - Controls how often the bot checks for trades (in seconds).
- `TRADING_ASSETS` (in `config_trading.py`):
  - List of assets and allocation per asset.
- `ENABLE_TRADING_BOT` (in `config.py`):
  - `False` disables live trading (safe for all tests).
  - `True` enables live trading (only after all validation is complete).

### Recommended Settings for Different Test Durations

**Test Scenario Quick Reference**

- **Quick Test**
  - `TEST_MODE_ENABLED = True`
  - `TRADING_CYCLE_INTERVAL = 5`
  - `TRADING_ASSETS = [("AAPL", "stock", 500), ("BTC/USD", "crypto", 200)]`
  - *Fastest cycles, minimal assets*

- **1 Hour Test**
  - `TEST_MODE_ENABLED = True`
  - `TRADING_CYCLE_INTERVAL = 15` to `30`
  - `TRADING_ASSETS = 2-4 assets`
  - *Simulates more realistic cycles*

- **1 Day Test**
  - `TEST_MODE_ENABLED = False`
  - `TRADING_CYCLE_INTERVAL = 60` to `300`
  - `TRADING_ASSETS = 5-10 assets`
  - *Use paper trading, realistic intervals*

- **1 Week+ Test**
  - `TEST_MODE_ENABLED = False`
  - `TRADING_CYCLE_INTERVAL = 300` to `900`
  - `TRADING_ASSETS = Full portfolio`
  - *Extended simulation, paper trading*

---

## How to Use the Visualizer and Log Monitoring 

**Steps:**

1. **Open a Command Prompt or PowerShell window in your project folder.**
   - You should see files like `main.py`, `bot_visualizer.py`, and `trading.log` in the folder.

2. **Start the Python interactive shell:**
   - Type (one at a time):
    ```powershell
     python

    ```
    
   - If you get an error like `'python' is not recognized`, you may need to install Python 3.10 and add it to your PATH. See the Prerequisites section above.

3. **Run the following commands one at a time:**
   - At the `>>>` prompt, type:
     ```python
     from bot_visualizer import VisualizerBot
     visualizer = VisualizerBot()
     visualizer.display_reflection_insights()
     visualizer.display_performance_trends()
     # IMPORTANT: To view the log, do NOT use visualizer.display_interactive_log_viewer()
     # Instead, import and call the function directly as shown below:
     from bot_visualizer import display_interactive_log_viewer
     display_interactive_log_viewer('trading.log')
     # You can also filter for important events:
     display_interactive_log_viewer('trading.log', filter_keywords=['TRADE_OUTCOME', 'DECISION', 'ERROR'])
     ```
   - Press Enter after each line. If you see an error, check that you are in the correct folder and that all dependencies are installed.

**Troubleshooting:**
- If you see `ModuleNotFoundError`, make sure you are in the folder with your project files and that you have installed all requirements with:
  ```powershell
  pip install -r requirements.txt
  ```
- If you see a permissions error or the log file is missing, make sure `main.py` has been run at least once to generate `trading.log`.
- If you see a long error message, copy the first line and search for it in the README or online.

**How to Use Log Monitoring:**
- The `display_interactive_log_viewer()` function streams the log file in real time.
- You can filter for important events by passing keywords:
  ```python
  visualizer.display_interactive_log_viewer('trading.log', filter_keywords=['TRADE_OUTCOME', 'DECISION', 'ERROR'])
  ```
- This will only show lines containing those keywords, making it easier to spot trading decisions, outcomes, or errors.

**What the Log File Shows:**
- Each line in `trading.log` includes:
  - **Timestamp**: When the event happened
  - **Log Level**: INFO, WARNING, ERROR, etc.
  - **Logger Name**: Which bot or module generated the entry
  - **Message**: Details of the event (e.g., trade decision, error, insight)
- **INFO**: Normal operation (e.g., "Making trading decision for AAPL")
- **WARNING**: Non-critical issues (e.g., rejected positions, missing data)
- **ERROR**: Critical issues (e.g., failed API calls, unexpected exceptions)
- ReflectionBot and DecisionMakerBot log their insights and decisions for transparency.

**If you are stuck:**
- Ask for help with the exact error message you see.
- Double-check you are in the correct folder and have run all setup steps.
- You do NOT need to know Python to use the visualizer or monitor logs if you follow these instructions step by step.

---

## Log Files: Usage and Interpretation

- **`trading.log`**: Main log file for all system events, errors, and trading actions.
  - Written in plain text, ASCII-safe format for compatibility.
  - Each log entry includes:
    - Timestamp: When the event occurred.
    - Log Level: INFO, WARNING, ERROR, etc.
    - Logger Name: Which bot or module generated the entry.
    - Message: Details of the event (e.g., trade decision, error, reflection insight).
- **How to Read the Log:**
  - Look for `[INFO]` entries for normal operation (e.g., "Making trading decision for AAPL").
  - `[WARNING]` entries indicate non-critical issues (e.g., rejected positions, missing data).
  - `[ERROR]` entries require attention (e.g., failed API calls, unexpected exceptions).
  - ReflectionBot and DecisionMakerBot log their insights and decisions for transparency.
- **Log Monitoring:**
  - Use `bot_visualizer.py`'s `display_interactive_log_viewer()` to stream and filter log events in real time.
  - Filter for keywords like `TRADE_OUTCOME`, `DECISION`, or `ERROR` to focus on key events.
- **Log Retention:**
  - Logs are not automatically cleared. You may archive or delete old logs as needed for disk space.

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

# To view the log in real time, use the standalone function (not a method):
from bot_visualizer import display_interactive_log_viewer
display_interactive_log_viewer('trading.log')
# You can also filter for important events:
display_interactive_log_viewer('trading.log', filter_keywords=['TRADE_OUTCOME', 'DECISION', 'ERROR'])
```

**Note:**
- If you see `AttributeError: 'VisualizerBot' object has no attribute 'display_interactive_log_viewer'`, it means you tried to call the log viewer as a method. Instead, import and call `display_interactive_log_viewer` directly as shown above. It is a standalone function, not a method of the VisualizerBot class.

---

**This setup ensures anyone can replicate your environment with minimal manual steps and maximum safety.**
