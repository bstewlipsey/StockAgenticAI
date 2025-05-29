# StockAgenticAI - Setup & Installation Guide

## Project Status

**As of May 2025, StockAgenticAI is fully refactored, integrated, and tested.**
- All configuration is centralized in `config_system.py` and `config_trading.py` (system/API) (strategy/bot).
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
   - Edit `config_system.py` and set your Alpaca, Gemini, NewsAPI, and other API keys as needed.
   - **To avoid exceeding your Gemini free quota:**
     - Set `TRADING_CYCLE_INTERVAL` in `config_system.py` to at least `60` seconds (or higher, e.g., `120` seconds) for testing.
     - Limit the number of assets in `TRADING_ASSETS` in `config_trading.py` to 1-2 assets during tests.
     - Set `MAX_TOKENS` in `config_system.py` to a lower value (e.g., `500`).
     - Set `TEMPERATURE` in `config_system.py` to `0.5` or lower for more deterministic, shorter responses.
     - Avoid running multiple bots or tests in parallel that use the Gemini API.
     - Monitor your quota usage in the Google Cloud Console.

4. **Run the trading bot:**
   ```powershell
   python main.py
   ```

## Troubleshooting
- If you see errors about missing C headers (like `longintrepr.h` or `aiohttp` build failures), make sure you have installed the Microsoft C++ Build Tools.
- If you use a different Python version, some dependencies may not work. Python 3.10 is recommended.

## Alpaca US Stock Data Troubleshooting

- If you receive empty data for US stocks (e.g., AAPL) when using the free IEX feed, this is a limitation of Alpaca's IEX data coverage. Not all stocks are available, and data may be delayed or missing outside US market hours.
- **US Stock Market Hours:** 9:30am–4:00pm Eastern Time (ET), Monday–Friday (excluding market holidays). Data may be unavailable or delayed outside these hours.
- For full, real-time US stock data, upgrade to SIP (paid) or use another provider.
- To explicitly use the free IEX feed, set `feed=DataFeed.IEX` in your Alpaca API requests.

## Configuration Files Overview

StockAgenticAI uses two main configuration files:

- `config_system.py`: Controls system-level and API settings (API keys, trading mode, logging, timeframes, etc.).
- `config_trading.py`: Controls trading strategy, asset selection, risk management, and AI prompt templates.

**Key Settings:**
- `TEST_MODE_ENABLED` (in `config_system.py`):
  - `True` for rapid iteration and safe testing (paper trading, fast cycles).
  - `False` for realistic simulation or live trading.
- `TRADING_CYCLE_INTERVAL` (in both configs):
  - Controls how often the bot checks for trades (in seconds).
- `TRADING_ASSETS` (in `config_trading.py`):
  - List of assets and allocation per asset.
- `ENABLE_TRADING_BOT` (in `config_system.py`):
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

# Gemini API keys are now managed exclusively by GeminiKeyManagerBot.
# Do NOT set or reference GEMINI_API_KEY in the environment or code.
# All Gemini API access is handled via GeminiKeyManagerBot for quota rotation and failover.

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

## Configuration File Best Practices
- `config_system.py`: System/API config only. All variables are used by the system, logging, or AI prompt construction
- `config_trading.py`: Trading/strategy config only. All variables are referenced by the trading system, bots, or tests.
- If you add new variables, document them and ensure they are not duplicated across files.

## Capturing All Output to Log
To capture all output (stdout and stderr) to the log file while running the bot, use:

```
python main.py > trading.log 2>&1
```

This will help with debugging and long-term monitoring.

---

## Bot Interaction & Data Access Guide

This section provides instructions on how to directly access information from your bots for debugging and monitoring.

### How to Access Trading Data & Logs Manually (from Python interactive shell)

1. **Open a Python interactive shell** in your project's root directory (where `main.py` and `bot_visualizer.py` are located):
   ```powershell
   python
   ```

2. **Import necessary modules and initialize bots/connect to DB:**
   ```python
   from bot_database import DatabaseBot
   from bot_portfolio import PortfolioBot
   from bot_visualizer import VisualizerBot, display_interactive_log_viewer
   from config_system import TOTAL_CAPITAL, ALPACA_API_KEY, ALPACA_SECRET_KEY, PAPER_TRADING
   from bot_trade_executor import TradeExecutorBot
   from bot_ai import AIBot
   from bot_report import ReportBot
   from bot_gemini_key_manager import GeminiKeyManagerBot

   db_bot = DatabaseBot()
   portfolio_bot = PortfolioBot(initial_capital=TOTAL_CAPITAL)
   visualizer = VisualizerBot()
   trade_executor = TradeExecutorBot(api_key=ALPACA_API_KEY, api_secret=ALPACA_SECRET_KEY, paper_trading=PAPER_TRADING)
   ai_bot = AIBot()
   report_bot = ReportBot()
   gemini_key_manager = GeminiKeyManagerBot()
   ```

3. **Retrieve Trading Database Information:**
   - **Get Portfolio Metrics:**
     ```python
     portfolio_metrics = portfolio_bot.get_portfolio_metrics()
     print("\n--- Current Portfolio Metrics ---")
     for key, value in portfolio_metrics.items():
         print(f"{key}: {value}")
     ```

   - **Get Open Positions:**
     ```python
     open_positions_memory = portfolio_bot.get_open_positions()
     print("\n--- Open Positions (from PortfolioBot) ---")
     if open_positions_memory:
         for pos_symbol, pos_data in open_positions_memory.items():
             print(f"Symbol: {pos_symbol}, Data: {pos_data}")
     else:
         print("No open positions in PortfolioBot.")
     ```

   - **Get Trade History:**
     ```python
     trade_history_memory = portfolio_bot.get_trade_history()
     print("\n--- Trade History (from PortfolioBot) ---")
     if trade_history_memory:
         for trade in trade_history_memory:
             print(trade)
     else:
         print("No completed trades in PortfolioBot's current session.")

     trade_history_db = db_bot.get_trade_history()
     print("\n--- Trade History (from DatabaseBot) ---")
     if trade_history_db:
         for trade in trade_history_db:
             print(trade)
     else:
         print("No completed trades in DatabaseBot.")
     ```

   - **Get Reflection Insights:**
     ```python
     print("\n--- Latest Reflection Insights (via VisualizerBot) ---")
     visualizer.display_reflection_insights(limit=5)
     ```

   - **Get Account Information (via TradeExecutorBot):**
     ```python
     account_info = trade_executor.get_account()
     print("\n--- Alpaca Account Info ---")
     if account_info:
         for key, value in account_info.items():
             print(f"{key}: {value}")
     else:
         print("Could not retrieve account info.")
     ```

   - **Generate Comprehensive Bot Status Report:**
     ```python
     report_bot.generate_comprehensive_report() # This will save a file. Check for the timestamped .txt file.
     print("\n--- Triggering Comprehensive Report Generation (Check for new .txt file) ---")
     ```

   - **View Logs in Real-Time:**
     ```python
     print("\n--- Real-time Log Viewer (Press Ctrl+C to stop) ---")
     display_interactive_log_viewer('trading.log')
     # You can also filter for important events:
     # display_interactive_log_viewer('trading.log', filter_keywords=['TRADE_OUTCOME', 'DECISION', 'ERROR'])
     ```

---

# Bot Architecture & Capabilities

StockAgenticAI is a modular, agentic trading system composed of specialized bots, each responsible for a distinct aspect of the trading workflow. Key bots include:

- **DecisionMakerBot**: Generates trading decisions using RAG and news context.
- **TradeExecutorBot**: Executes trades via broker APIs (e.g., Alpaca).
- **PortfolioBot**: Tracks portfolio state and metrics.
- **DatabaseBot**: Stores trades, positions, and insights.
- **NewsRetrieverBot**: Fetches and summarizes relevant news.
- **ReflectionBot**: Analyzes past trades and logs insights for continual learning.
- **RiskManagerBot**: Monitors and enforces risk constraints on trades and portfolio.
- **ReportBot**: Generates comprehensive status and health reports for all bots.
- **KnowledgeGraphBot**: (New) Maintains a knowledge graph linking trading decisions and outcomes for AI learning and analysis.

All bots are integrated and communicate via shared data structures and logging. Each bot supports a `selftest()` method for health checks and regression testing.

---

## KnowledgeGraphBot: AI Learning & Decision Tracking

**File:** `bot_knowledge_graph.py`

The `KnowledgeGraphBot` is a pilot module that uses a knowledge graph (via NetworkX) to link trading decisions and trade outcomes. This enables advanced AI learning, pattern discovery, and future explainability features.

**Capabilities:**
- Add nodes for `TradingDecision` and `TradeOutcome`.
- Link decisions to their resulting outcomes.
- Query relationships for analysis or visualization.
- Includes a robust `selftest()` for graph operations.

**Example Usage:**
```python
from bot_knowledge_graph import KnowledgeGraphBot
kg_bot = KnowledgeGraphBot()
decision_id = kg_bot.add_decision(symbol="AAPL", action="BUY", reason="Strong earnings")
outcome_id = kg_bot.add_outcome(symbol="AAPL", result="WIN", pnl=42.0)
kg_bot.link_decision_to_outcome(decision_id, outcome_id)
print(kg_bot.query_decision_outcomes(decision_id))
kg_bot.selftest()  # Runs internal tests
```

---

## Selftest & Health Checks

All major bots implement a `selftest()` method for rapid health checks and regression testing. This ensures that each bot's core logic, integration points, and data flows are functioning as expected.

**How to run a bot's selftest:**
```python
from bot_reflection import ReflectionBot
ReflectionBot().selftest()

from bot_risk_manager import RiskManagerBot
RiskManagerBot().selftest()

from bot_report import ReportBot
ReportBot().selftest()

from bot_knowledge_graph import KnowledgeGraphBot
KnowledgeGraphBot().selftest()
```

**How to run all tests (including integration):**
```
pytest
```

---

## CI/CD & Automated Log Monitoring (Preview)

Automated test runs and log monitoring are being added for continuous integration and operational reliability. See Section 5.2 in the TODO.txt for upcoming details.

---

## Automated Test Runs & Log Monitoring

A PowerShell script is provided to automate running all tests and checking the trading log for errors or critical issues.

**To use:**

1. Open PowerShell in your project directory.
2. Run:
   ```powershell
   .\run_tests_and_log_check.ps1
   ```
   - This will:
     - Run all Python tests using `pytest`.
     - Scan `trading.log` for any lines containing `ERROR` or `CRITICAL`.
     - Print a summary of any issues found.

**Note:**
- Ensure you have run `main.py` at least once to generate `trading.log`.
- For continuous integration (CI), you can adapt this script for use in GitHub Actions or other CI/CD systems.

---
