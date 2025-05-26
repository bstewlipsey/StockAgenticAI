# StockAgenticAI - Setup & Installation Guide

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
   - Edit `config.py` and set your Alpaca, Gemini, and other API keys.

4. **Run the trading bot:**
   ```powershell
   python main.py
   ```

## Troubleshooting
- If you see errors about missing C headers (like `longintrepr.h` or `aiohttp` build failures), make sure you have installed the Microsoft C++ Build Tools.
- If you use a different Python version, some dependencies may not work. Python 3.10 is recommended.

---

**This setup ensures anyone can replicate your environment with minimal manual steps.**
