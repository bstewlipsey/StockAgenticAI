import sys
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import yfinance as yf
from colorama import init, Fore, Style
import ccxt  # Add this import

def test_imports():
    """Test if all required packages are installed correctly"""
    try:
        init()  # Initialize colorama
        print(f"{Fore.GREEN}Testing package imports...{Style.RESET_ALL}")
        
        # Check virtual environment
        venv = os.environ.get('VIRTUAL_ENV')
        if venv:
            print(f"Using virtual environment: {venv}")
        else:
            print(f"{Fore.RED}Warning: Not running in a virtual environment{Style.RESET_ALL}")
            
        # Test yfinance
        msft = yf.Ticker("MSFT")
        info = msft.info
        print(f"\nSuccessfully fetched {info.get('shortName', 'Microsoft')} data")
        
        # Test ccxt
        print("Testing ccxt...")
        exchange = ccxt.kraken()
        print(f"CCXT version: {ccxt.__version__}")
        
        print(f"\n{Fore.GREEN}✓ All imports successful!{Style.RESET_ALL}")
        print(f"\nPackage versions:")
        print(f"Python: {sys.version}")
        print(f"Pandas: {pd.__version__}")
        print(f"Numpy: {np.__version__}")
        return True
        
    except Exception as e:
        print(f"{Fore.RED}Error during import test: {str(e)}{Style.RESET_ALL}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)