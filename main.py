from agent import TradingAgent, analyze_asset, print_performance_summary, print_portfolio_summary
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, PAPER_TRADING
import logging
from colorama import init, Fore, Style

def main():
    # Initialize colorama for Windows
    init()
    
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    print(f"\n{Style.BRIGHT}🤖 Starting StockAgenticAI...{Style.RESET_ALL}")

    # List of assets to monitor (symbol, type)
    assets = [
        ("AAPL", "stock"),
        ("XXBTZUSD", "crypto"),
        ("XETHZUSD", "crypto")
    ]

    try:
        for symbol, asset_type in assets:
            print(f"\n📊 Analyzing {symbol} ({asset_type.upper()})...")
            
            # Get analysis
            analysis = analyze_asset(symbol, asset_type)
            
            # Print performance
            print_performance_summary(symbol, asset_type)
        
        # Print overall portfolio performance
        print_portfolio_summary()
        
    except KeyboardInterrupt:
        logger.info("Stopping the trading bot...")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()