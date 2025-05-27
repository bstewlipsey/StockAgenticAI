# Core imports
import logging
from colorama import init, Style

# Bot imports
from bot_orchestrator import OrchestratorBot

# Configuration imports
from config import ENABLE_TRADING_BOT

def main():
    # Initialize colorama for Windows
    init()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    print(f"\n{Style.BRIGHT}🤖 Starting StockAgenticAI...{Style.RESET_ALL}")

    # Instantiate the orchestrator and start the trading loop
    orchestrator = OrchestratorBot()
    orchestrator.start_trading_loop()

if __name__ == "__main__":
    main()