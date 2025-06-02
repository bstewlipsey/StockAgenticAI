# Core imports
import logging
from colorama import init, Style

# Bot imports
from bot_orchestrator import OrchestratorBot
from bot_report import ReportBot


def main():
    # Initialize colorama for Windows
    init()
    # Set up logging with the correct format for all logs
    from utils.logging_setup import setup_logging
    setup_logging()

    print(f"\n{Style.BRIGHT}[AI] Starting StockAgenticAI...{Style.RESET_ALL}")

    orchestrator = OrchestratorBot()
    report_bot = ReportBot()
    try:
        orchestrator.start_trading_loop()
    except KeyboardInterrupt:
        print("\n[AI] Trading loop interrupted by user.")
    finally:
        print("\n[AI] Generating shutdown report...")
        report_bot.generate_comprehensive_report(reason="shutdown")
        print("[AI] Shutdown report generated.")


if __name__ == "__main__":
    main()
