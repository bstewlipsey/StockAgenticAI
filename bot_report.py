import os
import datetime
from config_system import *
from config_trading import *
from bot_portfolio import PortfolioBot
from bot_database import DatabaseBot
from bot_visualizer import VisualizerBot
from bot_ai import AIBot

class ReportBot:
    def __init__(self):
        self.db_bot = DatabaseBot()
        self.portfolio_bot = PortfolioBot(initial_capital=TOTAL_CAPITAL)
        self.visualizer = VisualizerBot()
        # Use default AIBot instantiation; do not reference GEMINI_API_KEY
        self.ai_bot = AIBot()  # Will use key manager or default logic

    def _get_config_summary(self):
        config = {
            'System': {
                'TEST_MODE_ENABLED': TEST_MODE_ENABLED,
                'TRADING_CYCLE_INTERVAL': TRADING_CYCLE_INTERVAL,
                'TOTAL_CAPITAL': TOTAL_CAPITAL,
                # ...add more as needed
            },
            'Trading': {
                'MIN_CONFIDENCE': MIN_CONFIDENCE,
                'TRADING_ASSETS': TRADING_ASSETS,
                # ...add more as needed
            }
        }
        return config

    def _get_log_summary(self, log_path='trading.log', max_lines=200):
        if not os.path.exists(log_path):
            return 'Log file not found.'
        lines = []
        # Open the log file with errors='replace' to handle any encoding issues robustly
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f.readlines()[-max_lines:]:
                if any(x in line for x in ['ERROR', 'WARNING', 'TRADE_OUTCOME', 'DECISION']):
                    lines.append(line.strip())
        return '\n'.join(lines[-50:])

    def _get_portfolio_state(self):
        metrics = self.portfolio_bot.get_portfolio_metrics()
        open_positions = self.portfolio_bot.get_open_positions()
        return metrics, open_positions

    def _get_trade_history(self, limit=10):
        # Use PortfolioBot's trade history for now, as DatabaseBot does not have get_trade_history
        trades = self.portfolio_bot.get_trade_history()
        return trades[-limit:] if trades else []

    def _get_reflection_insights(self, limit=5, as_string=True):
        # Get latest reflection insights as a string summary for the report
        # Use cross-asset insights for general report
        insights = self.db_bot.get_cross_asset_insights(days=90)[:limit] if hasattr(self.db_bot, 'get_cross_asset_insights') else []
        if not insights:
            return "No recent reflection insights."
        if as_string:
            lines = []
            for i, insight in enumerate(insights, 1):
                lines.append(f"[{i}] {insight.get('symbol', '')} | AvgPnL: {insight.get('avg_pnl', 0):.2f}% | Trades: {insight.get('trade_count', 0)} | Insights: {insight.get('combined_insights', '')}")
            return '\n'.join(lines)
        return insights

    def _get_bot_status(self):
        """
        Returns a health/status summary for all major bots.
        Checks for instantiation, API key presence, and basic connectivity if possible.
        """
        status = {}
        # PortfolioBot
        try:
            metrics = self.portfolio_bot.get_portfolio_metrics()
            status['PortfolioBot'] = {'status': 'OK', 'metrics': metrics}
        except Exception as e:
            status['PortfolioBot'] = {'status': 'ERROR', 'error': str(e)}
        # DatabaseBot
        try:
            # Use a method that should exist: get_reflection_insights (with only required args)
            _ = self.db_bot.get_reflection_insights('AAPL')
            status['DatabaseBot'] = {'status': 'OK'}
        except Exception as e:
            status['DatabaseBot'] = {'status': 'ERROR', 'error': str(e)}
        # VisualizerBot
        try:
            if self.visualizer:
                status['VisualizerBot'] = {'status': 'OK'}
            else:
                status['VisualizerBot'] = {'status': 'ERROR', 'error': 'Not instantiated'}
        except Exception as e:
            status['VisualizerBot'] = {'status': 'ERROR', 'error': str(e)}
        # AIBot
        try:
            if self.ai_bot and hasattr(self.ai_bot, 'api_key') and self.ai_bot.api_key:
                status['AIBot'] = {'status': 'OK'}
            else:
                status['AIBot'] = {'status': 'ERROR', 'error': 'No API key'}
        except Exception as e:
            status['AIBot'] = {'status': 'ERROR', 'error': str(e)}
        return status

    def _get_llm_summary(self, metrics, trade_history, reflection_insights):
        if not self.ai_bot:
            return "[LLM summary unavailable: No API key configured]"
        try:
            prompt = f"""
            Summarize the following trading system status for a human reader. Highlight any risks, opportunities, and key lessons.\n\nPortfolio Metrics: {metrics}\n\nRecent Trades: {trade_history}\n\nReflection Insights: {reflection_insights}\n"""
            summary = self.ai_bot.call_llm(prompt)
            if not summary or not summary.strip():
                return "[LLM summary unavailable: No response from LLM]"
            return summary.strip()
        except Exception as e:
            return f"[LLM summary error: {e}]"

    def generate_comprehensive_report(self, reason=None):
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        # Ensure the 'reports' directory exists
        reports_dir = 'reports'
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        filename = os.path.join(reports_dir, f'{reason}_report_{now}.txt')
        config = self._get_config_summary()
        log_summary = self._get_log_summary()
        metrics, open_positions = self._get_portfolio_state()
        trade_history = self._get_trade_history()
        reflection_insights = self._get_reflection_insights()
        bot_status = self._get_bot_status()
        try:
            llm_summary = self._get_llm_summary(metrics, trade_history, reflection_insights)
        except Exception as e:
            llm_summary = f"[LLM summary error: {e}]"

        report_title = f'=== Agentic Stock AI Comprehensive Report ({reason.replace("_", " ").title()}) ===' if reason else '=== Agentic Stock AI Comprehensive Report ==='
        report_lines = [
            report_title,
            f'Generated: {now}',
            '',
            '--- System & Trading Configuration ---',
            str(config),
            '',
            '--- Recent Log Summary ---',
            log_summary,
            '',
            '--- Portfolio State ---',
            f'Metrics: {metrics}',
            f'Open Positions: {open_positions}',
            '',
            '--- Recent Trade History ---',
            '\n'.join(str(t) for t in trade_history),
            '',
            '--- Recent Reflection Insights ---',
            str(reflection_insights),
            '',
            '--- Bot Status ---',
            str(bot_status),
            '',
            '--- LLM Summary ---',
            llm_summary,
            '',
            '=== End of Report ==='
        ]
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f'Report saved to {filename}')
        return filename

    def generate_report_on_demand(self):
        """Public method to generate a report from the Python shell."""
        return self.generate_comprehensive_report(reason='on_demand')

def selftest_report_bot():
    """Standalone self-test for ReportBot: tests report generation and file output."""
    print(f"\n--- Running ReportBot Self-Test ---")
    import os
    try:
        bot = ReportBot()
        filename = bot.generate_comprehensive_report(reason='test')
        assert os.path.exists(filename), f"Report file {filename} not created."
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'Comprehensive Report' in content, "Report content missing expected keywords."
        print("    -> Report file creation and content logic passed.")
        
        print(f"--- ReportBot Self-Test PASSED ---")
    except AssertionError as e:
        print(f"--- ReportBot Self-Test FAILED: {e} ---")
    except Exception as e:
        print(f"--- ReportBot Self-Test encountered an ERROR: {e} ---")

if __name__ == "__main__":
    selftest_report_bot()
