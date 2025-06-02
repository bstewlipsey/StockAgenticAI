# bot_gemini_key_manager.py
"""
GeminiKeyManagerBot: Tracks Gemini API key quotas and rotates keys to avoid exceeding daily limits.
Limits usage per key based on time of day and total allowed requests.
"""
import threading
from datetime import datetime, timedelta
from config_system import GEMINI_API_KEYS, GEMINI_API_QUOTA_PER_MINUTE
from bot_quota_manager import QuotaManagerBot
from utils.logger_mixin import LoggerMixin
from utils.logging_decorators import log_method_calls

# Gemini API quota per key (default: 500 requests per day)
GEMINI_DAILY_QUOTA = 500
GEMINI_KEY_COUNT = len([k for k in GEMINI_API_KEYS if k])


class GeminiKeyManagerBot(LoggerMixin):
    """
    Tracks Gemini API key quotas and rotates keys to avoid exceeding daily limits.
    Limits usage per key based on time of day and total allowed requests.
    Integrates with QuotaManagerBot for global quota enforcement.
    """

    def __init__(self):
        """
        Initialize GeminiKeyManagerBot with all available keys and usage tracking.
        Sets up quota manager and logging.
        """
        super().__init__()
        self.keys = [k for k in GEMINI_API_KEYS if k]
        self.daily_quota = GEMINI_DAILY_QUOTA
        self.usage = {
            k: [] for k in self.keys
        }  # List of timestamps for each key's usage
        self.lock = threading.Lock()
        self.quota_manager = QuotaManagerBot()
        self.logger.info(f"Initialized with {len(self.keys)} Gemini API keys.")

    @log_method_calls
    def _prune_old_usage(self, key):
        """
        Remove usage records older than 24 hours for a given key.
        Keeps usage tracking accurate for daily quota enforcement.
        """
        now = datetime.utcnow()
        before = len(self.usage[key])
        self.usage[key] = [t for t in self.usage[key] if t > now - timedelta(days=1)]
        after = len(self.usage[key])
        if before != after:
            self.logger.debug(f"Pruned {before - after} old usage entries for key {key[:6]}...: {before} -> {after}")

    @log_method_calls
    def get_available_key(self):
        """
        Return an available Gemini API key that is under quota.
        Returns None if all keys are exhausted or quota is exceeded.
        """
        now = datetime.utcnow()
        if not self.quota_manager.can_make_request("gemini", quota_per_minute=GEMINI_API_QUOTA_PER_MINUTE):
            self.logger.info("Gemini quota exceeded (global). No key available.")
            return None
        for key in self.keys:
            self._prune_old_usage(key)
            used = len(self.usage[key])
            seconds_since_midnight = (
                now - now.replace(hour=0, minute=0, second=0, microsecond=0)
            ).total_seconds()
            max_allowed = int(self.daily_quota * (seconds_since_midnight / 86400))
            if used < min(self.daily_quota, max_allowed):
                self.logger.info(f"Selected key {key[:6]}... (used: {used}, max_allowed: {max_allowed})")
                return key
        self.logger.warning("All Gemini API keys exhausted for now.")
        return None

    @log_method_calls
    def record_usage(self, key):
        """
        Record usage of a Gemini API key and update quota manager.
        """
        with self.lock:
            before = len(self.usage[key])
            self.usage[key].append(datetime.utcnow())
            after = len(self.usage[key])
            self.logger.info(f"Recorded usage for key {key[:6]}...: {before} -> {after}")
            self.quota_manager.record_request("gemini")

    @log_method_calls
    def get_usage_report(self):
        """
        Return a report of usage for all Gemini API keys.
        Includes used and remaining quota for each key.
        """
        report = {}
        for key in self.keys:
            self._prune_old_usage(key)
            used = len(self.usage[key])
            report[key] = {"used": used, "remaining": self.daily_quota - used}
            self.logger.debug(f"Usage report for key {key[:6]}...: used={used}, remaining={self.daily_quota - used}")
        return report

    @log_method_calls
    def reset_usage(self):
        """
        Reset usage records for all Gemini API keys.
        Useful for daily resets or testing.
        """
        with self.lock:
            for key in self.keys:
                self.usage[key] = []
                self.logger.info(f"Reset usage for key {key[:6]}...")

    @log_method_calls
    def selftest(self):
        """
        Run a self-test to verify key availability, usage tracking, and reset logic.
        Prints test results and asserts correct behavior.
        """
        print(f"\n--- Running {self.__class__.__name__} Self-Test ---")
        try:
            print("  - Testing key availability and usage tracking...")
            assert len(self.keys) > 0, "No Gemini API keys configured."
            key = self.get_available_key()
            assert (
                key in self.keys
            ), f"get_available_key() did not return a valid key. Got: {key}"
            print(f"    -> get_available_key() returned: {key[:6]}... (truncated)")
            before = self.get_usage_report()[key]["used"]
            self.record_usage(key)
            after = self.get_usage_report()[key]["used"]
            assert after == before + 1, "record_usage() did not increment usage count."
            print(f"    -> record_usage() incremented usage count: {before} -> {after}")
            self.reset_usage()
            reset = self.get_usage_report()[key]["used"]
            assert reset == 0, "reset_usage() did not reset usage count."
            print("    -> reset_usage() set usage count to 0.")
            print(f"--- {self.__class__.__name__} Self-Test PASSED ---")
            self.logger.info("Self-test PASSED.")
        except AssertionError as e:
            print(f"--- {self.__class__.__name__} Self-Test FAILED: {e} ---")
            self.logger.error(f"Self-test FAILED: {e}")
        except Exception as e:
            print(
                f"--- {self.__class__.__name__} Self-Test encountered an ERROR: {e} ---"
            )
            self.logger.error(f"Self-test ERROR: {e}")


if __name__ == "__main__":
    test_bot = GeminiKeyManagerBot()
    test_bot.selftest()

# Example usage:
# gemini_key_manager = GeminiKeyManagerBot()
# key = gemini_key_manager.get_available_key()
# if key:
#     ... use key for API call ...
#     gemini_key_manager.record_usage(key)
# else:
#     print("All Gemini API keys exhausted for now. Wait before retrying.")
