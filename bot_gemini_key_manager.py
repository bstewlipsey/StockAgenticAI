# bot_gemini_key_manager.py
"""
GeminiKeyManagerBot: Tracks Gemini API key quotas and rotates keys to avoid exceeding daily limits.
Limits usage per key based on time of day and total allowed requests.
"""
import os
import time
import threading
from datetime import datetime, timedelta
from config_system import GEMINI_API_KEYS

# Gemini API quota per key (default: 500 requests per day)
GEMINI_DAILY_QUOTA = 500
GEMINI_KEY_COUNT = len([k for k in GEMINI_API_KEYS if k])

class GeminiKeyManagerBot:
    def __init__(self, daily_quota=GEMINI_DAILY_QUOTA):
        self.keys = [k for k in GEMINI_API_KEYS if k]
        self.daily_quota = daily_quota
        self.usage = {k: [] for k in self.keys}  # List of timestamps for each key's usage
        self.lock = threading.Lock()

    def _prune_old_usage(self, key):
        now = datetime.utcnow()
        self.usage[key] = [t for t in self.usage[key] if t > now - timedelta(days=1)]

    def get_available_key(self):
        now = datetime.utcnow()
        for key in self.keys:
            self._prune_old_usage(key)
            used = len(self.usage[key])
            # Evenly distribute usage over the day
            seconds_since_midnight = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
            max_allowed = int(self.daily_quota * (seconds_since_midnight / 86400))
            if used < min(self.daily_quota, max_allowed):
                return key
        return None  # All keys exhausted for now

    def record_usage(self, key):
        with self.lock:
            self.usage[key].append(datetime.utcnow())

    def get_usage_report(self):
        now = datetime.utcnow()
        report = {}
        for key in self.keys:
            self._prune_old_usage(key)
            used = len(self.usage[key])
            report[key] = {
                'used': used,
                'remaining': self.daily_quota - used
            }
        return report

    def reset_usage(self):
        with self.lock:
            for key in self.keys:
                self.usage[key] = []

    def selftest(self):
        print(f"\n--- Running {self.__class__.__name__} Self-Test ---")
        try:
            # 1. Test key availability and usage tracking
            print("  - Testing key availability and usage tracking...")
            assert len(self.keys) > 0, "No Gemini API keys configured."
            key = self.get_available_key()
            assert key in self.keys, f"get_available_key() did not return a valid key. Got: {key}"
            print(f"    -> get_available_key() returned: {key[:6]}... (truncated)")
            before = self.get_usage_report()[key]['used']
            self.record_usage(key)
            after = self.get_usage_report()[key]['used']
            assert after == before + 1, "record_usage() did not increment usage count."
            print(f"    -> record_usage() incremented usage count: {before} -> {after}")
            # 2. Test reset_usage
            self.reset_usage()
            reset = self.get_usage_report()[key]['used']
            assert reset == 0, "reset_usage() did not reset usage count."
            print(f"    -> reset_usage() set usage count to 0.")
            print(f"--- {self.__class__.__name__} Self-Test PASSED ---")
        except AssertionError as e:
            print(f"--- {self.__class__.__name__} Self-Test FAILED: {e} ---")
        except Exception as e:
            print(f"--- {self.__class__.__name__} Self-Test encountered an ERROR: {e} ---")

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
