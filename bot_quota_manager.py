import time
import threading
import logging
from config_system import (
    RATE_LIMIT_DELAY_SECONDS,
    MAX_RETRIES,
    RETRY_DELAY,
    BASE_RETRY_DELAY,
    MAX_RETRY_BACKOFF_DELAY,
    JITTER_DELAY,
    TEST_MODE_ENABLED,
    SELFTEST_LIVE_API_CALLS_ENABLED,
)

class QuotaManagerBot:
    """
    Centralized API quota and rate limit manager for all external APIs (NewsAPI, Gemini, etc).
    Tracks usage, enforces delays, and provides can_make_request/api_request logic.
    Handles exponential backoff and retry logic for 429 errors.
    """
    def __init__(self):
        """
        Initialize QuotaManagerBot with usage tracking and backoff state.
        Sets up thread-safe structures for API usage and error handling.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.lock = threading.Lock()
        self.usage = {}  # {api_name: [timestamps]}
        self.last_429 = {}  # {api_name: last_429_time}
        self.retry_backoff = {}  # {api_name: current_backoff}

    def can_make_request(self, api_name, quota_per_minute=None):
        """
        Returns True if a request can be made to the given API, else False (enforces quota).
        Checks for recent 429 errors and enforces rate limits.
        """
        now = time.time()
        with self.lock:
            if quota_per_minute is not None:
                window = 60
                timestamps = self.usage.get(api_name, [])
                # Remove old timestamps
                timestamps = [t for t in timestamps if now - t < window]
                if len(timestamps) >= quota_per_minute:
                    self.logger.info(f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [can_make_request(api_name='{api_name}', quota_per_minute={quota_per_minute})] Quota exceeded: {len(timestamps)}/{quota_per_minute} in last 60s.")
                    return False
                self.usage[api_name] = timestamps
            # Check for recent 429
            last_429 = self.last_429.get(api_name, 0)
            if now - last_429 < RATE_LIMIT_DELAY_SECONDS:
                self.logger.info(f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [can_make_request(api_name='{api_name}')] Waiting after 429 error.")
                return False
            return True

    def record_request(self, api_name):
        """
        Record a request for the given API for quota tracking.
        """
        now = time.time()
        with self.lock:
            self.usage.setdefault(api_name, []).append(now)
            self.logger.debug(f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [record_request(api_name='{api_name}')] Request recorded at {now}.")

    def record_429(self, api_name):
        """
        Record a 429 (rate limit) error for the API and increase backoff.
        Implements exponential backoff logic.
        """
        now = time.time()
        with self.lock:
            self.last_429[api_name] = now
            # Exponential backoff
            prev = self.retry_backoff.get(api_name, BASE_RETRY_DELAY)
            new_backoff = min(prev * 2, MAX_RETRY_BACKOFF_DELAY)
            self.retry_backoff[api_name] = new_backoff
            self.logger.warning(f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [record_429(api_name='{api_name}')] 429 received. Backoff set to {new_backoff}s.")

    def get_backoff(self, api_name):
        """
        Get the current backoff delay for the API (in seconds).
        """
        return self.retry_backoff.get(api_name, BASE_RETRY_DELAY)

    def reset_backoff(self, api_name):
        """
        Reset the backoff delay for the API to the base value.
        """
        with self.lock:
            self.retry_backoff[api_name] = BASE_RETRY_DELAY
            self.logger.debug(f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [reset_backoff(api_name='{api_name}')] Backoff reset.")

    def api_request(self, api_name, func, *args, quota_per_minute=None, **kwargs):
        """
        Wraps an API call with quota and retry logic. Returns (success, result).
        Handles retries, backoff, and error logging for robust API usage.
        """
        for attempt in range(1, MAX_RETRIES + 1):
            if not self.can_make_request(api_name, quota_per_minute=quota_per_minute):
                backoff = self.get_backoff(api_name)
                self.logger.info(f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [api_request(api_name='{api_name}')] Sleeping for {backoff}s before retry.")
                time.sleep(backoff)
            try:
                self.record_request(api_name)
                result = func(*args, **kwargs)
                self.reset_backoff(api_name)
                return True, result
            except Exception as e:
                if hasattr(e, 'response') and getattr(e.response, 'status_code', None) == 429:
                    self.record_429(api_name)
                    self.logger.error(f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [api_request(api_name='{api_name}')] 429 Rate Limit encountered. Attempt {attempt}/{MAX_RETRIES}.")
                    time.sleep(self.get_backoff(api_name))
                else:
                    self.logger.error(f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [api_request(api_name='{api_name}')] Exception: {e}")
                    break
        return False, None

    @staticmethod
    def selftest():
        """
        Run a self-test to verify quota and backoff logic. Logs results.
        """
        logger = logging.getLogger("QuotaManagerBot")
        logger.info(f"bot_quota_manager [QuotaManagerBot]: [selftest()] START")
        bot = QuotaManagerBot()
        api = "test_api"
        # Simulate quota
        for i in range(5):
            allowed = bot.can_make_request(api, quota_per_minute=3)
            logger.info(f"bot_quota_manager [QuotaManagerBot]: [selftest()] can_make_request {i}: {allowed}")
            if allowed:
                bot.record_request(api)
            else:
                logger.info(f"bot_quota_manager [QuotaManagerBot]: [selftest()] Quota exceeded at {i}")
                time.sleep(1)
        # Simulate 429
        bot.record_429(api)
        assert bot.get_backoff(api) > BASE_RETRY_DELAY
        bot.reset_backoff(api)
        assert bot.get_backoff(api) == BASE_RETRY_DELAY
        logger.info(f"bot_quota_manager [QuotaManagerBot]: [selftest()] END")

if __name__ == "__main__":
    QuotaManagerBot.selftest()
