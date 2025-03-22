import time
import logging
import threading


class APIRateLimiter:
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.request_timestamps = []
        self.lock = threading.Lock()

    def _clean_old_requests(self):
        """Remove timestamps older than 1 minute."""
        current_time = time.time()
        one_minute_ago = current_time - 60

        with self.lock:
            self.request_timestamps = [
                t for t in self.request_timestamps if t > one_minute_ago
            ]

    def wait_if_needed(self):
        """Wait if rate limits would be exceeded."""
        self._clean_old_requests()
        current_time = time.time()

        with self.lock:
            # Check request rate limit
            if len(self.request_timestamps) >= self.requests_per_minute:
                oldest_timestamp = min(self.request_timestamps)
                wait_time = 60 - (current_time - oldest_timestamp) + 0.1
                if wait_time > 0:
                    logging.info(
                        f"Rate limit approaching. Waiting {wait_time:.2f} seconds."
                    )
                    time.sleep(wait_time)

            # Record this request
            self.request_timestamps.append(time.time())


class APIManager:
    def __init__(self, config: dict):
        self.rate_limiters = {
            "openai": APIRateLimiter(
                config["openai"]["rate_limit"]["requests_per_minute"]
            ),
            "anthropic": APIRateLimiter(
                config["anthropic"]["rate_limit"]["requests_per_minute"]
            ),
            "gemini": APIRateLimiter(
                config["gemini"]["rate_limit"]["requests_per_minute"]
            ),
        }

    def wait_if_needed(self, provider: str):
        """Wait if needed for a specific provider."""
        if provider in self.rate_limiters:
            self.rate_limiters[provider].wait_if_needed()
        else:
            logging.warning(f"Unknown provider: {provider}")
