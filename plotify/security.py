"""Security headers and a small in-process rate limiter."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable

from flask import Response

# The pages need no JavaScript and no third-party resources, so the policy
# can be strict: an injected <script> or inline style would not run.
CONTENT_SECURITY_POLICY = "; ".join([
    "default-src 'none'",
    "style-src 'self'",
    "img-src 'self'",
    "form-action 'self'",
    "base-uri 'none'",
    "frame-ancestors 'none'",
])


def apply_security_headers(response: Response, *, hsts: bool = False) -> Response:
    headers = response.headers
    headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
    headers.setdefault("X-Content-Type-Options", "nosniff")
    headers.setdefault("X-Frame-Options", "DENY")
    headers.setdefault("Referrer-Policy", "same-origin")
    headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
    headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=()")
    if hsts:
        headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response


class RateLimiter:
    """Sliding-window limit of ``limit`` hits per ``window`` seconds per key.

    State lives in this process. That fits waitress (one process, many
    threads). For several processes or hosts, enforce limits at the reverse
    proxy instead.
    """

    def __init__(self, limit: int, window: float = 60.0, clock: Callable[[], float] = time.monotonic,
                 max_keys: int = 10_000):
        self.limit = limit
        self.window = window
        self._clock = clock
        self._max_keys = max_keys
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> float | None:
        """Record a hit; return seconds to wait if over the limit, else ``None``."""
        if self.limit <= 0:
            return None
        now = self._clock()
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                if len(self._hits) >= self._max_keys:
                    self._prune(now)
                hits = self._hits[key] = deque()
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return max(hits[0] + self.window - now, 0.0)
            hits.append(now)
            return None

    def _prune(self, now: float) -> None:
        for key in [k for k, v in self._hits.items() if not v or v[-1] <= now - self.window]:
            del self._hits[key]
        if len(self._hits) >= self._max_keys:  # still full: drop the oldest keys
            for key in sorted(self._hits, key=lambda k: self._hits[k][-1])[: len(self._hits) // 2]:
                del self._hits[key]
