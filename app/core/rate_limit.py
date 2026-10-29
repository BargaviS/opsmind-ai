"""Tiny in-memory sliding-window rate limiter (per client IP).

Good enough for a single-process demo. For several servers you would move
the counters to Redis. That trade-off is documented in the README.
"""
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, Request

from app.core.config import Settings, get_settings

_hits: dict[str, deque] = defaultdict(deque)


def reset_rate_limiter() -> None:
    _hits.clear()


def rate_limit(request: Request, settings: Settings = Depends(get_settings)) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _hits[ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= settings.rate_limit_per_minute:
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")
    window.append(now)
