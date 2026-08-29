"""Deterministic weekday-stock / weekend-crypto routing."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
VALID_MARKETS = {"auto", "stocks", "crypto", "all"}


def automatic_market(now: datetime | None = None) -> str:
    """Return crypto on Saturday/Sunday in New York, stocks otherwise."""
    current = now or datetime.now(EASTERN)
    if current.tzinfo is None:
        current = current.replace(tzinfo=EASTERN)
    else:
        current = current.astimezone(EASTERN)
    return "crypto" if current.weekday() >= 5 else "stocks"


def resolve_market(requested: str, *, auto_enabled: bool, now: datetime | None = None) -> str:
    requested = requested.lower()
    if requested not in VALID_MARKETS:
        raise ValueError(f"unknown market mode: {requested}")
    if requested != "auto":
        return requested
    return automatic_market(now) if auto_enabled else "all"
