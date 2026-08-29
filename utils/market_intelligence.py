"""
Reader for the first-party morning news report written by nodes/news_node.py.

JSON output:  data/market_intelligence/YYYY-MM-DD.json
Status file:  data/run_status.json

news_gate_status() is the single fail-closed choke point: strategy nodes and
risk_node must refuse new trades if the report is missing or stale, rather
than silently skipping the gate.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from config.settings import settings
from utils.timeutil import utcnow

# Repo-local by default; overridable via NEWS_DATA_DIR for tests/alternate setups.
def data_dir() -> Path:
    if settings.NEWS_DATA_DIR:
        return Path(settings.NEWS_DATA_DIR)
    return Path(__file__).parent.parent / "data" / "market_intelligence"


def status_file() -> Path:
    return data_dir().parent / "run_status.json"


# Labels that block opening a new trade
_BLOCK_LABELS: set[str] = {
    "avoid_new_trade",
    "high_volatility_warning",
}

# Crypto-specific blocking labels
_CRYPTO_BLOCK_LABELS: set[str] = _BLOCK_LABELS | {"crypto_risk"}

# Labels that are warnings only (noted in thesis, don't block)
_WARN_LABELS: set[str] = {
    "earnings_risk",
    "increase_risk_review",
    "require_confirmation",
    "macro_risk",
}


def is_ready(date_str: str | None = None) -> bool:
    """Return True if the news node ran successfully for date_str."""
    if date_str is None:
        date_str = date.today().isoformat()
    sf = status_file()
    if not sf.exists():
        return False
    try:
        s = json.loads(sf.read_text(encoding="utf-8"))
        return s.get("status") == "success" and s.get("date") == date_str
    except Exception:
        return False


def load(date_str: str | None = None) -> dict | None:
    """Load the market intelligence dict for date_str. Returns None if unavailable."""
    if date_str is None:
        date_str = date.today().isoformat()
    path = data_dir() / f"{date_str}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def is_stale(data: dict, max_age_hours: Optional[float] = None) -> bool:
    """
    Return True if data's generated_at timestamp is missing, unparseable, or
    older than max_age_hours (defaults to settings.NEWS_MAX_AGE_HOURS). A
    missing/bad timestamp is treated as stale - fail closed, not open.
    """
    if max_age_hours is None:
        max_age_hours = settings.NEWS_MAX_AGE_HOURS
    raw = data.get("generated_at")
    if not raw:
        return True
    try:
        generated_at = datetime.fromisoformat(raw)
    except Exception:
        return True
    return (utcnow() - generated_at) > timedelta(hours=max_age_hours)


def news_gate_status(date_str: str | None = None) -> tuple[Optional[dict], bool, str]:
    """
    Single fail-closed choke point for the morning news gate.

    Returns (mi_data, blocked, reason). blocked=True means the caller must
    refuse ALL new trades this run - the report is missing, unreadable, or
    stale, and NEWS_GATE_ENABLED is not overriding that.
    """
    if not settings.NEWS_GATE_ENABLED:
        return None, False, ""

    if date_str is None:
        date_str = date.today().isoformat()

    if not is_ready(date_str):
        return None, True, "morning news report is missing for today"

    data = load(date_str)
    if data is None:
        return None, True, "morning news report could not be read"

    if is_stale(data):
        return data, True, (
            f"morning news report is stale (older than "
            f"{settings.NEWS_MAX_AGE_HOURS:.0f}h)"
        )

    return data, False, ""


def _get_items_for(symbol: str, data: dict) -> list[dict]:
    """Return all intelligence items that reference symbol."""
    ticker = symbol.upper().replace("-USD", "")
    return [
        item for item in data.get("items", [])
        if (item.get("symbol") or "").upper() == ticker
        or ticker in (item.get("related_symbols") or [])
    ]


def check_ticker(symbol: str, data: dict, crypto: bool = False) -> tuple[bool, str]:
    """
    Returns (ok_to_trade, reason).
    ok_to_trade=False means the news gate blocks this ticker today.
    """
    if data.get("overall_market_risk") == "high":
        return False, "news gate: overall_market_risk=HIGH"

    block_set = _CRYPTO_BLOCK_LABELS if crypto else _BLOCK_LABELS
    items = _get_items_for(symbol, data)
    blocking = [
        item.get("bot_context_label")
        for item in items
        if item.get("bot_context_label") in block_set
    ]
    if blocking:
        reasons = ", ".join(sorted(set(blocking)))
        return False, f"news gate: {reasons}"
    return True, ""


def get_warnings(symbol: str, data: dict) -> list[str]:
    """Return warning labels for symbol (non-blocking, for thesis annotation)."""
    items = _get_items_for(symbol, data)
    return list({
        item.get("bot_context_label")
        for item in items
        if item.get("bot_context_label") in _WARN_LABELS
    })


def overall_risk(data: dict) -> str:
    """Return overall_market_risk: 'low' | 'medium' | 'high'."""
    return data.get("overall_market_risk", "unknown")


def format_for_llm(data: dict, ticker: str | None = None, max_items: int = 10) -> str:
    """Format the intelligence as a compact string ready to prepend to LLM context."""
    lines = [
        f"=== MARKET INTELLIGENCE BRIEF ({data['report_date']}) ===",
        f"Overall Market Risk: {data.get('overall_market_risk', 'unknown').upper()}",
        "",
    ]
    for key, label in [
        ("macro_summary",         "MACRO"),
        ("stock_market_summary",  "STOCKS"),
        ("crypto_market_summary", "CRYPTO"),
        ("commodity_summary",     "COMMODITIES"),
    ]:
        val = data.get(key, "")
        if val:
            lines.append(f"{label}: {val}")

    items = data.get("items", [])
    if items:
        if ticker:
            t = ticker.upper().replace("-USD", "")
            related = [i for i in items if t in (i.get("related_symbols") or []) or (i.get("symbol") or "").upper() == t]
            others  = [i for i in items if i not in related]
            ordered = related + others
        else:
            ordered = items

        lines += ["", "TOP SCORED ITEMS (pre-classified before market open):"]
        for n, item in enumerate(ordered[:max_items], 1):
            score = item.get("importance_score", 0)
            sym   = item.get("symbol") or "-"
            cat   = (item.get("catalyst_type") or "other").replace("_", " ")
            lbl   = item.get("bot_context_label", "")
            sent  = item.get("sentiment", "unknown")
            hl    = (item.get("headline") or "")[:110]
            lines.append(f"{n}. [score:{score} | {sym} | {cat} | {lbl} | {sent}] {hl}")

    errors = data.get("source_errors", [])
    if errors:
        lines += ["", f"[source errors: {', '.join(e['source'] for e in errors)}]"]

    lines.append("=== END MARKET INTELLIGENCE BRIEF ===")
    return "\n".join(lines)
