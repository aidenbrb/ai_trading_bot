"""
First-party morning news fetching and classification.

Pure fetch/classify functions - no DB, no side effects, easy to unit test with
mocked yfinance/httpx. Feeds nodes/news_node.py, which writes the JSON that
utils/market_intelligence.py reads (same schema as the old external reader).

Sources (all free, no API key required):
  - Per-ticker headlines : yfinance.Ticker(symbol).news
  - Macro / market-wide  : CNBC Top News, CNBC Economy, Federal Reserve press
                           releases (RSS)
  - Volatility proxy     : ^VIX level + SPY ATR-14 vs its own 20-day average
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from config.settings import settings

# -- Classification keyword sets -------------------------------------------------

_SEVERE_NEGATIVE_KEYWORDS = [
    "bankruptcy", "bankrupt", "chapter 11", "fraud", "sec investigation",
    "securities fraud", "recall", "data breach", "hacked", "ceo resign",
    "cfo resign", "resigns as ceo", "delisting", "delisted", "restatement",
    "restate", "going concern", "class action", "indictment", "indicted",
    "plunges", "plunge", "trading halt", "halted trading",
]

_EARNINGS_KEYWORDS = [
    "earnings", "quarterly results", "q1 results", "q2 results",
    "q3 results", "q4 results", "guidance cut", "profit warning",
    "misses estimates", "beats estimates", "earnings report",
]

_VOLATILITY_KEYWORDS = [
    "selloff", "sell-off", "circuit breaker", "volatility spike",
    "market rout", "crashes", "crash",
]

_MACRO_HIGH_KEYWORDS = [
    "recession", "banking collapse", "bank run", "market-wide circuit breaker",
    "financial crisis", "credit crunch", "systemic risk",
]

# Reassuring/hypothetical framing that downgrades an otherwise-"high" match to
# "warn" - e.g. a Fed stress-test press release saying banks are "well
# positioned to weather a severe recession" is good news, not a recession.
_MACRO_HIGH_FALSE_POSITIVE_GUARDS = [
    "stress test", "well positioned", "weather a", "resilient to", "avoid a",
]

_MACRO_WARN_KEYWORDS = [
    "federal reserve", "fed rate", "interest rate", "inflation", "tariff",
    "government shutdown", "credit downgrade", "downgrade", "rate hike",
    "rate cut", "jobs report", "cpi report",
]


def classify_company_headline(text: str) -> Optional[str]:
    """Return a bot_context_label for a per-ticker headline, or None."""
    t = (text or "").lower()
    if any(k in t for k in _SEVERE_NEGATIVE_KEYWORDS):
        return "avoid_new_trade"
    if any(k in t for k in _EARNINGS_KEYWORDS):
        return "earnings_risk"
    if any(k in t for k in _VOLATILITY_KEYWORDS):
        return "high_volatility_warning"
    return None


def classify_ticker_headline(
    symbol: str,
    headline: str,
    summary: str = "",
    *,
    is_etf: bool = False,
) -> Optional[str]:
    """
    Classify a Yahoo per-ticker item with instrument context.

    Yahoo ETF feeds contain broad articles about unrelated constituents. Those
    articles must not create company-specific blocks for the ETF; broad market
    risk is already handled by the dedicated macro and VIX checks.

    Earnings mentions remain auditable warnings. Actual earnings-date blocking
    stays in risk_node._days_to_earnings(), which uses the company calendar.
    """
    if is_etf:
        return None
    return classify_company_headline(f"{headline} {summary}")


def is_recent_headline(
    pub_date: str,
    *,
    now: Optional[datetime] = None,
    max_age_hours: Optional[float] = None,
) -> bool:
    """Fail closed for missing, malformed, future, or stale publication times."""
    if not pub_date:
        return False
    try:
        published = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        published = published.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return False

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    age = current.astimezone(timezone.utc) - published
    limit = settings.NEWS_ITEM_MAX_AGE_HOURS if max_age_hours is None else max_age_hours
    return timedelta(0) <= age <= timedelta(hours=limit)


def classify_macro_headline(text: str) -> Optional[str]:
    """Return 'high' | 'warn' severity for a macro headline, or None."""
    t = (text or "").lower()
    if any(k in t for k in _MACRO_HIGH_KEYWORDS):
        if any(g in t for g in _MACRO_HIGH_FALSE_POSITIVE_GUARDS):
            return "warn"
        return "high"
    if any(k in t for k in _MACRO_WARN_KEYWORDS):
        return "warn"
    return None


# -- Per-ticker news --------------------------------------------------------------

def fetch_ticker_news(symbol: str, max_items: int = 8) -> list[dict]:
    """
    Return normalized, explicitly-related Yahoo headlines for symbol.

    yf.Search exposes relatedTickers, unlike Ticker.news. Requiring that field
    prevents Yahoo's broad recommendation feed from attaching a SpaceX story to
    several unrelated bank tickers. Ticker.news remains a compatibility fallback
    for older yfinance versions, but its results are marked as relevance_unknown.
    """
    try:
        import yfinance as yf
        raw = yf.Search(symbol, news_count=max_items).news or []
        source = "search"
    except Exception:
        try:
            raw = yf.Ticker(symbol).news or []
            source = "ticker_fallback"
        except Exception:
            return []

    items = []
    for n in raw[:max_items]:
        content = n.get("content", n)   # tolerate older yfinance's flat shape
        title = content.get("title") or ""
        if not title:
            continue
        related = content.get("relatedTickers") or []
        if source == "search" and symbol.upper() not in {
            str(t).upper() for t in related
        }:
            continue
        pub_date = content.get("pubDate") or ""
        if not pub_date and content.get("providerPublishTime"):
            try:
                pub_date = datetime.fromtimestamp(
                    int(content["providerPublishTime"]), tz=timezone.utc
                ).isoformat()
            except (TypeError, ValueError, OSError):
                pub_date = ""
        items.append({
            "symbol":           symbol,
            "headline":         title,
            "summary":          content.get("summary") or "",
            "pub_date":         pub_date,
            "related_symbols":  related,
            "relevance_source": (
                "yahoo_related_tickers" if source == "search"
                else "ticker_feed_fallback"
            ),
        })
    return items


# -- Macro / market-wide feeds -----------------------------------------------------

_MACRO_FEEDS = [
    ("CNBC Top News",                 "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("CNBC Economy",                  "https://www.cnbc.com/id/20910258/device/rss/rss.html"),
    ("Federal Reserve Press Releases", "https://www.federalreserve.gov/feeds/press_all.xml"),
]
MACRO_FEED_COUNT = len(_MACRO_FEEDS)


def fetch_macro_feeds() -> tuple[list[dict], list[dict]]:
    """
    Fetch and parse the macro/market-wide RSS feeds.
    Returns (items, source_errors) - one dead feed doesn't stop the others.
    """
    import httpx
    import feedparser

    items: list[dict] = []
    errors: list[dict] = []

    for name, url in _MACRO_FEEDS:
        try:
            resp = httpx.get(
                url,
                timeout=settings.NEWS_RSS_TIMEOUT_SECONDS,
                headers={"User-Agent": "Mozilla/5.0"},
                follow_redirects=True,
            )
            resp.raise_for_status()
            parsed = feedparser.parse(resp.content)
            for entry in parsed.entries[:15]:
                items.append({
                    "source":   name,
                    "headline": entry.get("title", ""),
                    "summary":  entry.get("summary", ""),
                    "pub_date": entry.get("published", ""),
                })
        except Exception as exc:
            errors.append({"source": name, "error": str(exc)})

    return items, errors


# -- Volatility regime --------------------------------------------------------------

def get_volatility_regime() -> dict:
    """
    Classify current market volatility as 'low' | 'elevated' | 'high' from two
    independent signals: ^VIX level, and SPY's ATR-14 vs its own 20-day average
    (catches an idiosyncratic single-day spike even when VIX itself is calm).
    """
    vix_level: Optional[float] = None
    spy_atr_ratio: Optional[float] = None

    try:
        import yfinance as yf
        vix = yf.Ticker("^VIX").history(period="5d", interval="1d")
        if not vix.empty:
            vix_level = float(vix["Close"].iloc[-1])
    except Exception:
        pass

    try:
        import yfinance as yf
        from utils.indicators import atr as atr_fn
        spy = yf.Ticker("SPY").history(period="60d", interval="1d")
        if len(spy) >= 34:   # need enough bars for a 20-day rolling avg of ATR-14
            atr_series = atr_fn(spy["High"], spy["Low"], spy["Close"], 14)
            current_atr = float(atr_series.iloc[-1])
            avg_atr = float(atr_series.rolling(20).mean().iloc[-1])
            if avg_atr > 0:
                spy_atr_ratio = current_atr / avg_atr
    except Exception:
        pass

    regime = "low"
    if ((vix_level is not None and vix_level >= settings.NEWS_VIX_HIGH_THRESHOLD)
            or (spy_atr_ratio is not None and spy_atr_ratio >= 1.5)):
        regime = "high"
    elif ((vix_level is not None and vix_level >= settings.NEWS_VIX_ELEVATED_THRESHOLD)
            or (spy_atr_ratio is not None and spy_atr_ratio >= 1.25)):
        regime = "elevated"

    return {"regime": regime, "vix": vix_level, "spy_atr_ratio": spy_atr_ratio}
