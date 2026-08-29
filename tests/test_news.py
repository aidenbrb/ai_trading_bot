"""Tests for utils/news.py - keyword classification, fetch resilience, volatility regime."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch

import pandas as pd

from utils.news import (
    MACRO_FEED_COUNT,
    classify_company_headline,
    classify_ticker_headline,
    classify_macro_headline,
    fetch_macro_feeds,
    fetch_ticker_news,
    get_volatility_regime,
    is_recent_headline,
)


class TestClassifyCompanyHeadline:
    def test_bankruptcy_blocks(self):
        assert classify_company_headline("Company files for Chapter 11 bankruptcy") == "avoid_new_trade"

    def test_fraud_blocks(self):
        assert classify_company_headline("SEC investigation into accounting fraud") == "avoid_new_trade"

    def test_earnings_flagged(self):
        assert classify_company_headline("Company reports Q2 earnings results") == "earnings_risk"

    def test_volatility_flagged(self):
        assert classify_company_headline("Stock crashes amid market selloff") == "high_volatility_warning"

    def test_neutral_headline_returns_none(self):
        assert classify_company_headline("Company announces new product launch") is None

    def test_case_insensitive(self):
        assert classify_company_headline("COMPANY FILES FOR BANKRUPTCY") == "avoid_new_trade"

    def test_empty_or_none_returns_none(self):
        assert classify_company_headline("") is None
        assert classify_company_headline(None) is None

    def test_severe_takes_priority_over_earnings(self):
        # A headline matching both severe and earnings keywords must classify severe
        text = "Company fraud uncovered ahead of earnings report"
        assert classify_company_headline(text) == "avoid_new_trade"


class TestClassifyMacroHeadline:
    def test_recession_is_high(self):
        assert classify_macro_headline("Economists warn of recession ahead") == "high"

    def test_fed_rate_is_warn(self):
        assert classify_macro_headline("Federal Reserve raises interest rate") == "warn"

    def test_neutral_headline_returns_none(self):
        assert classify_macro_headline("Local team wins championship") is None

    def test_stress_test_recession_language_is_downgraded_to_warn(self):
        # Regression: a Fed stress-test press release saying banks are well
        # positioned to weather a severe recession is reassuring, not a real
        # recession signal - it must not drive overall_market_risk to "high"
        # and block every ticker on a false alarm.
        headline = (
            "Federal Reserve Board's annual bank stress test confirms that "
            "large banks are well positioned to weather a severe recession"
        )
        assert classify_macro_headline(headline) == "warn"


class TestTickerAwareClassification:
    def test_etf_not_blocked_by_unrelated_constituent_earnings(self):
        headline = (
            "Dow Futures Rise As Investors Turn To AI Hyperscaler Earnings: "
            "INTC, TSLA, TTD In Focus"
        )
        assert classify_ticker_headline("DIA", headline, is_etf=True) is None

    def test_etf_not_blocked_by_generic_plunge_article(self):
        assert classify_ticker_headline(
            "DIA", "Is This KWEB Plunge A Bargain Or A Trap?", is_etf=True
        ) is None

    def test_company_earnings_item_remains_auditable(self):
        assert classify_ticker_headline(
            "XOM", "Exxon Mobil Earnings Expected Next Week"
        ) == "earnings_risk"


class TestNewsFreshness:
    def test_recent_iso_z_timestamp_passes(self):
        from datetime import datetime, timezone
        now = datetime(2026, 7, 25, 16, tzinfo=timezone.utc)
        assert is_recent_headline(
            "2026-07-24T22:00:00Z", now=now, max_age_hours=24
        )

    def test_stale_timestamp_fails(self):
        from datetime import datetime, timezone
        now = datetime(2026, 7, 25, 16, tzinfo=timezone.utc)
        assert not is_recent_headline(
            "2026-07-20T12:00:00Z", now=now, max_age_hours=24
        )

    def test_missing_or_future_timestamp_fails(self):
        from datetime import datetime, timezone
        now = datetime(2026, 7, 25, 16, tzinfo=timezone.utc)
        assert not is_recent_headline("", now=now)
        assert not is_recent_headline("2026-07-26T16:00:00Z", now=now)


class TestFetchTickerNews:
    def test_search_requires_explicit_related_ticker(self):
        mock_search = MagicMock()
        mock_search.news = [
            {
                "title": "AAPL headline",
                "providerPublishTime": 1767225600,
                "relatedTickers": ["AAPL", "MSFT"],
            },
            {
                "title": "Unrelated headline",
                "providerPublishTime": 1767225600,
                "relatedTickers": ["TSLA"],
            },
        ]
        with patch("yfinance.Search", return_value=mock_search):
            items = fetch_ticker_news("AAPL")
        assert len(items) == 1
        assert items[0]["headline"] == "AAPL headline"
        assert items[0]["related_symbols"] == ["AAPL", "MSFT"]
        assert items[0]["relevance_source"] == "yahoo_related_tickers"

    def test_normalizes_ticker_fallback_shape(self):
        mock_search = MagicMock()
        type(mock_search).news = property(lambda self: (_ for _ in ()).throw(Exception("unsupported")))
        mock_ticker = MagicMock()
        mock_ticker.news = [
            {"content": {"title": "Big headline", "summary": "Summary text", "pubDate": "2026-01-01"}},
        ]
        with patch("yfinance.Search", side_effect=Exception("unsupported")), \
             patch("yfinance.Ticker", return_value=mock_ticker):
            items = fetch_ticker_news("AAPL")
        assert len(items) == 1
        assert items[0] == {
            "symbol": "AAPL", "headline": "Big headline",
            "summary": "Summary text", "pub_date": "2026-01-01",
            "related_symbols": [],
            "relevance_source": "ticker_feed_fallback",
        }

    def test_returns_empty_list_on_failure(self):
        with patch("yfinance.Search", side_effect=Exception("boom")), \
             patch("yfinance.Ticker", side_effect=Exception("boom")):
            assert fetch_ticker_news("AAPL") == []

    def test_skips_items_without_title(self):
        mock_ticker = MagicMock()
        mock_ticker.news = [{"content": {"title": "", "summary": "x"}}]
        with patch("yfinance.Search", side_effect=Exception("unsupported")), \
             patch("yfinance.Ticker", return_value=mock_ticker):
            assert fetch_ticker_news("AAPL") == []


class TestFetchMacroFeeds:
    def test_one_feed_failing_does_not_stop_others(self):
        good_rss = (
            b'<?xml version="1.0"?><rss><channel><item>'
            b"<title>Test Headline</title><description>Test summary</description>"
            b"<pubDate>Mon, 01 Jan 2026 00:00:00 GMT</pubDate>"
            b"</item></channel></rss>"
        )

        def fake_get(url, **kwargs):
            if "federalreserve" in url:
                raise Exception("connection refused")
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            resp.content = good_rss
            return resp

        with patch("httpx.get", side_effect=fake_get):
            items, errors = fetch_macro_feeds()

        assert len(errors) == 1
        assert errors[0]["source"] == "Federal Reserve Press Releases"
        assert len(items) > 0
        assert items[0]["headline"] == "Test Headline"


def test_news_node_total_macro_outage_forces_high_risk(tmp_path):
    import json
    import nodes.news_node as nn

    errors = [
        {"source": f"feed-{i}", "error": "connection refused"}
        for i in range(MACRO_FEED_COUNT)
    ]
    report_dir = tmp_path / "market_intelligence"

    with patch.object(nn, "fetch_ticker_news", return_value=[]), \
         patch.object(nn, "fetch_macro_feeds", return_value=([], errors)), \
         patch.object(nn, "get_volatility_regime", return_value={
             "regime": "low", "vix": 12.0, "spy_atr_ratio": 1.0,
         }), \
         patch.object(nn, "data_dir", return_value=report_dir), \
         patch.object(nn, "status_file", return_value=report_dir / "run_status.json"), \
         patch.object(nn, "_write_log"):
        result = nn.run(tickers=["AAPL"])

    report_files = [p for p in report_dir.glob("*.json") if p.name != "run_status.json"]
    report = json.loads(report_files[0].read_text(encoding="utf-8"))
    assert result["overall_market_risk"] == "high"
    assert report["overall_market_risk"] == "high"
    assert "All configured macro news feeds failed" in report["macro_summary"]
    assert any(item["catalyst_type"] == "macro_data_outage" for item in report["items"])


def _mock_ticker_factory(vix_close, spy_rows=10):
    """
    yf.Ticker(...) stand-in whose .history() returns a small real DataFrame.
    SPY history is deliberately shorter than the 34-bar minimum
    get_volatility_regime() requires, so spy_atr_ratio stays None and only
    the VIX threshold logic is exercised.
    """
    def factory(symbol):
        m = MagicMock()
        if symbol == "^VIX":
            m.history.return_value = pd.DataFrame({"Close": [vix_close]})
        else:
            close = pd.Series([100.0] * spy_rows)
            m.history.return_value = pd.DataFrame({
                "High": close + 1, "Low": close - 1, "Close": close,
            })
        return m
    return factory


class TestGetVolatilityRegime:
    def test_high_vix_gives_high_regime(self):
        with patch("yfinance.Ticker", side_effect=_mock_ticker_factory(35.0)):
            regime = get_volatility_regime()
        assert regime["regime"] == "high"
        assert regime["vix"] == 35.0
        assert regime["spy_atr_ratio"] is None

    def test_elevated_vix_gives_elevated_regime(self):
        with patch("yfinance.Ticker", side_effect=_mock_ticker_factory(27.0)):
            regime = get_volatility_regime()
        assert regime["regime"] == "elevated"

    def test_low_vix_gives_low_regime(self):
        with patch("yfinance.Ticker", side_effect=_mock_ticker_factory(12.0)):
            regime = get_volatility_regime()
        assert regime["regime"] == "low"

    def test_yfinance_failure_defaults_to_low_with_none_values(self):
        with patch("yfinance.Ticker", side_effect=Exception("network error")):
            regime = get_volatility_regime()
        assert regime["regime"] == "low"
        assert regime["vix"] is None
        assert regime["spy_atr_ratio"] is None
