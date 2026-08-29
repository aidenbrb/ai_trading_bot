"""Tests for utils/market_intelligence.py"""
import pytest
import json
import sys
import os
from datetime import timedelta
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.market_intelligence import (
    check_ticker, format_for_llm, get_warnings, is_stale, news_gate_status, overall_risk,
)
from utils.timeutil import utcnow


def _data(items=None, risk="low"):
    return {
        "report_date": "2026-06-01",
        "overall_market_risk": risk,
        "stock_market_summary": "SPY +0.5%",
        "crypto_market_summary": "BTC $60k",
        "macro_summary": "Fed 4%",
        "commodity_summary": "Gold $3000",
        "items": items or [],
    }


def _item(symbol, label, sentiment="unknown", score=30, related=None):
    return {
        "symbol": symbol,
        "bot_context_label": label,
        "sentiment": sentiment,
        "importance_score": score,
        "headline": f"Test headline for {symbol}",
        "related_symbols": related or [],
        "catalyst_type": "other",
    }


class TestCheckTicker:
    def test_no_items_passes(self):
        ok, reason = check_ticker("AAPL", _data())
        assert ok
        assert reason == ""

    def test_avoid_new_trade_blocks(self):
        data = _data([_item("AAPL", "avoid_new_trade")])
        ok, reason = check_ticker("AAPL", data)
        assert not ok
        assert "avoid_new_trade" in reason

    def test_earnings_risk_warns_but_calendar_gate_is_authoritative(self):
        data = _data([_item("TSLA", "earnings_risk")])
        ok, reason = check_ticker("TSLA", data)
        assert ok
        assert reason == ""
        assert "earnings_risk" in get_warnings("TSLA", data)

    def test_high_volatility_blocks(self):
        data = _data([_item("NVDA", "high_volatility_warning")])
        ok, reason = check_ticker("NVDA", data)
        assert not ok

    def test_monitor_label_passes(self):
        data = _data([_item("AAPL", "monitor")])
        ok, reason = check_ticker("AAPL", data)
        assert ok

    def test_macro_risk_passes_for_stocks(self):
        data = _data([_item("AAPL", "macro_risk")])
        ok, reason = check_ticker("AAPL", data)
        assert ok

    def test_crypto_risk_blocks_crypto(self):
        data = _data([_item("BTC", "crypto_risk")])
        ok, reason = check_ticker("BTC-USD", data, crypto=True)
        assert not ok

    def test_crypto_risk_passes_stocks(self):
        data = _data([_item("BTC", "crypto_risk")])
        ok, reason = check_ticker("BTC", data, crypto=False)
        assert ok

    def test_related_symbols_block(self):
        item = _item("SPY", "avoid_new_trade", related=["AAPL"])
        data = _data([item])
        ok, reason = check_ticker("AAPL", data)
        assert not ok

    def test_case_insensitive(self):
        data = _data([_item("aapl", "avoid_new_trade")])
        ok, _ = check_ticker("AAPL", data)
        assert not ok

    def test_usd_suffix_stripped(self):
        data = _data([_item("BTC", "avoid_new_trade")])
        ok, _ = check_ticker("BTC-USD", data)
        assert not ok

    def test_high_overall_risk_blocks_even_with_no_items(self):
        ok, reason = check_ticker("AAPL", _data(risk="high"))
        assert not ok
        assert "overall_market_risk" in reason.lower()

    def test_medium_overall_risk_does_not_block(self):
        ok, reason = check_ticker("AAPL", _data(risk="medium"))
        assert ok


class TestGetWarnings:
    def test_returns_warning_labels(self):
        data = _data([_item("AAPL", "macro_risk")])
        warnings = get_warnings("AAPL", data)
        assert "macro_risk" in warnings

    def test_blocking_labels_not_in_warnings(self):
        data = _data([_item("AAPL", "avoid_new_trade")])
        warnings = get_warnings("AAPL", data)
        assert "avoid_new_trade" not in warnings

    def test_empty_when_no_items(self):
        warnings = get_warnings("AAPL", _data())
        assert warnings == []


class TestOverallRisk:
    def test_returns_value(self):
        assert overall_risk(_data(risk="high")) == "high"

    def test_unknown_default(self):
        assert overall_risk({}) == "unknown"


class TestFormatForLLM:
    def test_returns_string(self):
        result = format_for_llm(_data())
        assert isinstance(result, str)
        assert "MARKET INTELLIGENCE" in result

    def test_includes_ticker_items_first(self):
        items = [
            _item("AAPL", "monitor", score=40),
            _item("MSFT", "monitor", score=30),
        ]
        data = _data(items)
        result = format_for_llm(data, ticker="AAPL", max_items=5)
        aapl_pos = result.find("AAPL")
        msft_pos = result.find("MSFT")
        assert aapl_pos < msft_pos

    def test_no_crash_on_empty_data(self):
        result = format_for_llm(_data())
        assert "END MARKET INTELLIGENCE" in result


class TestIsStale:
    def test_missing_timestamp_is_stale(self):
        assert is_stale(_data())

    def test_unparseable_timestamp_is_stale(self):
        data = _data()
        data["generated_at"] = "not-a-timestamp"
        assert is_stale(data)

    def test_fresh_timestamp_not_stale(self):
        data = _data()
        data["generated_at"] = utcnow().isoformat()
        assert not is_stale(data, max_age_hours=18)

    def test_old_timestamp_is_stale(self):
        data = _data()
        data["generated_at"] = (utcnow() - timedelta(hours=5)).isoformat()
        assert is_stale(data, max_age_hours=1)


class TestNewsGateStatus:
    def _write_report(self, tmp_path, status="success", generated_at=None, date_str=None):
        from datetime import date as _date
        date_str = date_str or _date.today().isoformat()
        mi_dir = tmp_path / "market_intelligence"
        mi_dir.mkdir(parents=True, exist_ok=True)
        report = _data(risk="low")
        report["report_date"] = date_str
        report["generated_at"] = generated_at or utcnow().isoformat()
        (mi_dir / f"{date_str}.json").write_text(json.dumps(report), encoding="utf-8")
        (tmp_path / "run_status.json").write_text(
            json.dumps({"status": status, "date": date_str}), encoding="utf-8"
        )

    def _patch_data_dir(self, monkeypatch, tmp_path):
        from config.settings import settings
        monkeypatch.setattr(settings, "NEWS_DATA_DIR", str(tmp_path / "market_intelligence"))
        monkeypatch.setattr(settings, "NEWS_GATE_ENABLED", True)
        monkeypatch.setattr(settings, "NEWS_MAX_AGE_HOURS", 18)

    def test_missing_report_blocks(self, tmp_path, monkeypatch):
        self._patch_data_dir(monkeypatch, tmp_path)
        data, blocked, reason = news_gate_status()
        assert blocked
        assert "missing" in reason
        assert data is None

    def test_fresh_report_does_not_block(self, tmp_path, monkeypatch):
        self._patch_data_dir(monkeypatch, tmp_path)
        self._write_report(tmp_path, generated_at=utcnow().isoformat())
        data, blocked, reason = news_gate_status()
        assert not blocked
        assert reason == ""
        assert data is not None

    def test_stale_report_blocks(self, tmp_path, monkeypatch):
        self._patch_data_dir(monkeypatch, tmp_path)
        from config.settings import settings
        monkeypatch.setattr(settings, "NEWS_MAX_AGE_HOURS", 1)
        self._write_report(tmp_path, generated_at=(utcnow() - timedelta(hours=5)).isoformat())
        data, blocked, reason = news_gate_status()
        assert blocked
        assert "stale" in reason

    def test_error_status_blocks_like_missing(self, tmp_path, monkeypatch):
        self._patch_data_dir(monkeypatch, tmp_path)
        self._write_report(tmp_path, status="error")
        data, blocked, reason = news_gate_status()
        assert blocked

    def test_gate_disabled_never_blocks(self, tmp_path, monkeypatch):
        from config.settings import settings
        monkeypatch.setattr(settings, "NEWS_GATE_ENABLED", False)
        data, blocked, reason = news_gate_status()
        assert not blocked
        assert reason == ""
