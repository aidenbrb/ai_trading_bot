"""Tests for nodes/intraday_preflight_node.py - morning setup for day mode."""
from datetime import date
from unittest.mock import patch

import nodes.intraday_preflight_node as preflight_node


def test_skips_on_non_trading_day():
    with patch.object(preflight_node, "is_trading_day", return_value=False):
        result = preflight_node.run(as_of=date(2025, 6, 7))
    assert result["status"] == "skipped"


def test_fails_loudly_when_news_node_errors():
    import nodes.news_node as news_mod

    with patch.object(preflight_node, "is_trading_day", return_value=True), \
         patch.object(news_mod, "run", return_value={"status": "error"}):
        result = preflight_node.run(as_of=date(2025, 6, 2))
    assert result["status"] == "error"
    assert result["reason"] == "news_node_failed"


def test_success_runs_news_then_reference():
    import nodes.news_node as news_mod
    import nodes.intraday_reference_node as ref_mod

    with patch.object(preflight_node, "is_trading_day", return_value=True), \
         patch.object(news_mod, "run", return_value={"status": "success", "overall_market_risk": "low"}), \
         patch.object(ref_mod, "run", return_value={"computed": ["AAPL"], "skipped": [], "failed": []}):
        result = preflight_node.run(as_of=date(2025, 6, 2))
    assert result["status"] == "success"
    assert result["reference_result"]["computed"] == ["AAPL"]
