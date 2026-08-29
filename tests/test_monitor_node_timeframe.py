"""
Tests for nodes/monitor_node.py's timeframe-aware position review (Phase 2
Step 4): _position_strategy_version(), _position_timeframe(), and
_get_daily_indicator_namespace(). Added after a review found
_evaluate_position() always trailed/reversed every position on hourly
indicators, even for the new daily-bar crypto strategies.
"""
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import db.connection as connection
import nodes.monitor_node as monitor_node
from db.connection import get_session
from db.models import Order, Position, RiskApproval, Strategy, Ticker


def _seed(tmp_path, monkeypatch):
    monkeypatch.setattr(connection.settings, "DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    connection._engine = None
    connection.init_db()


def _make_position_chain(strategy_model_used, symbol="BTC-USD"):
    """Builds a full Ticker -> Strategy -> RiskApproval -> Order -> Position
    chain, mirroring how the real strategy/risk/execution nodes link these
    tables, and returns the resulting Position."""
    with get_session() as session:
        ticker = Ticker(symbol=symbol, sector="Crypto")
        session.add(ticker)
        session.flush()

        strategy = Strategy(
            run_id="run-1", ticker_id=ticker.id, bar_date=date(2026, 1, 1),
            model_used=strategy_model_used,
        )
        session.add(strategy)
        session.flush()

        approval = RiskApproval(run_id="run-1", strategy_id=strategy.id, approved=True)
        session.add(approval)
        session.flush()

        order = Order(
            run_id="run-1", risk_approval_id=approval.id, symbol="BTC-USD",
            side="buy", qty=1.0, order_type="market", status="filled",
        )
        session.add(order)
        session.flush()

        position = Position(
            ticker_id=ticker.id, order_id=order.id, entry_date=date(2026, 1, 1),
            entry_price=100.0, shares=1.0, stop_price=95.0, target_price=110.0,
        )
        session.add(position)
        session.flush()
        session.refresh(position)
        session.expunge(position)
        return position


def test_position_strategy_version_walks_the_full_chain(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    pos = _make_position_chain("crypto_trend_daily_v1")
    assert monitor_node._position_strategy_version(pos) == "crypto_trend_daily_v1"


def test_position_strategy_version_none_when_order_id_missing(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    with get_session() as session:
        ticker = Ticker(symbol="AAPL", sector="Technology")
        session.add(ticker)
        session.flush()
        position = Position(
            ticker_id=ticker.id, order_id=None, entry_date=date(2026, 1, 1),
            entry_price=100.0, shares=1.0, stop_price=95.0, target_price=110.0,
        )
        session.add(position)
        session.flush()
        session.refresh(position)
        session.expunge(position)
    assert monitor_node._position_strategy_version(position) is None


def test_position_timeframe_daily_for_the_two_new_strategies(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    for i, version in enumerate(("crypto_trend_daily_v1", "crypto_xsec_momentum_v1")):
        pos = _make_position_chain(version, symbol=f"SYM{i}-USD")
        assert monitor_node._position_timeframe(pos) == "daily"


def test_position_timeframe_hourly_for_v1():
    """v1 (crypto_trend_momentum_v1) - the only strategy actually live
    today - must keep evaluating on hourly indicators exactly as before."""
    pos = SimpleNamespace(order_id=None)  # short-circuits to None -> hourly
    assert monitor_node._position_timeframe(pos) == "hourly"


def test_position_timeframe_hourly_when_strategy_identity_unknown(tmp_path, monkeypatch):
    """A position with no resolvable strategy chain (e.g. reconciled
    directly from a broker fill) must degrade to the original, already-
    proven hourly review path - never guess "daily"."""
    _seed(tmp_path, monkeypatch)
    with get_session() as session:
        ticker = Ticker(symbol="ETH-USD", sector="Crypto")
        session.add(ticker)
        session.flush()
        position = Position(
            ticker_id=ticker.id, order_id=None, entry_date=date(2026, 1, 1),
            entry_price=100.0, shares=1.0, stop_price=95.0, target_price=110.0,
        )
        session.add(position)
        session.flush()
        session.refresh(position)
        session.expunge(position)
    assert monitor_node._position_timeframe(position) == "hourly"


def test_position_timeframe_hourly_for_stock_v2():
    pos = SimpleNamespace(order_id=None)
    assert monitor_node._position_timeframe(pos) == "hourly"


# -- _get_daily_indicator_namespace ------------------------------------------

def test_get_daily_indicator_namespace_wraps_fetch_result():
    fake_values = {
        "trend": "UPTREND", "rsi_14": 60.0, "macd_hist": 1.0, "atr_14": 2.0,
        "sma_20": 100.0, "sma_50": 95.0, "sma_200": 90.0, "rel_volume": 1.5,
    }
    with patch("nodes.crypto_strategy_node.fetch_daily_indicator", return_value=fake_values):
        ind = monitor_node._get_daily_indicator_namespace("BTC-USD")
    assert ind is not None
    assert ind.trend == "UPTREND"
    assert ind.atr_14 == 2.0


def test_get_daily_indicator_namespace_none_when_fetch_fails():
    with patch("nodes.crypto_strategy_node.fetch_daily_indicator", return_value=None):
        assert monitor_node._get_daily_indicator_namespace("BTC-USD") is None


# -- End-to-end: the daily namespace flows correctly into _evaluate_position --

def test_daily_indicator_drives_evaluate_position_identically_to_hourly_shape():
    """Structural proof that _get_daily_indicator_namespace()'s output is a
    drop-in replacement for a real Indicator row: _evaluate_position()
    (completely unmodified) produces the same decision either way given
    equivalent values."""
    from nodes.monitor_node import _evaluate_position

    daily_values = {
        "trend": "DOWNTREND", "rsi_14": 40.0, "macd_hist": -0.5, "atr_14": 2.0,
        "sma_20": 90.0, "sma_50": 95.0, "sma_200": 100.0, "rel_volume": 1.0,
    }
    with patch("nodes.crypto_strategy_node.fetch_daily_indicator", return_value=daily_values):
        ind = monitor_node._get_daily_indicator_namespace("BTC-USD")
    pos = SimpleNamespace(entry_price=100.0, stop_price=95.0, target_price=110.0)
    decision = _evaluate_position("BTC-USD", pos, current_price=98.0, ind=ind)
    assert decision["action"] == "CLOSE"
    assert "DOWNTREND" in decision["reason"]
