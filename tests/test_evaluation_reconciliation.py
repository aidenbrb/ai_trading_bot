from datetime import date
from unittest.mock import MagicMock

from sqlmodel import select

import db.connection as connection
from db.models import EvaluationLedger, Order, Position, RiskApproval, Strategy, Ticker
from nodes import evaluation_reconciliation_node as reconciliation


def _seed(tmp_path, monkeypatch):
    monkeypatch.setattr(connection.settings, "DATABASE_URL", f"sqlite:///{tmp_path / 'ledger.db'}")
    connection._engine = None
    connection.init_db()
    with connection.get_session() as session:
        ticker = Ticker(symbol="AAPL", sector="Technology")
        session.add(ticker); session.flush()
        strategy = Strategy(
            run_id="srun", ticker_id=ticker.id, bar_date=date(2026, 6, 1),
            signal="BUY", entry=100, stop=95, target=110, rr=2,
            conviction_score=80, model_used="stock_trend_momentum_v1",
        )
        session.add(strategy); session.flush()
        approval = RiskApproval(
            run_id="rrun", strategy_id=strategy.id, approved=True,
            shares=10, risk_amount=50, position_value=1000,
        )
        session.add(approval); session.flush()
        order = Order(
            run_id="orun", risk_approval_id=approval.id, alpaca_order_id="broker-1",
            symbol="AAPL", side="buy", qty=10, order_type="limit",
            status="accepted", dry_run=False,
        )
        session.add(order); session.flush()
        session.add(Position(
            ticker_id=ticker.id, order_id=order.id, entry_date=date(2026, 6, 1),
            entry_price=100, shares=10, stop_price=95, target_price=110,
            status="open",
        ))
    return order.id


def test_reconciliation_is_read_only_at_broker_and_detects_protection(tmp_path, monkeypatch):
    order_id = _seed(tmp_path, monkeypatch)
    broker = MagicMock()
    broker.get_all_positions.return_value = [MagicMock(symbol="AAPL")]
    protective = MagicMock(symbol="AAPL", side="sell", type="stop", legs=[])
    broker.get_orders.return_value = [protective]
    broker.get_order_by_id.return_value = MagicMock(
        status="filled", filled_avg_price="100.10", filled_at=None,
    )

    result = reconciliation.run(client=broker)
    assert result["broker_read_only"] is True
    assert result["safe_for_scheduler_change"] is True
    assert result["forward_paper_gate"]["passed"] is False
    assert "at_least_90_calendar_days" in result["forward_paper_gate"]["failed_checks"]
    broker.submit_order.assert_not_called()
    broker.cancel_order_by_id.assert_not_called()
    broker.close_position.assert_not_called()
    with connection.get_session() as session:
        ledger = session.exec(
            select(EvaluationLedger).where(EvaluationLedger.order_id == order_id)
        ).one()
        assert ledger.reconciliation_status == "consistent_open"
        assert ledger.entry_slippage_bps > 0


def test_reconciliation_fails_closed_when_open_position_has_no_stop(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    broker = MagicMock()
    broker.get_all_positions.return_value = [MagicMock(symbol="AAPL")]
    broker.get_orders.return_value = []
    broker.get_order_by_id.return_value = MagicMock(
        status="filled", filled_avg_price="100", filled_at=None,
    )
    result = reconciliation.run(client=broker)
    assert result["safe_for_scheduler_change"] is False
    assert "protective stop" in result["discrepancies"][0]["reason"]


def test_nested_held_bracket_stop_counts_as_protection(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    broker = MagicMock()
    broker.get_all_positions.return_value = [MagicMock(symbol="AAPL")]
    broker.get_orders.return_value = []  # Alpaca may omit the HELD sibling here.
    held_stop = MagicMock(symbol="AAPL", side="sell", type="stop", legs=[])
    broker.get_order_by_id.return_value = MagicMock(
        status="filled", filled_avg_price="100", filled_at=None,
        symbol="AAPL", side="buy", type="limit", legs=[held_stop],
    )
    result = reconciliation.run(client=broker)
    assert result["safe_for_scheduler_change"] is True
    assert result["discrepancies"] == []
