from math import floor

from live_slc.risk import (
    AccountSnapshot,
    OrderInfo,
    PositionInfo,
    SLC_MAX_POSITIONS,
    SLC_MAX_NEW_TRADES_PER_DAY,
    account_wide_entries_today,
    available_buying_power,
    check_new_entry_capacity,
    classify_order,
    daily_loss_breached,
    open_position_count,
    remaining_daily_entry_slots,
    size_order,
)


def _snapshot(**overrides):
    defaults = dict(
        account_id="acct1", equity=100000.0, cash=100000.0, non_marginable_buying_power=100000.0,
        start_of_day_equity=100000.0, daily_realized_pnl=0.0, daily_unrealized_pnl=0.0,
        positions=[], today_orders=[],
    )
    defaults.update(overrides)
    return AccountSnapshot(**defaults)


def test_buying_power_floored_at_zero_when_other_strategies_over_committed():
    snap = _snapshot(equity=10000.0, cash=10000.0, non_marginable_buying_power=10000.0,
                      positions=[PositionInfo(symbol="TSLA", qty=200, market_value=50000.0)])
    assert available_buying_power(snap) == 0.0  # would be negative unfloored


def test_buying_power_accounts_for_positions_and_pending_orders_from_any_source():
    snap = _snapshot(
        positions=[PositionInfo(symbol="TSLA", qty=10, market_value=5000.0)],
        today_orders=[OrderInfo(symbol="NVDA", classification="entry", notional=2000.0)],
    )
    assert available_buying_power(snap) == 100000.0 - 5000.0 - 2000.0


def test_daily_loss_halt_is_account_wide_realized_plus_unrealized():
    snap = _snapshot(daily_realized_pnl=-500.0, daily_unrealized_pnl=-1600.0)  # -2.1%
    assert daily_loss_breached(snap)
    snap2 = _snapshot(daily_realized_pnl=-500.0, daily_unrealized_pnl=-1000.0)  # -1.5%
    assert not daily_loss_breached(snap2)


def test_daily_loss_fails_closed_on_zero_start_of_day_equity():
    snap = _snapshot(start_of_day_equity=0.0)
    assert daily_loss_breached(snap)


def test_size_order_respects_risk_concentration_and_cash_caps():
    snap = _snapshot(equity=100000.0)
    qty = size_order(snap, reference_price=100.0, stop_price=99.0)
    # by_risk = 100000*0.0025/1 = 250; by_concentration = 100000*0.20/100 = 200; by_cash = 100000/100=1000
    assert qty == 200


def test_size_order_zero_on_invalid_inputs():
    snap = _snapshot()
    assert size_order(snap, reference_price=100.0, stop_price=100.0) == 0  # zero risk-per-share
    assert size_order(snap, reference_price=0.0, stop_price=1.0) == 0


def test_size_order_accounts_for_hypothetical_additional_notional():
    snap = _snapshot(equity=100000.0, cash=1000.0, non_marginable_buying_power=1000.0)
    qty_alone = size_order(snap, reference_price=100.0, stop_price=99.0)
    qty_with_prior = size_order(snap, reference_price=100.0, stop_price=99.0, hypothetical_additional_notional=500.0)
    assert qty_with_prior < qty_alone


def test_classify_order_slc_own_entry_vs_protect():
    assert classify_order({"client_order_id": "slc-abc123-entry"}) == "entry"
    assert classify_order({"client_order_id": "slc-abc123-protect"}) == "exit_leg"


def test_classify_order_foreign_orders():
    assert classify_order({"parent_id": "xyz"}) == "exit_leg"
    assert classify_order({"position_intent": "buy_to_open"}) == "entry"
    assert classify_order({"position_intent": "sell_to_close"}) == "exit_leg"


def test_classify_order_unclassifiable_counts_conservatively_as_entry():
    assert classify_order({}) == "unclassifiable"
    snap = _snapshot(today_orders=[OrderInfo(symbol="XYZ", classification="unclassifiable", notional=100.0)])
    assert account_wide_entries_today(snap) == 1


def test_exit_legs_never_counted_toward_daily_entries():
    snap = _snapshot(today_orders=[
        OrderInfo(symbol="AAPL", classification="entry", notional=100.0),
        OrderInfo(symbol="AAPL", classification="exit_leg", notional=100.0),
    ])
    assert account_wide_entries_today(snap) == 1
    assert remaining_daily_entry_slots(snap) == SLC_MAX_NEW_TRADES_PER_DAY - 1


def test_account_wide_entries_count_toward_shared_daily_cap_even_if_not_slcs_own():
    """Rev. 6: account-wide opening fills count toward the 2-new-trades/day
    cap, not just SLC's own fills - a foreign entry still consumes budget."""
    snap = _snapshot(today_orders=[OrderInfo(symbol="MSFT", classification="entry", notional=100.0, is_slc=False)])
    assert account_wide_entries_today(snap) == 1


def test_capacity_check_blocks_at_max_positions():
    positions = [PositionInfo(symbol=f"S{i}", qty=1, market_value=100.0) for i in range(SLC_MAX_POSITIONS)]
    snap = _snapshot(positions=positions)
    ok, reason = check_new_entry_capacity(snap, "NEW", [])
    assert not ok and reason == "skipped_capacity_max_positions"


def test_capacity_check_blocks_symbol_already_has_position_or_order():
    snap = _snapshot(positions=[PositionInfo(symbol="AAPL", qty=1, market_value=100.0)])
    ok, reason = check_new_entry_capacity(snap, "AAPL", [])
    assert not ok and reason == "skipped_symbol_already_has_position"

    snap2 = _snapshot(today_orders=[OrderInfo(symbol="MSFT", classification="entry", notional=1.0)])
    ok2, reason2 = check_new_entry_capacity(snap2, "MSFT", [])
    assert not ok2 and reason2 == "skipped_symbol_already_has_order_today"


def test_capacity_check_accounts_for_admitted_this_cycle():
    snap = _snapshot(positions=[PositionInfo(symbol=f"S{i}", qty=1, market_value=100.0) for i in range(SLC_MAX_POSITIONS - 1)])
    ok, _ = check_new_entry_capacity(snap, "NEW1", [])
    assert ok
    ok2, reason2 = check_new_entry_capacity(snap, "NEW2", ["already-admitted-placeholder"])
    assert not ok2 and reason2 == "skipped_capacity_max_positions"


def test_capacity_check_blocks_on_daily_loss_halt():
    snap = _snapshot(daily_realized_pnl=-3000.0)
    ok, reason = check_new_entry_capacity(snap, "AAPL", [])
    assert not ok and reason == "skipped_daily_loss_halt"


def test_open_position_count_ignores_zero_qty():
    snap = _snapshot(positions=[PositionInfo(symbol="X", qty=0, market_value=0.0)])
    assert open_position_count(snap) == 0
