from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from backtest.v2_readonly_adapters import CacheCoverageError
from backtest.whole_bot_engine import (
    ActiveOrder,
    BASELINE_COST,
    Candidate,
    ResearchPortfolio,
    _indicator_object,
    _snapshot,
    _transaction_cost,
    completed_bar_cutoff,
    decision_time_utc,
    indicator_frame,
    simulate_order_outcome,
    simulate_portfolio,
)
import backtest.whole_bot_engine as engine
from utils.market_calendar import next_trading_day, session_for, trading_days_between


def _bars(rows):
    return pd.DataFrame(
        rows,
        columns=["bar_time", "open", "high", "low", "close", "volume"],
    ).set_index("bar_time")


def _candidate(market="stock", day=date(2026, 6, 1), symbol="AAPL"):
    return Candidate(
        symbol=symbol,
        market=market,
        strategy_version=f"{market}_v1",
        decision_time=decision_time_utc(day),
        signal_bar_end=decision_time_utc(day) - timedelta(minutes=16),
        entry=100.0,
        stop=95.0,
        target=110.0,
        conviction=80,
        atr=2.0,
    )


def test_decision_cutoff_handles_dst_and_fifteen_minute_sip_delay():
    assert decision_time_utc(date(2026, 6, 1)) == datetime(2026, 6, 1, 15, 16)
    assert completed_bar_cutoff(date(2026, 6, 1), "stock") == datetime(2026, 6, 1, 14, 1)
    assert decision_time_utc(date(2026, 1, 5)) == datetime(2026, 1, 5, 16, 16)
    assert completed_bar_cutoff(date(2026, 1, 5), "stock") == datetime(2026, 1, 5, 15, 1)


def test_snapshot_excludes_partial_or_not_yet_available_bar():
    idx = pd.to_datetime(["2026-06-01 14:00", "2026-06-01 15:00"])
    frame = pd.DataFrame({
        "close": [100.0, 999.0], "sma_20": [90.0, 90.0],
        "sma_50": [80.0, 80.0], "sma_200": [70.0, 70.0],
        "rsi_14": [60.0, 60.0], "macd_hist": [1.0, 1.0],
        "atr_14": [2.0, 2.0], "rel_volume": [1.5, 1.5],
    }, index=idx)
    row = _snapshot(frame, date(2026, 6, 1), "stock")
    assert row["close"] == 100.0


def test_stock_same_minute_stop_and_target_is_adverse_and_ambiguous():
    candidate = _candidate()
    minute = _bars([
        (datetime(2026, 6, 1, 15, 17), 100.0, 111.0, 94.0, 105.0, 1000),
    ])
    fetch = lambda symbols, start, end, amount, unit: {"AAPL": minute}
    outcome = simulate_order_outcome(
        candidate, date(2026, 6, 1), {"AAPL": pd.DataFrame()}, fetch_stock=fetch
    )
    assert outcome["exit_reason"] == "stop"
    assert outcome["exit_price"] == 95.0
    assert outcome["ambiguous"] is True


def test_stock_unfilled_gtc_prefill_scan_uses_one_calendar_lookup_per_day(monkeypatch):
    candidate = _candidate(day=date(2026, 6, 1))
    index = pd.date_range("2026-06-01 13:30", periods=390, freq="min")
    minute = pd.DataFrame({
        "open": 101.0, "high": 102.0, "low": 100.5,
        "close": 101.0, "volume": 1000.0,
    }, index=index)
    fetch = lambda symbols, start, end, amount, unit: {"AAPL": minute}
    original = engine.session_for
    calls = []

    def counted(day):
        calls.append(day)
        return original(day)

    monkeypatch.setattr(engine, "session_for", counted)
    outcome = simulate_order_outcome(
        candidate, date(2026, 6, 1), {"AAPL": pd.DataFrame()}, fetch_stock=fetch
    )

    assert outcome["status"] == "unfilled_end"
    assert calls == [date(2026, 6, 1)]


def test_crypto_stop_limit_waits_for_price_to_recover_to_limit():
    candidate = _candidate("crypto", symbol="BTC-USD")
    minute = _bars([
        (datetime(2026, 6, 1, 15, 17), 100.0, 92.0, 90.0, 91.0, 1000),
        (datetime(2026, 6, 1, 15, 18), 92.0, 94.0, 91.0, 93.5, 1000),
    ])
    fetch = lambda symbols, start, end, amount, unit: {"BTC-USD": minute}
    outcome = simulate_order_outcome(
        candidate, date(2026, 6, 1), {"BTC-USD": pd.DataFrame()}, fetch_crypto=fetch
    )
    assert outcome["exit_reason"] == "stop_limit"
    assert outcome["exit_time"] == datetime(2026, 6, 1, 15, 18)
    assert outcome["exit_price"] == pytest.approx(93.1)


def test_crypto_target_is_checked_only_on_thirty_minute_monitor_tick():
    candidate = _candidate("crypto", symbol="BTC-USD")
    minute = _bars([
        (datetime(2026, 6, 1, 15, 17), 100.0, 101.0, 99.0, 100.0, 1000),
        (datetime(2026, 6, 1, 15, 20), 105.0, 111.0, 104.0, 111.0, 1000),
        (datetime(2026, 6, 1, 15, 30), 110.0, 111.0, 109.0, 110.5, 1000),
    ])
    fetch = lambda symbols, start, end, amount, unit: {"BTC-USD": minute}
    outcome = simulate_order_outcome(
        candidate, date(2026, 6, 1), {"BTC-USD": pd.DataFrame()}, fetch_crypto=fetch
    )
    assert outcome["exit_reason"] == "monitor_target"
    assert outcome["exit_time"] == datetime(2026, 6, 1, 15, 30)


def test_crypto_baseline_cost_is_35_bps_on_each_leg():
    candidate = _candidate("crypto", symbol="BTC-USD")
    active = ActiveOrder(
        candidate=candidate,
        quantity=10.0,
        reserved_notional=1000.0,
        risk_amount=50.0,
        outcome={"status": "closed", "fill_price": 100.0, "exit_price": 110.0},
    )
    assert _transaction_cost(active, BASELINE_COST) == pytest.approx(7.35)


def test_data_source_failure_is_missing_outcome_not_no_fill():
    def failed(*args, **kwargs):
        raise RuntimeError("data API unavailable")

    outcome = simulate_order_outcome(
        _candidate(), date(2026, 6, 1), {"AAPL": pd.DataFrame()}, fetch_stock=failed
    )
    assert outcome["status"] == "outcome_data_missing"
    assert outcome["outcome_data_missing"] is True


def _hourly_frame(n=45, seed=3):
    idx = pd.date_range("2026-01-05 10:00", periods=n, freq="h")
    close = pd.Series(100 + np.cumsum(np.random.default_rng(seed).standard_normal(n) * 0.3 + 0.05), index=idx)
    return pd.DataFrame({
        "open": close, "high": close + 0.3, "low": close - 0.3,
        "close": close, "volume": 1_000_000.0,
    }, index=idx)


def test_indicator_frame_include_adx_is_opt_in():
    df = _hourly_frame()
    default = indicator_frame(df)
    assert "adx_14" not in default.columns
    with_adx = indicator_frame(df, include_adx=True)
    assert "adx_14" in with_adx.columns


def test_indicator_object_include_adx_is_opt_in():
    row = pd.Series({
        "trend": "UPTREND", "rsi_14": 60.0, "macd_hist": 0.1, "rel_volume": 1.5,
        "atr_14": 2.0, "sma_20": 105.0, "sma_50": 100.0, "adx_14": 30.0,
    })
    default = _indicator_object(row)
    assert not hasattr(default, "adx_14")
    assert not hasattr(default, "bars_observed")
    with_adx = _indicator_object(row, include_adx=True, bars_observed=45)
    assert with_adx.adx_14 == 30.0
    assert with_adx.bars_observed == 45


def test_fills_on_session_two_and_targets_after_session_five():
    """The exact scenario an unconditional expires_at filter (rev. 2's
    bug) would get wrong: a fill well before expiration must not have its
    post-fill exit search truncated at the expiration boundary."""
    day1 = date(2026, 6, 1)
    candidate = _candidate(day=day1)
    trading_days = trading_days_between(day1, day1 + timedelta(days=30))
    session2, session7 = trading_days[1], trading_days[6]
    expires_at = session_for(trading_days[4])["close"]
    open2 = session_for(session2)["open"]
    open7 = session_for(session7)["open"]

    def fetch(symbols, start, end, amount, unit):
        rows = [
            (open2, 100.5, 100.5, 99.9, 100.0, 1000),   # fills, no stop/target hit
            (open7, 105.0, 111.0, 104.0, 110.5, 1000),  # target hit, well after expires_at
        ]
        return {"AAPL": _bars(rows)}

    outcome = simulate_order_outcome(
        candidate, day1 + timedelta(days=60), {"AAPL": pd.DataFrame()},
        fetch_stock=fetch, expiration_sessions=5,
    )
    assert outcome["status"] == "closed"
    assert outcome["exit_reason"] == "target"
    assert outcome["filled_at"] == open2
    assert outcome["exit_time"] == open7
    assert outcome["exit_time"] > expires_at


def test_no_fill_before_expiration_resolves_expired_unfilled():
    """A price touch that only ever occurs after expires_at must not
    register as a fill - the vectorized scan is bounded to the
    pre-expiration window while searching for the entry."""
    day1 = date(2026, 6, 1)
    candidate = _candidate(day=day1)
    trading_days = trading_days_between(day1, day1 + timedelta(days=30))
    session7 = trading_days[6]
    expires_at = session_for(trading_days[4])["close"]
    open7 = session_for(session7)["open"]

    def fetch(symbols, start, end, amount, unit):
        return {"AAPL": _bars([(open7, 100.5, 100.5, 99.0, 100.0, 1000)])}

    outcome = simulate_order_outcome(
        candidate, day1 + timedelta(days=60), {"AAPL": pd.DataFrame()},
        fetch_stock=fetch, expiration_sessions=5,
    )
    assert outcome["status"] == "expired_unfilled"
    assert outcome["filled_at"] is None
    assert outcome["expires_at"] == expires_at


def test_expired_unfilled_does_not_request_a_further_chunk():
    day1 = date(2026, 6, 1)
    candidate = _candidate(day=day1)
    calls = []

    def fetch(symbols, start, end, amount, unit):
        calls.append((start, end))
        return {"AAPL": pd.DataFrame(columns=["open", "high", "low", "close", "volume"])}

    outcome = simulate_order_outcome(
        candidate, day1 + timedelta(days=400), {"AAPL": pd.DataFrame()},
        fetch_stock=fetch, expiration_sessions=5,
    )
    assert outcome["status"] == "expired_unfilled"
    assert len(calls) == 1


def test_data_fetch_failure_with_expiration_set_is_still_missing_not_expired():
    def failed(*args, **kwargs):
        raise RuntimeError("data API unavailable")

    outcome = simulate_order_outcome(
        _candidate(), date(2026, 6, 1), {"AAPL": pd.DataFrame()},
        fetch_stock=failed, expiration_sessions=5,
    )
    assert outcome["status"] == "outcome_data_missing"
    assert outcome["outcome_data_missing"] is True


def test_cache_coverage_error_is_missing_not_expired_unfilled():
    """A read-only cache-coverage gap (unknown) must resolve the same way
    as any other data-fetch failure - never expired_unfilled (a known,
    definitive non-event) - even once expires_at has already passed."""
    def raises_coverage_error(*args, **kwargs):
        raise CacheCoverageError("AAPL research-stock-sip-1Minute not fully cached")

    outcome = simulate_order_outcome(
        _candidate(), date(2026, 6, 1), {"AAPL": pd.DataFrame()},
        fetch_stock=raises_coverage_error, expiration_sessions=5,
    )
    assert outcome["status"] == "outcome_data_missing"
    assert outcome["outcome_data_missing"] is True


def test_crypto_ignores_expiration_sessions_entirely():
    candidate = _candidate("crypto", symbol="BTC-USD")
    minute = _bars([
        (datetime(2026, 6, 1, 15, 17), 100.0, 101.0, 99.0, 100.0, 1000),
    ])
    fetch = lambda symbols, start, end, amount, unit: {"BTC-USD": minute}
    without_expiration = simulate_order_outcome(
        candidate, date(2026, 6, 1), {"BTC-USD": pd.DataFrame()}, fetch_crypto=fetch,
    )
    with_expiration = simulate_order_outcome(
        candidate, date(2026, 6, 1), {"BTC-USD": pd.DataFrame()}, fetch_crypto=fetch,
        expiration_sessions=5,
    )
    assert without_expiration == with_expiration


def test_portfolio_enforces_two_new_trades_per_day():
    day = date(2026, 6, 1)
    candidates = [_candidate(symbol=s) for s in ("AAPL", "MSFT", "SPY")]
    calendar = {day: {"stock": candidates, "crypto": []}}

    def closed(candidate, end_date, frames, expiration_sessions=None):
        return {
            "status": "closed", "outcome_data_missing": False,
            "filled_at": candidate.decision_time + timedelta(minutes=1),
            "fill_price": candidate.entry,
            "exit_time": candidate.decision_time + timedelta(hours=1),
            "exit_price": candidate.target, "exit_reason": "target",
            "ambiguous": False,
        }

    result = simulate_portfolio(
        calendar,
        {"stock": {}, "crypto": {}},
        day,
        day,
        ResearchPortfolio("test", max_new_trades_per_day=2),
        BASELINE_COST,
        "stock_only",
        outcome_simulator=closed,
    )
    assert len(result["trades"]) == 2
    assert any(r["reason"] == "max_new_trades_per_day" for r in result["rejected"])


def _unfilled(candidate, end_date, frames, expiration_sessions=None):
    return {"status": "unfilled_end", "outcome_data_missing": False,
            "filled_at": None, "fill_price": None}


def test_same_symbol_blocked_while_first_order_is_still_unfilled():
    """Same-symbol duplicate guard is unaffected by the capacity fix."""
    day1 = date(2026, 6, 1)
    day2 = next_trading_day(day1)
    calendar = {
        day1: {"stock": [_candidate(day=day1, symbol="AAPL")], "crypto": []},
        day2: {"stock": [_candidate(day=day2, symbol="AAPL")], "crypto": []},
    }

    result = simulate_portfolio(
        calendar, {"stock": {}, "crypto": {}}, day1, day2,
        ResearchPortfolio("test"), BASELINE_COST, "stock_only",
        outcome_simulator=_unfilled,
    )
    assert len(result["trades"]) == 1
    assert any(r["symbol"] == "AAPL" and r["reason"] == "already_held_or_pending"
               for r in result["rejected"])


def test_unfilled_order_does_not_occupy_capacity_slot():
    """Core bug reproduction: a never-filled resting order must not
    permanently block a later, different-symbol candidate."""
    day1 = date(2026, 6, 1)
    day2 = next_trading_day(day1)
    calendar = {
        day1: {"stock": [_candidate(day=day1, symbol="AAPL")], "crypto": []},
        day2: {"stock": [_candidate(day=day2, symbol="MSFT")], "crypto": []},
    }

    result = simulate_portfolio(
        calendar, {"stock": {}, "crypto": {}}, day1, day2,
        ResearchPortfolio("test", max_positions=1), BASELINE_COST, "stock_only",
        outcome_simulator=_unfilled,
    )
    assert {t["symbol"] for t in result["trades"]} == {"AAPL", "MSFT"}
    assert not any(r["reason"] == "max_positions" for r in result["rejected"])


def test_same_day_approvals_consume_capacity_immediately():
    """Two same-day approvals must consume capacity in order, even though
    neither has technically 'filled' relative to that decision yet -
    matches risk_node.py incrementing current_positions right after
    approval, before any fill."""
    day1 = date(2026, 6, 1)
    calendar = {
        day1: {"stock": [
            _candidate(day=day1, symbol="AAPL"),
            _candidate(day=day1, symbol="MSFT"),
        ], "crypto": []},
    }

    result = simulate_portfolio(
        calendar, {"stock": {}, "crypto": {}}, day1, day1,
        ResearchPortfolio("test", max_positions=1), BASELINE_COST, "stock_only",
        outcome_simulator=_unfilled,
    )
    assert {t["symbol"] for t in result["trades"]} == {"AAPL"}
    assert any(r["symbol"] == "MSFT" and r["reason"] == "max_positions"
               for r in result["rejected"])


def test_pending_order_starts_occupying_slot_once_historically_filled():
    """A resting order that hasn't filled yet must not occupy a slot, but
    once its precomputed fill time arrives it must occupy one, same as a
    real open position."""
    day1 = date(2026, 6, 1)
    day2 = next_trading_day(day1)
    day3 = next_trading_day(day2)
    day4 = next_trading_day(day3)
    fill_time = decision_time_utc(day3)
    calendar = {
        day1: {"stock": [_candidate(day=day1, symbol="AAPL")], "crypto": []},
        day2: {"stock": [_candidate(day=day2, symbol="MSFT")], "crypto": []},
        day3: {"stock": [], "crypto": []},
        day4: {"stock": [_candidate(day=day4, symbol="SPY")], "crypto": []},
    }

    def outcome_for(candidate, end_date, frames, expiration_sessions=None):
        if candidate.symbol == "AAPL":
            return {"status": "open", "outcome_data_missing": False,
                    "filled_at": fill_time, "fill_price": 100.0, "exit_time": None}
        return _unfilled(candidate, end_date, frames)

    result = simulate_portfolio(
        calendar, {"stock": {}, "crypto": {}}, day1, day4,
        ResearchPortfolio("test", max_positions=1), BASELINE_COST, "stock_only",
        outcome_simulator=outcome_for,
    )
    assert {t["symbol"] for t in result["trades"]} == {"AAPL", "MSFT"}
    assert any(r["symbol"] == "SPY" and r["reason"] == "max_positions"
               for r in result["rejected"])


def test_exited_position_releases_its_slot():
    """Once a filled position's modeled exit is in the past relative to a
    later decision, its slot must be free for a different-symbol
    candidate."""
    day1 = date(2026, 6, 1)
    day2 = next_trading_day(day1)
    day3 = next_trading_day(day2)
    exit_time = decision_time_utc(day2)
    calendar = {
        day1: {"stock": [_candidate(day=day1, symbol="AAPL")], "crypto": []},
        day2: {"stock": [], "crypto": []},
        day3: {"stock": [_candidate(day=day3, symbol="MSFT")], "crypto": []},
    }

    def outcome_for(candidate, end_date, frames, expiration_sessions=None):
        if candidate.symbol == "AAPL":
            return {"status": "closed", "outcome_data_missing": False,
                    "filled_at": candidate.decision_time, "fill_price": 100.0,
                    "exit_time": exit_time, "exit_price": 110.0,
                    "exit_reason": "target", "ambiguous": False}
        return _unfilled(candidate, end_date, frames)

    result = simulate_portfolio(
        calendar, {"stock": {}, "crypto": {}}, day1, day3,
        ResearchPortfolio("test", max_positions=1), BASELINE_COST, "stock_only",
        outcome_simulator=outcome_for,
    )
    assert {t["symbol"] for t in result["trades"]} == {"AAPL", "MSFT"}
    assert not any(r["symbol"] == "MSFT" for r in result["rejected"])


def test_expired_order_releases_cash_not_just_its_capacity_slot():
    """_occupies_slot already frees a never-filled order's position slot
    on its own; this proves the SEPARATE cash-release branch actually
    returns its reserved notional too - without it, a later candidate
    that needs that specific cash would be rejected for
    insufficient_cash_or_quantity, not max_positions."""
    day1 = date(2026, 6, 1)
    day2 = next_trading_day(day1)
    portfolio = ResearchPortfolio(
        "test", starting_equity=20_000.0, risk_per_trade=1.0,
        max_position_pct=1.0, max_positions=10,
    )
    calendar = {
        day1: {"stock": [_candidate(day=day1, symbol="AAPL")], "crypto": []},
        day2: {"stock": [_candidate(day=day2, symbol="MSFT")], "crypto": []},
    }
    expires_at = decision_time_utc(day2)

    def expired_then_unfilled(candidate, end_date, frames, expiration_sessions=None):
        if candidate.symbol == "AAPL":
            return {"status": "expired_unfilled", "outcome_data_missing": False,
                    "filled_at": None, "fill_price": None, "expires_at": expires_at}
        return _unfilled(candidate, end_date, frames)

    result = simulate_portfolio(
        calendar, {"stock": {}, "crypto": {}}, day1, day2,
        portfolio, BASELINE_COST, "stock_only",
        outcome_simulator=expired_then_unfilled,
    )
    assert {t["symbol"] for t in result["trades"]} == {"AAPL", "MSFT"}
    assert not any(r["symbol"] == "MSFT" and r["reason"] == "insufficient_cash_or_quantity"
                   for r in result["rejected"])


def test_expiration_sessions_changes_the_cache_key():
    day1 = date(2026, 6, 1)
    calendar = {day1: {"stock": [_candidate(day=day1, symbol="AAPL")], "crypto": []}}
    outcome_cache: dict = {}

    simulate_portfolio(
        calendar, {"stock": {}, "crypto": {}}, day1, day1,
        ResearchPortfolio("test"), BASELINE_COST, "stock_only", outcome_cache,
        outcome_simulator=_unfilled, expiration_sessions=None,
    )
    simulate_portfolio(
        calendar, {"stock": {}, "crypto": {}}, day1, day1,
        ResearchPortfolio("test"), BASELINE_COST, "stock_only", outcome_cache,
        outcome_simulator=_unfilled, expiration_sessions=5,
    )
    assert len(outcome_cache) == 2
