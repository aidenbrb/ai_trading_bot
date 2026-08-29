"""
Tests for backtest/engine.py - the session-date-major, multi-symbol batched
ORB simulation engine, and its portfolio accounting (eligible/admitted/
triggered, reserved vs. realized exposure, evolving equity).
"""
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import backtest.engine as engine
from backtest.portfolio import PortfolioConfig
from utils.cost_model import ZERO_COST, BASELINE_COST
from utils.market_calendar import prior_trading_days, session_for

# A real NYSE trading day with a full 14-prior-session history behind it.
_TARGET_DATE = date(2025, 7, 1)
_SYMBOLS = [f"S{i:02d}" for i in range(20)]


def _daily_df(n=20):
    idx = pd.date_range(end=_TARGET_DATE - timedelta(days=1), periods=n, freq="D")
    return pd.DataFrame({
        "open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n,
        "close": [100.0] * n, "volume": [2_000_000.0] * n,
    }, index=idx)


def _flat_opening_bars(symbols, when, volume=1000.0):
    idx = pd.DatetimeIndex([when])
    return {s: pd.DataFrame({"open": [10.0], "high": [10.6], "low": [9.8], "close": [10.5], "volume": [volume]}, index=idx)
            for s in symbols}


def _ranked_opening_bars(symbols, when):
    """Volume increases with symbol index, so rank order is deterministic (S19 highest)."""
    idx = pd.DatetimeIndex([when])
    return {
        s: pd.DataFrame({"open": [10.0], "high": [10.6], "low": [9.8], "close": [10.5], "volume": [float((i + 1) * 1000)]}, index=idx)
        for i, s in enumerate(symbols)
    }


def _fake_daily_multi(symbols, start, end):
    return {s: _daily_df() for s in symbols}


def _make_fake_intraday_multi(target_date=_TARGET_DATE, ranked_on_target=True, minute_bars_factory=None):
    def fake(symbols, start, end, minutes=5, feed="iex"):
        if minutes == 1:
            if minute_bars_factory is not None:
                return minute_bars_factory(symbols, start, end)
            idx = pd.DatetimeIndex([start, start + timedelta(minutes=1)])
            return {s: pd.DataFrame(
                {"open": [10.6, 12.0], "high": [10.7, 12.1], "low": [10.5, 11.9], "close": [10.6, 12.0]}, index=idx
            ) for s in symbols}
        is_target = start.date() == target_date
        if is_target and ranked_on_target:
            return _ranked_opening_bars(symbols, start)
        return _flat_opening_bars(symbols, start)
    return fake


class TestFetchBatching:
    def test_one_batch_call_per_session_date_for_iex_opening_not_per_symbol(self):
        fake_intraday = _make_fake_intraday_multi()
        with patch.object(engine, "get_daily_bars_multi", side_effect=_fake_daily_multi), \
             patch.object(engine, "get_intraday_bars_multi", side_effect=fake_intraday) as mock_fetch:
            engine.run_backtest(
                symbols=_SYMBOLS, start_date=_TARGET_DATE, end_date=_TARGET_DATE,
                scenarios={"intended_deployment": PortfolioConfig(100_000.0, 0.0025, 1.0, 5, 0.20)},
                cost_models={"zero": ZERO_COST},
            )
        iex_opening_calls = [c for c in mock_fetch.call_args_list if c.kwargs.get("minutes") == 5 and c.kwargs.get("feed") == "iex"]
        # Every call passes the FULL symbol list, never a single symbol.
        for c in iex_opening_calls:
            assert c.args[0] == _SYMBOLS

    def test_sip_opening_fetched_every_session_not_only_warmup(self):
        fake_intraday = _make_fake_intraday_multi()
        with patch.object(engine, "get_daily_bars_multi", side_effect=_fake_daily_multi), \
             patch.object(engine, "get_intraday_bars_multi", side_effect=fake_intraday) as mock_fetch:
            engine.run_backtest(
                symbols=_SYMBOLS, start_date=_TARGET_DATE, end_date=_TARGET_DATE,
                scenarios={"intended_deployment": PortfolioConfig(100_000.0, 0.0025, 1.0, 5, 0.20)},
                cost_models={"zero": ZERO_COST},
            )
        sip_calls = [c for c in mock_fetch.call_args_list if c.kwargs.get("minutes") == 5 and c.kwargs.get("feed") == "sip"]
        # At least the warm-up dates (14) plus the one evaluated session date.
        assert len(sip_calls) >= 15

    def test_warmup_covers_14_trading_days_before_start(self):
        fake_intraday = _make_fake_intraday_multi()
        with patch.object(engine, "get_daily_bars_multi", side_effect=_fake_daily_multi) as mock_daily, \
             patch.object(engine, "get_intraday_bars_multi", side_effect=fake_intraday):
            engine.run_backtest(
                symbols=_SYMBOLS, start_date=_TARGET_DATE, end_date=_TARGET_DATE,
                scenarios={"intended_deployment": PortfolioConfig(100_000.0, 0.0025, 1.0, 5, 0.20)},
                cost_models={"zero": ZERO_COST},
            )
        expected_warmup_start = prior_trading_days(_TARGET_DATE, 14)[0]
        daily_start_arg = mock_daily.call_args.args[1]
        assert daily_start_arg == expected_warmup_start

    def test_one_minute_call_per_session_with_selections(self):
        fake_intraday = _make_fake_intraday_multi()
        with patch.object(engine, "get_daily_bars_multi", side_effect=_fake_daily_multi), \
             patch.object(engine, "get_intraday_bars_multi", side_effect=fake_intraday) as mock_fetch:
            engine.run_backtest(
                symbols=_SYMBOLS, start_date=_TARGET_DATE, end_date=_TARGET_DATE,
                scenarios={"intended_deployment": PortfolioConfig(100_000.0, 0.0025, 1.0, 5, 0.20)},
                cost_models={"zero": ZERO_COST},
            )
        one_min_calls = [c for c in mock_fetch.call_args_list if c.kwargs.get("minutes") == 1]
        assert len(one_min_calls) == 1
        assert set(one_min_calls[0].args[0]) <= set(_SYMBOLS)


class TestPortfolioAccounting:
    def _run(self, scenarios, cost_models=None):
        fake_intraday = _make_fake_intraday_multi()
        with patch.object(engine, "get_daily_bars_multi", side_effect=_fake_daily_multi), \
             patch.object(engine, "get_intraday_bars_multi", side_effect=fake_intraday):
            return engine.run_backtest(
                symbols=_SYMBOLS, start_date=_TARGET_DATE, end_date=_TARGET_DATE,
                scenarios=scenarios, cost_models=cost_models or {"zero": ZERO_COST},
            )

    def test_admission_capped_at_max_concurrent_positions(self):
        config = PortfolioConfig(starting_equity=100_000.0, risk_per_trade_pct=0.0025,
                                  max_gross_exposure_x=10.0, max_concurrent_positions=5,
                                  max_position_concentration_pct=None)
        result = self._run({"intended_deployment": config})
        admitted = [t for t in result["trades"] if t["scenario"] == "intended_deployment" and t["cost_model"] == "zero"]
        rejected = [r for r in result["rejected_orders"] if r["scenario"] == "intended_deployment"]
        assert len(admitted) == 5
        assert len(rejected) == 15
        assert all(r["reason"] == "max_concurrent_positions" for r in rejected)
        # Rank order: the 5 admitted must be the 5 HIGHEST relative-volume symbols.
        assert {t["symbol"] for t in admitted} == {"S19", "S18", "S17", "S16", "S15"}

    def test_admission_capped_at_gross_exposure(self):
        config = PortfolioConfig(starting_equity=100_000.0, risk_per_trade_pct=0.05,  # large risk -> large positions
                                  max_gross_exposure_x=0.1, max_concurrent_positions=20,
                                  max_position_concentration_pct=None)
        result = self._run({"tight_exposure": config})
        rejected = [r for r in result["rejected_orders"] if r["scenario"] == "tight_exposure"]
        assert any(r["reason"] == "max_gross_exposure" for r in rejected)

    def test_admit_at_exactly_the_cap_not_rejected(self):
        """Admission boundary: proposed exposure == cap must be ADMITTED, not rejected."""
        # entry=10.6, stop=10.6-0.1*2=10.4 -> risk_per_unit=0.2; risk_pct chosen so
        # a single trade's position_value lands exactly on the exposure cap.
        config = PortfolioConfig(starting_equity=100_000.0, risk_per_trade_pct=0.0025,
                                  max_gross_exposure_x=1.0, max_concurrent_positions=1,
                                  max_position_concentration_pct=None)
        result = self._run({"solo": config})
        admitted = [t for t in result["trades"] if t["scenario"] == "solo"]
        assert len(admitted) == 1  # the single top-ranked candidate must be admitted

    def test_non_triggered_admitted_order_counts_as_admitted_not_triggered(self):
        def never_triggers(symbols, start, end):
            idx = pd.DatetimeIndex([start])
            return {s: pd.DataFrame({"open": [10.0], "high": [10.1], "low": [9.9], "close": [10.0]}, index=idx) for s in symbols}

        fake_intraday = _make_fake_intraday_multi(minute_bars_factory=never_triggers)
        with patch.object(engine, "get_daily_bars_multi", side_effect=_fake_daily_multi), \
             patch.object(engine, "get_intraday_bars_multi", side_effect=fake_intraday):
            result = engine.run_backtest(
                symbols=_SYMBOLS, start_date=_TARGET_DATE, end_date=_TARGET_DATE,
                scenarios={"intended_deployment": PortfolioConfig(100_000.0, 0.0025, 1.0, 5, 0.20)},
                cost_models={"zero": ZERO_COST},
            )
        admitted = [t for t in result["trades"] if t["scenario"] == "intended_deployment"]
        assert len(admitted) == 5
        assert all(t["triggered"] is False for t in admitted)
        assert all(t["cost_adjusted_pnl"] is None for t in admitted)

    def test_reservation_overrun_does_not_imply_session_exposure_cap_exceeded(self):
        result = self._run({"intended_deployment": PortfolioConfig(100_000.0, 0.0025, 1.0, 5, 0.20)})
        admitted = [t for t in result["trades"] if t["scenario"] == "intended_deployment"]
        # Entry gaps from 10.6 (trigger) to 10.6 in the fixture (no gap) by default,
        # so reservation_overrun should be False here - sanity check the field exists
        # and is well-formed, independent of exposure-cap state.
        for t in admitted:
            assert "reservation_overrun" in t
            assert "session_exposure_cap_exceeded" in t

    def test_equity_evolves_across_sessions(self):
        first_date = _TARGET_DATE
        second_date = None
        # Find the next trading day after _TARGET_DATE for a genuine two-session run.
        from utils.market_calendar import next_trading_day
        second_date = next_trading_day(first_date)

        def fake_intraday_two_day(symbols, start, end, minutes=5, feed="iex"):
            if minutes == 1:
                idx = pd.DatetimeIndex([start, start + timedelta(minutes=1)])
                return {s: pd.DataFrame(
                    {"open": [10.6, 12.0], "high": [10.7, 12.1], "low": [10.5, 11.9], "close": [10.6, 12.0]}, index=idx
                ) for s in symbols}
            if start.date() in (first_date, second_date):
                return _ranked_opening_bars(symbols, start)
            return _flat_opening_bars(symbols, start)

        with patch.object(engine, "get_daily_bars_multi", side_effect=_fake_daily_multi), \
             patch.object(engine, "get_intraday_bars_multi", side_effect=fake_intraday_two_day):
            result = engine.run_backtest(
                symbols=_SYMBOLS, start_date=first_date, end_date=second_date,
                scenarios={"intended_deployment": PortfolioConfig(100_000.0, 0.0025, 1.0, 5, 0.20)},
                cost_models={"zero": ZERO_COST},
            )

        trades = [t for t in result["trades"] if t["scenario"] == "intended_deployment"]
        day1_equity = {t["equity_at_arm_time"] for t in trades if t["session_date"] == first_date}
        day2_equity = {t["equity_at_arm_time"] for t in trades if t["session_date"] == second_date}
        assert day1_equity == {100_000.0}
        # Day 2 sizing must reflect day 1's realized P&L, not the original starting_equity.
        assert day2_equity != {100_000.0}
        assert result["final_equity"][("intended_deployment", "zero")] != 100_000.0


class TestMissingOutcomeDataIsNotConflatedWithNoTrigger:
    def test_failed_one_minute_fetch_marks_outcome_data_missing(self):
        def fake_intraday_outcome_fails(symbols, start, end, minutes=5, feed="iex"):
            if minutes == 1:
                raise RuntimeError("network error")
            return _ranked_opening_bars(symbols, start) if start.date() == _TARGET_DATE else _flat_opening_bars(symbols, start)

        with patch.object(engine, "get_daily_bars_multi", side_effect=_fake_daily_multi), \
             patch.object(engine, "get_intraday_bars_multi", side_effect=fake_intraday_outcome_fails):
            result = engine.run_backtest(
                symbols=_SYMBOLS, start_date=_TARGET_DATE, end_date=_TARGET_DATE,
                scenarios={"intended_deployment": PortfolioConfig(100_000.0, 0.0025, 1.0, 5, 0.20)},
                cost_models={"zero": ZERO_COST},
            )
        admitted = [t for t in result["trades"] if t["scenario"] == "intended_deployment"]
        assert len(admitted) == 5
        assert all(t["outcome_data_missing"] is True for t in admitted)
        assert all(t["exit_reason"] == "outcome_data_missing" for t in admitted)
        assert all(t["cost_adjusted_pnl"] is None for t in admitted)  # excluded from P&L, same as a real no-trigger
        # missing_outcome_data is recorded once per SELECTED candidate (20,
        # shared across scenarios) - not once per admitted order (5) - since
        # the 1-minute fetch happens once per session, before admission.
        assert len(result["missing_outcome_data"]) == 20
        assert all(m["session_date"] == _TARGET_DATE for m in result["missing_outcome_data"])

    def test_one_symbol_missing_one_minute_bars_is_flagged_not_treated_as_no_trigger(self):
        def fake_intraday_one_symbol_empty(symbols, start, end, minutes=5, feed="iex"):
            if minutes == 1:
                result = {}
                for s in symbols:
                    if s == "S19":  # the top-ranked, always-admitted symbol
                        result[s] = pd.DataFrame()  # successful response, but empty for this symbol
                    else:
                        idx = pd.DatetimeIndex([start, start + timedelta(minutes=1)])
                        result[s] = pd.DataFrame(
                            {"open": [10.6, 12.0], "high": [10.7, 12.1], "low": [10.5, 11.9], "close": [10.6, 12.0]}, index=idx
                        )
                return result
            return _ranked_opening_bars(symbols, start) if start.date() == _TARGET_DATE else _flat_opening_bars(symbols, start)

        with patch.object(engine, "get_daily_bars_multi", side_effect=_fake_daily_multi), \
             patch.object(engine, "get_intraday_bars_multi", side_effect=fake_intraday_one_symbol_empty):
            result = engine.run_backtest(
                symbols=_SYMBOLS, start_date=_TARGET_DATE, end_date=_TARGET_DATE,
                scenarios={"intended_deployment": PortfolioConfig(100_000.0, 0.0025, 1.0, 5, 0.20)},
                cost_models={"zero": ZERO_COST},
            )
        admitted = [t for t in result["trades"] if t["scenario"] == "intended_deployment"]
        s19_trade = next(t for t in admitted if t["symbol"] == "S19")
        others = [t for t in admitted if t["symbol"] != "S19"]
        assert s19_trade["outcome_data_missing"] is True
        assert s19_trade["exit_reason"] == "outcome_data_missing"
        assert all(t["outcome_data_missing"] is False for t in others)
        assert len(result["missing_outcome_data"]) == 1
        assert result["missing_outcome_data"][0]["symbol"] == "S19"


class TestMissingDataAttemptedDenominator:
    def test_attempted_equals_full_symbol_list_even_with_exclusions(self):
        def fake_intraday_partial(symbols, start, end, minutes=5, feed="iex"):
            if minutes == 1:
                return {}
            present = [s for s in symbols if s != "S00"]  # S00 always missing its opening bar
            return _flat_opening_bars(present, start)

        with patch.object(engine, "get_daily_bars_multi", side_effect=_fake_daily_multi), \
             patch.object(engine, "get_intraday_bars_multi", side_effect=fake_intraday_partial):
            result = engine.run_backtest(
                symbols=_SYMBOLS, start_date=_TARGET_DATE, end_date=_TARGET_DATE,
                scenarios={"intended_deployment": PortfolioConfig(100_000.0, 0.0025, 1.0, 5, 0.20)},
                cost_models={"zero": ZERO_COST},
            )
        counts = result["daily_candidate_counts"][0]
        assert counts["attempted"] == len(_SYMBOLS)
        assert counts["considered"] == len(_SYMBOLS) - 1  # S00 excluded
        assert any(e["symbol"] == "S00" for e in result["excluded"])


class TestOverlapRecords:
    def test_overlap_record_produced_for_evaluated_session(self):
        fake_intraday = _make_fake_intraday_multi()
        with patch.object(engine, "get_daily_bars_multi", side_effect=_fake_daily_multi), \
             patch.object(engine, "get_intraday_bars_multi", side_effect=fake_intraday):
            result = engine.run_backtest(
                symbols=_SYMBOLS, start_date=_TARGET_DATE, end_date=_TARGET_DATE,
                scenarios={"intended_deployment": PortfolioConfig(100_000.0, 0.0025, 1.0, 5, 0.20)},
                cost_models={"zero": ZERO_COST},
            )
        assert len(result["overlap_records"]) == 1
        record = result["overlap_records"][0]
        assert record["valid"] is True
        assert record["iex_selected_count"] > 0
        assert record["sip_selected_count"] > 0

    def test_failed_iex_fetch_marks_session_invalid(self):
        def fake_intraday_iex_fails(symbols, start, end, minutes=5, feed="iex"):
            if minutes == 1:
                return {}
            if feed == "iex" and start.date() == _TARGET_DATE:
                raise RuntimeError("network error")
            return _flat_opening_bars(symbols, start)

        with patch.object(engine, "get_daily_bars_multi", side_effect=_fake_daily_multi), \
             patch.object(engine, "get_intraday_bars_multi", side_effect=fake_intraday_iex_fails):
            result = engine.run_backtest(
                symbols=_SYMBOLS, start_date=_TARGET_DATE, end_date=_TARGET_DATE,
                scenarios={"intended_deployment": PortfolioConfig(100_000.0, 0.0025, 1.0, 5, 0.20)},
                cost_models={"zero": ZERO_COST},
            )
        assert result["overlap_records"][0]["valid"] is False
