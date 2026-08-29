"""Tests for utils/orb_signals.py - pure ORB signal functions."""
import pandas as pd
import pytest

from utils.orb_signals import (
    classify_opening_candle,
    orb_direction,
    same_time_opening_volume_avg,
    opening_relative_volume,
    passes_orb_filters,
    rank_and_select,
    compute_stop_price,
    simulate_intraday_outcome,
    build_candidate_fields,
    compute_daily_reference_stats,
)


def _bars(rows: list[tuple]) -> pd.DataFrame:
    """rows: list of (timestamp_str, open, high, low, close)."""
    idx = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows])
    return pd.DataFrame(
        {
            "open": [r[1] for r in rows], "high": [r[2] for r in rows],
            "low": [r[3] for r in rows], "close": [r[4] for r in rows],
        },
        index=idx,
    )


class TestClassifyOpeningCandle:
    def test_bullish_when_close_above_open(self):
        assert classify_opening_candle(open_=10.0, close=10.5) == "bullish"

    def test_bearish_when_close_below_open(self):
        assert classify_opening_candle(open_=10.0, close=9.5) == "bearish"

    def test_doji_when_exactly_equal(self):
        assert classify_opening_candle(open_=10.0, close=10.0) == "doji"

    def test_near_equal_is_not_doji(self):
        # Locks in the "exact spec, no ratio threshold" decision as a
        # regression test - a near-equal-but-not-exact candle must NOT be
        # classified as doji in v1.
        assert classify_opening_candle(open_=10.00, close=10.001) == "bullish"
        assert classify_opening_candle(open_=10.00, close=9.999) == "bearish"


class TestOrbDirection:
    def test_bullish_maps_to_long(self):
        assert orb_direction("bullish") == "long"

    def test_bearish_maps_to_short(self):
        assert orb_direction("bearish") == "short"

    def test_doji_maps_to_none(self):
        assert orb_direction("doji") == "none"


class TestSameTimeOpeningVolumeAvg:
    def test_averages_the_lookback_window(self):
        vols = [100.0] * 10 + [200.0] * 4
        assert same_time_opening_volume_avg(vols, lookback=14) == pytest.approx(128.57, rel=1e-3)

    def test_fewer_than_lookback_sessions_returns_none(self):
        """
        Regression test: an incomplete window must fail closed (None), not
        silently average whatever is available - a short sample could be
        biased by whichever specific days happened to be missing.
        """
        assert same_time_opening_volume_avg([100.0, 200.0], lookback=14) is None

    def test_exactly_lookback_sessions_averages(self):
        assert same_time_opening_volume_avg([100.0] * 14, lookback=14) == 100.0

    def test_no_prior_sessions_returns_none(self):
        assert same_time_opening_volume_avg([], lookback=14) is None

    def test_uses_only_most_recent_lookback_sessions(self):
        vols = [1000.0] + [100.0] * 14  # oldest session should be dropped
        assert same_time_opening_volume_avg(vols, lookback=14) == 100.0


class TestOpeningRelativeVolume:
    def test_computes_ratio(self):
        assert opening_relative_volume(200.0, 100.0) == 2.0

    def test_none_average_returns_none(self):
        assert opening_relative_volume(200.0, None) is None

    def test_zero_average_returns_none(self):
        assert opening_relative_volume(200.0, 0.0) is None


class TestPassesOrbFilters:
    def _base(self, **overrides):
        base = dict(price=10.0, avg_daily_volume_14d=2_000_000,
                    daily_atr_14=1.0, opening_rel_volume=1.5)
        base.update(overrides)
        return base

    def test_all_pass(self):
        ok, reason = passes_orb_filters(**self._base())
        assert ok, reason

    def test_price_must_be_strictly_above_5(self):
        ok, reason = passes_orb_filters(**self._base(price=5.0))
        assert not ok
        assert "price" in reason

    def test_price_exactly_above_5_passes(self):
        ok, _ = passes_orb_filters(**self._base(price=5.01))
        assert ok

    def test_avg_volume_at_exactly_minimum_passes(self):
        ok, _ = passes_orb_filters(**self._base(avg_daily_volume_14d=1_000_000))
        assert ok

    def test_avg_volume_below_minimum_fails(self):
        ok, reason = passes_orb_filters(**self._base(avg_daily_volume_14d=999_999))
        assert not ok
        assert "volume" in reason

    def test_atr_must_be_strictly_above_050(self):
        ok, reason = passes_orb_filters(**self._base(daily_atr_14=0.50))
        assert not ok
        assert "ATR" in reason

    def test_atr_exactly_above_050_passes(self):
        ok, _ = passes_orb_filters(**self._base(daily_atr_14=0.51))
        assert ok

    def test_opening_rel_volume_at_exactly_1x_passes(self):
        ok, _ = passes_orb_filters(**self._base(opening_rel_volume=1.0))
        assert ok

    def test_opening_rel_volume_below_1x_fails(self):
        ok, reason = passes_orb_filters(**self._base(opening_rel_volume=0.99))
        assert not ok
        assert "relative volume" in reason

    def test_missing_atr_fails_closed(self):
        ok, _ = passes_orb_filters(**self._base(daily_atr_14=None))
        assert not ok

    def test_missing_avg_volume_fails_closed(self):
        ok, _ = passes_orb_filters(**self._base(avg_daily_volume_14d=None))
        assert not ok

    def test_missing_opening_rel_volume_fails_closed(self):
        ok, _ = passes_orb_filters(**self._base(opening_rel_volume=None))
        assert not ok


class TestRankAndSelect:
    def test_ranks_by_descending_relative_volume(self):
        candidates = [
            {"symbol": "A", "opening_rel_volume": 1.5},
            {"symbol": "B", "opening_rel_volume": 3.0},
            {"symbol": "C", "opening_rel_volume": 2.0},
        ]
        ranked = rank_and_select(candidates, top_n=2)
        assert [c["symbol"] for c in ranked] == ["B", "C", "A"]
        assert [c["rank"] for c in ranked] == [1, 2, 3]

    def test_top_n_truncation(self):
        candidates = [{"symbol": str(i), "opening_rel_volume": float(i)} for i in range(30)]
        ranked = rank_and_select(candidates, top_n=20)
        selected = [c for c in ranked if c["selected"]]
        assert len(selected) == 20
        assert selected[0]["symbol"] == "29"

    def test_none_relative_volume_sorts_last_and_never_selected(self):
        candidates = [
            {"symbol": "A", "opening_rel_volume": None},
            {"symbol": "B", "opening_rel_volume": 1.0},
        ]
        ranked = rank_and_select(candidates, top_n=20)
        assert ranked[-1]["symbol"] == "A"
        assert ranked[-1]["selected"] is False

    def test_does_not_mutate_input(self):
        candidates = [{"symbol": "A", "opening_rel_volume": 1.0}]
        rank_and_select(candidates, top_n=20)
        assert "rank" not in candidates[0]


class TestComputeStopPrice:
    def test_long_stop_is_below_entry(self):
        stop = compute_stop_price(entry_trigger=100.0, daily_atr_14=2.0, direction="long")
        assert stop == pytest.approx(99.8)  # 100 - 0.10*2.0

    def test_short_stop_is_above_entry(self):
        stop = compute_stop_price(entry_trigger=100.0, daily_atr_14=2.0, direction="short")
        assert stop == pytest.approx(100.2)  # 100 + 0.10*2.0

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError):
            compute_stop_price(entry_trigger=100.0, daily_atr_14=2.0, direction="none")

    def test_custom_stop_fraction(self):
        stop = compute_stop_price(
            entry_trigger=100.0, daily_atr_14=2.0, direction="long", stop_atr_fraction=0.05
        )
        assert stop == pytest.approx(99.9)


class TestComputeDailyReferenceStats:
    def _daily_df(self, n=20, volume=1_000_000.0):
        idx = pd.date_range("2025-05-01", periods=n, freq="D")
        return pd.DataFrame({
            "high": [101.0] * n, "low": [99.0] * n, "close": [100.0] * n, "volume": [volume] * n,
        }, index=idx)

    def test_computes_all_three_stats(self):
        stats = compute_daily_reference_stats(self._daily_df(20), [10_000.0] * 14, lookback=14)
        assert stats is not None
        assert stats["avg_daily_volume_14d"] == pytest.approx(1_000_000.0)
        assert stats["daily_atr_14"] is not None
        assert stats["avg_opening_volume_14d"] == pytest.approx(10_000.0)

    def test_empty_daily_bars_returns_none(self):
        assert compute_daily_reference_stats(pd.DataFrame(), [10_000.0], lookback=14) is None

    def test_too_few_daily_bars_returns_none(self):
        assert compute_daily_reference_stats(self._daily_df(1), [10_000.0], lookback=14) is None

    def test_no_opening_volumes_returns_none(self):
        assert compute_daily_reference_stats(self._daily_df(20), [], lookback=14) is None


class TestBuildCandidateFields:
    """
    The shared, I/O-free core used identically by nodes/day_strategy_node.py
    (live) and backtest/engine.py (historical) - this is what keeps
    backtested and live signal logic from ever drifting apart.
    """

    def _base(self, **overrides):
        base = dict(
            opening_open=10.0, opening_high=10.6, opening_low=9.8, opening_close=10.5,
            opening_volume=50_000.0, avg_daily_volume_14d=2_000_000.0,
            daily_atr_14=1.0, avg_opening_volume_14d=10_000.0,
        )
        base.update(overrides)
        return base

    def test_tradable_long_candidate(self):
        fields = build_candidate_fields(**self._base())
        assert fields["candle_type"] == "bullish"
        assert fields["direction"] == "long"
        assert fields["passed_filters"] is True
        assert fields["entry_trigger_price"] == 10.6
        assert fields["stop_price"] == pytest.approx(10.6 - 0.10 * 1.0)
        assert fields["rejection_reason"] is None

    def test_doji_is_never_tradable(self):
        fields = build_candidate_fields(**self._base(opening_close=10.0))
        assert fields["candle_type"] == "doji"
        assert fields["passed_filters"] is False
        assert fields["entry_trigger_price"] is None
        assert fields["rejection_reason"] == "doji - no trade"

    def test_price_filter_uses_opening_price_not_close(self):
        """
        Regression test: the $5 filter (and the resulting opening_price
        field elsewhere) must use the 9:30 OPEN, not the 5-minutes-later
        CLOSE. opening_open straddles $5 in the opposite direction from
        opening_close here - only the open-based filter is correct.
        """
        fields = build_candidate_fields(**self._base(opening_open=4.50, opening_close=5.50))
        assert fields["passed_filters"] is False
        assert "price" in fields["rejection_reason"]

    def test_failing_filter_is_not_tradable_even_with_direction(self):
        fields = build_candidate_fields(**self._base(daily_atr_14=0.10))  # below $0.50 minimum
        assert fields["direction"] == "long"
        assert fields["passed_filters"] is False
        assert fields["entry_trigger_price"] is None
        assert "ATR" in fields["rejection_reason"]


class TestSimulateIntradayOutcome:
    def test_never_triggers(self):
        bars = _bars([
            ("2025-06-02 09:35", 9.9, 10.0, 9.8, 9.9),
            ("2025-06-02 09:36", 9.9, 10.0, 9.8, 9.9),
        ])
        result = simulate_intraday_outcome(bars, entry_trigger=11.0, stop_price=9.0, direction="long")
        assert result["breakout_triggered"] is False
        assert result["exit_reason"] == "no_trigger"

    def test_long_triggers_then_stops_out_on_later_bar_no_gap(self):
        bars = _bars([
            ("2025-06-02 09:35", 10.2, 10.6, 10.2, 10.5),   # non-gap trigger (open 10.2 < 10.5)
            ("2025-06-02 09:36", 10.4, 10.5, 10.3, 10.4),
            ("2025-06-02 09:37", 10.1, 10.3, 10.0, 10.1),   # stop hit intrabar, open 10.1 > stop 10.0 (no gap)
        ])
        result = simulate_intraday_outcome(bars, entry_trigger=10.5, stop_price=10.0, direction="long")
        assert result["breakout_triggered"] is True
        assert result["entry_gapped"] is False
        assert result["simulated_entry_price"] == 10.5
        assert result["stop_hit"] is True
        assert result["exit_price"] == 10.0
        assert result["exit_gapped"] is False
        assert result["exit_reason"] == "stop"
        assert result["outcome_ambiguous"] is False
        assert str(result["trigger_time"]) == "2025-06-02 09:35:00"

    def test_long_triggers_and_runs_to_eod(self):
        bars = _bars([
            ("2025-06-02 09:35", 10.2, 10.6, 10.2, 10.5),
            ("2025-06-02 15:59", 11.7, 12.0, 11.7, 11.9),   # never touches stop; last bar of session
        ])
        result = simulate_intraday_outcome(bars, entry_trigger=10.5, stop_price=10.0, direction="long")
        assert result["breakout_triggered"] is True
        assert result["stop_hit"] is False
        assert result["exit_reason"] == "eod_flatten"
        assert result["exit_price"] == 11.9

    def test_same_bar_trigger_and_stop_no_gap_is_ambiguous_and_assumes_adverse(self):
        # A single volatile bar sweeps both the breakout trigger and the stop,
        # with a non-gap open (between stop and trigger) - genuinely unknowable order.
        bars = _bars([("2025-06-02 09:35", 10.2, 10.6, 9.9, 10.2)])
        result = simulate_intraday_outcome(bars, entry_trigger=10.5, stop_price=10.0, direction="long")
        assert result["breakout_triggered"] is True
        assert result["entry_gapped"] is False
        assert result["outcome_ambiguous"] is True
        assert result["exit_reason"] == "stop"  # adverse outcome assumed, not continuation
        assert result["exit_price"] == 10.0
        assert result["exit_gapped"] is False

    def test_short_direction_mirrors_high_low(self):
        bars = _bars([
            ("2025-06-02 09:35", 9.7, 10.0, 9.4, 9.5),   # triggers short (low <= 9.5), no gap (open 9.7 > 9.5)
            ("2025-06-02 09:36", 9.6, 10.2, 9.5, 10.1),  # stop hit (high >= 10.2), no gap (open 9.6 < 10.2)
        ])
        result = simulate_intraday_outcome(bars, entry_trigger=9.5, stop_price=10.2, direction="short")
        assert result["breakout_triggered"] is True
        assert result["stop_hit"] is True
        assert result["exit_price"] == 10.2
        assert result["exit_gapped"] is False

    def test_entry_gap_fills_at_open_not_trigger(self):
        # Price gaps up through the trigger before the bar even starts.
        bars = _bars([("2025-06-02 09:35", 10.8, 10.9, 10.7, 10.85)])
        result = simulate_intraday_outcome(bars, entry_trigger=10.5, stop_price=10.0, direction="long")
        assert result["breakout_triggered"] is True
        assert result["entry_gapped"] is True
        assert result["simulated_entry_price"] == 10.8  # the bar's open, not the nominal trigger

    def test_exit_gap_on_a_later_bar_fills_at_open(self):
        bars = _bars([
            ("2025-06-02 09:35", 10.2, 10.6, 10.2, 10.5),   # non-gap trigger
            ("2025-06-02 09:36", 10.4, 10.5, 10.3, 10.4),   # no stop touch
            ("2025-06-02 09:37", 9.5, 9.6, 9.3, 9.4),       # gapped down through stop (open 9.5 <= stop 10.0)
        ])
        result = simulate_intraday_outcome(bars, entry_trigger=10.5, stop_price=10.0, direction="long")
        assert result["stop_hit"] is True
        assert result["exit_gapped"] is True
        assert result["exit_price"] == 9.5  # the later bar's open, worse than the nominal stop
        assert result["outcome_ambiguous"] is False

    def test_same_bar_entry_gapped_then_intrabar_stop_touch_is_not_ambiguous(self):
        # Entry gaps up (open already >= trigger), and the SAME bar also dips
        # to touch the stop intrabar - order is airtight (entry at the gap-open,
        # stop touch necessarily after), so this is NOT ambiguous, and the
        # stop still fills at the nominal stop_price (the open didn't gap past it).
        bars = _bars([("2025-06-02 09:35", 10.7, 10.8, 9.9, 10.0)])
        result = simulate_intraday_outcome(bars, entry_trigger=10.5, stop_price=10.0, direction="long")
        assert result["breakout_triggered"] is True
        assert result["entry_gapped"] is True
        assert result["outcome_ambiguous"] is False
        assert result["stop_hit"] is True
        assert result["exit_price"] == 10.0
        assert result["exit_gapped"] is False

    def test_same_bar_non_gap_entry_never_reuses_open_as_stop_gap_price(self):
        """
        The precise edge case from round 4: this bar's open (9.9) is already
        at/below the stop (10.0) - BEFORE any position exists - and the bar
        subsequently rises to trigger entry (high 10.6 >= 10.5). Because
        entry was a non-gap intrabar touch on this SAME bar, the position did
        not exist when the bar opened, so 9.9 must NEVER be used to price a
        stop exit here, even though it numerically satisfies open<=stop_price.
        Must be treated as the ordinary same-bar ambiguity case instead.
        """
        bars = _bars([("2025-06-02 09:35", 9.9, 10.6, 9.8, 10.2)])
        result = simulate_intraday_outcome(bars, entry_trigger=10.5, stop_price=10.0, direction="long")
        assert result["entry_gapped"] is False
        assert result["outcome_ambiguous"] is True
        assert result["exit_price"] == 10.0  # plain stop_price, NOT the bar's open (9.9)
        assert result["exit_gapped"] is False
