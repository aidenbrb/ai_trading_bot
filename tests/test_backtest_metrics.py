"""Tests for backtest/metrics.py - backtest reporting."""
from datetime import date, datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

import backtest.metrics as metrics


def _trade(**overrides):
    base = dict(
        scenario="intended_deployment", cost_model="baseline", direction="long",
        breakout_triggered=True, cost_adjusted_pnl=10.0, pnl_r=0.2,
        session_date=date(2025, 6, 2), outcome_ambiguous=False,
    )
    base.update(overrides)
    return base


class TestSummarize:
    def test_computes_win_rate_and_expectancy(self):
        trades = [
            _trade(cost_adjusted_pnl=100.0, pnl_r=2.0),
            _trade(cost_adjusted_pnl=-50.0, pnl_r=-1.0),
        ]
        result = metrics.summarize(trades, "intended_deployment", "baseline")
        assert result["closed_count"] == 2
        assert result["win_rate"] == 0.5
        assert result["expectancy"] == pytest.approx(25.0)
        assert result["total_pnl"] == pytest.approx(50.0)

    def test_only_includes_matching_scenario_and_cost_model(self):
        trades = [
            _trade(scenario="research_fidelity", cost_adjusted_pnl=999.0),
            _trade(scenario="intended_deployment", cost_model="zero", cost_adjusted_pnl=999.0),
            _trade(cost_adjusted_pnl=10.0),
        ]
        result = metrics.summarize(trades, "intended_deployment", "baseline")
        assert result["closed_count"] == 1
        assert result["total_pnl"] == pytest.approx(10.0)

    def test_no_closed_trades_returns_none_stats(self):
        result = metrics.summarize([], "intended_deployment", "baseline")
        assert result["closed_count"] == 0
        assert result["win_rate"] is None

    def test_never_triggered_trades_excluded_from_closed_count(self):
        trades = [_trade(breakout_triggered=False, cost_adjusted_pnl=None, pnl_r=None)]
        result = metrics.summarize(trades, "intended_deployment", "baseline")
        assert result["admitted_count"] == 1
        assert result["triggered_count"] == 0
        assert result["closed_count"] == 0

    def test_counts_ambiguous_outcomes(self):
        trades = [_trade(outcome_ambiguous=True), _trade(outcome_ambiguous=False)]
        result = metrics.summarize(trades, "intended_deployment", "baseline")
        assert result["ambiguous_count"] == 1


class TestSummarizeByDirection:
    def test_reports_long_and_short_separately(self):
        trades = [
            _trade(direction="long", cost_adjusted_pnl=100.0),
            _trade(direction="short", cost_adjusted_pnl=-20.0),
        ]
        result = metrics.summarize_by_direction(trades, "intended_deployment", "baseline")
        assert result["long"]["expectancy"] == pytest.approx(100.0)
        assert result["short"]["expectancy"] == pytest.approx(-20.0)


class TestMissingDataReport:
    def test_reports_rate_and_reasons(self):
        excluded = [
            {"symbol": "A", "session_date": date(2025, 6, 2), "reason": "missing opening bar or reference stats"},
            {"symbol": "B", "session_date": date(2025, 6, 2), "reason": "missing opening bar or reference stats"},
            {"symbol": "C", "session_date": date(2025, 6, 2), "reason": "some other error"},
        ]
        report = metrics.missing_data_report(excluded, total_symbol_days=10)
        assert report["excluded_count"] == 3
        assert report["excluded_rate"] == pytest.approx(0.3)
        assert report["reasons"]["missing opening bar or reference stats"] == 2

    def test_zero_total_symbol_days_does_not_divide_by_zero(self):
        report = metrics.missing_data_report([], total_symbol_days=0)
        assert report["excluded_rate"] is None


class TestEquityCurve:
    def test_accumulates_pnl_chronologically(self):
        trades = [
            _trade(session_date=date(2025, 6, 3), cost_adjusted_pnl=100.0),
            _trade(session_date=date(2025, 6, 2), cost_adjusted_pnl=-50.0),
        ]
        curve = metrics.equity_curve(trades, "intended_deployment", "baseline", starting_equity=1000.0)
        assert [c["session_date"] for c in curve] == [date(2025, 6, 2), date(2025, 6, 3)]
        assert curve[0]["equity"] == pytest.approx(950.0)
        assert curve[1]["equity"] == pytest.approx(1050.0)

    def test_tracks_drawdown_from_peak(self):
        trades = [
            _trade(session_date=date(2025, 6, 2), cost_adjusted_pnl=100.0),
            _trade(session_date=date(2025, 6, 3), cost_adjusted_pnl=-50.0),
        ]
        curve = metrics.equity_curve(trades, "intended_deployment", "baseline", starting_equity=1000.0)
        assert curve[1]["drawdown_pct"] == pytest.approx(-50.0 / 1100.0)


def _overlap_record(**overrides):
    base = dict(session_date=date(2025, 6, 2), valid=True, overlap_rate=0.5, jaccard=0.33)
    base.update(overrides)
    return base


class TestAggregateOverlap:
    def test_averages_only_comparable_valid_sessions(self):
        records = [
            _overlap_record(overlap_rate=0.5, jaccard=0.33),
            _overlap_record(overlap_rate=1.0, jaccard=1.0),
        ]
        result = metrics.aggregate_overlap(records)
        assert result["avg_overlap_rate"] == pytest.approx(0.75)
        assert result["avg_jaccard"] == pytest.approx(0.665)
        assert result["comparable_sessions"] == 2

    def test_both_empty_sessions_excluded_from_average_but_counted(self):
        records = [
            _overlap_record(overlap_rate=1.0, jaccard=1.0),
            _overlap_record(overlap_rate=None, jaccard=None),  # both feeds selected zero
        ]
        result = metrics.aggregate_overlap(records)
        assert result["avg_overlap_rate"] == pytest.approx(1.0)  # only the comparable session counts
        assert result["both_empty_sessions"] == 1
        assert result["valid_sessions"] == 2

    def test_zero_overlap_sessions_pull_the_average_down(self):
        """
        Critical distinction: a session where one feed found real
        opportunities and the other found none (overlap_rate=0.0) is NOT
        the same as 'nothing to compare' (None) - it must count in the
        average and pull it down, not be silently excluded.
        """
        records = [
            _overlap_record(overlap_rate=1.0, jaccard=1.0),
            _overlap_record(overlap_rate=0.0, jaccard=0.0),
        ]
        result = metrics.aggregate_overlap(records)
        assert result["avg_overlap_rate"] == pytest.approx(0.5)
        assert result["comparable_sessions"] == 2

    def test_invalid_sessions_excluded_entirely(self):
        records = [
            _overlap_record(overlap_rate=1.0, jaccard=1.0),
            _overlap_record(valid=False, overlap_rate=None, jaccard=None, reason="missing/incomplete feed data"),
        ]
        result = metrics.aggregate_overlap(records)
        assert result["avg_overlap_rate"] == pytest.approx(1.0)
        assert result["invalid_sessions"] == 1
        assert result["valid_sessions"] == 1

    def test_no_comparable_sessions_returns_none_average(self):
        records = [_overlap_record(valid=False, overlap_rate=None, jaccard=None)]
        result = metrics.aggregate_overlap(records)
        assert result["avg_overlap_rate"] is None
        assert result["avg_jaccard"] is None

    def test_empty_records_list(self):
        result = metrics.aggregate_overlap([])
        assert result["total_sessions"] == 0
        assert result["avg_overlap_rate"] is None
