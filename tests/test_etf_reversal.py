"""Required tests for etf_reversal_v1, per
research/etf_reversal_v1_preregistration.md Section 11."""
from __future__ import annotations

import ast
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.whole_bot_engine import (
    COSTS,
    EtfReversalConfig,
    simulate_etf_reversal_portfolio,
)
from backtest.whole_bot_metrics import qualify_strategy, summarize_run
from utils.indicators import rsi, sma


def _frame(opens: dict[str, float], closes: dict[str, float]) -> pd.DataFrame:
    """opens/closes: {"YYYY-MM-DD": value}. Builds Open/High/Low/Close with
    High=max(O,C), Low=min(O,C) - fine for these synthetic tests, which
    never touch High/Low."""
    dates = sorted(set(opens) | set(closes))
    index = pd.to_datetime(dates)
    rows = []
    for d in dates:
        o = opens.get(d)
        c = closes.get(d)
        rows.append({"Open": o, "High": max(o or c, c or o), "Low": min(o or c, c or o), "Close": c})
    return pd.DataFrame(rows, index=index)


def _flat(index: pd.DatetimeIndex, value: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame({"Open": value, "High": value, "Low": value, "Close": value}, index=index)


def _uptrend(index: pd.DatetimeIndex, low: float = 50.0, high: float = 150.0) -> pd.Series:
    """A steady linear uptrend, low->high across the whole index - close
    stays comfortably above its own SMA200 throughout, so a short one-day
    dip (see _with_dip) can drive RSI(2) low without close falling below
    SMA200 (verified numerically: a 200-day-average base near ~100 barely
    moves from one dip day, while an uptrend's recent price is already well
    above that average, leaving genuine room for "oversold but still above
    the 200-day trend" - the actual entry condition)."""
    n = len(index)
    return pd.Series([low + (high - low) * i / max(n - 1, 1) for i in range(n)], index=index)


def _with_dip(series: pd.Series, idx: int, pct: float = 0.05) -> pd.Series:
    """Single-day pct drop at position idx, relative to the PRIOR close -
    verified dip%->RSI(2) mapping on this exact _uptrend shape: 0.01->~23,
    0.02->~13, 0.03->~9, 0.05->~5.7, 0.10->~2.9, all with close > SMA200."""
    out = series.copy()
    out.iloc[idx] = out.iloc[idx - 1] * (1.0 - pct)
    return out


def _ohlc(close: pd.Series, open_: pd.Series | None = None) -> pd.DataFrame:
    open_series = open_ if open_ is not None else close
    return pd.DataFrame(
        {"Open": open_series, "High": pd.concat([open_series, close], axis=1).max(axis=1),
         "Low": pd.concat([open_series, close], axis=1).min(axis=1), "Close": close},
    )


# -- RSI(2) hand-verified vector -----------------------------------------------


def test_rsi2_matches_hand_verified_vector():
    closes = pd.Series([10.0, 11.0, 10.5, 12.0, 11.0, 11.5, 10.0, 9.5, 10.5, 12.0])
    r = rsi(closes, period=2)
    expected = [np.nan, np.nan, 66.666667, 88.888889, 47.058824, 64.0, 21.917808, 15.238095, 61.802575, 85.575365]
    for actual, exp in zip(r.tolist(), expected):
        if np.isnan(exp):
            assert np.isnan(actual)
        else:
            assert actual == pytest.approx(exp, rel=1e-5)


def test_rsi2_intermediate_avg_gain_avg_loss_match_hand_verified_vector():
    closes = pd.Series([10.0, 11.0, 10.5, 12.0, 11.0, 11.5, 10.0, 9.5, 10.5, 12.0])
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=0.5, min_periods=2, adjust=False).mean()
    avg_loss = loss.ewm(alpha=0.5, min_periods=2, adjust=False).mean()
    expected_gain = [np.nan, np.nan, 0.5, 1.0, 0.5, 0.5, 0.25, 0.125, 0.5625, 1.03125]
    expected_loss = [np.nan, np.nan, 0.25, 0.125, 0.5625, 0.28125, 0.890625, 0.6953125, 0.34765625, 0.173828125]
    for actual, exp in zip(avg_gain.tolist(), expected_gain):
        if np.isnan(exp):
            assert np.isnan(actual)
        else:
            assert actual == pytest.approx(exp, rel=1e-9)
    for actual, exp in zip(avg_loss.tolist(), expected_loss):
        if np.isnan(exp):
            assert np.isnan(actual)
        else:
            assert actual == pytest.approx(exp, rel=1e-9)


# -- max_hold worked example, exact dates --------------------------------------


def test_max_hold_5_worked_example_friday_signal_monday_fill_following_friday_exit():
    """Preregistration Section 4's exact worked example: entry signal fires
    at a Friday close, fills the following Monday's open (day 1), time_exit
    first fires at the close of the NEXT Friday (day 5), exit fills the
    Monday after that - a full week held."""
    days = pd.bdate_range("2020-01-06", periods=260, freq="B")  # starts on a Monday
    base = _uptrend(days)
    entry_signal_idx = None
    for idx in range(210, len(days) - 20):
        if days[idx].weekday() == 4:  # Friday
            entry_signal_idx = idx
            break
    assert entry_signal_idx is not None

    closes = _with_dip(base, entry_signal_idx, pct=0.03)   # verified: RSI2~8.9, close > SMA200
    dipped_value = closes.iloc[entry_signal_idx]
    closes.iloc[entry_signal_idx + 1:] = dipped_value   # flat afterward -> RSI2 stays constant, never crosses exit_threshold=200
    aaa = _ohlc(closes)
    spy = _flat(days, 100.0)

    start = days[entry_signal_idx - 5].date()
    # End exactly at the exit fill day - RSI2 stays constant at ~8.9 forever
    # (flat post-dip prices), which is BELOW entry_threshold=15, so AAA would
    # immediately re-enter the day after exiting if the window ran longer;
    # ending here isolates the one worked-example trade this test checks.
    end = days[entry_signal_idx + 6].date()
    result = simulate_etf_reversal_portfolio(
        {"SPY": spy, "AAA": aaa}, start, end, 100_000.0, COSTS["zero"],
        EtfReversalConfig(entry_threshold=15, exit_threshold=200, max_hold=5),  # exit_threshold=200: RSI can never trigger target_exit
        max_positions=5,
    )
    aaa_trades = [t for t in result["trades"] if t["symbol"] == "AAA"]
    assert len(aaa_trades) == 1
    trade = aaa_trades[0]

    entry_fill_monday = days[entry_signal_idx + 1].date()
    assert entry_fill_monday.weekday() == 0
    exit_fill_monday = days[entry_signal_idx + 6].date()
    assert exit_fill_monday.weekday() == 0

    assert trade["entry_date"] == entry_fill_monday
    assert trade["exit_reason"] == "time_exit"
    assert trade["exit_date"] == exit_fill_monday


def test_max_hold_counts_trading_days_not_calendar_days_across_a_holiday():
    """A holiday inside the hold window must not shift the trading-day
    count - max_hold=5 still means 5 trading SESSIONS, holiday or not."""
    # Business-day range spanning New Year's - but the reference index
    # itself only contains real trading days (no Jan 1), so max_hold=5
    # counts 5 entries in that index regardless of the calendar gap. 260+
    # days of warmup needed for SMA200, with the entry signal placed a few
    # trading days before year-end so the max_hold=5 hold window spans the
    # New Year's holiday.
    days = pd.bdate_range("2020-01-01", "2021-01-15", freq="B")
    days = days[days != pd.Timestamp("2021-01-01")]  # remove New Year's Day (observed non-trading day)
    assert pd.Timestamp("2020-12-31") in days and pd.Timestamp("2021-01-04") in days  # holiday sits between them
    entry_signal_idx = list(days).index(pd.Timestamp("2020-12-29"))

    base = _uptrend(days)
    closes = _with_dip(base, entry_signal_idx, pct=0.03)
    dipped_value = closes.iloc[entry_signal_idx]
    closes.iloc[entry_signal_idx + 1:] = dipped_value
    aaa = _ohlc(closes)
    spy = _flat(days, 100.0)

    # End exactly at the exit fill day - flat RSI2 (~8.9, constant, below
    # entry_threshold=15) would trigger an immediate re-entry the day after
    # exiting if the window ran longer; ending here isolates the one trade.
    end = days[entry_signal_idx + 6].date()
    result = simulate_etf_reversal_portfolio(
        {"SPY": spy, "AAA": aaa}, days[0].date(), end, 100_000.0, COSTS["zero"],
        EtfReversalConfig(entry_threshold=15, exit_threshold=200, max_hold=5), max_positions=5,
    )
    aaa_trades = [t for t in result["trades"] if t["symbol"] == "AAA" and t["exit_reason"] == "time_exit"]
    assert len(aaa_trades) == 1
    entry_idx = list(days.date).index(aaa_trades[0]["entry_date"])
    exit_idx = list(days.date).index(aaa_trades[0]["exit_date"])
    # entry fill is day 1; time_exit fires at close of day 5 (index entry_idx+4), fills at entry_idx+5 -
    # exactly 5 TRADING-day index steps, regardless of the holiday sitting inside that span.
    assert exit_idx - entry_idx == 5
    assert pd.Timestamp("2021-01-01") not in days  # holiday genuinely absent from the reference calendar


# -- SMA200 boundary --------------------------------------------------------


def test_sma200_exactly_equal_to_close_does_not_trigger_trend_exit():
    """close < SMA200 is strict - close == SMA200 must not exit."""
    days = pd.bdate_range("2020-01-01", periods=210, freq="B")
    closes = pd.Series(100.0, index=days)
    aaa = pd.DataFrame({"Open": closes, "High": closes, "Low": closes, "Close": closes})
    spy = _flat(days, 100.0)
    sma_series = sma(closes, 200)
    boundary_idx = sma_series.first_valid_index()
    assert boundary_idx is not None
    # close is already exactly 100.0 == its own flat SMA200 at every point -
    # confirm no trend_exit ever fires purely from equality.
    result = simulate_etf_reversal_portfolio(
        {"SPY": spy, "AAA": aaa}, days[199].date(), days[-1].date(), 100_000.0, COSTS["zero"],
        EtfReversalConfig(entry_threshold=200, exit_threshold=200, max_hold=1000), max_positions=5,
    )
    trend_exits = [t for t in result["trades"] if t["symbol"] == "AAA" and t["exit_reason"] == "trend_exit"]
    assert trend_exits == []


def test_symbol_becomes_eligible_exactly_200_trading_days_after_its_first_bar():
    days = pd.bdate_range("2020-01-01", periods=260, freq="B")
    closes = pd.Series(100.0, index=days)
    closes.iloc[220] = 10.0  # deep oversold dip once enough history exists
    aaa = pd.DataFrame({"Open": closes, "High": closes, "Low": closes, "Close": closes})
    spy = _flat(days, 100.0)
    sma_series = sma(closes, 200)
    first_valid = sma_series.first_valid_index()
    assert first_valid is not None
    assert (sma_series.index.get_loc(first_valid)) == 199  # 200th trading day, 0-indexed


# -- Slot ranking by RSI(2), alphabetic tie-break ------------------------------


def test_capacity_ranks_by_lowest_rsi2_first_ties_alphabetical():
    days = pd.bdate_range("2020-01-01", periods=210, freq="B")
    spy = _flat(days, 100.0)
    base = _uptrend(days)
    dip_idx = 205

    def dip_frame(pct):
        return _ohlc(_with_dip(base, dip_idx, pct=pct))

    universe = {
        "SPY": spy,
        "AAA": dip_frame(0.01),   # mildest dip -> highest RSI2 among candidates (~23)
        "BBB": dip_frame(0.05),   # deepest dip -> lowest RSI2, admitted first (~5.7)
        "CCC": dip_frame(0.05),   # exact RSI2 tie with BBB -> alphabetical: BBB before CCC
        "DDD": dip_frame(0.03),   # middle (~9)
    }
    result = simulate_etf_reversal_portfolio(
        universe, days[204].date(), days[206].date(), 100_000.0, COSTS["zero"],
        EtfReversalConfig(entry_threshold=90, exit_threshold=200, max_hold=1000), max_positions=2,
    )
    entries = sorted(
        (t for t in result["trades"] if t["entry_date"] == days[206].date()),
        key=lambda t: t["symbol"],
    )
    admitted_symbols = {t["symbol"] for t in entries}
    assert admitted_symbols == {"BBB", "CCC"}   # the two lowest (tied) RSI2 candidates, not AAA/DDD
    rejected_capacity = {r["symbol"] for r in result["rejected"] if r["reason"] == "insufficient_capacity"}
    assert rejected_capacity == {"AAA", "DDD"}


# -- Next-open fill, no lookahead ----------------------------------------------


def test_entry_fills_at_next_open_not_signal_day_close():
    days = pd.bdate_range("2020-01-01", periods=210, freq="B")
    base = _uptrend(days)
    closes = _with_dip(base, 205, pct=0.03)   # signal day close, RSI2~9
    signal_day_close = closes.iloc[205]
    opens = closes.copy()
    opens.iloc[206] = 777.0   # distinctive next-day open price, unrelated to the close series
    aaa = _ohlc(closes, open_=opens)
    spy = _flat(days, 100.0)
    result = simulate_etf_reversal_portfolio(
        {"SPY": spy, "AAA": aaa}, days[205].date(), days[207].date(), 100_000.0, COSTS["zero"],
        EtfReversalConfig(entry_threshold=90, exit_threshold=200, max_hold=1000), max_positions=5,
    )
    trade = [t for t in result["trades"] if t["symbol"] == "AAA"][0]
    assert trade["entry_price"] == pytest.approx(777.0)   # next day's OPEN, not the signal day's close
    assert trade["entry_price"] != pytest.approx(signal_day_close)
    assert trade["entry_date"] == days[206].date()


# -- Missing-bar asymmetry: entry dropped, exit persists -----------------------


def test_missing_next_day_bar_drops_a_queued_entry_permanently():
    days = pd.bdate_range("2020-01-01", periods=210, freq="B")
    base = _uptrend(days)
    closes = _with_dip(base, 205, pct=0.03)
    aaa_full = _ohlc(closes)
    # Remove AAA's row entirely for the fill day (day 206) - a genuine data gap.
    aaa = aaa_full.drop(aaa_full.index[206])
    spy = _flat(days, 100.0)
    result = simulate_etf_reversal_portfolio(
        {"SPY": spy, "AAA": aaa}, days[205].date(), days[209].date(), 100_000.0, COSTS["zero"],
        EtfReversalConfig(entry_threshold=90, exit_threshold=200, max_hold=1000), max_positions=5,
    )
    assert [t for t in result["trades"] if t["symbol"] == "AAA"] == []
    assert any(r["symbol"] == "AAA" and r["reason"] == "missing_bar_entry_fill" for r in result["rejected"])
    # And it is never retried on a later day either.
    assert not any(t["symbol"] == "AAA" for t in result["trades"])


def test_missing_next_day_bar_leaves_an_exit_pending_position_open_not_stuck():
    """A dropped row means both Open and Close are missing that day - and
    since utils.indicators.rsi()'s EWM recursion has no defined value once
    it hits a NaN gain/loss (pandas propagates NaN through .ewm(adjust=False)
    indefinitely from a single NaN input), RSI(2) for this symbol never
    becomes computable again after a genuine mid-series gap under the
    frozen, reused-unmodified rsi() implementation - confirmed directly
    against the real function, not assumed. The POSITION itself is still
    never stuck, per Section 4: it survives to the end-of-test forced close
    rather than being incorrectly force-closed by the data gap itself or
    left in an inconsistent state - which is exactly what this test checks."""
    days = pd.bdate_range("2020-01-01", periods=215, freq="B")
    base = _uptrend(days, low=50.0, high=250.0)
    closes = _with_dip(base, 205, pct=0.03)   # entry signal day, RSI2~9
    for j in range(206, len(days)):
        closes.iloc[j] = 5000.0  # deeply overbought right after entry -> immediate target_exit signal queued
    aaa_full = _ohlc(closes)
    # Remove the row for day 207 (the day the queued exit would have filled) -
    # a genuine gap right at the exit fill day.
    aaa = aaa_full.drop(aaa_full.index[207])
    spy = _flat(days, 100.0)
    result = simulate_etf_reversal_portfolio(
        {"SPY": spy, "AAA": aaa}, days[205].date(), days[212].date(), 100_000.0, COSTS["zero"],
        EtfReversalConfig(entry_threshold=90, exit_threshold=65, max_hold=1000), max_positions=5,
    )
    aaa_trades = [t for t in result["trades"] if t["symbol"] == "AAA"]
    assert len(aaa_trades) == 1                  # never stuck - closes exactly once
    assert aaa_trades[0]["exit_date"] > days[207].date()   # not force-closed AT the gap itself
    assert aaa_trades[0]["exit_reason"] == "end_of_test"   # RSI(2) never recovers after the gap (see docstring);
                                                            # max_hold=1000 rules out time_exit; SMA200 stays satisfied


# -- free_cash sizing -----------------------------------------------------------


def test_free_cash_sizing_caps_at_equity_over_five_and_at_available_cash():
    days = pd.bdate_range("2020-01-01", periods=210, freq="B")
    spy = _flat(days, 100.0)
    base = _uptrend(days)

    def dip_frame():
        return _ohlc(_with_dip(base, 205, pct=0.03))

    universe = {"SPY": spy, "AAA": dip_frame(), "BBB": dip_frame(), "CCC": dip_frame()}
    # Starting equity small enough that equity/5 exceeds what full cash allows
    # once several slots fill the same morning.
    result = simulate_etf_reversal_portfolio(
        universe, days[205].date(), days[206].date(), 1_000.0, COSTS["zero"],
        EtfReversalConfig(entry_threshold=90, exit_threshold=200, max_hold=1000), max_positions=5,
    )
    entries = [t for t in result["trades"]]
    assert len(entries) == 3
    total_notional = sum(t["entry_price"] * t["quantity"] for t in entries)
    assert total_notional <= 1_000.0 + 1e-6   # never sized past available cash


def test_free_cash_zero_or_negative_skips_admission_rather_than_negative_sizing():
    days = pd.bdate_range("2020-01-01", periods=210, freq="B")
    base = _uptrend(days)
    aaa = _ohlc(_with_dip(base, 205, pct=0.03))
    spy = _flat(days, 100.0)
    result = simulate_etf_reversal_portfolio(
        {"SPY": spy, "AAA": aaa}, days[205].date(), days[206].date(), 0.0, COSTS["zero"],
        EtfReversalConfig(entry_threshold=90, exit_threshold=200, max_hold=1000), max_positions=5,
    )
    assert result["trades"] == []
    assert any(r["reason"] in ("insufficient_free_cash", "insufficient_cash_or_quantity") for r in result["rejected"])


def test_same_day_exit_proceeds_fund_that_same_days_entries():
    """Exit-before-entry ordering: cash freed by a morning exit fill must be
    available to size that SAME morning's entry fills."""
    days = pd.bdate_range("2020-01-01", periods=215, freq="B")
    spy = _flat(days, 100.0)
    base = _uptrend(days, low=50.0, high=250.0)
    # AAA: entered earlier, exits (target_exit) exactly on the fill day we test.
    aaa_close = _with_dip(base, 200, pct=0.03)   # entry signal at index 200
    for j in range(201, len(days)):
        aaa_close.iloc[j] = 5000.0  # overbought right after -> queued target_exit, fills day 202
    aaa = _ohlc(aaa_close)
    # BBB: signals a fresh entry on day 201's close, so it competes for cash
    # at day 202's open - the SAME morning AAA's exit proceeds land.
    bbb_close = _with_dip(base, 201, pct=0.03)
    bbb = _ohlc(bbb_close)

    starting_equity = 1_000.0   # small enough that AAA's exit proceeds are needed to fund BBB
    result = simulate_etf_reversal_portfolio(
        {"SPY": spy, "AAA": aaa, "BBB": bbb}, days[200].date(), days[203].date(), starting_equity,
        COSTS["zero"], EtfReversalConfig(entry_threshold=90, exit_threshold=65, max_hold=1000),
        max_positions=5,
    )
    bbb_trades = [t for t in result["trades"] if t["symbol"] == "BBB"]
    assert len(bbb_trades) == 1
    assert bbb_trades[0]["entry_date"] == days[202].date()   # funded same morning as AAA's exit


# -- Trading-day-only equity series --------------------------------------------


def test_equity_series_has_no_weekend_or_holiday_rows():
    days = pd.bdate_range("2020-01-01", periods=30, freq="B")
    spy = _flat(days, 100.0)
    result = simulate_etf_reversal_portfolio(
        {"SPY": spy}, days[0].date(), days[-1].date(), 100_000.0, COSTS["zero"], EtfReversalConfig(),
    )
    equity_dates = {row["date"] for row in result["daily_equity"]}
    for d in equity_dates:
        assert pd.Timestamp(d).weekday() < 5   # every row is a real trading day
    assert len(result["daily_equity"]) == len(days)


# -- Cost-model selection --------------------------------------------------------


def test_cost_uses_stock_bps_per_leg():
    days = pd.bdate_range("2020-01-01", periods=210, freq="B")
    closes = pd.Series(100.0, index=days)
    closes.iloc[205] = 10.0
    aaa = pd.DataFrame({"Open": closes, "High": closes, "Low": closes, "Close": closes})
    spy = _flat(days, 100.0)
    baseline = COSTS["baseline"]
    result = simulate_etf_reversal_portfolio(
        {"SPY": spy, "AAA": aaa}, days[205].date(), days[207].date(), 100_000.0, baseline,
        EtfReversalConfig(entry_threshold=90, exit_threshold=200, max_hold=1000), max_positions=5,
    )
    trade = [t for t in result["trades"] if t["exit_reason"] == "end_of_test"][0]
    entry_notional = trade["entry_price"] * trade["quantity"]
    exit_notional = trade["exit_price"] * trade["quantity"]
    expected_cost = (entry_notional + exit_notional) * baseline.stock_bps_per_leg / 10_000.0
    assert trade["transaction_cost"] == pytest.approx(expected_cost)


# -- pnl_r, mode, qualify_strategy compatibility -------------------------------


def test_result_mode_is_stock_only_and_pnl_r_uses_entry_notional():
    days = pd.bdate_range("2020-01-01", periods=210, freq="B")
    closes = pd.Series(100.0, index=days)
    closes.iloc[205] = 10.0
    aaa = pd.DataFrame({"Open": closes, "High": closes, "Low": closes, "Close": closes})
    spy = _flat(days, 100.0)
    result = simulate_etf_reversal_portfolio(
        {"SPY": spy, "AAA": aaa}, days[205].date(), days[207].date(), 100_000.0, COSTS["zero"],
        EtfReversalConfig(entry_threshold=90, exit_threshold=200, max_hold=1000), max_positions=5,
    )
    assert result["mode"] == "stock_only"
    trade = [t for t in result["trades"] if t["symbol"] == "AAA"][0]
    entry_notional = trade["entry_price"] * trade["quantity"]
    assert trade["pnl_r"] == pytest.approx(trade["net_pnl"] / entry_notional)

    benchmark = {"symbol": "SPY", "sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0}
    summary = summarize_run(
        result, starting_equity=100_000.0, start_date=days[205].date(), end_date=days[207].date(),
        benchmark=benchmark,
    )
    qualify_strategy(summary, summary, 1.0)   # must not raise


# -- Deterministic repeated-run output -----------------------------------------


def test_repeated_run_is_byte_identical():
    days = pd.bdate_range("2020-01-01", periods=230, freq="B")
    rng = np.random.default_rng(11)
    spy = _flat(days, 100.0)

    def rand_frame():
        closes = pd.Series(100.0 + np.cumsum(rng.normal(0, 1.5, len(days))), index=days)
        return pd.DataFrame({"Open": closes, "High": closes, "Low": closes, "Close": closes})

    universe = {"SPY": spy, "AAA": rand_frame(), "BBB": rand_frame(), "CCC": rand_frame()}
    config = EtfReversalConfig(entry_threshold=15, exit_threshold=65, max_hold=5)
    first = simulate_etf_reversal_portfolio(
        universe, days[200].date(), days[-1].date(), 100_000.0, COSTS["baseline"], config,
    )
    second = simulate_etf_reversal_portfolio(
        universe, days[200].date(), days[-1].date(), 100_000.0, COSTS["baseline"], config,
    )
    assert first["trades"] == second["trades"]
    assert first["daily_equity"] == second["daily_equity"]


# -- Static prohibition on broker/order-client imports --------------------------


def test_no_broker_trading_client_import():
    for path in [Path("backtest/whole_bot_engine.py")]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                assert "alpaca.trading" not in name
                assert "execution_node" not in name
                assert "robin_stocks" not in name
