"""Required tests for etf_momentum_v1, per
research/etf_momentum_v1_preregistration.md Section 10."""
from __future__ import annotations

import ast
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import backtest.whole_bot_engine as engine
from backtest.whole_bot_engine import (
    COSTS,
    EtfMomentumConfig,
    _etf_month_end_series,
    _etf_rebalance_days,
    _etf_trailing_return,
    rank_etf_universe,
    resolve_etf_target_weights,
    simulate_etf_momentum_portfolio,
)
from backtest.whole_bot_metrics import qualify_strategy, summarize_run


def _daily_series(prices: dict[str, float]) -> pd.Series:
    """prices: {"YYYY-MM-DD": close}."""
    index = pd.to_datetime(sorted(prices.keys()))
    return pd.Series([prices[str(idx.date())] for idx in index], index=index)


def _monthly_flat_series(start: str, end: str, monthly_close: dict[str, float]) -> pd.Series:
    """One bar per calendar day at each month's first business day mapped
    from monthly_close (month 'YYYY-MM' -> close), forward-filled daily -
    enough for _etf_month_end_series() to pick up month-end values."""
    days = pd.bdate_range(start, end)
    out = {}
    last = None
    for day in days:
        key = f"{day.year:04d}-{day.month:02d}"
        if key in monthly_close:
            last = monthly_close[key]
        if last is not None:
            out[day] = last
    return pd.Series(out)


# -- Skip-month trailing-return calculation ----------------------------------


def test_trailing_return_skip_zero_uses_most_recently_completed_month():
    month_end = pd.Series(
        {pd.Period("2019-12", "M"): 100.0, pd.Period("2020-06", "M"): 100.0,
         pd.Period("2020-12", "M"): 150.0},
    )
    as_of = pd.Period("2021-01", "M")  # rebalance month; most recent completed = 2020-12
    result = _etf_trailing_return(month_end, as_of, lookback_months=12, skip_last_month=0)
    assert result == pytest.approx(150.0 / 100.0 - 1.0)


def test_trailing_return_skip_one_excludes_most_recent_completed_month():
    month_end = pd.Series({
        pd.Period("2019-11", "M"): 100.0,
        pd.Period("2020-11", "M"): 140.0,
        pd.Period("2020-12", "M"): 999.0,  # excluded by skip_last_month=1
    })
    as_of = pd.Period("2021-01", "M")
    result = _etf_trailing_return(month_end, as_of, lookback_months=12, skip_last_month=1)
    assert result == pytest.approx(140.0 / 100.0 - 1.0)


def test_trailing_return_none_when_start_bar_unavailable():
    month_end = pd.Series({pd.Period("2020-12", "M"): 150.0})
    as_of = pd.Period("2021-01", "M")
    assert _etf_trailing_return(month_end, as_of, lookback_months=12, skip_last_month=0) is None


def test_month_end_series_takes_last_close_of_each_month():
    daily = _daily_series({"2020-01-30": 10.0, "2020-01-31": 11.0, "2020-02-03": 12.0})
    month_end = _etf_month_end_series(daily)
    assert month_end[pd.Period("2020-01", "M")] == 11.0
    assert month_end[pd.Period("2020-02", "M")] == 12.0


# -- Absolute-momentum-to-BIL substitution, per-slot -------------------------


def test_slot_below_bil_return_substitutes_bil():
    ranked = [("AAA", 0.20), ("BBB", 0.01), ("CCC", -0.05)]
    weights = resolve_etf_target_weights(ranked, bil_return=0.02, config=EtfMomentumConfig(top_n=3))
    # AAA beats BIL -> stays; BBB and CCC are below BIL's 0.02 -> both go to BIL
    assert weights == {"AAA": pytest.approx(1 / 3), "BIL": pytest.approx(2 / 3)}


def test_bil_itself_can_win_a_slot_on_its_own_merit():
    ranked = [("BIL", 0.05), ("AAA", 0.03)]
    weights = resolve_etf_target_weights(ranked, bil_return=0.05, config=EtfMomentumConfig(top_n=2))
    # BIL ranks #1 on its own return; AAA (0.03) is below BIL's own 0.05 -> also substituted to BIL
    assert weights == {"BIL": pytest.approx(1.0)}


def test_missing_rankable_candidates_fill_remaining_slots_with_bil():
    ranked = [("AAA", 0.10)]  # only 1 rankable candidate but top_n=3
    weights = resolve_etf_target_weights(ranked, bil_return=0.02, config=EtfMomentumConfig(top_n=3))
    assert weights == {"AAA": pytest.approx(1 / 3), "BIL": pytest.approx(2 / 3)}


def test_rank_etf_universe_includes_bil_as_a_regular_candidate():
    adjusted_close = {
        "BIL": _monthly_flat_series("2020-01-01", "2021-02-01", {
            "2020-01": 100.0, "2020-06": 101.0, "2021-01": 105.0,
        }),
        "SPY": _monthly_flat_series("2020-01-01", "2021-02-01", {
            "2020-01": 100.0, "2020-06": 90.0, "2021-01": 95.0,
        }),
    }
    ranked, bil_return = rank_etf_universe(
        adjusted_close, pd.Period("2021-02", "M"), EtfMomentumConfig(lookback_months=12, skip_last_month=0),
    )
    symbols = [s for s, _ in ranked]
    assert "BIL" in symbols
    assert bil_return == pytest.approx(105.0 / 100.0 - 1.0)


# -- Late-starting-member exclusion -------------------------------------------


def test_symbol_with_insufficient_history_excluded_from_ranking():
    adjusted_close = {
        "OLD": _monthly_flat_series("2019-01-01", "2021-02-01", {
            "2019-01": 100.0, "2020-01": 110.0, "2021-01": 120.0,
        }),
        "NEW": _monthly_flat_series("2020-06-01", "2021-02-01", {  # started too recently
            "2020-06": 50.0, "2021-01": 55.0,
        }),
        "BIL": _monthly_flat_series("2019-01-01", "2021-02-01", {"2019-01": 100.0, "2021-01": 101.0}),
    }
    ranked, _ = rank_etf_universe(
        adjusted_close, pd.Period("2021-02", "M"), EtfMomentumConfig(lookback_months=12, skip_last_month=0),
    )
    symbols = [s for s, _ in ranked]
    assert "OLD" in symbols
    assert "NEW" not in symbols  # not enough history for a 12-month lookback yet


def test_xlre_and_xlc_excluded_before_their_real_inception_using_synthetic_dates():
    """Mirrors the real XLRE (2015-10-08) / XLC (2018-06-19) late starts
    without depending on the live snapshot - a synthetic case with the same
    shape (one very-late starter, one merely-late starter)."""
    adjusted_close = {
        "SEASONED": _monthly_flat_series("2007-01-01", "2010-01-01", {"2007-01": 50.0, "2009-12": 60.0}),
        "LATE_2009": _monthly_flat_series("2009-06-01", "2010-01-01", {"2009-06": 20.0, "2009-12": 22.0}),
        "BIL": _monthly_flat_series("2007-01-01", "2010-01-01", {"2007-01": 100.0, "2009-12": 101.0}),
    }
    ranked, _ = rank_etf_universe(
        adjusted_close, pd.Period("2010-01", "M"), EtfMomentumConfig(lookback_months=12, skip_last_month=0),
    )
    symbols = [s for s, _ in ranked]
    assert "SEASONED" in symbols
    assert "LATE_2009" not in symbols  # only ~7 months of history, needs 12


# -- Equal-weight sizing arithmetic -------------------------------------------


def test_equal_weight_sizing_splits_equity_evenly_across_top_n():
    adjusted_close = {
        "SPY": _monthly_flat_series("2019-01-01", "2020-03-01", {"2019-01": 100.0, "2020-02": 100.0}),
        "BIL": _monthly_flat_series("2019-01-01", "2020-03-01", {"2019-01": 100.0, "2020-02": 100.0}),
        "AAA": _monthly_flat_series("2019-01-01", "2020-03-01", {"2019-01": 10.0, "2020-01": 20.0, "2020-02": 20.0}),
        "BBB": _monthly_flat_series("2019-01-01", "2020-03-01", {"2019-01": 10.0, "2020-01": 15.0, "2020-02": 15.0}),
        "CCC": _monthly_flat_series("2019-01-01", "2020-03-01", {"2019-01": 10.0, "2020-01": 12.0, "2020-02": 12.0}),
    }
    result = simulate_etf_momentum_portfolio(
        adjusted_close, date(2020, 2, 3), date(2020, 2, 3), 90_000.0, COSTS["zero"],
        EtfMomentumConfig(lookback_months=1, skip_last_month=0, top_n=3),
    )
    entries = [t for t in result["trades"] if t["exit_reason"] == "end_of_test"]
    assert len(entries) == 3
    for t in entries:
        assert t["entry_price"] * t["quantity"] == pytest.approx(30_000.0, rel=1e-6)


# -- No-stop position lifecycle -----------------------------------------------


def test_position_survives_intramonth_crash_with_no_stop():
    """A 90% intra-month drawdown must not close the position early -
    there is no stop on this engine (preregistration Section 1, item 6).
    Run over a single month (start_date == the month's rebalance day, end_date
    the day before the NEXT rebalance) so the only closing event possible is
    the forced end-of-test close, never an early stop."""
    days = pd.bdate_range("2019-12-01", "2020-02-28")
    prices = []
    for d in days:
        if d < pd.Timestamp("2020-01-02"):
            prices.append(100.0)
        elif d < pd.Timestamp("2020-02-04"):
            prices.append(110.0)  # AAA up 10% in January (incl. month-end 1/31); still 110 on the Feb 3 entry day
        elif d < pd.Timestamp("2020-02-18"):
            prices.append(11.0)  # catastrophic mid-February plunge (-90%), AFTER entry
        else:
            prices.append(99.0)  # partial recovery by month end
    aaa = pd.Series(prices, index=days)
    adjusted_close = {
        "SPY": pd.Series(100.0, index=days),
        "BIL": pd.Series(100.0, index=days),
        "AAA": aaa,
    }
    result = simulate_etf_momentum_portfolio(
        adjusted_close, date(2020, 2, 3), date(2020, 2, 28), 100_000.0, COSTS["zero"],
        EtfMomentumConfig(lookback_months=1, skip_last_month=0, top_n=1),
    )
    aaa_trades = [t for t in result["trades"] if t["symbol"] == "AAA"]
    assert len(aaa_trades) == 1
    trade = aaa_trades[0]
    assert trade["exit_date"] == date(2020, 2, 28)
    assert trade["exit_reason"] == "end_of_test"
    assert trade["entry_price"] == pytest.approx(110.0)
    assert trade["exit_price"] == pytest.approx(99.0)


# -- pnl_r resolution ----------------------------------------------------------


def test_pnl_r_is_net_pnl_over_entry_notional_not_a_stop_distance():
    days = pd.bdate_range("2020-01-01", "2020-03-02")
    aaa = pd.Series([100.0 if d < pd.Timestamp("2020-02-03") else 110.0 for d in days], index=days)
    adjusted_close = {"SPY": pd.Series(100.0, index=days), "BIL": pd.Series(100.0, index=days), "AAA": aaa}
    result = simulate_etf_momentum_portfolio(
        adjusted_close, date(2020, 2, 3), date(2020, 3, 2), 100_000.0, COSTS["zero"],
        EtfMomentumConfig(lookback_months=1, skip_last_month=0, top_n=1),
    )
    trade = [t for t in result["trades"] if t["symbol"] == "AAA"][0]
    entry_notional = trade["entry_price"] * trade["quantity"]
    assert trade["pnl_r"] == pytest.approx(trade["net_pnl"] / entry_notional)
    assert "stop" not in trade  # no stop field exists on this engine's trade dict at all


def test_result_mode_is_stock_only_so_sharpe_uses_252_trading_day_annualization():
    """Regression test: simulate_etf_momentum_portfolio() must report
    mode="stock_only" so summarize_run() annualizes with periods=252
    (trading days), not its periods=365 default (calendar days, meant for
    crypto). This was a real bug found after the first qualification run -
    it inflated every reported Sharpe/Sortino by sqrt(365/252) =~ 1.204x."""
    days = pd.bdate_range("2020-01-01", "2020-06-01")
    aaa = pd.Series([100.0 if d < pd.Timestamp("2020-02-03") else 110.0 for d in days], index=days)
    adjusted_close = {"SPY": pd.Series(100.0, index=days), "BIL": pd.Series(100.0, index=days), "AAA": aaa}
    result = simulate_etf_momentum_portfolio(
        adjusted_close, date(2020, 2, 3), date(2020, 6, 1), 100_000.0, COSTS["zero"],
        EtfMomentumConfig(lookback_months=1, skip_last_month=0, top_n=1),
    )
    assert result["mode"] == "stock_only"

    benchmark = {"symbol": "SPY", "sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0}
    summary = summarize_run(
        result, starting_equity=100_000.0, start_date=date(2020, 2, 3), end_date=date(2020, 6, 1),
        benchmark=benchmark,
    )
    # Cross-check: recompute Sharpe directly from the same equity curve with
    # an explicit periods=252 and confirm it matches summarize_run()'s own
    # value - proving the annualization constant it actually used was 252,
    # not the 365 that produced the original bug.
    from backtest.whole_bot_metrics import _sharpe
    equity = pd.Series(
        [row["equity"] for row in result["daily_equity"]],
        index=pd.to_datetime([row["date"] for row in result["daily_equity"]]),
    )
    returns = equity.pct_change().fillna(0.0)
    expected_252 = _sharpe(returns.to_numpy(), 252)
    assert summary["sharpe"] == pytest.approx(expected_252)


def test_summarize_run_and_qualify_strategy_are_denominator_agnostic_for_pnl_r():
    """Nothing in the shared metrics path assumes pnl_r came from a
    stop-distance - it must consume the entry-notional-based pnl_r this
    engine emits with no special-casing."""
    days = pd.bdate_range("2020-01-01", "2020-06-01")
    aaa = pd.Series([100.0 if d < pd.Timestamp("2020-02-03") else 120.0 for d in days], index=days)
    adjusted_close = {"SPY": pd.Series(100.0, index=days), "BIL": pd.Series(100.0, index=days), "AAA": aaa}
    result = simulate_etf_momentum_portfolio(
        adjusted_close, date(2020, 2, 3), date(2020, 6, 1), 100_000.0, COSTS["zero"],
        EtfMomentumConfig(lookback_months=1, skip_last_month=0, top_n=1),
    )
    benchmark = {"symbol": "SPY", "sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0}
    summary = summarize_run(
        result, starting_equity=100_000.0, start_date=date(2020, 2, 3), end_date=date(2020, 6, 1),
        benchmark=benchmark,
    )
    assert summary["average_pnl_r"] is not None
    # qualify_strategy must not raise or special-case a missing "stop"/"risk_amount" field
    qualify_strategy(summary, summary, 1.0)


# -- Monthly first-trading-day timing -----------------------------------------


def test_rebalance_days_are_first_trading_day_of_each_month():
    index = pd.bdate_range("2021-01-01", "2021-03-31")  # business days, no holiday calendar
    days = _etf_rebalance_days(index, date(2021, 1, 1), date(2021, 3, 31))
    assert days == [date(2021, 1, 1), date(2021, 2, 1), date(2021, 3, 1)]


def test_rebalance_day_skips_a_month_start_holiday():
    """2021-01-01 was a Friday but the exchange was closed for New Year's
    Day (observed) - the reference index (built to reflect real trading
    days, not just weekdays) must land the rebalance on the next actual
    trading day, 2021-01-04."""
    index = pd.DatetimeIndex(
        [d for d in pd.bdate_range("2021-01-01", "2021-01-10") if d != pd.Timestamp("2021-01-01")]
    )
    days = _etf_rebalance_days(index, date(2021, 1, 1), date(2021, 1, 10))
    assert days == [date(2021, 1, 4)]


# -- Cost-model selection -------------------------------------------------------


def test_cost_uses_stock_bps_per_leg_not_crypto():
    days = pd.bdate_range("2020-01-01", "2020-03-02")
    aaa = pd.Series([100.0 if d < pd.Timestamp("2020-02-03") else 100.0 for d in days], index=days)
    adjusted_close = {"SPY": pd.Series(100.0, index=days), "BIL": pd.Series(100.0, index=days), "AAA": aaa}
    baseline = COSTS["baseline"]
    assert baseline.stock_bps_per_leg == 5.0 and baseline.crypto_bps_per_leg == 35.0
    result = simulate_etf_momentum_portfolio(
        adjusted_close, date(2020, 2, 3), date(2020, 3, 2), 100_000.0, baseline,
        EtfMomentumConfig(lookback_months=1, skip_last_month=0, top_n=1),
    )
    trade = [t for t in result["trades"] if t["symbol"] == "AAA"][0]
    entry_notional = trade["entry_price"] * trade["quantity"]
    exit_notional = trade["exit_price"] * trade["quantity"]
    expected_cost = (entry_notional + exit_notional) * baseline.stock_bps_per_leg / 10_000.0
    assert trade["transaction_cost"] == pytest.approx(expected_cost)


# -- Snapshot: total-return canary (BIL, TLT, XLU vs published distributions) -


def test_snapshot_total_return_canary_bil_tlt_xlu():
    from backtest.etf_momentum_snapshot import load_snapshot

    snapshot = load_snapshot()
    for ticker in ("BIL", "TLT", "XLU"):
        raw = snapshot[ticker]["raw"]
        adjusted = snapshot[ticker]["adjusted"]
        raw_return = raw["Close"].iloc[-1] / raw["Close"].iloc[0] - 1.0
        adjusted_return = adjusted["Close"].iloc[-1] / adjusted["Close"].iloc[0] - 1.0

        # Compound each distribution as a (div / prior_close) return factor -
        # the correct reconciliation (a naive sum-of-dividends/avg-price
        # approximation under-states the gap for a ticker with meaningful
        # price appreciation, since the adjustment factors compound
        # multiplicatively with the price return, not additively).
        distributions = raw["Dividends"]
        div_dates = distributions[distributions > 0]
        factor = 1.0
        for ts, amount in div_dates.items():
            prior = raw.loc[:ts, "Close"]
            prior_close = prior.iloc[-2] if len(prior) > 1 else raw["Close"].iloc[0]
            factor *= 1.0 + amount / prior_close
        implied_total_return = (1 + raw_return) * factor - 1.0

        assert len(div_dates) > 0, f"{ticker}: no distributions found - adjustment cannot be canaried"
        # The adjusted series must be a genuinely different, materially
        # larger series than the raw close (the whole point of the canary).
        assert adjusted_return > raw_return + 0.01, (
            f"{ticker}: adjusted return ({adjusted_return}) is not materially "
            f"above raw return ({raw_return}) - dividend adjustment may not be applied"
        )
        # And it must reconcile with the distribution-implied total return
        # to within 5% relative (compounding/reinvestment-timing tolerance,
        # not exact-to-the-cent).
        assert adjusted_return == pytest.approx(implied_total_return, rel=0.05)


def test_late_starters_match_documented_inception_in_snapshot():
    from backtest.etf_momentum_snapshot import load_snapshot

    snapshot = load_snapshot()
    xlre_first = snapshot["XLRE"]["raw"].index.min().date()
    xlc_first = snapshot["XLC"]["raw"].index.min().date()
    assert xlre_first == date(2015, 10, 8)
    assert xlc_first == date(2018, 6, 19)


# -- No live yfinance calls outside the snapshot loader -----------------------


def test_no_module_outside_snapshot_calls_yfinance_at_runtime():
    """Static check: only backtest/etf_momentum_snapshot.py may import
    yfinance among the etf_momentum_v1 modules."""
    for path in [
        Path("backtest/whole_bot_engine.py"),
        Path("backtest/whole_bot_metrics.py"),
    ]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names.add(node.module)
        assert not any("yfinance" in name for name in imported_names), (
            f"{path} must not import yfinance directly - only "
            "backtest/etf_momentum_snapshot.py may (preregistration Section 4)"
        )


# -- Deterministic repeated-run output ----------------------------------------


def test_repeated_run_is_byte_identical():
    days = pd.bdate_range("2020-01-01", "2020-06-01")
    rng = np.random.default_rng(7)
    aaa = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, len(days))), index=days)
    bbb = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, len(days))), index=days)
    adjusted_close = {
        "SPY": pd.Series(100.0, index=days), "BIL": pd.Series(100.0, index=days),
        "AAA": aaa, "BBB": bbb,
    }
    config = EtfMomentumConfig(lookback_months=1, skip_last_month=0, top_n=2)
    first = simulate_etf_momentum_portfolio(
        adjusted_close, date(2020, 2, 3), date(2020, 6, 1), 100_000.0, COSTS["baseline"], config,
    )
    second = simulate_etf_momentum_portfolio(
        adjusted_close, date(2020, 2, 3), date(2020, 6, 1), 100_000.0, COSTS["baseline"], config,
    )
    assert first["trades"] == second["trades"]
    assert first["daily_equity"] == second["daily_equity"]


# -- Static prohibition on broker/order-client imports ------------------------


def test_no_broker_trading_client_import():
    for path in [
        Path("backtest/whole_bot_engine.py"),
        Path("backtest/etf_momentum_snapshot.py"),
    ]:
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
