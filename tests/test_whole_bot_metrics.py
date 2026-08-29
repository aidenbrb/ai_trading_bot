from datetime import date, datetime

from backtest.whole_bot_metrics import qualify_strategy, summarize_run


def _trade(day, pnl, symbol="AAPL"):
    return {
        "status": "closed", "net_pnl": pnl, "pnl_r": pnl / 100.0,
        "gross_pnl": pnl + 2.0, "quantity": 1.0, "fill_price": 100.0,
        "exit_price": 100.0 + pnl, "exit_time": datetime.combine(day, datetime.min.time()),
        "symbol": symbol, "mode": "stock_only", "portfolio": "current_1pct",
        "cost_model": "baseline", "ambiguous": False,
    }


def test_summary_is_deterministic_and_reports_break_even_cost():
    trades = [_trade(date(2025, 1, 2), 10), _trade(date(2025, 4, 2), -5)]
    result = {"trades": trades, "missing_outcomes": [], "daily_equity": []}
    benchmark = {"symbol": "SPY", "sharpe": 0.1, "total_return": 0.1, "max_drawdown": -0.2}
    first = summarize_run(
        result, starting_equity=100_000, start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 31), benchmark=benchmark,
    )
    second = summarize_run(
        result, starting_equity=100_000, start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 31), benchmark=benchmark,
    )
    assert first["bootstrap_95pct_lower_mean_r"] == second["bootstrap_95pct_lower_mean_r"]
    assert first["profit_factor"] == 2.0
    assert first["break_even_cost_bps_per_leg_pool"] is not None


def _qualifying_baseline():
    return {
        "closed_count": 100, "missing_outcome_rate": 0.0, "net_expectancy": 10.0,
        "profit_factor": 1.5, "sharpe": 1.5, "benchmark": {"sharpe": 0.8},
        "max_drawdown": -0.10, "recent_12m_net_pnl": 100.0,
        "positive_quarter_fraction": 0.75, "bootstrap_95pct_lower_mean_r": 0.1,
        "max_symbol_profit_contribution": 0.20,
    }


def test_qualification_requires_every_gate():
    baseline = _qualifying_baseline()
    stressed = {"net_expectancy": 1.0}
    assert qualify_strategy(baseline, stressed, 0.96)["passed"] is True
    baseline["recent_12m_net_pnl"] = -1.0
    result = qualify_strategy(baseline, stressed, 0.96)
    assert result["passed"] is False
    assert "recent_12m_positive" in result["failed_checks"]


# -- Phase 5 Step 4: walk-forward OOS + sensitivity plateau checks --------


def test_omitting_walk_forward_and_sensitivity_reproduces_the_original_13_checks():
    """Default call shape (every pre-existing call site) must be byte-for-byte
    unaffected - none of the 3 new checks appear at all when the new
    parameters are omitted."""
    baseline = _qualifying_baseline()
    stressed = {"net_expectancy": 1.0}
    result = qualify_strategy(baseline, stressed, 0.96)
    assert len(result["checks"]) == 13
    assert "walk_forward_oos_sharpe_beats_benchmark" not in result["checks"]
    assert "walk_forward_oos_profit_factor_at_least_1_0" not in result["checks"]
    assert "sensitivity_plateau_within_25pct_of_neighbor_median" not in result["checks"]


def _walk_forward(oos_closed_trades=30, oos_sharpe=0.5, oos_benchmark_sharpe=0.3, oos_profit_factor=1.2):
    return {
        "oos_closed_trades": oos_closed_trades, "oos_sharpe": oos_sharpe,
        "oos_benchmark_sharpe": oos_benchmark_sharpe, "oos_profit_factor": oos_profit_factor,
    }


def test_walk_forward_checks_pass_with_enough_trades_and_good_metrics():
    baseline = _qualifying_baseline()
    stressed = {"net_expectancy": 1.0}
    result = qualify_strategy(baseline, stressed, 0.96, walk_forward=_walk_forward())
    assert result["checks"]["walk_forward_oos_sharpe_beats_benchmark"] is True
    assert result["checks"]["walk_forward_oos_profit_factor_at_least_1_0"] is True
    assert result["passed"] is True


def test_walk_forward_checks_fail_below_30_oos_trades_even_with_good_metrics():
    """Phase 5 Step 4 amendment (b): a nominal OOS pass on too few trades
    must not count - matches Phase 3's xsec finding where a headline OOS
    Sharpe pass was a single-period artifact, not a real result."""
    baseline = _qualifying_baseline()
    stressed = {"net_expectancy": 1.0}
    walk_forward = _walk_forward(oos_closed_trades=29, oos_sharpe=5.0, oos_profit_factor=5.0)
    result = qualify_strategy(baseline, stressed, 0.96, walk_forward=walk_forward)
    assert result["checks"]["walk_forward_oos_sharpe_beats_benchmark"] is False
    assert result["checks"]["walk_forward_oos_profit_factor_at_least_1_0"] is False
    assert result["passed"] is False


def test_walk_forward_sharpe_check_fails_when_oos_sharpe_below_benchmark():
    baseline = _qualifying_baseline()
    stressed = {"net_expectancy": 1.0}
    walk_forward = _walk_forward(oos_sharpe=0.1, oos_benchmark_sharpe=0.3)
    result = qualify_strategy(baseline, stressed, 0.96, walk_forward=walk_forward)
    assert result["checks"]["walk_forward_oos_sharpe_beats_benchmark"] is False
    assert result["checks"]["walk_forward_oos_profit_factor_at_least_1_0"] is True


def test_walk_forward_profit_factor_check_fails_below_1_0():
    baseline = _qualifying_baseline()
    stressed = {"net_expectancy": 1.0}
    walk_forward = _walk_forward(oos_profit_factor=0.9)
    result = qualify_strategy(baseline, stressed, 0.96, walk_forward=walk_forward)
    assert result["checks"]["walk_forward_oos_profit_factor_at_least_1_0"] is False
    assert result["checks"]["walk_forward_oos_sharpe_beats_benchmark"] is True


def _neighbor(sharpe=0.4, closed_trades=30, in_grid=True):
    return {"in_grid": in_grid, "sharpe": sharpe, "closed_trades": closed_trades}


def test_sensitivity_plateau_passes_with_four_solid_neighbors_near_selected():
    baseline = _qualifying_baseline()
    stressed = {"net_expectancy": 1.0}
    sensitivity = {
        "selected_sharpe": 0.5,
        "neighbors": [_neighbor(sharpe=0.4) for _ in range(4)],  # median 0.4 >= 0.75*0.5=0.375
    }
    result = qualify_strategy(baseline, stressed, 0.96, sensitivity=sensitivity)
    assert result["checks"]["sensitivity_plateau_within_25pct_of_neighbor_median"] is True


def test_sensitivity_plateau_fails_with_fewer_than_four_evaluable_neighbors():
    """Amendment (a): out-of-grid neighbors (an edge/corner selected cell)
    are excluded from the evaluable count, not padded to reach 4."""
    baseline = _qualifying_baseline()
    stressed = {"net_expectancy": 1.0}
    sensitivity = {
        "selected_sharpe": 0.5,
        "neighbors": [_neighbor(sharpe=0.5)] * 3 + [_neighbor(in_grid=False)],
    }
    result = qualify_strategy(baseline, stressed, 0.96, sensitivity=sensitivity)
    assert result["checks"]["sensitivity_plateau_within_25pct_of_neighbor_median"] is False


def test_sensitivity_plateau_fails_when_a_neighbor_has_too_few_trades():
    """Amendment (a): a thin neighbor (<30 trades) fails the whole check
    outright rather than being dropped as 'missing' - it still counts
    toward the evaluable-neighbor total, but its unreliable Sharpe can't
    be laundered into a passing median."""
    baseline = _qualifying_baseline()
    stressed = {"net_expectancy": 1.0}
    sensitivity = {
        "selected_sharpe": 0.5,
        "neighbors": [_neighbor(sharpe=0.9)] * 3 + [_neighbor(sharpe=0.9, closed_trades=5)],
    }
    result = qualify_strategy(baseline, stressed, 0.96, sensitivity=sensitivity)
    assert result["checks"]["sensitivity_plateau_within_25pct_of_neighbor_median"] is False


def test_sensitivity_plateau_fails_when_neighbor_median_too_far_below_selected():
    baseline = _qualifying_baseline()
    stressed = {"net_expectancy": 1.0}
    sensitivity = {
        "selected_sharpe": 1.0,
        "neighbors": [_neighbor(sharpe=0.1) for _ in range(4)],  # 0.1 < 0.75*1.0
    }
    result = qualify_strategy(baseline, stressed, 0.96, sensitivity=sensitivity)
    assert result["checks"]["sensitivity_plateau_within_25pct_of_neighbor_median"] is False

