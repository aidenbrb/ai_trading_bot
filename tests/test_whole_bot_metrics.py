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


def test_qualification_requires_every_gate():
    baseline = {
        "closed_count": 100, "missing_outcome_rate": 0.0, "net_expectancy": 10.0,
        "profit_factor": 1.5, "sharpe": 1.5, "benchmark": {"sharpe": 0.8},
        "max_drawdown": -0.10, "recent_12m_net_pnl": 100.0,
        "positive_quarter_fraction": 0.75, "bootstrap_95pct_lower_mean_r": 0.1,
        "max_symbol_profit_contribution": 0.20,
    }
    stressed = {"net_expectancy": 1.0}
    assert qualify_strategy(baseline, stressed, 0.96)["passed"] is True
    baseline["recent_12m_net_pnl"] = -1.0
    result = qualify_strategy(baseline, stressed, 0.96)
    assert result["passed"] is False
    assert "recent_12m_positive" in result["failed_checks"]

