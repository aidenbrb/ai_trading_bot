"""Safety and accounting tests for the research-only SLC runner."""
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import backtest.run_slc_backtest as runner
from backtest.run_slc_backtest import (
    _rth, simulate_portfolio, verify_deployment_baseline, verify_rule_freeze,
)
from backtest.whole_bot_engine import COSTS, PORTFOLIOS
from utils.slc_signals import SlcSignal


def _signal(symbol, minute, direction="long"):
    entry = 10.0
    stop = 9.0 if direction == "long" else 11.0
    target = 12.0 if direction == "long" else 8.0
    timestamp = pd.Timestamp(f"2025-06-02 14:{minute:02d}")
    return SlcSignal(
        "slc_4h_5m_stock_v1", symbol, direction, f"{symbol}-level", "fresh",
        9.5, 10.5, timestamp - pd.Timedelta(minutes=20),
        timestamp - pd.Timedelta(minutes=5), timestamp, entry, stop, target,
        1.0, 2.0, 21.0, 20.0, 1.0,
        "uptrend" if direction == "long" else "downtrend", 1.5,
    )


def _outcome(signal, exit_price):
    gross = exit_price - signal.entry if signal.direction == "long" else signal.entry - exit_price
    return {
        "status": "closed", "exit_time": signal.entry_time + pd.Timedelta(minutes=10),
        "exit_price": exit_price, "exit_reason": "target", "ambiguous": False,
        "gross_per_share": gross, "gross_r": gross / signal.initial_risk,
    }


def _key(signal):
    return signal.symbol, signal.level_id, signal.entry_time.isoformat()


def test_frozen_documents_still_match_embedded_hashes():
    hashes = verify_rule_freeze()
    assert hashes["preregistration"].startswith("d453ae")
    assert hashes["amendment_001"].startswith("3f4617")


def test_actual_deployment_baseline_includes_parent_run_bot():
    result = verify_deployment_baseline()
    assert "../run_bot.bat" in result["guardrails"]
    assert result["baseline_type"] == "post-implementation_pre-full-run"


def test_deployment_baseline_detects_drift_before_run(tmp_path, monkeypatch):
    run_bot = tmp_path / "run_bot.bat"
    run_bot.write_text("frozen launcher", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        '{"baseline_type":"test","guardrails":{"../run_bot.bat":"'
        + runner._sha256(run_bot)
        + '"}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "GUARDRAILS", {"../run_bot.bat": run_bot})
    monkeypatch.setattr(runner, "DEPLOYMENT_BASELINE", baseline)
    # verify_deployment_baseline()'s success path computes
    # DEPLOYMENT_BASELINE.relative_to(REPO_ROOT) for the returned
    # baseline_path - tmp_path (pytest's per-test temp dir) is never under
    # the real REPO_ROOT, so that call raises ValueError unless REPO_ROOT
    # is also repointed here. REPO_ROOT is read only in that one line (not
    # used to build GUARDRAILS, which is separately monkeypatched above),
    # so this is safe and doesn't change what the test actually exercises.
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        runner, "EXPECTED_DEPLOYMENT_BASELINE_SHA256", runner._sha256(baseline)
    )
    assert verify_deployment_baseline()["guardrails"]["../run_bot.bat"]
    run_bot.write_text("changed before run", encoding="utf-8")
    with pytest.raises(RuntimeError, match="drift"):
        verify_deployment_baseline()


def test_deployment_baseline_detects_tampered_baseline(tmp_path, monkeypatch):
    run_bot = tmp_path / "run_bot.bat"
    run_bot.write_text("frozen launcher", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        '{"baseline_type":"test","guardrails":{"../run_bot.bat":"'
        + runner._sha256(run_bot)
        + '"}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "GUARDRAILS", {"../run_bot.bat": run_bot})
    monkeypatch.setattr(runner, "DEPLOYMENT_BASELINE", baseline)
    monkeypatch.setattr(runner, "EXPECTED_DEPLOYMENT_BASELINE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="baseline file hash mismatch"):
        verify_deployment_baseline()


def test_research_modules_contain_no_order_submission_client():
    root = Path(__file__).resolve().parents[1]
    files = [root / "utils" / "slc_signals.py", root / "backtest" / "slc_data.py",
             root / "backtest" / "run_slc_backtest.py"]
    forbidden = ("TradingClient", "submit_order", "alpaca.trading", "MarketOrderRequest")
    text = "\n".join(path.read_text(encoding="utf-8") for path in files)
    assert not any(token in text for token in forbidden)


def test_baseline_cost_is_five_bps_per_leg():
    signal = _signal("AAPL", 30)
    result = simulate_portfolio(
        [signal], {_key(signal): _outcome(signal, 12.0)},
        portfolio=PORTFOLIOS["safe_0_25pct"], cost=COSTS["baseline"],
    )
    trade = result["trades"][0]
    expected = trade["quantity"] * (10.0 + 12.0) * 5 / 10_000
    assert trade["transaction_cost"] == pytest.approx(expected)
    assert trade["net_pnl"] == pytest.approx(trade["gross_pnl"] - expected)


def test_short_pnl_uses_entry_minus_exit():
    signal = _signal("AAPL", 30, direction="short")
    result = simulate_portfolio(
        [signal], {_key(signal): _outcome(signal, 8.0)},
        portfolio=PORTFOLIOS["safe_0_25pct"], cost=COSTS["zero"],
    )
    assert result["trades"][0]["gross_pnl"] > 0


def test_only_two_new_trades_are_approved_per_day():
    signals = [_signal(symbol, 30) for symbol in ("AAPL", "MSFT", "NVDA")]
    outcomes = {_key(signal): _outcome(signal, 12.0) for signal in signals}
    result = simulate_portfolio(
        signals, outcomes, portfolio=PORTFOLIOS["safe_0_25pct"], cost=COSTS["zero"],
    )
    assert len(result["trades"]) == 2
    assert result["rejected"] == [{
        "date": date(2025, 6, 2), "symbol": "NVDA",
        "reason": "max_new_trades_per_day", "entry_time": pd.Timestamp("2025-06-02 14:30"),
    }]


def test_missing_outcome_remains_auditable_and_has_no_pnl():
    signal = _signal("AAPL", 30)
    result = simulate_portfolio(
        [signal], {}, portfolio=PORTFOLIOS["safe_0_25pct"], cost=COSTS["zero"],
    )
    trade = result["trades"][0]
    assert trade["status"] == "outcome_data_missing"
    assert trade["net_pnl"] is None
    assert len(result["missing_outcomes"]) == 1


def test_empty_cache_frame_fails_closed_without_timestamp_comparison_error():
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    result = _rth(empty, [date(2025, 6, 2)])
    assert result.empty
    assert isinstance(result.index, pd.DatetimeIndex)
