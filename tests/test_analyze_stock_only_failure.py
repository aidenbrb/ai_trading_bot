import json

import pandas as pd
import pytest

import backtest.analyze_stock_only_failure as analyze_mod
from backtest.analyze_stock_only_failure import (
    build_manifest,
    compute_trade_excursion,
    entry_cohort_performance_by_period,
    realized_performance_by_period,
    reserved_capital_ledger,
    run_analysis,
)


# -- 1. Exit-time vs. decision-time attribution --------------------------

def test_exit_vs_decision_attribution_are_independent():
    closed = pd.DataFrame([{
        "decision_time": pd.Timestamp("2023-11-01"),
        "exit_time": pd.Timestamp("2024-01-15"),
        "net_pnl": 100.0,
    }])
    realized = realized_performance_by_period(closed)
    entry_cohort = entry_cohort_performance_by_period(closed)
    assert str(realized.iloc[0]["period"]) == "2024Q1"
    assert str(entry_cohort.iloc[0]["period"]) == "2023Q4"


# -- 2. Reserved-capital ledger, including same-timestamp tie-breaking ----

def test_reserved_capital_ledger_releases_before_reserving_at_same_timestamp():
    shared_ts = pd.Timestamp("2023-06-01 15:16:00")
    trades = pd.DataFrame([
        {"decision_time": pd.Timestamp("2023-05-01 15:16:00"), "exit_time": shared_ts,
         "reserved_notional": 10_000.0, "symbol": "AAA"},
        {"decision_time": shared_ts, "exit_time": pd.NaT,
         "reserved_notional": 5_000.0, "symbol": "BBB"},
    ])
    ledger = reserved_capital_ledger(trades)

    at_shared_ts = ledger[ledger["timestamp"] == shared_ts]
    assert list(at_shared_ts["event"]) == ["release", "reserve"]
    assert list(ledger["reserved_total"]) == [10_000.0, 0.0, 5_000.0]
    # If reserve were processed before release, this would transiently hit 15,000.
    assert ledger["reserved_total"].max() == 10_000.0


def test_reserved_capital_ledger_never_released_for_open_position():
    trades = pd.DataFrame([
        {"decision_time": pd.Timestamp("2023-05-01 15:16:00"), "exit_time": pd.NaT,
         "reserved_notional": 18_000.0, "symbol": "LLY"},
    ])
    ledger = reserved_capital_ledger(trades)
    assert len(ledger) == 1
    assert ledger.iloc[0]["event"] == "reserve"
    assert ledger.iloc[0]["reserved_total"] == 18_000.0


# -- 3. MFE/MAE, general correctness with a known price path -------------

def _stock_trade(filled_at, exit_time, entry=100.0, stop=95.0, fill_price=100.0):
    return pd.Series({
        "symbol": "AAPL", "market": "stock", "entry": entry, "stop": stop,
        "fill_price": fill_price, "filled_at": filled_at, "exit_time": exit_time,
    })


def test_mfe_mae_known_price_path(monkeypatch):
    trade = _stock_trade(
        pd.Timestamp("2026-06-01 13:31:00"), pd.Timestamp("2026-06-01 13:35:00"),
    )
    bars = pd.DataFrame({
        "open": [100, 103, 108, 96, 102],
        "high": [101, 104, 110, 99, 103],   # max favorable 110 -> (110-100)/5 = 2.0R
        "low": [99, 102, 106, 94, 101],     # max adverse 94 -> (100-94)/5 = 1.2R
        "close": [100.5, 103.5, 109, 97, 102.5],
        "volume": [1000] * 5,
    }, index=pd.date_range("2026-06-01 13:31", periods=5, freq="min"))

    monkeypatch.setattr(analyze_mod, "read_cached_bars_or_none", lambda *a, **k: bars)
    monkeypatch.setattr(analyze_mod, "_stock_regular_bars", lambda b: b)

    result = compute_trade_excursion(trade)
    assert result.missing is False
    assert result.mfe_r == pytest.approx(2.0)
    assert result.mae_r == pytest.approx(1.2)


# -- 4. Exit-bar inclusion: the corrected bug, proven with a real repro --

def test_mfe_regression_exit_bar_must_be_included(monkeypatch):
    filled_at = pd.Timestamp("2026-06-01 13:31:00")
    exit_time = pd.Timestamp("2026-06-01 13:33:00")
    trade = _stock_trade(filled_at, exit_time)

    # The true maximum favorable excursion is on the EXIT bar itself
    # (13:33, high=120) - exactly the scenario a [filled_at, exit_time)
    # exclusive-of-exit-bar query would miss.
    all_bars = pd.DataFrame({
        "open": [100, 101, 108],
        "high": [101, 102, 120],
        "low": [99, 100, 107],
        "close": [100.5, 101.5, 119],
        "volume": [1000] * 3,
    }, index=pd.date_range("2026-06-01 13:31", periods=3, freq="min"))

    monkeypatch.setattr(analyze_mod, "_stock_regular_bars", lambda b: b)

    def fake_reader(symbol, interval, start, end):
        return all_bars[(all_bars.index >= start) & (all_bars.index < end)]

    monkeypatch.setattr(analyze_mod, "read_cached_bars_or_none", fake_reader)

    result = compute_trade_excursion(trade)
    assert result.mfe_r == pytest.approx(4.0), (
        "true MFE is (120-100)/5=4.0R; the exit bar must be included"
    )

    # Prove this is actually testing the fix: the OLD buggy window
    # [filled_at, exit_time) reads only the first 2 bars and drastically
    # understates MFE.
    buggy_bars = all_bars[(all_bars.index >= filled_at) & (all_bars.index < exit_time)]
    buggy_mfe = (buggy_bars["high"].max() - trade["fill_price"]) / (trade["entry"] - trade["stop"])
    assert buggy_mfe == pytest.approx(0.4)
    assert buggy_mfe != pytest.approx(result.mfe_r)


# -- 5. Cache-miss safety: mfe_data_missing, zero network calls ----------

def test_cache_miss_yields_missing_flag_with_no_network_call(monkeypatch):
    import utils.alpaca_bars as alpaca_bars

    def explode(*args, **kwargs):
        raise AssertionError("network call attempted - cache-only guarantee violated")

    monkeypatch.setattr(alpaca_bars, "fetch_bars", explode)
    monkeypatch.setattr(alpaca_bars, "fetch_crypto_bars", explode)
    # Simulate exactly what read_cached_bars_or_none returns on a real
    # cache miss (incomplete coverage) - None, never a fetch.
    monkeypatch.setattr(analyze_mod, "read_cached_bars_or_none", lambda *a, **k: None)

    trade = _stock_trade(
        pd.Timestamp("2026-06-01 13:31:00"), pd.Timestamp("2026-06-01 13:33:00"),
    )
    result = compute_trade_excursion(trade)
    assert result.missing is True
    assert result.mfe_r is None
    assert result.mae_r is None


# -- 6. Reconciliation with the source summary.json -----------------------

def _write_synthetic_run(run_dir, *, baseline_total, expectancy):
    run_dir.mkdir(parents=True, exist_ok=True)
    trades_df = pd.DataFrame([
        {"mode": "stock_only", "portfolio": "current_1pct", "cost_model": "baseline",
         "status": "closed", "symbol": "AAA", "net_pnl": 100.0, "market": "stock",
         "decision_time": "2023-01-01T15:16:00", "exit_time": "2023-01-02T15:16:00",
         "filled_at": "2023-01-01T15:17:00", "signal_bar_end": "2023-01-01T14:16:00",
         "entry": 100.0, "stop": 95.0, "fill_price": 100.0, "exit_price": 105.0,
         "quantity": 10, "atr": 2.0, "gross_pnl": 50.0, "reserved_notional": 1000.0,
         "exit_reason": "target", "pnl_r": 1.0},
        {"mode": "stock_only", "portfolio": "current_1pct", "cost_model": "baseline",
         "status": "closed", "symbol": "BBB", "net_pnl": -40.0, "market": "stock",
         "decision_time": "2023-01-03T15:16:00", "exit_time": "2023-01-04T15:16:00",
         "filled_at": "2023-01-03T15:17:00", "signal_bar_end": "2023-01-03T14:16:00",
         "entry": 50.0, "stop": 48.0, "fill_price": 50.0, "exit_price": 48.0,
         "quantity": 20, "atr": 1.0, "gross_pnl": -40.0, "reserved_notional": 1000.0,
         "exit_reason": "stop", "pnl_r": -1.0},
    ])
    trades_df.to_csv(run_dir / "trades.csv", index=False)

    summary = {"summaries": {"stock_only": {
        "current_1pct": {
            "baseline": {"closed_count": 2, "total_net_pnl": baseline_total, "net_expectancy": expectancy},
            "zero": {"closed_count": 2, "total_net_pnl": 100.0, "net_expectancy": 50.0},
            "stressed": {"closed_count": 2, "total_net_pnl": 20.0, "net_expectancy": 10.0},
        },
        "safe_0_25pct": {"baseline": {"closed_count": 2, "total_net_pnl": 15.0, "net_expectancy": 7.5}},
    }}}
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    config = {"start": "2023-01-01", "end": "2023-01-05", "config_sha256": "deadbeef"}
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "rejected_orders.csv").write_text("", encoding="utf-8")
    (run_dir / "data_coverage.json").write_text("{}", encoding="utf-8")


def _patch_out_regime_and_manifest(monkeypatch):
    monkeypatch.setattr(analyze_mod, "load_spy_hourly", lambda config: pd.DataFrame())
    monkeypatch.setattr(
        analyze_mod, "compute_spy_regime",
        lambda spy, decision_times: pd.Series(["UNKNOWN"] * len(decision_times), index=decision_times.index),
    )
    monkeypatch.setattr(analyze_mod, "build_manifest", lambda run_dir, config: {"stub": True})


def test_reconciliation_passes_when_totals_match(tmp_path, monkeypatch):
    run_dir = tmp_path / "source_run"
    _write_synthetic_run(run_dir, baseline_total=60.0, expectancy=30.0)  # 100 + (-40) = 60
    _patch_out_regime_and_manifest(monkeypatch)

    out_dir = run_analysis(run_dir, output_root=tmp_path / "out")
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "report.md").exists()
    report_text = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "POST-HOC EXPLORATORY ANALYSIS" in report_text


def test_reconciliation_catches_a_real_mismatch(tmp_path, monkeypatch):
    run_dir = tmp_path / "source_run"
    _write_synthetic_run(run_dir, baseline_total=999.0, expectancy=499.5)  # deliberately wrong
    _patch_out_regime_and_manifest(monkeypatch)

    with pytest.raises(AssertionError):
        run_analysis(run_dir, output_root=tmp_path / "out")


# -- Manifest hashing (supporting correctness check for requirement #2 of v2) --

def test_manifest_hashes_expected_files_and_accepts_injected_cache_path(tmp_path):
    run_dir = tmp_path / "source_run"
    _write_synthetic_run(run_dir, baseline_total=60.0, expectancy=30.0)
    cache_stub = tmp_path / "tiny_cache.db"
    cache_stub.write_bytes(b"not a real sqlite file, just bytes to hash")

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    manifest = build_manifest(run_dir, config, cache_db_path=cache_stub)

    for name in ("trades.csv", "rejected_orders.csv", "summary.json", "data_coverage.json", "config.json"):
        assert name in manifest["source_file_sha256"]
        assert len(manifest["source_file_sha256"][name]) == 64  # sha256 hex digest length
    assert manifest["config_internal_sha256_field"] == "deadbeef"
    assert manifest["cache_db_sha256"] == analyze_mod.sha256_file(cache_stub)
