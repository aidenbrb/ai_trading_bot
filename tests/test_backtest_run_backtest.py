"""
End-to-end smoke test for backtest/run_backtest.py's CLI, entirely with
synthetic/mocked bar data - proves the whole pipeline (engine -> metrics ->
file output) wires together and writes real result files, without any
network access.
"""
import json
import sys
from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

import backtest.run_backtest as run_backtest_mod

_TARGET_DATE = date(2025, 6, 2)


def _daily_multi(symbols, start, end):
    idx = pd.date_range(end=_TARGET_DATE - timedelta(days=1), periods=20, freq="D")
    df = pd.DataFrame({
        "open": [100.0] * 20, "high": [101.0] * 20, "low": [99.0] * 20,
        "close": [100.0] * 20, "volume": [2_000_000.0] * 20,
    }, index=idx)
    return {s: df.copy() for s in symbols}


def _intraday_multi(symbols, start, end, minutes=5, feed="iex"):
    if minutes == 1:
        idx = pd.DatetimeIndex([start, start + timedelta(minutes=1)])
        return {s: pd.DataFrame(
            {"open": [10.5, 11.9], "high": [10.7, 12.0], "low": [10.5, 11.8], "close": [10.6, 11.9]}, index=idx
        ) for s in symbols}
    idx = pd.DatetimeIndex([start])
    return {s: pd.DataFrame(
        {"open": [10.0], "high": [10.6], "low": [9.8], "close": [10.5], "volume": [50_000.0]}, index=idx
    ) for s in symbols}


def test_full_cli_run_writes_result_files(tmp_path, monkeypatch):
    monkeypatch.setattr(run_backtest_mod, "_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "run_backtest.py", "--symbols", "AAPL", "--start", str(_TARGET_DATE), "--end", str(_TARGET_DATE),
    ])

    with patch("backtest.engine.get_daily_bars_multi", side_effect=_daily_multi), \
         patch("backtest.engine.get_intraday_bars_multi", side_effect=_intraday_multi):
        run_backtest_mod.main()

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    out = run_dirs[0]
    assert (out / "config.json").exists()
    assert (out / "summary.json").exists()
    assert (out / "trades.csv").exists()
    # Auditability: the per-session detail behind the averaged overlap stats
    # must be written to disk, not just the aggregate in summary.json.
    assert (out / "overlap_records.csv").exists()

    summary = json.loads((out / "summary.json").read_text())
    assert "universe_caveat" in summary
    assert "long_only_note" in summary
    assert "portfolio_model_note" in summary
    assert "missing_outcome_data_count" in summary
    assert len(summary["summaries"]) == 6  # 2 scenarios x 3 cost tiers
    assert "avg_overlap_rate" in summary["iex_sip_overlap"]

    import csv
    with open(out / "overlap_records.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert "iex_eligible_count" in rows[0]
    assert "sip_eligible_count" in rows[0]
