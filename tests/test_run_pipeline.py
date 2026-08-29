"""Regression tests for pipeline-level crash isolation and audit logging."""
import sys
from unittest.mock import MagicMock, patch

import pytest

import run_pipeline
import nodes.monitor_node as monitor_mod
import nodes.data_node as data_mod
import nodes.indicator_node as indicator_mod
import nodes.news_node as news_mod
import nodes.stock_strategy_node as stock_strat_mod
import nodes.day_strategy_node as day_strat_mod
import nodes.risk_node as risk_mod
import nodes.execution_node as exec_mod
import nodes.options_research_node as options_mod
from utils.strategy_registry import registration
from utils.strategy_signals import CRYPTO_STRATEGY_VERSION, STOCK_STRATEGY_VERSION


def test_run_phase_logs_crash_and_reraises(capsys):
    mock_session = MagicMock()
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_session
    mock_cm.__exit__.return_value = False

    def crash():
        raise RuntimeError("simulated data failure")

    with patch.object(run_pipeline, "init_db"), \
         patch.object(run_pipeline, "get_session", return_value=mock_cm):
        with pytest.raises(RuntimeError, match="simulated data failure"):
            run_pipeline._run_phase("data", crash)

    log = mock_session.add.call_args.args[0]
    assert log.node_name == "data"
    assert log.status == "error"
    assert log.error_message == "simulated data failure"
    assert log.finished_at is not None

    output = capsys.readouterr().out
    assert "PHASE 'data' CRASHED - remaining phases will not run" in output


class TestDayModeSafetyGate:
    """
    Day-mode (--stock-strategy day) signals must be structurally unable to
    reach risk/execution/options, even with EXECUTION_ENABLED=true - this is
    the critical safety property the whole day-trading feature depends on.
    """

    def _patch_all_phase_runs(self, stack):
        mocks = {}
        for mod, name in [
            (monitor_mod, "monitor"), (data_mod, "data"), (indicator_mod, "indicator"),
            (news_mod, "news"), (stock_strat_mod, "strategy"), (day_strat_mod, "day_strategy"),
            (risk_mod, "risk"), (exec_mod, "execution"), (options_mod, "options"),
        ]:
            mocks[name] = MagicMock(return_value={})
            stack.enter_context(patch.object(mod, "run", mocks[name]))
        return mocks

    def test_default_nodes_never_calls_risk_execution_or_options(self, monkeypatch):
        from contextlib import ExitStack

        monkeypatch.setattr(run_pipeline.settings, "EXECUTION_ENABLED", True)
        monkeypatch.setattr(sys, "argv", [
            "run_pipeline.py", "--stock-strategy", "day", "--market", "stocks",
        ])

        with ExitStack() as stack:
            stack.enter_context(patch.object(run_pipeline.settings, "validate"))
            stack.enter_context(patch.object(
                run_pipeline, "_check_market_regime",
                return_value={"label": "BULL", "quality_boost": 0, "description": ""},
            ))
            mocks = self._patch_all_phase_runs(stack)
            run_pipeline.main()

        assert mocks["monitor"].called
        assert mocks["data"].called
        assert mocks["indicator"].called
        assert mocks["news"].called
        assert mocks["day_strategy"].called
        mocks["strategy"].assert_not_called()  # swing strategy must NOT run in day mode
        mocks["risk"].assert_not_called()
        mocks["execution"].assert_not_called()
        mocks["options"].assert_not_called()

    def test_swing_mode_still_calls_stock_strategy_not_day_strategy(self, monkeypatch):
        """Swing research signals still run even though entry phases are locked."""
        from contextlib import ExitStack

        monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--market", "stocks", "--nodes", "strategy"])

        with ExitStack() as stack:
            stack.enter_context(patch.object(run_pipeline.settings, "validate"))
            mocks = self._patch_all_phase_runs(stack)
            run_pipeline.main()

        assert mocks["strategy"].called
        mocks["day_strategy"].assert_not_called()

    def test_explicit_risk_node_hard_errors(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", [
            "run_pipeline.py", "--stock-strategy", "day", "--market", "stocks",
            "--nodes", "risk",
        ])
        with patch.object(run_pipeline.settings, "validate"):
            with pytest.raises(SystemExit):
                run_pipeline.main()

    def test_explicit_execution_node_hard_errors(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", [
            "run_pipeline.py", "--stock-strategy", "day", "--market", "stocks",
            "--nodes", "execution",
        ])
        with patch.object(run_pipeline.settings, "validate"):
            with pytest.raises(SystemExit):
                run_pipeline.main()

    @pytest.mark.parametrize("market", ["all", "crypto"])
    def test_requires_resolved_market_stocks(self, monkeypatch, market):
        monkeypatch.setattr(sys, "argv", [
            "run_pipeline.py", "--stock-strategy", "day", "--market", market,
        ])
        with patch.object(run_pipeline.settings, "validate"):
            with pytest.raises(SystemExit):
                run_pipeline.main()


class TestEvidenceDeploymentGate:
    @pytest.mark.parametrize(
        "version",
        [STOCK_STRATEGY_VERSION, CRYPTO_STRATEGY_VERSION, "orb_v1"],
    )
    def test_failed_strategies_are_not_execution_eligible(self, version):
        record = registration(version)
        assert record.evidence_status == "rejected"
        assert record.execution_eligible is False

    @pytest.mark.parametrize("market", ["stocks", "crypto", "all"])
    @pytest.mark.parametrize("phase", ["risk", "execution"])
    def test_explicit_entry_phase_hard_errors(self, monkeypatch, market, phase):
        monkeypatch.setattr(sys, "argv", [
            "run_pipeline.py", "--market", market, "--nodes", phase,
        ])
        with patch.object(run_pipeline.settings, "validate"):
            with pytest.raises(SystemExit):
                run_pipeline.main()

    def test_default_swing_run_keeps_research_and_monitor_but_blocks_entries(
        self, monkeypatch,
    ):
        from contextlib import ExitStack

        monkeypatch.setattr(run_pipeline.settings, "EXECUTION_ENABLED", True)
        monkeypatch.setattr(sys, "argv", [
            "run_pipeline.py", "--market", "stocks",
        ])

        modules = [
            (monitor_mod, "monitor"), (data_mod, "data"),
            (indicator_mod, "indicator"), (news_mod, "news"),
            (stock_strat_mod, "strategy"), (risk_mod, "risk"),
            (exec_mod, "execution"), (options_mod, "options"),
        ]
        with ExitStack() as stack:
            stack.enter_context(patch.object(run_pipeline.settings, "validate"))
            stack.enter_context(patch.object(
                run_pipeline, "_check_market_regime",
                return_value={"label": "BULL", "quality_boost": 0, "description": ""},
            ))
            mocks = {}
            for module, name in modules:
                mocks[name] = MagicMock(return_value={})
                stack.enter_context(patch.object(module, "run", mocks[name]))
            run_pipeline.main()

        assert mocks["monitor"].called
        assert mocks["strategy"].called
        assert mocks["options"].called
        mocks["risk"].assert_not_called()
        mocks["execution"].assert_not_called()
