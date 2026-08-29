import ast
from pathlib import Path


def test_research_modules_do_not_import_trading_or_execution_clients():
    root = Path(__file__).resolve().parents[1]
    for name in (
        "backtest/whole_bot_engine.py",
        "backtest/whole_bot_metrics.py",
        "backtest/run_strategy_comparison.py",
    ):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        assert not any(module.startswith("alpaca.trading") for module in imported)
        assert "nodes.execution_node" not in imported
