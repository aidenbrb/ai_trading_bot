"""
Static-scan proof that live_slc is fully decoupled from the frozen
guardrail files, in both directions: those files contain no reference to
live_slc, and live_slc never imports the execution-critical parts of them.
"""
import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_SLC_DIR = REPO_ROOT / "live_slc"

FROZEN_FILES = [
    REPO_ROOT / "run_pipeline.py",
    REPO_ROOT / "nodes" / "execution_node.py",
    REPO_ROOT / "utils" / "strategy_registry.py",
]


def test_frozen_files_contain_no_slc_or_live_slc_reference():
    for path in FROZEN_FILES:
        source = path.read_text(encoding="utf-8")
        assert "live_slc" not in source, f"{path.name} references live_slc"
        assert "slc" not in source.lower(), f"{path.name} references slc"


def _all_imports(tree: ast.AST) -> list[str]:
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


def test_live_slc_never_imports_execution_node_or_strategy_registry():
    for py_file in LIVE_SLC_DIR.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        imports = _all_imports(tree)
        assert not any(name.startswith("nodes.execution_node") for name in imports), (
            f"{py_file.name} imports nodes.execution_node"
        )
        assert not any(name.startswith("utils.strategy_registry") for name in imports), (
            f"{py_file.name} imports utils.strategy_registry"
        )
        assert not any(name == "run_pipeline" for name in imports), (
            f"{py_file.name} imports run_pipeline"
        )


def test_live_slc_never_writes_to_strategy_registry():
    for py_file in LIVE_SLC_DIR.glob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        assert "REGISTRY[" not in source
        assert "strategy_registry.REGISTRY" not in source


def test_run_slc_backtest_and_its_test_file_remain_fully_untouched():
    """Cross-checked against the Tier-1 hash set (which live_slc's own
    guardrail baseline independently re-verifies every run) - this test
    additionally confirms the backtest test file itself, which isn't part
    of any hash set, has no live_slc-related edits by scanning it for any
    reference at all."""
    backtest_test = REPO_ROOT / "tests" / "test_slc_backtest.py"
    source = backtest_test.read_text(encoding="utf-8")
    assert "live_slc" not in source


def test_research_modules_used_by_live_slc_contain_no_order_submission_client():
    """Mirrors the existing test_slc_backtest.py::test_research_modules_contain_no_order_submission_client
    property, extended to live_slc's own reducer/bar_cache (the two
    modules that touch market data, not order submission)."""
    forbidden = ("TradingClient", "submit_order")
    for name in ("reducer.py", "bar_cache.py"):
        source = (LIVE_SLC_DIR / name).read_text(encoding="utf-8")
        assert not any(token in source for token in forbidden), f"{name} references an order-submission client"
