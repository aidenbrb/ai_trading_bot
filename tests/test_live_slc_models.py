import sqlite3

import live_slc.models as models


def test_init_live_slc_db_creates_only_slc_tables_even_with_shared_app_models_imported(tmp_path, monkeypatch):
    """The core isolation property: live_slc never opens or writes
    trading.db. Importing db.models into the same process must not leak
    the shared app's tables into live_slc's own database file."""
    import db.models  # noqa: F401 - import the shared app's models into this process

    db_path = tmp_path / "isolated.db"
    monkeypatch.setattr(models, "LIVE_SLC_DB_PATH", db_path)
    models.init_live_slc_db()

    con = sqlite3.connect(db_path)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()

    assert tables == {
        "slc_five_min_bars", "slc_reducer_state", "slc_zones", "slc_signal_records",
        "slc_orders", "slc_positions", "slc_trades", "slc_deployment_status",
        "slc_activation_events", "slc_reauth_nonces", "slc_session_stats", "slc_cycle_runs",
        "slc_audit_events", "slc_schema_version",
    }
    assert "tickers" not in tables
    assert "ohlcv" not in tables
    assert "positions" not in tables  # the shared app's table name, not slc_positions


def test_multiple_db_paths_never_collide_via_engine_cache(tmp_path, monkeypatch):
    path_a = tmp_path / "a.db"
    path_b = tmp_path / "b.db"

    monkeypatch.setattr(models, "LIVE_SLC_DB_PATH", path_a)
    models.init_live_slc_db()
    with models.get_live_slc_session() as s:
        s.add(models.SlcDeploymentStatus(strategy_version="marker-a"))

    monkeypatch.setattr(models, "LIVE_SLC_DB_PATH", path_b)
    models.init_live_slc_db()
    with models.get_live_slc_session() as s:
        s.add(models.SlcDeploymentStatus(strategy_version="marker-b"))

    con_a = sqlite3.connect(path_a)
    con_b = sqlite3.connect(path_b)
    rows_a = [r[0] for r in con_a.execute("SELECT strategy_version FROM slc_deployment_status")]
    rows_b = [r[0] for r in con_b.execute("SELECT strategy_version FROM slc_deployment_status")]
    con_a.close()
    con_b.close()

    assert rows_a == ["marker-a"]
    assert rows_b == ["marker-b"]


def test_live_slc_never_imports_the_shared_db_connection_module():
    """Process-level invariant, not a file-hash claim (rev. 6 correction):
    db/connection.py is what actually manages trading.db (via
    config.settings.DATABASE_URL) - live_slc never imports it, verified
    via AST scan of every import statement in the package (catches lazy
    imports inside function bodies too, not just module-level ones)."""
    import ast
    from pathlib import Path

    package_dir = Path(models.__file__).parent
    for py_file in package_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert not name.startswith("db.connection"), (
                    f"{py_file.name} imports db.connection (manages trading.db)"
                )
                assert name != "db.models", f"{py_file.name} imports the shared app's db.models"


def test_live_slc_db_path_is_its_own_file_not_trading_db():
    assert models.LIVE_SLC_DB_PATH.name == "live_slc.db"
