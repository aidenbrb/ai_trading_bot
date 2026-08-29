"""
Real, transactional SQLite migration tests for live_slc.

These are not testing SQLModel/SQLAlchemy in the abstract - they exist
because the first draft of this migration used `with engine.begin():` and
was verified (empirically, by the user) to NOT actually roll back DDL on
this project's pysqlite setup. Every test here proves the corrected
BEGIN-IMMEDIATE-based implementation genuinely has the properties it
claims, checked directly against the raw SQLite file via PRAGMA - never
inferred from the schema-version bookkeeping alone.
"""
import sqlite3
import uuid

import pytest

from live_slc.migrations import (
    MigrationBlockedByDuplicates,
    MigrationBlockedByUnresolvableNaturalKey,
    migrate_live_slc_db,
)
from live_slc.models import LIVE_SLC_DB_PATH, init_live_slc_db, _engines


def _old_schema_db(path):
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE slc_positions (id TEXT PRIMARY KEY, symbol TEXT, direction TEXT, "
        "signal_id TEXT, qty REAL, entry_price REAL, stop_price REAL, target_price REAL, "
        "status TEXT, session_date TEXT, opened_at TEXT, closed_at TEXT)"
    )
    con.execute(
        "CREATE TABLE slc_signal_records (id TEXT PRIMARY KEY, cycle_run_id TEXT, symbol TEXT, "
        "level_id TEXT, direction TEXT, level_state TEXT, level_low REAL, level_high REAL, "
        "level_active_time TEXT, confirmation_time TEXT, entry_time TEXT, stop REAL, "
        "stochastic_k REAL, atr14 REAL, structure TEXT, impulse_atr REAL)"
    )
    con.execute(
        "CREATE TABLE slc_reducer_state (symbol TEXT PRIMARY KEY, bootstrap_completed INTEGER, "
        "last_processed_bar_time TEXT, in_progress_4h_bar_json TEXT, completed_4h_bars_json TEXT, "
        "raw_bar_tail_json TEXT, active_levels_json TEXT, last_split_check_at TEXT, "
        "last_split_check_close REAL, updated_at TEXT)"
    )
    con.commit()
    con.close()


def _columns(db_path, table):
    con = sqlite3.connect(db_path)
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
    con.close()
    return cols


def _indexes(db_path, table):
    con = sqlite3.connect(db_path)
    idx = [r[1] for r in con.execute(f"PRAGMA index_list({table})")]
    con.close()
    return idx


def test_fresh_db_creates_full_schema(tmp_path):
    db_path = tmp_path / "fresh.db"
    init_live_slc_db(db_path)
    con = sqlite3.connect(db_path)
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    con.close()
    assert "slc_schema_version" in tables
    assert "slc_audit_events" in tables
    assert "level_id" in _columns(db_path, "slc_positions")


def test_migration_backfills_natural_key_from_linked_signal(tmp_path):
    db_path = tmp_path / "old.db"
    _old_schema_db(db_path)
    con = sqlite3.connect(db_path)
    sig_id = str(uuid.uuid4())
    con.execute(
        "INSERT INTO slc_signal_records (id, cycle_run_id, symbol, level_id, direction, "
        "level_state, level_low, level_high, level_active_time, confirmation_time, entry_time, "
        "stop, stochastic_k, atr14, structure, impulse_atr) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (sig_id, "run1", "AAPL", "demand:x", "long", "fresh", 100.0, 101.0,
         "2024-01-01T10:00:00", "2024-01-01T10:00:00", "2024-01-01T10:05:00", 99.0, 15.0, 1.0, "uptrend", 1.5),
    )
    pos_id = str(uuid.uuid4())
    con.execute(
        "INSERT INTO slc_positions (id, symbol, direction, signal_id, qty, entry_price, "
        "stop_price, target_price, status, session_date, opened_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (pos_id, "AAPL", "long", sig_id, 10, 100.5, 99.0, 102.5, "open", "2024-01-01", "2024-01-01T10:05:00"),
    )
    con.commit()
    con.close()

    result = migrate_live_slc_db(db_path)
    assert result.applied and result.from_version == 0 and result.to_version == 1

    con2 = sqlite3.connect(db_path)
    row = con2.execute("SELECT level_id, confirmation_time FROM slc_positions WHERE id=?", (pos_id,)).fetchone()
    con2.close()
    assert row == ("demand:x", "2024-01-01T10:00:00")
    assert "uq_slc_positions_symbol_level_id_confirmation_time" in _indexes(db_path, "slc_positions")


def test_migration_aborts_on_unresolvable_natural_key_no_placeholder_written(tmp_path):
    db_path = tmp_path / "unresolvable.db"
    _old_schema_db(db_path)
    con = sqlite3.connect(db_path)
    orphan_id = str(uuid.uuid4())
    con.execute(
        "INSERT INTO slc_positions (id, symbol, direction, signal_id, qty, entry_price, "
        "stop_price, target_price, status, session_date, opened_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (orphan_id, "TSLA", "long", None, 5, 200.0, 195.0, 210.0, "open", "2024-01-01", "2024-01-01T10:00:00"),
    )
    con.commit()
    con.close()

    with pytest.raises(MigrationBlockedByUnresolvableNaturalKey):
        migrate_live_slc_db(db_path)

    assert "level_id" not in _columns(db_path, "slc_positions")  # nothing partially applied


def test_migration_aborts_on_preexisting_duplicate_natural_keys(tmp_path):
    db_path = tmp_path / "dup.db"
    _old_schema_db(db_path)
    con = sqlite3.connect(db_path)
    sig_id = str(uuid.uuid4())
    con.execute(
        "INSERT INTO slc_signal_records (id, cycle_run_id, symbol, level_id, direction, "
        "level_state, level_low, level_high, level_active_time, confirmation_time, entry_time, "
        "stop, stochastic_k, atr14, structure, impulse_atr) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (sig_id, "run1", "AAPL", "demand:x", "long", "fresh", 100.0, 101.0,
         "2024-01-01T10:00:00", "2024-01-01T10:00:00", "2024-01-01T10:05:00", 99.0, 15.0, 1.0, "uptrend", 1.5),
    )
    # two positions pointing at the SAME signal -> same natural key after backfill -> duplicate
    for _ in range(2):
        con.execute(
            "INSERT INTO slc_positions (id, symbol, direction, signal_id, qty, entry_price, "
            "stop_price, target_price, status, session_date, opened_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), "AAPL", "long", sig_id, 10, 100.5, 99.0, 102.5, "open", "2024-01-01", "2024-01-01T10:05:00"),
        )
    con.commit()
    con.close()

    with pytest.raises(MigrationBlockedByDuplicates):
        migrate_live_slc_db(db_path)
    assert "level_id" not in _columns(db_path, "slc_positions")


def test_interrupted_migration_leaves_schema_genuinely_unchanged(tmp_path, monkeypatch):
    """The core property the user's local verification demanded: a crash
    mid-migration must leave the ACTUAL SQLite schema (not just the
    schema-version bookkeeping) exactly as it was."""
    import live_slc.migrations as migrations

    db_path = tmp_path / "interrupt.db"
    _old_schema_db(db_path)
    cols_before = sorted(_columns(db_path, "slc_positions"))
    idx_before = sorted(_indexes(db_path, "slc_positions"))

    def _crash(conn):
        raise RuntimeError("simulated crash mid-migration")

    monkeypatch.setattr(migrations, "_backfill_position_natural_key_or_abort", _crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        migrations.migrate_live_slc_db(db_path)

    con = sqlite3.connect(db_path)
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    con.close()
    assert sorted(_columns(db_path, "slc_positions")) == cols_before
    assert sorted(_indexes(db_path, "slc_positions")) == idx_before
    assert "slc_schema_version" not in tables  # even the version table's own creation rolled back

    # a clean second attempt (with the crash removed) then succeeds
    monkeypatch.undo()
    result = migrations.migrate_live_slc_db(db_path)
    assert result.applied and result.to_version == 1
    assert "level_id" in _columns(db_path, "slc_positions")


def test_second_migration_is_idempotent_and_verifies_actual_schema(tmp_path):
    db_path = tmp_path / "idempotent.db"
    _old_schema_db(db_path)
    first = migrate_live_slc_db(db_path)
    assert first.applied
    second = migrate_live_slc_db(db_path)
    assert not second.applied  # version already current
    assert second.from_version == second.to_version == 1


def test_fresh_db_survives_init_live_slc_db_called_twice(tmp_path):
    """Regression: create_all() on a brand-new DB satisfies each
    UniqueConstraint via a SQLite sqlite_autoindex_* index (inline UNIQUE
    in CREATE TABLE), not one named uq_<table>_<cols>. The second
    init_live_slc_db() call must recognize that as equivalent, not raise
    SchemaDriftDetected."""
    db_path = tmp_path / "fresh_twice.db"
    init_live_slc_db(db_path)
    init_live_slc_db(db_path)  # must not raise


def test_sqlite_autoindex_satisfies_unique_constraint_verification(tmp_path):
    """Directly proves the mechanism behind the bug: an index named
    sqlite_autoindex_* (SQLite's own name for a UNIQUE declared inline in
    CREATE TABLE) is accepted as satisfying the constraint, purely by
    column match, not by name."""
    from sqlalchemy import create_engine

    from live_slc.migrations import _has_equivalent_unique_index

    db_path = tmp_path / "autoindex.db"
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE t (a TEXT, b TEXT, UNIQUE(a, b))")
    con.commit()
    idx_name = con.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='t'"
    ).fetchone()[0]
    assert idx_name.startswith("sqlite_autoindex_")
    con.close()

    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        assert _has_equivalent_unique_index(conn, "t", ["a", "b"])
    engine.dispose()


def test_explicitly_named_migration_created_index_still_accepted(tmp_path):
    """The pre-existing migration path (ALTER TABLE + CREATE UNIQUE INDEX
    uq_<table>_<cols>, exercised by an old DB migrating forward) must
    continue to be recognized - this fix must not regress that path while
    fixing the autoindex case."""
    db_path = tmp_path / "old_migrated_twice.db"
    _old_schema_db(db_path)
    first = migrate_live_slc_db(db_path)
    assert first.applied
    assert "uq_slc_positions_symbol_level_id_confirmation_time" in _indexes(db_path, "slc_positions")
    second = migrate_live_slc_db(db_path)  # must not raise SchemaDriftDetected
    assert not second.applied


def test_wrong_column_unique_index_still_raises_schema_drift(tmp_path):
    """A unique index that exists but covers the wrong columns must not be
    mistaken for satisfying the constraint - fail closed."""
    from live_slc.migrations import SchemaDriftDetected

    db_path = tmp_path / "wrong_cols.db"
    _old_schema_db(db_path)
    migrate_live_slc_db(db_path)

    con = sqlite3.connect(db_path)
    con.execute('DROP INDEX "uq_slc_positions_symbol_level_id_confirmation_time"')
    con.execute('CREATE UNIQUE INDEX "uq_slc_positions_symbol_level_id_confirmation_time" ON slc_positions (symbol, level_id)')
    con.commit()
    con.close()

    with pytest.raises(SchemaDriftDetected):
        migrate_live_slc_db(db_path)


def test_non_unique_index_over_right_columns_still_raises_schema_drift(tmp_path):
    """A plain (non-unique) index over the correct columns does not
    satisfy a UNIQUE constraint - fail closed rather than treat column
    match alone as sufficient."""
    from live_slc.migrations import SchemaDriftDetected

    db_path = tmp_path / "non_unique.db"
    _old_schema_db(db_path)
    migrate_live_slc_db(db_path)

    con = sqlite3.connect(db_path)
    con.execute('DROP INDEX "uq_slc_positions_symbol_level_id_confirmation_time"')
    con.execute('CREATE INDEX "not_unique_idx" ON slc_positions (symbol, level_id, confirmation_time)')
    con.commit()
    con.close()

    with pytest.raises(SchemaDriftDetected):
        migrate_live_slc_db(db_path)


def test_duplicate_rows_still_block_missing_unique_index_creation(tmp_path):
    """Column-equivalence detection must not weaken the existing
    duplicate-data guard: if no equivalent unique index exists yet and
    duplicate rows are present, creation must still abort rather than
    silently build an index over colliding data."""
    db_path = tmp_path / "dup_no_autoindex.db"
    _old_schema_db(db_path)
    con = sqlite3.connect(db_path)
    sig_id = str(uuid.uuid4())
    con.execute(
        "INSERT INTO slc_signal_records (id, cycle_run_id, symbol, level_id, direction, "
        "level_state, level_low, level_high, level_active_time, confirmation_time, entry_time, "
        "stop, stochastic_k, atr14, structure, impulse_atr) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (sig_id, "run1", "AAPL", "demand:x", "long", "fresh", 100.0, 101.0,
         "2024-01-01T10:00:00", "2024-01-01T10:00:00", "2024-01-01T10:05:00", 99.0, 15.0, 1.0, "uptrend", 1.5),
    )
    for _ in range(2):
        con.execute(
            "INSERT INTO slc_positions (id, symbol, direction, signal_id, qty, entry_price, "
            "stop_price, target_price, status, session_date, opened_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), "AAPL", "long", sig_id, 10, 100.5, 99.0, 102.5, "open", "2024-01-01", "2024-01-01T10:05:00"),
        )
    con.commit()
    con.close()

    with pytest.raises(MigrationBlockedByDuplicates):
        migrate_live_slc_db(db_path)
    assert "level_id" not in _columns(db_path, "slc_positions")


def test_schema_version_lying_about_drift_is_detected(tmp_path):
    """Schema-version bookkeeping must never conceal actual drift: if the
    version row claims current but a real column is missing, that's a
    hard failure, not silently trusted."""
    from live_slc.migrations import SchemaDriftDetected

    db_path = tmp_path / "drift.db"
    _old_schema_db(db_path)
    migrate_live_slc_db(db_path)  # brings it to version 1 correctly

    # simulate drift: someone/something removed a column the version row
    # claims exists (SQLite has no DROP COLUMN in old versions, so simulate
    # via a rebuilt table missing it)
    con = sqlite3.connect(db_path)
    con.execute("ALTER TABLE slc_positions RENAME TO slc_positions_old")
    cols = [c for c in _columns(db_path, "slc_positions_old") if c != "level_id"]
    con.execute(f"CREATE TABLE slc_positions ({', '.join(cols)})")
    con.execute(f"INSERT INTO slc_positions SELECT {', '.join(cols)} FROM slc_positions_old")
    con.execute("DROP TABLE slc_positions_old")
    con.commit()
    con.close()

    with pytest.raises(SchemaDriftDetected):
        migrate_live_slc_db(db_path)
