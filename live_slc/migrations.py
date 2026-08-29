"""
Real, transactional, idempotent SQLite migration for live_slc.db.

SQLModel.metadata.create_all() only creates tables that don't yet exist -
it never alters an existing table to add a column or index. A real
`live_slc.db` that predates a schema change needs actual migration.

Two SQLite-specific gotchas this module works around, both found by direct
verification rather than assumed:

1. **DDL is not genuinely transactional through SQLAlchemy's default
   pysqlite driver.** `with engine.begin(): conn.exec_driver_sql("ALTER
   TABLE ...")` does not roll back on failure - pysqlite issues an
   implicit commit before DDL regardless of SQLAlchemy's own transaction
   wrapping. Fixed here by disabling pysqlite's own transaction handling
   (`dbapi_connection.isolation_level = None`) and issuing a real `BEGIN
   IMMEDIATE` by hand, on a dedicated, short-lived engine used only for
   migration - the shared engine `models.get_live_slc_session()` uses is
   never touched, so normal ORM session behavior elsewhere is unaffected.

2. **`ALTER TABLE ADD COLUMN` cannot add a NOT NULL column without a
   default to a populated table**, and a placeholder default is forbidden
   for `slc_positions`'s new natural-key columns (level_id,
   confirmation_time) - a fake value would be worse than no constraint at
   all. Handled by adding them nullable, backfilling from the linked
   SlcSignalRecord, auditing for remaining NULLs/duplicates, and only then
   creating the unique index - the column stays nullable at the SQLite
   schema level by deliberate choice (see _backfill_position_natural_key_or_abort's
   docstring), not because NOT NULL wasn't considered.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from sqlalchemy import UniqueConstraint, create_engine, event
from sqlmodel import SQLModel

SCHEMA_VERSION = 1


class MigrationBlockedByDuplicates(RuntimeError):
    def __init__(self, table: str, columns: str, rows) -> None:
        super().__init__(f"unique constraint on {table}({columns}) blocked by existing duplicate rows: {rows}")
        self.table = table
        self.columns = columns
        self.rows = rows


class MigrationBlockedByUnresolvableNaturalKey(RuntimeError):
    def __init__(self, row_ids: list[str]) -> None:
        super().__init__(
            f"slc_positions row(s) {row_ids} have no signal_id, or no matching "
            "SlcSignalRecord, so level_id/confirmation_time cannot be derived "
            "unambiguously - manual resolution required before migrating."
        )
        self.row_ids = row_ids


class SchemaDriftDetected(RuntimeError):
    def __init__(self, detail: str) -> None:
        super().__init__(f"schema-version row claims current, but actual schema disagrees: {detail}")


@dataclass(frozen=True)
class MigrationResult:
    applied: bool
    from_version: int
    to_version: int


def _migration_engine(db_path):
    """A dedicated engine, never the shared cached one from models.py -
    the event listeners below change fundamental transaction behavior and
    must not leak into normal ORM session usage."""
    engine = create_engine(f"sqlite:///{db_path}")

    @event.listens_for(engine, "connect")
    def _disable_pysqlite_implicit_txn(dbapi_connection, _record) -> None:
        dbapi_connection.isolation_level = None  # stop pysqlite's commit-before-DDL behavior

    return engine


def _existing_columns(conn, table_name: str) -> set[str]:
    rows = conn.exec_driver_sql(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def _unique_indexes_by_columns(conn, table_name: str) -> list[list[str]]:
    """The ordered column list of every genuinely UNIQUE index on
    `table_name`, discovered via PRAGMA index_list/index_info rather than
    by name. SQLite creates automatic unique indexes (named
    sqlite_autoindex_<table>_<n>) for UNIQUE constraints declared inline
    in CREATE TABLE - exactly what create_all() emits - so a name-only
    ("uq_<table>_<cols>") check misses them entirely even though the
    constraint they enforce is identical."""
    index_rows = conn.exec_driver_sql(f'PRAGMA index_list("{table_name}")').fetchall()
    result = []
    for row in index_rows:
        index_name, is_unique = row[1], row[2]
        if not is_unique:
            continue
        info_rows = conn.exec_driver_sql(f'PRAGMA index_info("{index_name}")').fetchall()
        cols = [info_row[2] for info_row in sorted(info_rows, key=lambda r: r[0])]
        result.append(cols)
    return result


def _has_equivalent_unique_index(conn, table_name: str, expected_cols: list[str]) -> bool:
    """True iff some UNIQUE index (any name) covers exactly these columns,
    in this order. A non-unique index over the same columns, or a unique
    index over different columns, does not satisfy the constraint."""
    return expected_cols in _unique_indexes_by_columns(conn, table_name)


def _table_exists(conn, table_name: str) -> bool:
    row = conn.exec_driver_sql(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    ).fetchone()
    return row is not None


_SQLITE_TYPE_MAP = {
    "VARCHAR": "TEXT", "TEXT": "TEXT", "BOOLEAN": "BOOLEAN", "DATETIME": "DATETIME",
    "DATE": "DATE", "FLOAT": "FLOAT", "INTEGER": "INTEGER", "JSON": "TEXT",
}


def _column_ddl(column, engine) -> str:
    compiled = column.type.compile(engine.dialect)
    return f'"{column.name}" {compiled}'


def _ensure_schema_version_table(conn) -> None:
    if not _table_exists(conn, "slc_schema_version"):
        conn.exec_driver_sql(
            "CREATE TABLE slc_schema_version (id INTEGER PRIMARY KEY, version INTEGER NOT NULL, updated_at DATETIME)"
        )
        conn.exec_driver_sql("INSERT INTO slc_schema_version (id, version) VALUES (1, 0)")


def _read_schema_version(conn) -> int:
    row = conn.exec_driver_sql("SELECT version FROM slc_schema_version WHERE id=1").fetchone()
    return row[0] if row else 0


def _write_schema_version(conn, version: int) -> None:
    conn.exec_driver_sql("UPDATE slc_schema_version SET version=?, updated_at=CURRENT_TIMESTAMP WHERE id=1", (version,))


def _backfill_position_natural_key_or_abort(conn) -> None:
    """Never a placeholder value. Aborts (raises - the caller's transaction
    rolls back entirely) if any row's natural key can't be derived from
    its linked SlcSignalRecord."""
    if not _table_exists(conn, "slc_positions") or not _table_exists(conn, "slc_signal_records"):
        return
    rows = conn.exec_driver_sql(
        "SELECT id, signal_id FROM slc_positions WHERE level_id IS NULL OR confirmation_time IS NULL"
    ).fetchall()
    unresolvable = []
    for row_id, signal_id in rows:
        if not signal_id:
            unresolvable.append(row_id)
            continue
        signal = conn.exec_driver_sql(
            "SELECT level_id, confirmation_time FROM slc_signal_records WHERE id=?", (signal_id,)
        ).fetchone()
        if signal is None:
            unresolvable.append(row_id)
            continue
        conn.exec_driver_sql(
            "UPDATE slc_positions SET level_id=?, confirmation_time=? WHERE id=?",
            (signal[0], signal[1], row_id),
        )
    if unresolvable:
        raise MigrationBlockedByUnresolvableNaturalKey(unresolvable)

    # Explicit post-backfill audit - checked directly, not inferred from
    # the unique-index-creation step's own success/failure.
    remaining_nulls = conn.exec_driver_sql(
        "SELECT COUNT(*) FROM slc_positions WHERE level_id IS NULL OR confirmation_time IS NULL"
    ).fetchone()[0]
    if remaining_nulls:
        raise MigrationBlockedByUnresolvableNaturalKey(["<unexpected residual NULLs after backfill>"])


def _create_unique_index_safely(conn, table_name: str, constraint) -> None:
    cols = list(constraint.columns.keys()) if hasattr(constraint.columns, "keys") else [c.name for c in constraint.columns]
    cols_sql = ", ".join(f'"{c}"' for c in cols)
    if _has_equivalent_unique_index(conn, table_name, cols):
        # Already satisfied - possibly by a sqlite_autoindex_* SQLite
        # created automatically for an inline UNIQUE, or by a
        # differently-named index from an earlier migration. Creating a
        # second, redundant unique index over the same columns would add
        # nothing but drift risk.
        return
    duplicates = conn.exec_driver_sql(
        f'SELECT {cols_sql}, COUNT(*) c FROM "{table_name}" GROUP BY {cols_sql} HAVING c > 1'
    ).fetchall()
    if duplicates:
        # Never silently delete data to make an index succeed.
        raise MigrationBlockedByDuplicates(table_name, cols_sql, duplicates)
    index_name = f"uq_{table_name}_{'_'.join(cols)}"
    conn.exec_driver_sql(f'CREATE UNIQUE INDEX "{index_name}" ON "{table_name}" ({cols_sql})')


def _verify_schema_matches_expected(conn, live_slc_table_names: set[str], preexisting_tables: set[str]) -> None:
    """Always run, even when the version row already claims current -
    schema-version bookkeeping must never conceal actual drift. Scoped to
    `preexisting_tables` only: a table that didn't exist before this
    migration attempt is create_all()'s responsibility (called right after
    this function returns, on a brand-new DB or a brand-new table added in
    this release) and correctly doesn't exist yet at this point - that is
    not drift, it's sequencing."""
    for table in SQLModel.metadata.tables.values():
        if table.name not in live_slc_table_names or table.name not in preexisting_tables:
            continue
        if not _table_exists(conn, table.name):
            raise SchemaDriftDetected(f"table {table.name} is missing")
        existing_cols = _existing_columns(conn, table.name)
        expected_cols = {c.name for c in table.columns}
        missing_cols = expected_cols - existing_cols
        if missing_cols:
            raise SchemaDriftDetected(f"{table.name} missing columns: {sorted(missing_cols)}")
        for constraint in table.constraints:
            if isinstance(constraint, UniqueConstraint):
                cols = [c.name for c in constraint.columns]
                if not _has_equivalent_unique_index(conn, table.name, cols):
                    raise SchemaDriftDetected(f"{table.name} missing unique index on columns {cols}")


def migrate_live_slc_db(db_path, *, live_slc_table_names=None) -> MigrationResult:
    """Called from models.init_live_slc_db(), which is itself called only
    from inside the process lock (run_slc_live.main()) - two processes can
    never run this concurrently."""
    from live_slc.models import _LIVE_SLC_TABLE_NAMES

    table_names = live_slc_table_names or _LIVE_SLC_TABLE_NAMES
    engine = _migration_engine(db_path)
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("BEGIN IMMEDIATE")
            try:
                _ensure_schema_version_table(conn)
                preexisting = {name for name in table_names if _table_exists(conn, name)}
                from_version = _read_schema_version(conn)
                if from_version >= SCHEMA_VERSION:
                    # Nothing to write, but ALWAYS verify actual schema
                    # before trusting the version number.
                    _verify_schema_matches_expected(conn, table_names, preexisting)
                    conn.exec_driver_sql("ROLLBACK")  # no-op transaction, nothing was written
                    return MigrationResult(applied=False, from_version=from_version, to_version=from_version)

                for table in SQLModel.metadata.tables.values():
                    if table.name not in preexisting:
                        continue  # doesn't exist yet - create_all() handles it separately, with all constraints baked in
                    existing_cols = _existing_columns(conn, table.name)
                    for column in table.columns:
                        if column.name not in existing_cols:
                            conn.exec_driver_sql(f'ALTER TABLE "{table.name}" ADD COLUMN {_column_ddl(column, engine)}')

                _backfill_position_natural_key_or_abort(conn)

                for table in SQLModel.metadata.tables.values():
                    if table.name not in preexisting:
                        continue
                    for constraint in table.constraints:
                        if isinstance(constraint, UniqueConstraint):
                            _create_unique_index_safely(conn, table.name, constraint)

                _write_schema_version(conn, SCHEMA_VERSION)
                _verify_schema_matches_expected(conn, table_names, preexisting)
                conn.exec_driver_sql("COMMIT")
                return MigrationResult(applied=True, from_version=from_version, to_version=SCHEMA_VERSION)
            except Exception:
                conn.exec_driver_sql("ROLLBACK")
                raise
    finally:
        engine.dispose()
