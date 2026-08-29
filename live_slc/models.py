"""
Database models and isolated engine for live_slc.

Deliberately independent of db/models.py and db/connection.py: its own
SQLModel table classes, its own SQLite file (live_slc/live_slc.db), and
its own init function. live_slc never opens or writes trading.db - that
is a process-level invariant, not merely a convention (see
tests/test_live_slc_isolation.py).

SQLModel.metadata is a single global registry shared by every SQLModel
subclass in the process. If db/models.py and this module are ever both
imported, a naive `SQLModel.metadata.create_all(engine)` would attempt to
create every registered table - including the main app's - on whichever
engine it's given. init_live_slc_db() below scopes create_all() explicitly
to this module's own table objects to avoid that.
"""
from __future__ import annotations

import datetime as dt
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, Session, SQLModel, create_engine

LIVE_SLC_DB_PATH = Path(__file__).parent / "live_slc.db"


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


# -- Rolling 5-minute bar cache (isolated from the SIP research cache and the
#    shared whole-bot cache - this is a live IEX feed) ------------------------

class SlcFiveMinBar(SQLModel, table=True):
    __tablename__ = "slc_five_min_bars"
    __table_args__ = (UniqueConstraint("symbol", "bar_time", name="uq_slc_bar_symbol_time"),)

    id: str = Field(default_factory=_uuid, primary_key=True)
    symbol: str = Field(index=True)
    bar_time: dt.datetime = Field(index=True)
    open: float
    high: float
    low: float
    close: float
    volume: float
    created_at: dt.datetime = Field(default_factory=_now)


# -- Reducer persisted state: one row per symbol, reloaded every cycle -------

class SlcReducerState(SQLModel, table=True):
    __tablename__ = "slc_reducer_state"

    symbol: str = Field(primary_key=True)
    bootstrap_completed: bool = Field(default=False)
    last_processed_bar_time: Optional[dt.datetime] = None
    # JSON-serialized: in-progress 4h OHLCV aggregate + expected/observed bar count
    in_progress_4h_bar_json: str = Field(default="null")
    # JSON-serialized list of ALL completed 4h bars since bootstrap (retained
    # in full - small, ~2/day - so classify_structure()/confirmed_pivots()
    # can be called on it directly, unmodified, with no separate derived
    # pivot cache needed)
    completed_4h_bars_json: str = Field(default="[]")
    # JSON-serialized short tail of raw 5-min bars: only as many as
    # atr14()/stochastic_5_3_3()/the 3-bar impulse window need, pruned
    # immediately after each bar is incorporated - recomputed via the
    # frozen functions on this tail, never hand-rolled incrementally
    raw_bar_tail_json: str = Field(default="[]")
    # JSON-serialized list of active (unexpired) level states
    active_levels_json: str = Field(default="[]")
    last_split_check_at: Optional[dt.datetime] = None
    last_split_check_close: Optional[float] = None
    # Trading days that already produced this symbol's one signal for the
    # session (reducer.py's cross-level per-day dedup) - JSON list of ISO
    # date strings. Reloading this every cycle (rather than starting from
    # an empty set, as every prior process invocation did) is what makes
    # "at most one signal per symbol per session" actually hold across the
    # real multi-process operating model, not just a single in-memory run.
    signaled_sessions_json: str = Field(default="[]")
    # True the instant a validated split is detected (preflight) - blocks
    # this symbol's reducer processing and entries entirely until a full
    # 120-day atomic rebuild succeeds and clears it. The prior (now
    # stale-scale) state is never erased before that success.
    split_pending: bool = Field(default=False)
    updated_at: dt.datetime = Field(default_factory=_now)


# -- Audit copy of active levels (observability - reducer's own state above is
#    authoritative for resuming; this is for inspection/debugging) ----------

class SlcZone(SQLModel, table=True):
    __tablename__ = "slc_zones"

    id: str = Field(default_factory=_uuid, primary_key=True)
    symbol: str = Field(index=True)
    level_id: str = Field(index=True)
    direction: str
    low: float
    high: float
    base_time: dt.datetime
    active_time: dt.datetime
    impulse_atr: float
    activation_session: dt.date
    state: str  # fresh | waiting_reclaim | reclaimed | expired
    recorded_at: dt.datetime = Field(default_factory=_now)


class SlcSignalRecord(SQLModel, table=True):
    __tablename__ = "slc_signal_records"

    id: str = Field(default_factory=_uuid, primary_key=True)
    cycle_run_id: str = Field(index=True)
    symbol: str = Field(index=True)
    level_id: str
    direction: str
    level_state: str
    level_low: float
    level_high: float
    level_active_time: dt.datetime
    confirmation_time: dt.datetime = Field(index=True)
    entry_time: dt.datetime  # deterministic (confirmation_time + 5min) - always known at confirmation
    stop: float  # computed from the level + ATR, independent of entry price - always known
    # Neither the theoretical entry nor the target is knowable at confirmation
    # time in live operation (the entry is the actual broker fill, per
    # amendment_004) - these stay None on the live path. They are only ever
    # populated when this reducer is run in an offline/parity-testing mode
    # against historical data that already has a "next bar" to read from.
    theoretical_entry: Optional[float] = None
    theoretical_target: Optional[float] = None
    stochastic_k: float
    stochastic_d: Optional[float]
    atr14: float
    structure: str
    impulse_atr: float
    detected_at_utc: dt.datetime = Field(default_factory=_now)
    acted_on: bool = Field(default=False)
    action_result: Optional[str] = None  # e.g. submitted, dry_run, skipped_capacity,
    # skipped_not_shortable, stale_or_missing_data, skipped_ambiguous_ownership
    rank_position: Optional[int] = None


class SlcOrder(SQLModel, table=True):
    __tablename__ = "slc_orders"
    # status values: submission_intent_pending | submitted | submitted_unresolved
    # | confirmed_rejected | ambiguous_submission | existing_broker_order_detected
    # | confirmed_no_order_resulted | dry_run_proposed | filled | canceled_unfilled
    # | ambiguous_cancel | target_replaced | target_replacement_pending
    # The first three (plus existing_broker_order_detected) are the states
    # risk.system_wide_entry_block_reasons() watches for.

    id: str = Field(default_factory=_uuid, primary_key=True)
    client_order_id: str = Field(index=True, unique=True)
    alpaca_order_id: Optional[str] = Field(default=None, index=True)
    signal_id: Optional[str] = Field(default=None, index=True)
    symbol: str = Field(index=True)
    leg: str  # entry | protect | exit
    side: str
    position_intent: str
    qty: float
    order_class: str
    status: str
    dry_run: bool = Field(default=True)
    submitted_at: dt.datetime = Field(default_factory=_now)
    filled_at: Optional[dt.datetime] = None
    fill_price: Optional[float] = None
    error_message: Optional[str] = None
    # Audit schema (rev. 11 Step 11) - naturally tabular per-order detail.
    expected_quote: Optional[float] = None
    stop_submitted: Optional[float] = None  # tick-rounded, Decimal-derived
    target_submitted: Optional[float] = None  # tick-rounded provisional (pre-fill) or final (post-replacement)
    notional: Optional[float] = None
    monetary_risk: Optional[float] = None  # qty * abs(fill - stop_submitted)
    slippage: Optional[float] = None  # actual_fill - expected_quote, signed
    fees: Optional[float] = None
    effective_reward_risk: Optional[float] = None  # from submitted broker prices, never labeled "exactly 2R"


class SlcPosition(SQLModel, table=True):
    __tablename__ = "slc_positions"
    # Natural key, mirrors the entry signal's identity - lets any
    # recovery/reconcile path find-or-update idempotently instead of
    # blind-inserting (rev. 11 Step 1's migration handles backfilling
    # this for any pre-existing row, aborting if it can't be derived
    # unambiguously; nullable at the SQLite schema level by deliberate
    # choice - see migrations.py's docstring - enforced by the unique
    # index + forward-going application code instead of a NOT NULL DDL
    # constraint).
    __table_args__ = (UniqueConstraint("symbol", "level_id", "confirmation_time", name="uq_slc_position_natural_key"),)

    id: str = Field(default_factory=_uuid, primary_key=True)
    symbol: str = Field(index=True)
    level_id: Optional[str] = Field(default=None, index=True)
    confirmation_time: Optional[dt.datetime] = Field(default=None, index=True)
    direction: str
    signal_id: Optional[str] = Field(default=None, index=True)
    qty: float
    entry_price: float
    stop_price: float
    target_price: float
    # status: open | closed | ambiguous | protected_degraded
    # protected_degraded = exact-2R target never confirmed after a fill -
    # never described as a healthy open position; blocks all new SLC
    # entries system-wide until resolved via the guarded emergency-exit
    # path (never from preflight).
    status: str
    session_date: dt.date = Field(index=True)
    opened_at: dt.datetime = Field(default_factory=_now)
    closed_at: Optional[dt.datetime] = None
    entry_order_id: Optional[str] = None
    protective_order_id: Optional[str] = None
    target_order_id: Optional[str] = None  # overwritten on every successful replace_order_by_id (new id each time)
    # Audit schema (rev. 11 Step 11).
    expected_quote: Optional[float] = None
    actual_fill: Optional[float] = None
    notional: Optional[float] = None
    monetary_risk: Optional[float] = None
    slippage: Optional[float] = None
    fees: Optional[float] = None


class SlcTrade(SQLModel, table=True):
    __tablename__ = "slc_trades"
    # A position closes exactly once - the unique constraint is the
    # idempotency mechanism for crash-recovery/reconciliation writes.
    __table_args__ = (UniqueConstraint("position_id", name="uq_slc_trade_position"),)

    id: str = Field(default_factory=_uuid, primary_key=True)
    position_id: str = Field(index=True)
    symbol: str = Field(index=True)
    direction: str
    entry_price: float
    exit_price: float
    exit_time: dt.datetime
    exit_reason: str
    exit_order_id: Optional[str] = None
    qty: float
    gross_pnl: float
    net_pnl: float
    pnl_r: float
    fees: Optional[float] = None
    session_date: dt.date = Field(index=True)


class SlcDeploymentStatus(SQLModel, table=True):
    __tablename__ = "slc_deployment_status"

    strategy_version: str = Field(primary_key=True, default="slc_4h_5m_stock_v1")
    status: str = Field(default="not_authorized")  # not_authorized|dry_run|paper_active|suspended|halted
    updated_at: dt.datetime = Field(default_factory=_now)
    updated_by: str = Field(default="")
    observed_account_id: Optional[str] = None
    live_baseline_sha256: Optional[str] = None
    activation_proposal_sha256: Optional[str] = None
    notes: str = Field(default="")


class SlcActivationEvent(SQLModel, table=True):
    __tablename__ = "slc_activation_events"

    id: str = Field(default_factory=_uuid, primary_key=True)
    from_status: str
    to_status: str
    occurred_at: dt.datetime = Field(default_factory=_now)
    reason: str
    session_gate_metrics_json: str = Field(default="{}")
    guardrail_baseline_sha256_at_transition: Optional[str] = None
    activation_proposal_sha256_at_transition: Optional[str] = None
    observed_account_id: Optional[str] = None
    operator_note: str = Field(default="")
    # Phase 6 Step 1: human-only re-authorization signature (ed25519-sk via
    # ssh-keygen -Y sign/-Y verify - see live_slc/reauth_signature.py).
    # Required whenever a Tier-1/Tier-2 baseline is (re-)pinned or
    # to_status is paper_active/live; NULL on every event that predates
    # this scheme (2026-08-29 and earlier) - those are legacy, unsigned,
    # and are exactly what the new read-time check in guardrails.py must
    # refuse to trust once enforcement is armed.
    signed_payload: Optional[str] = Field(default=None)
    signature_blob: Optional[str] = Field(default=None)
    git_commit_sha: Optional[str] = Field(default=None)
    changed_guardrail_paths_json: Optional[str] = Field(default=None)
    nonce: Optional[str] = Field(default=None, index=True)
    signer_identity: Optional[str] = Field(default=None)
    # Phase 6 Step 1 (circularity fix): the repo-relative path of the
    # dated baseline JSON file this event pins. Previously guardrails.py
    # hardcoded a single "current" path/hash as literal module constants -
    # meaning every re-baseline changed guardrails.py's own bytes, making
    # it impossible for guardrails.py to ever safely be included in its
    # own GUARDRAILS_TIER1 (a stable input can't depend on the value it's
    # about to produce). Storing the active path here instead means
    # guardrails.py's source never needs to change on a re-baseline - the
    # active baseline is DB state (resolve_active_baseline()), not
    # compiled-in state.
    baseline_file_relative_path: Optional[str] = Field(default=None)


class SlcReauthNonce(SQLModel, table=True):
    """Phase 6 Step 1: server-issued, single-use nonces for re-authorization
    signatures. issue_reauth_nonce() (authorization.py) creates a pending
    row before any payload is ever shown to a human for signing;
    record_transition() consumes it exactly once when a valid signature
    referencing it is recorded. A nonce's created_at (server clock, never
    client/payload-supplied) is the sole authority for the 24h validity
    window - a payload's own embedded timestamp is for human-readable
    audit only and is never itself trusted for freshness enforcement."""
    __tablename__ = "slc_reauth_nonces"

    nonce: str = Field(primary_key=True)
    created_at: dt.datetime = Field(default_factory=_now)
    consumed_at: Optional[dt.datetime] = None
    consumed_by_event_id: Optional[str] = None


class SlcSessionStat(SQLModel, table=True):
    __tablename__ = "slc_session_stats"

    session_date: dt.date = Field(primary_key=True)
    trades_opened: int = Field(default=0)
    trades_closed: int = Field(default=0)
    wins: int = Field(default=0)
    losses: int = Field(default=0)
    net_pnl: float = Field(default=0.0)
    daily_loss_limit_breached: bool = Field(default=False)
    signals_generated: int = Field(default=0)
    signals_acted_on: int = Field(default=0)
    signals_skipped_not_shortable: int = Field(default=0)
    signals_skipped_stale_or_missing_data: int = Field(default=0)
    signals_skipped_capacity: int = Field(default=0)
    valid_bar_coverage_pct: float = Field(default=0.0)
    cycles_run: int = Field(default=0)
    cycles_over_budget: int = Field(default=0)
    duplicate_or_stale_signal_count: int = Field(default=0)
    guardrail_check_passed: bool = Field(default=False)
    engine_parity_check_passed: bool = Field(default=False)
    closeout_left_no_open_state: Optional[bool] = None
    # rev. 11 Step 11/12 - required by the corrected engineering gate
    # (authorization.evaluate_dry_run_session_gate()) so a never-populated,
    # all-default row fails closed rather than trivially passing.
    expected_symbol_count: int = Field(default=0)
    valid_symbol_count: int = Field(default=0)
    failed_cycles: int = Field(default=0)
    overlapping_cycles: int = Field(default=0)
    reconciliation_discrepancy_count: int = Field(default=0)
    unprotected_position_incident_count: int = Field(default=0)
    dry_run_proposal_count: int = Field(default=0)
    closeout_confirmed_flat_by_broker_readback: Optional[bool] = None
    updated_at: dt.datetime = Field(default_factory=_now)


class SlcCycleRun(SQLModel, table=True):
    __tablename__ = "slc_cycle_runs"

    id: str = Field(default_factory=_uuid, primary_key=True)
    cycle_time_utc: dt.datetime = Field(default_factory=_now, index=True)
    stage: str  # preflight | cycle | closeout
    symbols_scanned: int = Field(default=0)
    new_bars_ingested: int = Field(default=0)
    confirmations_collected: int = Field(default=0)
    orders_submitted: int = Field(default=0)
    orders_dry_run: int = Field(default=0)
    errors_json: str = Field(default="[]")
    duration_seconds: Optional[float] = None
    status: str = Field(default="running")  # running | completed | failed
    heartbeat_at: dt.datetime = Field(default_factory=_now)


class SlcAuditEvent(SQLModel, table=True):
    """Versioned incident log (rev. 11 Step 11): reconciliation
    discrepancies, unprotected-position incidents, ambiguous-submission
    events, protected_degraded transitions, closeout/broker-flat results -
    irregular/incident-style records, distinct from the naturally-tabular
    per-order/position/trade audit fields above."""
    __tablename__ = "slc_audit_events"

    id: str = Field(default_factory=_uuid, primary_key=True)
    schema_version: int = Field(default=1)
    event_type: str
    symbol: Optional[str] = Field(default=None, index=True)
    occurred_at: dt.datetime = Field(default_factory=_now, index=True)
    payload_json: str = Field(default="{}")
    resolved: bool = Field(default=False)
    resolved_at: Optional[dt.datetime] = None


class SlcSchemaVersion(SQLModel, table=True):
    """Single-row table tracking the applied migration version explicitly
    (migrations.py) - not re-derived purely from column/index
    introspection, though that introspection is still what the migration
    verifies against on every start (schema-version bookkeeping never
    conceals actual schema drift)."""
    __tablename__ = "slc_schema_version"

    id: int = Field(default=1, primary_key=True)
    version: int = Field(default=0)
    updated_at: dt.datetime = Field(default_factory=_now)


_LIVE_SLC_TABLES = [
    SlcFiveMinBar.__table__,
    SlcReducerState.__table__,
    SlcZone.__table__,
    SlcSignalRecord.__table__,
    SlcOrder.__table__,
    SlcPosition.__table__,
    SlcTrade.__table__,
    SlcDeploymentStatus.__table__,
    SlcActivationEvent.__table__,
    SlcReauthNonce.__table__,
    SlcSessionStat.__table__,
    SlcCycleRun.__table__,
    SlcAuditEvent.__table__,
    SlcSchemaVersion.__table__,
]
_LIVE_SLC_TABLE_NAMES = {table.name for table in _LIVE_SLC_TABLES}

# Keyed by resolved path, not a single global - so multiple DB paths (e.g.
# the real live_slc.db and a test's tmp_path) can coexist correctly within
# one process. db_path defaults to None, resolved to LIVE_SLC_DB_PATH
# *inside* each function body (a live lookup) rather than as a bound
# default-parameter value, so monkeypatching module.LIVE_SLC_DB_PATH in a
# test actually takes effect - a bound default would freeze the value that
# was current at import time and silently ignore the monkeypatch.
_engines: dict[str, "Engine"] = {}


def _get_engine(db_path: Optional[Path] = None):
    if db_path is None:
        db_path = LIVE_SLC_DB_PATH
    key = str(db_path)
    if key not in _engines:
        _engines[key] = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )
    return _engines[key]


def init_live_slc_db(db_path: Optional[Path] = None) -> None:
    """Create only live_slc's own tables - never the shared app's tables,
    even if db/models.py has also been imported into this process.
    Migrates any already-existing tables first (adds missing columns/
    unique indexes via a real, transactional migration - see
    migrations.py), then create_all() picks up any genuinely new table
    that doesn't exist at all yet (a fresh CREATE TABLE already includes
    every constraint correctly, no separate migration needed for those).
    Must only ever be called from inside the process lock - see
    run_slc_live.main()."""
    resolved_path = db_path if db_path is not None else LIVE_SLC_DB_PATH
    from live_slc.migrations import migrate_live_slc_db
    migrate_live_slc_db(resolved_path)
    engine = _get_engine(db_path)
    SQLModel.metadata.create_all(engine, tables=_LIVE_SLC_TABLES)


@contextmanager
def get_live_slc_session(db_path: Optional[Path] = None) -> Generator[Session, None, None]:
    engine = _get_engine(db_path)
    with Session(engine, expire_on_commit=False) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
