"""
Database models for ai_trading_bot.

Every node reads and writes through these models - nodes never talk
directly to each other, only through the database.

Table overview:
  tickers       - universe of symbols being tracked
  ohlcv         - raw price/volume bars (written by data_node)
  indicators    - computed technicals per bar (written by indicator_node)
  analyses      - Claude's per-ticker market read (written by analysis_node)
  strategies    - Claude-generated trade setups (written by strategy_node)
  risk_approvals- risk node decisions per setup (written by risk_node)
  orders        - every order submitted to Alpaca (written by execution_node)
  positions     - currently open positions (managed by execution/monitor nodes)
  trades        - completed round-trips with PnL (written by monitor_node)
  intraday_daily_stats - prior-session-only ORB reference stats (written by intraday_reference_node)
  intraday_signals     - 5-min ORB day-mode signals + shadow outcomes (written by day_strategy_node / intraday_shadow_node)
  evaluation_ledger    - paper-order research observations reconciled read-only from Alpaca
  run_log       - one row per node per pipeline run (written by every node)
"""
import uuid
import datetime as dt
from typing import List, Optional

from sqlmodel import Field, Relationship, SQLModel
from sqlalchemy import UniqueConstraint


# -- Helpers -------------------------------------------------------------------

def _uuid() -> str:
    return str(uuid.uuid4())

def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


# -- Tickers -------------------------------------------------------------------

class Ticker(SQLModel, table=True):
    __tablename__ = "tickers"

    id:      str = Field(default_factory=_uuid, primary_key=True)
    symbol:  str = Field(index=True, unique=True)
    sector:  Optional[str] = None
    active:  bool = Field(default=True)
    created_at: dt.datetime = Field(default_factory=_now)

    ohlcv:      List["OHLCV"]      = Relationship(back_populates="ticker")
    indicators: List["Indicator"]  = Relationship(back_populates="ticker")
    analyses:   List["Analysis"]   = Relationship(back_populates="ticker")
    strategies: List["Strategy"]   = Relationship(back_populates="ticker")
    positions:  List["Position"]   = Relationship(back_populates="ticker")
    trades:     List["Trade"]      = Relationship(back_populates="ticker")
    intraday_daily_stats: List["IntradayDailyStats"] = Relationship(back_populates="ticker")
    intraday_signals:     List["IntradaySignal"]      = Relationship(back_populates="ticker")


# -- OHLCV ---------------------------------------------------------------------

class OHLCV(SQLModel, table=True):
    __tablename__ = "ohlcv"

    id:        str = Field(default_factory=_uuid, primary_key=True)
    ticker_id: str = Field(foreign_key="tickers.id", index=True)
    bar_time:  dt.datetime = Field(index=True)  # hourly timestamp (UTC, naive)

    open:   float
    high:   float
    low:    float
    close:  float
    volume: float

    created_at: dt.datetime = Field(default_factory=_now)

    ticker: Optional[Ticker] = Relationship(back_populates="ohlcv")


# -- Indicators ----------------------------------------------------------------

class Indicator(SQLModel, table=True):
    __tablename__ = "indicators"

    id:        str     = Field(default_factory=_uuid, primary_key=True)
    ticker_id: str     = Field(foreign_key="tickers.id", index=True)
    bar_date:  dt.date = Field(index=True)

    # Trend
    sma_20:  Optional[float] = None
    sma_50:  Optional[float] = None
    sma_100: Optional[float] = None
    sma_200: Optional[float] = None

    # Momentum
    rsi_14: Optional[float] = None

    # MACD (12, 26, 9)
    macd_line:   Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist:   Optional[float] = None

    # Volatility
    atr_14:    Optional[float] = None
    bb_upper:  Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower:  Optional[float] = None
    bb_width:  Optional[float] = None

    # Volume
    avg_volume_20: Optional[float] = None
    rel_volume:    Optional[float] = None

    # Derived
    trend:    Optional[str] = None   # UPTREND | DOWNTREND | SIDEWAYS
    momentum: Optional[str] = None   # STRONG | NEUTRAL | WEAK | OVERBOUGHT

    created_at: dt.datetime = Field(default_factory=_now)

    ticker: Optional[Ticker] = Relationship(back_populates="indicators")


# -- Analyses (Claude output) --------------------------------------------------

class Analysis(SQLModel, table=True):
    __tablename__ = "analyses"

    id:        str     = Field(default_factory=_uuid, primary_key=True)
    run_id:    str     = Field(index=True)
    ticker_id: str     = Field(foreign_key="tickers.id", index=True)
    bar_date:  dt.date = Field(index=True)

    # Claude's structured output
    trend:           Optional[str]   = None
    momentum:        Optional[str]   = None
    key_support:     Optional[float] = None
    key_resistance:  Optional[float] = None
    setup_quality:   Optional[int]   = None   # 0-100
    thesis:          Optional[str]   = None

    # Full raw response saved for auditing
    raw_response:    Optional[str]   = None
    model_used:      Optional[str]   = None
    prompt_tokens:   Optional[int]   = None
    response_tokens: Optional[int]   = None

    created_at: dt.datetime = Field(default_factory=_now)

    ticker:     Optional[Ticker]   = Relationship(back_populates="analyses")
    strategies: List["Strategy"]   = Relationship(back_populates="analysis")


# -- Strategies (Claude output) ------------------------------------------------

class Strategy(SQLModel, table=True):
    __tablename__ = "strategies"

    id:          str     = Field(default_factory=_uuid, primary_key=True)
    run_id:      str     = Field(index=True)
    ticker_id:   str     = Field(foreign_key="tickers.id", index=True)
    analysis_id: Optional[str] = Field(default=None, foreign_key="analyses.id")
    bar_date:    dt.date = Field(index=True)

    # Trade setup
    strategy_name:    Optional[str]   = None
    signal:           Optional[str]   = None   # BUY | NONE
    entry:            Optional[float] = None
    stop:             Optional[float] = None
    target:           Optional[float] = None
    rr:               Optional[float] = None
    conviction_score: Optional[int]   = None   # 0-100
    thesis:           Optional[str]   = None

    # Full raw response saved for auditing
    raw_response:     Optional[str]   = None
    model_used:       Optional[str]   = None

    created_at: dt.datetime = Field(default_factory=_now)

    ticker:        Optional[Ticker]          = Relationship(back_populates="strategies")
    analysis:      Optional[Analysis]        = Relationship(back_populates="strategies")
    risk_approval: Optional["RiskApproval"]  = Relationship(back_populates="strategy")


# -- Risk Approvals ------------------------------------------------------------

class RiskApproval(SQLModel, table=True):
    __tablename__ = "risk_approvals"

    id:          str = Field(default_factory=_uuid, primary_key=True)
    run_id:      str = Field(index=True)
    strategy_id: str = Field(foreign_key="strategies.id", index=True)

    approved:         bool
    rejection_reason: Optional[str]   = None
    # Float supports Alpaca's fractional crypto quantities while remaining
    # compatible with whole-share stock orders.
    shares:           Optional[float] = None
    risk_amount:      Optional[float] = None
    position_value:   Optional[float] = None

    created_at: dt.datetime = Field(default_factory=_now)

    strategy: Optional[Strategy] = Relationship(back_populates="risk_approval")
    order:    Optional["Order"]  = Relationship(back_populates="risk_approval")


# -- Orders --------------------------------------------------------------------

class Order(SQLModel, table=True):
    __tablename__ = "orders"

    id:               str  = Field(default_factory=_uuid, primary_key=True)
    run_id:           str  = Field(index=True)
    risk_approval_id: str  = Field(foreign_key="risk_approvals.id", index=True)

    alpaca_order_id:  Optional[str]   = None
    symbol:           str
    side:             str
    qty:              float
    order_type:       str
    status:           str
    filled_price:     Optional[float] = None
    stop_price:       Optional[float] = None
    target_price:     Optional[float] = None
    dry_run:          bool            = Field(default=True)
    error_message:    Optional[str]   = None

    created_at: dt.datetime          = Field(default_factory=_now)
    filled_at:  Optional[dt.datetime] = None

    risk_approval: Optional[RiskApproval] = Relationship(back_populates="order")
    position:      Optional["Position"]   = Relationship(back_populates="order")
    evaluation:    Optional["EvaluationLedger"] = Relationship(back_populates="order")


# -- Positions -----------------------------------------------------------------

class Position(SQLModel, table=True):
    __tablename__ = "positions"

    id:        str  = Field(default_factory=_uuid, primary_key=True)
    ticker_id: str  = Field(foreign_key="tickers.id", index=True)
    order_id:  Optional[str] = Field(default=None, foreign_key="orders.id")

    entry_date:     dt.date
    entry_price:    float
    shares:         float
    stop_price:     float
    target_price:   float
    current_price:  Optional[float] = None
    unrealized_pnl: Optional[float] = None

    alpaca_position_id: Optional[str] = None
    status: str = Field(default="open")

    created_at: dt.datetime = Field(default_factory=_now)
    updated_at: dt.datetime = Field(default_factory=_now)

    ticker: Optional[Ticker]  = Relationship(back_populates="positions")
    order:  Optional[Order]   = Relationship(back_populates="position")
    trade:  Optional["Trade"] = Relationship(back_populates="position")


# -- Trades (completed round-trips) --------------------------------------------

class Trade(SQLModel, table=True):
    __tablename__ = "trades"

    id:          str  = Field(default_factory=_uuid, primary_key=True)
    ticker_id:   str  = Field(foreign_key="tickers.id", index=True)
    position_id: str  = Field(foreign_key="positions.id")

    entry_date:   dt.date
    entry_price:  float
    exit_date:    dt.date
    exit_price:   float
    shares:       float
    pnl:          float
    pnl_pct:      float
    exit_reason:  str
    holding_days: int

    created_at: dt.datetime = Field(default_factory=_now)

    ticker:   Optional[Ticker]   = Relationship(back_populates="trades")
    position: Optional[Position] = Relationship(back_populates="trade")


# -- Paper evaluation ledger --------------------------------------------------

class EvaluationLedger(SQLModel, table=True):
    """One auditable research observation per non-dry-run paper order.

    Alpaca reconciliation updates this local row using GET-only broker calls.
    It is intentionally separate from the operational Order/Position/Trade
    state so a research audit can never change how a live position is managed.
    """
    __tablename__ = "evaluation_ledger"
    __table_args__ = (UniqueConstraint("order_id"),)

    id:          str = Field(default_factory=_uuid, primary_key=True)
    order_id:    str = Field(foreign_key="orders.id", index=True)
    strategy_id: str = Field(foreign_key="strategies.id", index=True)
    strategy_version: str = Field(index=True)
    market:      str = Field(index=True)  # "stock" | "crypto"
    symbol:      str = Field(index=True)

    decision_time_utc:  dt.datetime
    signal_bar_end_utc: Optional[dt.datetime] = None
    expected_entry:     float
    expected_stop:      float
    expected_target:    float
    expected_qty:       float

    broker_order_id:    Optional[str] = Field(default=None, index=True)
    broker_order_status: Optional[str] = None
    reconciliation_status: str = Field(default="unreconciled", index=True)
    discrepancy:        Optional[str] = None

    filled_at_utc:      Optional[dt.datetime] = None
    actual_fill_price:  Optional[float] = None
    entry_slippage_bps: Optional[float] = None
    fee_amount:         Optional[float] = None
    fee_source:         Optional[str] = None

    exit_time_utc:      Optional[dt.datetime] = None
    exit_price:         Optional[float] = None
    exit_reason:        Optional[str] = None
    gross_pnl:          Optional[float] = None
    net_pnl:            Optional[float] = None

    last_reconciled_at: Optional[dt.datetime] = None
    created_at:         dt.datetime = Field(default_factory=_now)
    updated_at:         dt.datetime = Field(default_factory=_now)

    order: Optional[Order] = Relationship(back_populates="evaluation")


# -- TradingAgents Signals -----------------------------------------------------

class TradingAgentsSignal(SQLModel, table=True):
    __tablename__ = "tradingagents_signals"

    id:        str     = Field(default_factory=_uuid, primary_key=True)
    run_id:    str     = Field(index=True)
    ticker_id: str     = Field(foreign_key="tickers.id", index=True)
    bar_date:  dt.date = Field(index=True)

    # Top-level verdict: Buy / Overweight / Hold / Underweight / Sell
    signal:           Optional[str] = None
    # Full Portfolio Manager decision text
    final_decision:   Optional[str] = None
    # Individual analyst reports (truncated to 2000 chars each to stay DB-friendly)
    market_report:       Optional[str] = None
    news_report:         Optional[str] = None
    fundamentals_report: Optional[str] = None
    sentiment_report:    Optional[str] = None

    model_used:  Optional[str] = None
    created_at:  dt.datetime   = Field(default_factory=_now)


# -- Options research ----------------------------------------------------------

class OptionsResearch(SQLModel, table=True):
    """Auditable, non-executable options contract recommendation."""
    __tablename__ = "options_research"

    id:            str = Field(default_factory=_uuid, primary_key=True)
    run_id:        str = Field(index=True)
    strategy_id:   str = Field(foreign_key="strategies.id", index=True)
    underlying:    str = Field(index=True)
    contract_symbol: str = Field(index=True)
    contract_type: str
    expiration_date: dt.date = Field(index=True)
    strike_price:  float
    multiplier:    int = Field(default=100)

    bid:           float
    ask:           float
    midpoint:      float
    spread_pct:    float
    delta:         Optional[float] = None
    implied_volatility: Optional[float] = None
    open_interest: int
    days_to_expiration: int

    contracts:     int
    premium_per_contract: float
    total_premium: float
    max_loss:      float
    status:        str = Field(default="research_only")
    rationale:     Optional[str] = None
    created_at:    dt.datetime = Field(default_factory=_now)


# -- Intraday day-trading mode (5-min ORB, signal-only) ------------------------
#
# Deliberately separate from OHLCV/Indicator: those are hourly-bar, one-row-
# per-ticker-per-DAY tables used by the swing pipeline. Day-mode signals are
# NEVER written to Strategy/RiskApproval - see nodes/day_strategy_node.py and
# run_pipeline.py's --stock-strategy day handling for why that isolation is
# structural, not just a config flag.

class IntradayDailyStats(SQLModel, table=True):
    """
    Prior-session-only reference stats (14-day avg volume, daily ATR-14,
    14-session avg opening-bar volume) computed by intraday_reference_node
    before the market opens, so the time-critical opening job only has to
    fetch today's single opening bar. `interval` is recorded because
    avg_opening_volume_14d is timeframe-specific (5-min vs 15-min opening
    volume are different numbers).
    """
    __tablename__ = "intraday_daily_stats"
    __table_args__ = (UniqueConstraint("ticker_id", "as_of_session", "interval"),)

    id:             str     = Field(default_factory=_uuid, primary_key=True)
    ticker_id:      str     = Field(foreign_key="tickers.id", index=True)
    as_of_session:  dt.date = Field(index=True)  # session these stats apply to; computed from data through the PRIOR session only
    interval:       str     = Field(default="5Min", index=True)

    avg_daily_volume_14d:   Optional[float] = None
    daily_atr_14:           Optional[float] = None
    avg_opening_volume_14d: Optional[float] = None

    computed_at: dt.datetime = Field(default_factory=_now)

    ticker: Optional[Ticker] = Relationship(back_populates="intraday_daily_stats")


class IntradaySignal(SQLModel, table=True):
    """
    One row per ticker per session-date-interval-strategy_version. Written/
    upserted by day_strategy_node at the open (candle/filter/rank fields) and
    by intraday_shadow_node after the close (outcome fields, all None until
    then). This table is the ONLY output of day-mode signal generation -
    never Strategy/RiskApproval - which is what makes it structurally
    impossible for day-mode to reach risk_node/execution_node.
    """
    __tablename__ = "intraday_signals"
    __table_args__ = (
        UniqueConstraint("ticker_id", "session_date", "interval", "strategy_version"),
    )

    id:            str     = Field(default_factory=_uuid, primary_key=True)
    run_id:        str     = Field(index=True)
    ticker_id:     str     = Field(foreign_key="tickers.id", index=True)
    session_date:  dt.date = Field(index=True)
    interval:          str = Field(default="5Min", index=True)
    strategy_version:  str = Field(default="orb_v1", index=True)
    strategy_mode:     str = Field(default="day")

    opening_bar_time: dt.datetime
    opening_open:     float
    opening_high:     float
    opening_low:      float
    opening_close:    float
    opening_volume:   float
    candle_type:      str             # "bullish" | "bearish" | "doji"
    direction:        Optional[str]   = None   # "long" | "short" | None

    opening_price: float   # the 9:30 opening price used by the price>$5 filter
    avg_daily_volume_14d:   Optional[float] = None
    daily_atr_14:           Optional[float] = None
    avg_opening_volume_14d: Optional[float] = None
    opening_rel_volume:     Optional[float] = None
    passed_filters:  bool           = Field(default=False)
    rank:            Optional[int]  = None
    selected:        bool           = Field(default=False)
    entry_trigger_price: Optional[float] = None
    stop_price:          Optional[float] = None
    rejection_reason:    Optional[str]   = None

    # Simulated sizing context, sized using the INTENDED_DEPLOYMENT backtest
    # scenario (0.25% risk, $100k assumed equity) - not both backtest
    # scenarios; those remain backtest-report-only outputs.
    simulated_quantity:       Optional[float] = None
    simulated_risk_amount:    Optional[float] = None
    simulated_position_value: Optional[float] = None
    account_equity_used:      Optional[float] = None
    cost_model_version:       Optional[str]   = None

    # Populated by intraday_shadow_node after the close; all None until then.
    breakout_triggered:   Optional[bool]     = None
    trigger_time:         Optional[dt.datetime] = None
    simulated_entry_price: Optional[float]   = None
    stop_hit:             Optional[bool]     = None
    exit_time:            Optional[dt.datetime] = None
    exit_price:           Optional[float]    = None
    exit_reason:          Optional[str]      = None   # "stop" | "eod_flatten" | "no_trigger"
    gross_pnl:            Optional[float]    = None
    cost_adjusted_pnl:    Optional[float]    = None
    pnl_r:                Optional[float]    = None
    outcome_ambiguous:    Optional[bool]     = None

    created_at: dt.datetime = Field(default_factory=_now)

    ticker: Optional[Ticker] = Relationship(back_populates="intraday_signals")


# -- Run Log -------------------------------------------------------------------

class RunLog(SQLModel, table=True):
    __tablename__ = "run_log"

    id:                str  = Field(default_factory=_uuid, primary_key=True)
    run_id:            str  = Field(index=True)
    node_name:         str
    status:            str
    tickers_processed: Optional[int]   = None
    records_written:   Optional[int]   = None
    error_message:     Optional[str]   = None
    duration_seconds:  Optional[float] = None

    started_at:  dt.datetime          = Field(default_factory=_now)
    finished_at: Optional[dt.datetime] = None
