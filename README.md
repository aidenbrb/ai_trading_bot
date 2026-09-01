# AI Trading Bot

An autonomous, multi-asset trading research and execution platform, combining a
rule-based signal pipeline with LLM-driven analysis and a formal, pre-registration-based
research methodology for deciding what earns real capital.

The project's central design bet: **most trading ideas are wrong, and the job of the
system is to prove that quickly, honestly, and in writing** — before any strategy ever
touches an order. Every strategy tested against this codebase is preregistered (rules,
universe, cost model, and pass/fail thresholds frozen and hashed *before* the backtest
runs), evaluated exactly once against a 16-check quantitative gate, and recorded in an
evidence ledger whether it passes or fails. As of this writing, four independent strategy
families — intraday stock structure/level/confirmation trading, crypto trend and
cross-sectional momentum, monthly ETF momentum, and short-term ETF mean reversion — have
gone through that process. All four failed the gate, and all four failures are documented
in detail rather than discarded. That is the point: the system is built to prevent a
plausible-looking backtest from becoming a live strategy on vibes alone.

## What this is

- A daily/intraday **pipeline** that scans a configurable universe, computes technical
  indicators, generates trade candidates, runs them through risk gates, and (when enabled)
  submits bracket orders to a live broker.
- A **multi-broker, multi-asset execution layer** — Alpaca (stocks/ETFs, paper trading) and
  Robinhood (stocks/ETFs, live account) on the equity side, Coinbase Advanced Trade for
  crypto — with routing decided per-symbol and a hard master dry-run switch that gates all
  order submission regardless of which broker is selected.
- An **LLM research layer**: Claude writes market analysis and generates strategy proposals,
  and an integration with the open-source [TradingAgents](https://github.com/TauricResearch/TradingAgents)
  framework runs a multi-agent bull/bear/risk-manager debate as an additional, independent
  signal before any trade proposal reaches the risk gate.
- A **from-scratch backtesting and qualification framework** (`backtest/`) purpose-built
  around evidentiary discipline rather than curve-fitting: frozen preregistrations with
  SHA-256 provenance, walk-forward out-of-sample validation, bootstrap confidence intervals,
  parameter-sensitivity plateau checks, and a one-shot rule that forbids re-running a
  strategy after seeing its result.
- A **live-deployment safety system** (`live_slc/`) for the one strategy that reached
  paper-trading: a two-tier (Tier-1/Tier-2) file-integrity guardrail, SSH-key cryptographic
  signing required for any transition into an active trading state, and an independently
  written verifier script with zero imports from the main codebase, run by the scheduler
  before every live process launch.

## Why it's built this way

Most retail trading-bot projects backtest a strategy, like the number, and ship it. This
one treats "does this strategy actually have an edge" as a research question that has to
survive:

1. **A frozen specification.** Every rule, every threshold, and the full parameter grid are
   written down and hashed *before* any result exists — see `research/*_preregistration.md`.
   A strategy's rules cannot be quietly adjusted after seeing how it performs.
2. **A single qualification run.** Parameter selection happens once, mechanically, by
   in-sample Sharpe over a frozen grid — never by picking whichever run looked best.
3. **A 16-check evidence gate** (`backtest/whole_bot_metrics.py::qualify_strategy`) covering
   trade-count sufficiency, data coverage, expectancy under multiple cost models, profit
   factor, Sharpe (absolute and vs. a passive benchmark), max drawdown, quarter-over-quarter
   consistency, a bootstrap lower-bound on mean risk-adjusted return, single-symbol
   concentration, out-of-sample Sharpe/profit-factor, and a sensitivity-plateau check that
   fails a strategy outright if its selected parameters sit on a knife's edge with too few
   robust neighbors.
4. **Honest failure reporting.** Every strategy's result — pass or fail — is written into a
   published evidence ledger with the actual numbers, including where the original
   hypothesis was simply wrong. A strategy that trailed its own zero-cost benchmark, or
   whose out-of-sample Sharpe collapsed, or that turned out to be untestable because a paper
   trading parity check caught a bug in the offline/live comparison, is reported that way —
   not quietly dropped.

## Architecture

```
monitor_node        -> scan universe, score setup quality, rank candidates
data_node            -> fetch OHLCV (Alpaca / yfinance)
indicator_node       -> RSI, MACD, Bollinger Bands, ATR, SMA/EMA, relative volume
analysis_node        -> Claude-generated market analysis report
tradingagents_node   -> multi-agent bull/bear/risk-manager debate (TauricResearch/TradingAgents)
strategy_node        -> Claude-generated entry/stop/target, informed by the debate verdict
risk_node            -> daily loss limit, max positions, earnings-proximity buffer,
                         TradingAgents veto, minimum reward:risk
execution_node       -> bracket orders to Alpaca / Robinhood (stocks) or Coinbase (crypto)
```

Additional nodes (`nodes/`) support day-trading-specific flows (preflight, shadow
evaluation, intraday reference data), options research, and post-trade reconciliation.

A market-regime filter adjusts the setup-quality threshold based on SPY's position
relative to its 20/50-day moving averages, tightening candidate selection in caution/bear
regimes.

## Research and backtesting

`backtest/` is a separate, self-contained simulation and qualification framework — it never
imports a broker's order-submission client, only market-data clients, so research code is
structurally incapable of placing a trade.

Highlights:

- **`whole_bot_engine.py`** — deterministic portfolio simulators for stock, crypto, and ETF
  strategies, each purpose-built for its own execution shape (intraday stop/target lifecycle,
  monthly cross-sectional rebalancing, or daily signal-driven admission with next-bar fills)
  rather than forced through one generic engine.
- **`whole_bot_metrics.py`** — the 16-check qualification gate, shared across every strategy
  family so the bar is identical regardless of asset class or timeframe.
- **Per-strategy preregistration documents** (`research/*_preregistration.md`) — each one
  pins the hypothesis, universe, data window, exact rule formulas (down to hand-verified
  indicator test vectors), the frozen parameter grid, and the qualification gate's exact
  thresholds, with a SHA-256 manifest recording the frozen file and every dependency it was
  built against.
- **Walk-forward validation** — parameters are always selected on an in-sample window and
  evaluated once, unmodified, on a later out-of-sample window.
- **Sensitivity/plateau analysis** — the selected parameter cell is checked against its
  immediate grid neighbors; a result that only "works" at one exact, isolated setting fails
  the gate even if its headline number looks good.
- **1,000+ automated tests** (`tests/`) covering indicator math against hand-computed
  vectors, no-lookahead execution timing, portfolio accounting edge cases, and the
  qualification gate's own logic.

## Live safety design (`live_slc/`)

The one strategy that progressed to live paper trading is guarded by a dedicated safety
layer, independent of the research code:

- **Tier-1 / Tier-2 file guardrails** — a hash-verified split between files that can never
  change without a fresh cryptographic signature (guardrails, execution-critical modules)
  and files that can change under normal development, with an explicit, documented gap
  analysis for edge cases like key rotation.
- **SSH-key signature required for any transition into an active trading state** — moving
  *out* of an active state (e.g., halting a strategy) requires no signature by design, so an
  operator can never be blocked from de-risking; moving *in* always does.
- **An independently written verifier** (`scripts/slc_live/verify_tier1_independent.py`)
  with zero imports from the rest of the codebase, its own read-only database connection,
  and its own subprocess call to `ssh-keygen -Y verify` — run by the scheduler before every
  live process launch, so a bug in the main authorization code can't silently disable the
  check meant to catch it.
- **Multiple independent dry-run switches** — `EXECUTION_ENABLED`, per-broker enable flags,
  and Alpaca's own hardcoded paper-mode client all have to agree before any order reaches a
  broker.

## Tech stack

Python 3.12 · SQLModel/SQLAlchemy (SQLite) · pandas/NumPy · Alpaca & Robinhood & Coinbase
Advanced Trade SDKs · yfinance · Claude (Anthropic API) · pytest · Windows Task Scheduler
for live orchestration.

## Project layout

```
run_pipeline.py        entry point for the daily rule-based (+ optional --llm) pipeline
nodes/                 pipeline stages (monitor, data, indicator, strategy, risk, execution, ...)
config/                typed settings, universe definitions (stocks by sector, crypto list)
db/                     SQLModel schema and connection handling
backtest/               simulators, qualification gate, per-strategy research runners
research/               frozen preregistration documents and their SHA-256 manifests
live_slc/               live-deployment guardrails, authorization, and the paper-trading reducer
utils/                  indicators, market calendar, broker/data-client helpers
scripts/                scheduled-task entry points (.bat wrappers) and independent verifiers
tests/                  1,000+ tests across indicators, engines, guardrails, and the gate
```

## Getting started

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env
# fill in ALPACA_API_KEY / ALPACA_SECRET_KEY for paper-account data (optional —
# the pipeline runs against a simulated $100k account without them)

python run_pipeline.py --tickers AAPL MSFT NVDA   # rule-based, no API key required
python -m pytest tests/ -q                         # run the test suite
```

`EXECUTION_ENABLED`, `ROBINHOOD_ENABLED`, and `COINBASE_ENABLED` all default to `false` —
the full pipeline runs end-to-end without ever placing a real order. `LIVE_TRADING=true` is
additionally blocked in code. See `.env.example` for the complete list of configuration
flags.

The `--llm` flag enables Claude-driven analysis and strategy generation (requires
`ANTHROPIC_API_KEY`); the TradingAgents multi-agent debate additionally requires cloning
[TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) and
pointing `TRADINGAGENTS_REPO` at it.

## Current status

Four strategy families have been run through the qualification gate; none currently
qualifies for live capital. Full results, including the exact numbers behind each pass/fail
check, are written up as an evidence ledger covering every strategy tested to date. This is
treated as the expected, healthy outcome of a research process designed to reject weak
ideas rather than as a setback — the value of the project is the process that reliably says
no, not a specific strategy that happened to pass.

## Disclaimer

This is a research and engineering project, not financial advice, and is not intended for
use with real capital without independent review. `EXECUTION_ENABLED` and the broker-enable
flags are `false` by default for this reason.
