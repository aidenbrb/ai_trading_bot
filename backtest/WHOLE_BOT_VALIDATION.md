# Whole-bot evidence workflow

This workflow evaluates the versioned stock and crypto swing rules. It is
research-only: the comparison modules import Alpaca market-data clients but no
trading client or execution node.

## Commands

```powershell
# Small live-data and deterministic-repeatability check
python -m backtest.run_strategy_comparison --smoke --determinism-check

# Locked January 2022 through August 9, 2026 evaluation
python -m backtest.run_strategy_comparison --determinism-check --label whole_bot_v1_fixed

# GET-only paper-order/position audit
python -m nodes.evaluation_reconciliation_node --json-out data\evaluation_reconciliation.json
```

The full run uses split-adjusted Alpaca SIP stock data and Alpaca-US crypto
data. It never falls back to Yahoo or Coinbase. Hourly history is cached in
bounded, independently committed chunks; minute bars are downloaded only for
admitted orders.

Results are written under `backtest/results/whole_bot/<timestamp>/`. The
configuration includes a SHA-256 digest, the exact universes, decision clock,
cost assumptions, risk scenarios, strategy registry, and the historical
news/earnings limitation. Reports include trades, exclusions, missing outcome
data, rejected orders, equity curves, quarterly/yearly results, benchmarks,
and every qualification check.

Historical qualification never enables automatic entries by itself. A passing
strategy is only a paper-rollout candidate and must subsequently pass the
versioned 90-day/30-closed-trade forward gate reported by the reconciliation
command. No entry Scheduled Task is installed by this workflow.
