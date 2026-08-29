# slc_4h_5m_stock_v1 — Paper-Forward Activation Proposal

**Status: PAPER-FORWARD ACTIVATION PROPOSAL. This document provides no authorization until explicitly approved by the operator, hash-pinned in `SlcDeploymentStatus`, and accompanied by a recorded `dry_run -> paper_active` transition.**

## 1. Scope of this proposal

This document requests authorization to transition `slc_4h_5m_stock_v1` from `dry_run` to `paper_active` — i.e., to allow it to submit real (paper) orders to the shared Alpaca paper trading account, under the existing, unmodified strategy rules and risk limits.

**This proposal does not, and cannot, authorize:**
- Real-money trading of any kind.
- Options trading.
- Any tuning of the strategy's signal, entry, exit, sizing, or risk logic.
- Any change to the frozen rules in `research/slc_4h_5m_stock_v1_preregistration.md` or its amendments.

## 2. Strategy version and account binding

- **Strategy version**: `slc_4h_5m_stock_v1`
- **Broker**: Alpaca, paper trading account `2c2b4f39-f521-4594-80b9-b7e8ab7acd57`
- This is the same account already pinned as `SLC_EXPECTED_ACCOUNT_ID` in `.env` and referenced in every prior `SlcActivationEvent` for this strategy. Every entry submission gate (`live_slc/guardrails.py`) verifies the observed account ID against this value before acting, and blocks on any mismatch.

## 3. Monday 2026-08-17 dry-run evidence (complete session record)

The full `SlcSessionStat` row for `2026-08-17`, read directly from `live_slc/live_slc.db`, every column, no omissions:

| Field | Value |
|---|---|
| `session_date` | 2026-08-17 |
| `trades_opened` | 0 |
| `trades_closed` | 0 |
| `wins` | 0 |
| `losses` | 0 |
| `net_pnl` | 0.0 |
| `daily_loss_limit_breached` | False |
| `signals_generated` | 72 |
| `signals_acted_on` | 2 |
| `signals_skipped_not_shortable` | 0 |
| `signals_skipped_stale_or_missing_data` | 1 |
| `signals_skipped_capacity` | 68 |
| `valid_bar_coverage_pct` | 97.23200312989046 |
| `cycles_run` | 72 |
| `cycles_over_budget` | 0 |
| `duplicate_or_stale_signal_count` | 0 |
| `guardrail_check_passed` | True |
| `engine_parity_check_passed` | True |
| `closeout_left_no_open_state` | True |
| `expected_symbol_count` | 10224 |
| `valid_symbol_count` | 9941 |
| `failed_cycles` | 0 |
| `overlapping_cycles` | 0 |
| `reconciliation_discrepancy_count` | 0 |
| `unprotected_position_incident_count` | 0 |
| `dry_run_proposal_count` | 2 |
| `closeout_confirmed_flat_by_broker_readback` | True |
| `updated_at` | 2026-08-17 12:36:53.273283 |

This session passed every check `live_slc/authorization.py::evaluate_dry_run_session_gate()` applies, independently re-confirmed via `python -m live_slc.check_session_gate --date 2026-08-17` → `PASS - dry-run session gate satisfied.`

### Frozen validation-corpus reducer parity

Independently re-verified against the frozen corpus (`research/slc_4h_5m_stock_v1_reducer_validation_corpus_manifest.json`), within the declared evaluation window:

- **AAPL**: 28 long, 10 short reference signals — reducer output matched the frozen batch engine exactly.
- **AMD**: 1 long, 26 short reference signals — reducer output matched the frozen batch engine exactly.

Both directions are represented for both symbols; the live reducer's signal generation logic is proven, not assumed, to agree with the frozen offline engine across long and short cases.

## 4. Risk parameters (unchanged, cited to the frozen source)

Every numeric limit below is quoted from `research/slc_4h_5m_stock_v1_preregistration.md`, Section 4 ("Portfolio and costs"), lines 116-119 — not merely from `live_slc/risk.py`'s constants, though the code implements exactly these values and nothing else:

> "Starting equity $100,000; cash-only; no leverage; maximum five positions; maximum two entries per day; one open position per symbol; maximum 20% of equity per position; stop approving entries after a 2% realized daily loss."
> "Primary research risk is 0.25% of equity per trade."

Reaffirmed unchanged by `research/slc_4h_5m_stock_v1_amendment_004.md`, lines 57-59: "This amendment does not change the 0.25% risk, five-position, two-entry, 20% concentration, or 2% daily-loss rules in Section 4."

Summary:
- **0.25%** of equity risked per trade.
- **Maximum 5** concurrent SLC positions.
- **Maximum 2** new entries per calendar day.
- **20%** maximum concentration per position.
- **2%** realized daily-loss entry halt (new entries stop; does not force-close existing positions).

This proposal changes none of these values in any way.

## 5. Trading behavior

- **Both directions**: long and short paper trades, per the frozen signal logic.
- **Intraday only**: no position is intentionally held overnight.
- **Mandatory end-of-day closeout**: the closeout guardian (Scheduled Task, calendar-derived trigger starting 12:55:45 ET, retrying every minute through market close plus a reconciliation window) is responsible for flattening every open SLC position before end of day and confirming a broker-side read-back that the account is flat of SLC-originated positions.

## 6. IEX data-source deviation (disclosed, not silently absorbed)

Quoted verbatim from `research/slc_4h_5m_stock_v1_amendment_004.md`, lines 15-19:

> "1. Data source. Section 5 specifies split-adjusted Alpaca SIP historical data. Live paper-forward decisions instead use Alpaca IEX real-time data, because SIP is not available in real time on this account tier. This is a genuine data-source substitution, disclosed here rather than silently absorbed."

This deviation was already authorized as part of amendment 004 and is not newly introduced by this proposal. It is repeated here so the activation record is self-contained.

## 7. Paper-fill limitations and shared-account considerations

**Paper-fill realism**: Alpaca's paper trading environment does not model slippage, partial fills, or market-impact behavior identically to live markets. Fill prices and fill timing in paper trading are a reasonable but imperfect proxy for what a live order would experience. This is inherent to any paper-trading venue, not specific to this strategy.

**Shared Alpaca account**: The paper account used by SLC is the same account the main bot pipeline (`run_pipeline.py`, `nodes/execution_node.py`) is configured to use. Isolation between the two systems rests on:

- **Separate databases**: `live_slc/live_slc.db` (SLC) vs. `trading.db` (main bot) — two physically separate SQLite files, no shared table, AST-verified (`tests/test_live_slc_models.py`) that `live_slc/` never imports `db.connection`/`db.models` anywhere in the package.
- **SLC's own account-wide entry check** (`live_slc/risk.py`, `check_new_entry_capacity`): SLC refuses to open a new position in any symbol that already has a position anywhere in the shared account, regardless of which system opened it — frozen logic, unmodified by this proposal.
- **Checked preconditions of this activation, not one-time assumptions**: at the time of activation, `trading.db`'s own `positions` table holds zero open or pending rows, and all three strategies registered in `utils/strategy_registry.py` (`stock_trend_momentum_v1`, `crypto_trend_momentum_v1`, `orb_v1`) are `execution_eligible=False` — meaning the main bot cannot currently open any new position at all, by an independent, already-existing gate in its own code. These facts are re-verified immediately before activation and again as part of Tuesday's operational checklist; activation does not proceed if either is no longer true.

A known, currently-unreachable defect in the main bot's own closing logic (`nodes/monitor_node.py`) is tracked separately and documented as a prerequisite for ever re-enabling any main-bot strategy while SLC is `paper_active` — it does not block this activation because it cannot be triggered under the current, checked state described above.

## 8. Historical backtest remains deferred

`research/slc_4h_5m_stock_v1_preregistration.md`, Section 6, lines 146-148:

> "Passing this historical test does not enable execution. It permits a separate paper-forward proposal at 0.25% risk for at least 90 calendar days and 30 closed trades. Until that proposal is separately reviewed, this version is unregistered, research-only, and structurally unable to submit an order."

**This document is exactly that separate paper-forward proposal.** It is not, and does not claim to be, evidence that the deferred historical backtest has been run or passed. The historical backtest remains a distinct, still-pending piece of work.

The full-scope historical backtest has not been completed. This is permitted, not overlooked: `research/slc_4h_5m_stock_v1_amendment_003.md`, lines 19-30, quoted verbatim:

> "This amendment permits proposing and running a paper-forward trial under Section 6 without first passing Section 6's historical qualification gates. The full-scope historical backtest begun under this preregistration is deferred, not abandoned: its cache and results directory remain untouched and the run remains resumable at any time.
>
> This does not weaken any other part of Section 6. Real order submission — including paper order submission — still requires a separate, human-authored, hash-recorded proposal document, separately reviewed, before this strategy is registered as anything other than unregistered and research-only. That proposal must commit any paper-forward trial to run for at least 90 calendar days and 30 closed trades before its results are used to evaluate any further step."

Amendment 003 permits this paper-forward trial to proceed without the historical qualification gates having passed first — it does not, and this proposal does not, remove or shorten the separately-reviewed/hash-recorded proposal requirement (Section 13 below) or the 90-calendar-day/30-closed-trade commitment (Section 11 below). This document satisfies both of those retained requirements; it does not bypass them.

## 9. No profitability or "optimal strategy" claim

This proposal makes no claim that `slc_4h_5m_stock_v1` is profitable, optimal, or superior to any alternative. It requests permission to observe real (paper) execution behavior under controlled risk limits — nothing more.

## 10. Automatic fail-closed behavior

The following are properties of the existing, already-implemented and tested code — not new commitments made by this document:

- **Ambiguous order submissions**: a broker response that isn't a definitive confirmed-rejected or confirmed-filled outcome (network error, timeout, unclear status) is never assumed successful or failed — recorded ambiguous and resolved only by explicit delayed reconciliation (`live_slc/execution.py`'s 3-outcome model).
- **Unprotected positions**: a fill that doesn't reach confirmed stop/target protection is marked `protected_degraded`, never described as a healthy open position, and blocks all new SLC entries system-wide until resolved via the guarded emergency-exit path.
- **Account mismatch**: every precondition gate (`live_slc/guardrails.py`) compares the observed Alpaca account ID against the pinned expected ID before any action; any mismatch blocks.
- **Data failures**: a broker account read that returns missing/non-numeric/NaN/infinite values, or a missing live quote for an in-flight order, raises `AccountSnapshotUnusable` and blocks — never a fabricated `$0`-notional fallback.
- **Guardrail drift**: any byte-level change to a Tier-1 or Tier-2 guarded file since the frozen baseline blocks operation until explicitly re-frozen through a new, dated baseline and a new `SlcActivationEvent`.
- **Broker/database discrepancies**: `risk.system_wide_entry_block_reasons()` blocks all new entries on any unresolved ambiguous order state, quantity mismatch, unresolved orphan broker position, or existing `protected_degraded` position — checked fresh before every single candidate, not once per cycle.

## 11. Commitment

A minimum of **90 calendar days** and **30 closed paper trades** under `paper_active` status is required before any results from this activation are used to justify moving to a further stage (e.g., live capital). This is not a target to be shortened based on early results in either direction.

## 12. Explicit prohibitions under this authorization

The following remain prohibited for `slc_4h_5m_stock_v1` under this proposal, without exception:
- Real-money (live) trading.
- Options trading.
- Any strategy tuning: changes to market structure, zone detection, stochastic confirmation, entry/stop/target construction, ranking, sizing, or risk-limit logic.
- Any change to the frozen rule documents this strategy is built on.

## 13. Append-immutability

Upon approval, this document's SHA-256 hash will be computed and pinned as `SlcDeploymentStatus.activation_proposal_sha256` via `authorization.record_transition("dry_run", "paper_active", ...)`. From that point forward, `live_slc/guardrails.py::assert_submission_preconditions()` independently re-hashes this exact file and compares it against the pinned hash **before every single future entry attempt** — not just once at activation. Any edit to this document after approval will cause every future entry attempt to block until either the document is reverted to its approved text or a new activation is separately authorized. This is enforced by the running code, not merely a stated intention.

---

## Reviewer checklist (for the human approving this document)

- [ ] Section 3's session data matches your own independent check of `live_slc/live_slc.db` (or the `check_session_gate.py`/`check_readiness.py` output you've already seen).
- [ ] Sections 4, 6, 8 accurately quote the frozen documents (spot-check against `research/slc_4h_5m_stock_v1_preregistration.md` and `amendment_004.md` directly if desired).
- [ ] You understand and accept the shared-account risk described in Section 7, including that it depends on the main bot's current state remaining as described (re-checked before activation, not merely assumed).
- [ ] You understand this document becomes hash-pinned and effectively immutable once approved and activated.
- [ ] **Robinhood credentials exposed earlier this session have been rotated or removed** — this is a stop condition for activation independent of this proposal's content, tracked separately in the activation plan (Part B0).

**No action beyond drafting this file has been taken.** Its SHA-256 has not been computed or recorded anywhere; `.env` is untouched; no `record_transition()` call has been made; Part B of the activation plan has not begun.
