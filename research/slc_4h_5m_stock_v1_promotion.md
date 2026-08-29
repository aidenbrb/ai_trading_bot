# slc_4h_5m_stock_v1 — Promotion and Kill Criteria

**Status: DRAFT, for operator review. This document authorizes nothing by
itself.** It defines the criteria a future promotion decision (paper_active
→ live, and any live-capital scaling step) must be evaluated against. Actually
moving to `live` still requires a separately-reviewed, human-signed
`record_transition(..., to_status="live", ...)` event under the Phase 6
Step 1 signature gate, exactly as `paper_active` itself does today.

Frozen: pending. This file becomes append-immutable once its SHA-256 is
recorded in `live_slc/guardrails.py`'s `GUARDRAILS_TIER2` set (Phase 6
Step 3) — any later change requires a new version or a separately hashed
amendment, matching the convention `slc_4h_5m_stock_v1_preregistration.md`
already established.

## 1. Why this document exists

`slc_4h_5m_stock_v1_preregistration.md` (Section 6) and
`slc_4h_5m_stock_v1_paper_forward_activation_proposal.md` (Section 11)
together establish a floor: **at least 90 calendar days and 30 closed
trades under `paper_active`** before results may be used to justify any
further stage. Neither document states what those results actually have
to show. This document closes that gap — stricter than the pre-registered
floor, and with an explicit kill switch, not just a promotion bar.

## 2. Promotion criteria (paper_active → live)

Evaluated only once **both** of the following are true:
- **≥90 calendar days** under `paper_active` (unchanged from Section 11).
- **≥50 closed trades** (stricter than Section 11's 30-trade floor — a
  deliberate choice, not an oversight: 30 trades is barely enough to
  compute a stable win rate, let alone compare a drawdown profile against
  a backtest distribution).

All of the following must hold simultaneously. This is a conjunction, not
a scorecard — failing any one criterion means promotion does not proceed,
regardless of how well the others look.

1. **Expectancy and profit factor, stressed cost model.** Recompute every
   paper trade's net P&L using the stressed cost model (13 bps per leg,
   applied to each trade's actual notional — `qty * (entry_price +
   exit_price) * 0.0013`, the same convention used throughout this
   project's backtests). Over the full ≥50-trade paper sample:
   `net_expectancy > 0` **and** `profit_factor >= 1.15`.
2. **Paper mean R inside the backtest's bootstrap interval, same-length
   window.** See Section 3 for what "same-length window" means precisely.
   The paper sample's mean `pnl_r` must fall within the 95% bootstrap
   confidence interval computed over same-length windows of the
   historical backtest's trade sequence — i.e., paper performance must
   not be a statistically distinguishable regime from what the backtest
   itself considers normal variation.
3. **Max drawdown no worse than the backtest's worst same-length rolling
   window.** The paper sample's max drawdown (in R, over its equity
   curve) must not exceed the single worst max-drawdown observed across
   every same-length rolling window in the historical backtest.
4. **Zero suspension-class incidents in the final 30 days; zero
   unreconciled positions.** No `SlcActivationEvent` transitioning
   `paper_active -> suspended` (or any transition originating from an
   `ambiguous`/`protected_degraded` incident) in the 30 calendar days
   immediately preceding the promotion evaluation. Zero `SlcPosition`
   rows in `ambiguous` or `protected_degraded` status at evaluation time,
   and zero unresolved entries in `risk.system_wide_entry_block_reasons()`.
5. **First live stage is capped, not a full switch.** Even after 1-4 pass,
   the first live-capital stage runs at **≤20% of intended capital** for
   **≥50 further closed trades under the identical rules** (same
   `slc_4h_5m_stock_v1` frozen signal logic, same 0.25% risk/5-position/
   2-entry-per-day/20%-concentration/2%-daily-loss limits from
   Section 4 of the preregistration) before any further scaling is even
   considered. That second evaluation is a fresh application of Section 2
   in full — the ≤20% stage does not skip re-evaluating criteria 1-4
   against its own trade sample.

## 3. "Same-length window" methodology

Criteria 2 and 3 reference the deferred historical backtest described in
`slc_4h_5m_stock_v1_preregistration.md` Section 6 and permitted to remain
deferred by `amendment_003.md`. **That backtest has not been completed as
of this document's drafting** — this section defines the methodology to
apply once it is, not a result available today.

Let *N* = the actual number of closed paper trades being evaluated at
promotion time (≥50, per Section 2). Let the backtest's full closed-trade
sequence, in chronological order, be *T* = (t₁, t₂, ..., tₘ).

- **Same-length window** = any contiguous run of *N* consecutive trades
  from *T*: (tᵢ, tᵢ₊₁, ..., tᵢ₊ₙ₋₁) for every valid starting index *i*.
  This is a rolling window over trade count, not calendar time — a
  50-trade paper sample is compared against every 50-trade stretch the
  backtest ever produced, regardless of how many calendar days each
  stretch took.
- **Bootstrap interval (criterion 2):** for each same-length window,
  compute its mean `pnl_r`. Across all windows, report the 2.5th and
  97.5th percentiles of that distribution as the 95% interval. (This is a
  distribution *over rolling windows of the backtest's own realized
  trades*, not a resample-with-replacement bootstrap of a single window —
  deliberately, since the question is "does paper look like a normal
  stretch of the backtest," not "what's the uncertainty on one stretch's
  mean.")
- **Worst rolling drawdown (criterion 3):** for each same-length window,
  compute its own max drawdown (in R, from that window's own trade
  sequence, not the full backtest equity curve). Report the single worst
  (most negative) value across all windows.
- If *N* exceeds the number of trades in the backtest's own sequence
  (*N* > *m*), same-length comparison is impossible — this is
  automatically a failure of criteria 2 and 3, not a case that falls back
  to a shorter window silently.

**This methodology cannot be evaluated today.** Promotion cannot honestly
be assessed against criteria 2-3 until the deferred historical backtest
is actually run. This document does not un-defer it or set a deadline for
it — it only makes explicit that promotion has a real prerequisite beyond
accumulating paper trades.

## 4. Kill criteria (paper_active or live — any one triggers flat-and-disable)

These apply from the moment `paper_active` begins, not only after
promotion — a kill condition during paper trading is exactly as real a
signal as one during live trading, just lower-stakes. "Flat-and-disable"
means: the emergency-exit path closes every open SLC-originated position,
`record_transition()` moves the deployment status to `suspended`, and the
Scheduled Tasks are disabled pending human review — never an automatic
resumption.

1. **Drawdown beyond the backtest's worst same-length rolling window**
   (Section 3's methodology, evaluated continuously against the trailing
   *N*-trade window where *N* = trades elapsed so far, once *N* ≥ 10 —
   below 10 trades the comparison is too noisy to act on).
2. **A losing streak beyond the backtest's worst** (longest consecutive
   run of losing trades observed anywhere in the backtest's full trade
   sequence).
3. **Trailing-30-trade expectancy after costs < 0** (stressed 13 bps/leg
   model, same formula as Section 2 criterion 1, evaluated on the most
   recent 30 closed trades specifically — a rolling check, re-evaluated
   after every close, not a one-time gate).
4. **Any reconciliation incident while live** — any
   `SlcAuditEvent.event_type` in the critical set already tracked by
   `live_slc/check_paper_session_audit.py`
   (`protected_degraded`, `ambiguous_quantity`,
   `discovered_fill_unresolvable_identity`,
   `orphan_broker_position_unresolved`, `split_evidence_conflict`,
   `split_rebuild_failed`) occurring while `status == "live"`. Unlike
   criteria 1-3, this has zero tolerance and zero trade-count minimum —
   one incident is sufficient.

None of these four are evaluated by new code as of this document's
drafting — they are criteria for a future implementation, not yet wired
into `live_slc/authorization.py` or any scheduled check. Implementing the
kill-switch enforcement itself is separate follow-on work, out of scope
for this document.

## 5. What this document does not do

- It does not itself authorize `paper_active`, `live`, or any capital
  change — see the status line above.
- It does not complete or schedule the deferred historical backtest.
- It does not implement the kill-criteria monitoring described in
  Section 4 — that is future code, not existing enforcement.
- It does not change any frozen rule in
  `slc_4h_5m_stock_v1_preregistration.md` or its amendments.

## Reviewer checklist (for the human approving this document)

- [ ] The 90-day/50-trade floor (stricter than pre-registration's 30) is
      the deliberate choice you want, not pre-registration's original
      number.
- [ ] Section 3's "same-length window" methodology is the comparison you
      want against the (still-deferred) historical backtest — not, e.g.,
      a fixed calendar-time window instead of a trade-count window.
- [ ] You understand the deferred historical backtest is a genuine
      prerequisite this document surfaces but does not resolve.
- [ ] You accept that this document, once its hash is added to
      `GUARDRAILS_TIER2`, becomes append-immutable under the same
      discipline as the preregistration and its amendments.
