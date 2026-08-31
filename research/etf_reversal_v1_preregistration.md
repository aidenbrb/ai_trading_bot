# ETF short-term mean reversion v1 — frozen preregistration

Frozen: 2026-08-30. This document is a research specification, not evidence
that the strategy is profitable and not authorization to submit orders.

Version identifier: `etf_reversal_v1`.

## 1. Hypothesis

Short-horizon reversal: after a sharp multi-day pullback within an
established long-term uptrend, equity index/sector ETFs tend to rebound
over the following days (Jegadeesh 1990; codified as a mechanical rule by
Connors' RSI(2) system). This is the complementary anomaly to the three
trend/momentum families already tested and rejected in this project
(crypto, `slc_4h_5m_stock_v1`, `etf_momentum_v1`) — it buys weakness
inside an uptrend rather than chasing strength. Long-only; no short side.

## 2. Universe

The 16 **equity** ETFs already present in the frozen `etf_momentum_v1`
data snapshot (manifest SHA-256 `c56f02f2cad8647f40813b6b012db660e37e559e67d7b90b0b35fc744a8b303b`,
verified unchanged — see Section 3): SPY, QQQ, IWM, EFA, EEM, XLB, XLC,
XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY.

Bonds (IEF, TLT), gold (GLD), commodities (DBC), and BIL are **excluded**
— the published short-term reversal evidence this hypothesis rests on is
equity-specific (index/sector ETF price action), and extending it to
rate-sensitive or commodity instruments without a distinct evidentiary
basis would be an unstated additional hypothesis, not this one.

**Late-starting members (XLC, XLRE)**, per `etf_momentum_v1`'s
already-established rule (reused unchanged, not re-derived): a member
without 200 trading days of prior history for the SMA200 filter (Section
4) is excluded from candidacy that day, never given a placeholder value.
XLC (first bar 2018-06-19) and XLRE (first bar 2015-10-08) become
eligible roughly 200 trading days after those dates — both well inside
the backtest window (Section 3), so both are absent from candidacy only
in their own early years.

## 3. Data

Reuses the `etf_momentum_v1` snapshot **unchanged — no new fetch**.
Verified directly: the snapshot's per-ticker parquet already carries
`Open`/`High`/`Low`/`Close` (not just `Close`), which is what this
strategy's next-open fill convention needs (Section 4) — nothing further
to add. Verified separately: 273 trading days of SPY history exist in the
snapshot before 2008-07-01, comfortably covering the 200-day SMA
lookback with no additional lead time required.

**Window: 2008-07-01 → 2026-08-01**, same effective window as
`etf_momentum_v1` (the 2005-01-01 nominal start remains unreachable for
the same reason established there — the snapshot's own BIL-driven lead
time — even though BIL itself isn't in this strategy's universe, the
snapshot was fetched once for all `etf_momentum_v1` candidates and is
being reused as-is here, not re-fetched with a different start).

**IS/OOS split, unchanged from `etf_momentum_v1`:**
- In-sample: 2008-07-01 → 2019-12-31.
- Out-of-sample: 2020-01-01 → 2026-08-01.

## 4. Rules (long-only)

**Indicators — reused unmodified.** `utils/indicators.py::rsi(close,
period=2)` already implements Wilder-smoothed RSI (exponential moving
average of gains/losses with `alpha=1/period`, `adjust=False`) —
parameterized by period, so `rsi(close, period=2)` **is** RSI(2), no new
indicator code needed. `utils/indicators.py::sma(close, period=200)` is
likewise reused unmodified for the SMA200 filter.

**RSI(2) formula, pinned precisely:**
```
delta      = close.diff()
gain       = max(delta, 0)
loss       = max(-delta, 0)
avg_gain   = EWM(gain, alpha=0.5, adjust=False, min_periods=2)
avg_loss   = EWM(loss, alpha=0.5, adjust=False, min_periods=2)
RS         = avg_gain / avg_loss
RSI(2)     = 100 - 100 / (1 + RS)
```

**Hand-verified test vector** (computed directly against the real
function, to catch any future silent change to `rsi()`'s behavior — not
an independent from-scratch derivation, since Wilder's recursive
smoothing is exactly what `.ewm(alpha=1/period, adjust=False)` computes):
closes `[10.0, 11.0, 10.5, 12.0, 11.0, 11.5, 10.0, 9.5, 10.5, 12.0]` give
RSI(2) `[NaN, NaN, 66.667, 88.889, 47.059, 64.000, 21.918, 15.238,
61.803, 85.575]` (index 0-1 are `NaN` — `min_periods=2` on the
gain/loss series, which itself starts at index 1 from `.diff()`, means
the first valid RSI(2) needs 3 price points). Intermediate values for
this same vector: `avg_gain = [NaN, NaN, 0.5, 1.0, 0.5, 0.5, 0.25, 0.125,
0.5625, 1.03125]`, `avg_loss = [NaN, NaN, 0.25, 0.125, 0.5625, 0.28125,
0.890625, 0.6953125, 0.34765625, 0.173828125]`.

**Entry signal**, checked at the close of day T for every universe member
not currently held: `RSI(2) < entry_threshold AND close > SMA200`.

**Exit signal**, checked at the close of day T for every held position,
in this fixed priority order (first true condition wins and is the
recorded `exit_reason` — only one can be first in a fixed evaluation
order, so this is deterministic even when multiple conditions are true
the same day):
1. `close < SMA200` → `"trend_exit"` (the long-term filter itself broke;
   checked first because losing the uptrend context invalidates the
   original entry thesis regardless of the RSI reading).
2. `RSI(2) > exit_threshold` → `"target_exit"` (the reversion played out).
3. Trading days elapsed since entry fill `>= max_hold` → `"time_exit"`
   (exact counting convention below).

**`max_hold` counting, pinned exactly.** The entry fill day is day 1 of
the hold (not day 0). The `time_exit` condition (`days_held >= max_hold`)
is first checked true at the close of the `max_hold`-th trading session
counting from and including the fill day; the resulting exit fills at the
*next* session's open, same as every other exit. Worked example
(`max_hold=5`, no holidays): entry signal fires at a Friday close, entry
fills the following Monday's open — that Monday is day 1. Tuesday=day 2,
Wednesday=day 3, Thursday=day 4, Friday (of the following week)=day 5:
`5 >= 5` first becomes true at that Friday's close, so the `time_exit`
signal fires there and fills at the Monday after — a full week after
entry. Section 11's required tests assert this exact case.

**Execution: next session's open, no lookahead.** A signal computed from
day T's completed close is frozen at that value and executes
unconditionally at day T+1's open — no re-validation of the entry/exit
condition at the fill bar itself (matching this project's existing
"next bar" conventions: SLC's next-5-minute-bar entry,
`etf_momentum_v1`'s decision/transaction-day separation). This was
checked, not assumed: this project's two existing daily-bar engines
(`simulate_portfolio()`, `simulate_xsec_momentum_portfolio()`) both fill
at the *same* decision day's close, not the next day's open — reusing
either engine's fill convention unmodified would mean same-close fills,
not what this hypothesis calls for. Implementing genuine next-open fills
instead is not blocked by anything: the snapshot already carries Open
prices (verified above), so this needs new simulator code (Section 6) but
no new data and no disproportionate complexity — adopted as specified,
not falling back to a next-close (or same-close) convention.

**Missing-bar handling is asymmetric between entries and exits, by
design.** If a T+1 bar is missing entirely (a genuine data gap, not a
holiday — holidays are never in the trading-day reference calendar to
begin with): an **entry** signal from day T is simply skipped and logged
as a rejection, not deferred or retried — the opportunity is gone, same
as `etf_momentum_v1`'s rebalance-entry handling. An **exit** signal
behaves differently: the position stays open, and *every* exit condition
(trend, target, time) is re-evaluated fresh at each subsequent close
until one of them fires against an actually-available next-day open, or
the position is forced closed at end-of-test. Exits are never
permanently dropped by a data gap the way a missed entry opportunity is —
a position is never left with no path to close short of the end-of-test
backstop.

**No price stop.** `max_hold` and the SMA200 trend-exit are the only
bounds on how long or how far a losing position can run. Per-trade loss
inside the hold window is explicitly unbounded by construction — a
position can decline arbitrarily before either exit condition triggers.
This is disclosed here, not mitigated: the qualification gate's drawdown
check (check 9) is what judges the real consequence of this design
choice, not a stop-loss.

## 5. Grid (frozen now) and selection

Frozen grid, no other axis, ever:
- `entry_threshold` ∈ {5, 10, 15}
- `exit_threshold` ∈ {55, 65, 75}
- `max_hold` ∈ {5, 10} (trading days)

18 cells. Selection is in-sample by baseline-cost (5bps) Sharpe only,
mechanical, one shot — identical method to `etf_momentum_v1`: run every
cell over the IS window, take the max-Sharpe cell, freeze it, evaluate
once on OOS without retuning.

**Check 16 neighbor definition and corner count, computed directly (not
estimated by analogy) for this grid's actual shape:** a neighbor changes
exactly one axis by one grid step, holding the other two fixed —
identical rule to `etf_momentum_v1`. Verified by direct enumeration: **8
of the 18 cells are corners** (`entry_threshold ∈ {5,15}` AND
`exit_threshold ∈ {55,75}`, any `max_hold`) with only 3 evaluable
neighbors — below `MIN_PLATEAU_NEIGHBORS=4` — and fail check 16 outright
by construction if selected. This mirrors `etf_momentum_v1`'s own 8/18
corner ratio exactly, because both grids share the same shape (two
three-valued axes plus one two-valued axis) — not a coincidence, a
structural property of this grid design. Accepted in advance, same as
`etf_momentum_v1`: if selection lands on a corner, check 16 fails by
construction and that is the recorded result, not grounds to revisit the
neighbor definition after the fact.

## 6. Engine

**Neither of this project's existing daily-bar simulators is a clean
fit**, and this section says so plainly rather than overstating reuse.
`simulate_portfolio()` (the general stock/crypto engine) is built around
intraday stop/target order-lifecycle machinery (`ActiveOrder`,
`outcome_simulator`, gap/adverse-ambiguity handling) that has no
counterpart here — this strategy has no stop and no intraday order
mechanics at all, only EOD signals and next-open fills.
`simulate_xsec_momentum_portfolio()` / `simulate_etf_momentum_portfolio()`
rebalance on a fixed monthly calendar to target weights, not on
per-symbol daily signals with a ranked admission queue. A new,
purpose-built simulator is needed — the same conclusion `etf_momentum_v1`
itself reached about reusing the crypto engine, applied again here.

**What genuinely is reused, unmodified:** `utils/indicators.py::rsi()`
and `sma()` (Section 4); `ResearchCost`/`COSTS` (`stock_bps_per_leg`,
Section 8); `whole_bot_metrics.summarize_run()`/`qualify_strategy()`
(the trade-dict aliasing pattern `etf_momentum_v1` established —
`status="closed"`, `exit_time`, `fill_price`, `mode="stock_only"` for
correct 252-trading-day Sharpe annualization, carrying forward the fix
made after `etf_momentum_v1`'s first qualification run); the general
shape of "rank candidates, admit within capacity, exit-before-entry
same-day ordering" already used by `simulate_portfolio()`, reimplemented
for this strategy's simpler (no intraday, no stop) daily-signal case
rather than copied.

**Same-day exit-before-entry ordering**, mirroring existing precedent:
each day, exit signals computed from that day's close are queued first
(freeing capacity), then entry signals are ranked against the
now-current free-slot count — both categories of signal fill at the
following day's open together.

**Equity series: trading days only, at adjusted closes.** The simulator
emits one mark-to-market equity row per trading day (from the SPY
reference index, same source used elsewhere in this project), not one
per calendar day — a deliberate choice, not `etf_momentum_v1`'s
calendar-day loop carried over by default. This strategy has no reason to
mark a weekend or holiday at all (no intraday component, no rebalance
concept spanning non-trading days), and trading-day-only granularity is
what `periods=252` annualization (Section 9's `mode="stock_only"`) is
actually scaled for — sampling every calendar day (including flat,
non-trading weekend rows) would dilute the return series with
zero-return observations under a scaling constant that assumes the
sample *is* trading days, a mismatch this design avoids by construction
rather than by coincidence. This equity series is what checks 7
(Sharpe), 9 (drawdown), 10 (recent 12 months), and 11 (positive quarters)
consume via `summarize_run()`.

**Idle cash earns zero.** Uncommitted cash (Section 7) accrues no
interest or yield for the duration it sits idle between positions — a
disclosed conservative simplification, not a claim that real idle cash
earns nothing; it biases the reported result downward relative to an
implementation that sweeps idle cash into a money-market or T-bill
equivalent, which this design does not model.

**Check 2 (`data_coverage_at_least_95pct`) definition**, stated
explicitly (this was left implicit for `etf_momentum_v1` — closed here):
`coverage_rate = filled_actions / (filled_actions + rejected_actions)`,
counted over every entry and exit fill attempt across the full window —
the same ratio `etf_momentum_v1`'s runner computed, now written down
rather than left implicit in code. `summarize_run()` itself does not
define a coverage concept; `coverage_rate` is always a caller-supplied
argument to `qualify_strategy()`, computed by whichever strategy's runner
script owns that data-quality question.

## 7. Sizing

- Cash-only, no leverage — same as `etf_momentum_v1`.
- Maximum 5 concurrent positions.
- Each new position sized at entry: `notional = min(equity / 5, free_cash)`,
  where `free_cash` is uncommitted, settled cash *after* that day's exit
  fills have already settled (Section 6's exit-before-entry ordering) —
  never a fixed fraction of starting equity, and never sized past actual
  available cash. If `free_cash <= 0` when a slot is otherwise available,
  that admission is skipped and logged, not partially filled or
  negatively sized.
- When entry signals on a given day exceed free slots, admit by **lowest
  RSI(2) first** (most oversold prioritized), ties broken alphabetically
  by ticker.
- `risk_per_trade` and `daily_loss_limit` (both used by SLC/crypto's
  per-symbol admission gates) are **not used** — there is no stop to size
  a "risk amount" against, and no per-day loss-halt concept for a
  5-slot, next-open-fill EOD strategy. `max_position_pct` similarly
  doesn't apply — the 5-slot cap is the sole concentration control, same
  reasoning `etf_momentum_v1` gave for not applying it there.
- **The 16 members are highly correlated** (broad equity-market beta
  shared across SPY, QQQ, IWM, the sector SPDRs, EFA, and EEM), so entry
  signals cluster in time rather than arriving independently — the five
  slots typically deploy together during market-wide selloffs rather than
  across five uncorrelated bets, meaning the slot cap is the *only*
  concentration control this design has, not a proxy for real
  diversification.

## 8. Costs and reporting

- Zero (0bps), baseline (5bps/leg), stressed (13bps/leg) — the existing
  `stock_bps_per_leg` `ResearchCost` models, unchanged, same as
  `etf_momentum_v1`.
- Report annualized turnover (closed trades/year, gross notional/year
  over average equity) and average hold duration in trading days,
  alongside the standard trade count.

## 9. Qualification gate

All 16 checks from `backtest/whole_bot_metrics.py::qualify_strategy()`,
identical thresholds to `etf_momentum_v1` (quoted verbatim there;
reproduced here for completeness): `closed_trades_at_least_100` (≥100),
`data_coverage_at_least_95pct` (≥95%), `missing_outcomes_below_1pct`
(<1%), `baseline_expectancy_positive` (>0), `stressed_expectancy_positive`
(>0), `baseline_profit_factor_at_least_1_15` (≥1.15), `sharpe_at_least_1`
(≥1.0), `sharpe_beats_benchmark` (> SPY), `max_drawdown_no_more_than_15pct`
(≥-15%), `recent_12m_positive` (>0), `positive_quarters_at_least_60pct`
(≥60%), `bootstrap_lower_mean_r_positive` (>0),
`single_symbol_profit_no_more_than_25pct` (≤25%),
`walk_forward_oos_sharpe_beats_benchmark` (> OOS SPY, ≥30 OOS trades),
`walk_forward_oos_profit_factor_at_least_1_0` (≥1.0),
`sensitivity_plateau_within_25pct_of_neighbor_median` (≥4 neighbors,
median ≥75% of selected).

**Benchmark**: SPY buy-and-hold, same convention as every prior strategy
in this ledger.

**`pnl_r` resolution, carried over from `etf_momentum_v1`'s amendment**
(Section 8 there): no stop exists here either (Section 4), so
`pnl_r = net_pnl / entry_notional` for every trade this engine emits, same
formula, same reasoning — checks 4, 5, and 12 consume `summarize_run()`'s
output exactly as before (check 12 uses `pnl_r` directly via
`bootstrap_95pct_lower_mean_r`; checks 4/5 use `net_expectancy`, the mean
of `net_pnl`, unaffected by the `pnl_r` definition either way — same
clarification made for `etf_momentum_v1`, since it's the actual behavior
of `summarize_run()`'s source, not specific to either strategy).

**Check 3 vacuous-note, carried over.** This engine shares the property
that sank check 3's meaningfulness in `etf_momentum_v1`: every position
closes with a real price by construction — a trend-exit, target-exit,
time-exit, or a forced end-of-test close if none of those trigger first —
so `missing_outcomes` is structurally always empty and check 3 passes
trivially, not on the merits. Run as specified anyway (interface
uniformity), the ledger entry marks it vacuous rather than presenting the
trivial pass as meaningful, identical treatment to `etf_momentum_v1`.

## 10. One-shot rule, stated prior, and freeze rule

A single qualification run, after this document is approved and frozen.
The 18-cell grid sweep is part of that one run — cell selection happens
mechanically by max in-sample Sharpe, no discretionary review of
intermediate per-cell results before the OOS/gate evaluation proceeds.
The result enters the evidence ledger exactly once. No post-hoc variants
of this family — a different grid, sizing rule, universe, or window is a
new, separately preregistered version, never a retroactive edit to this
result. A failed result may not be used to tune this version on the same
evaluation period.

**Stated prior, before the run.** Published short-term-reversal /
RSI(2)-style results concentrate in the pre-2010 literature and
practitioner era; this project's own OOS window (2020-01-01 onward) sits
exactly where reported edge decay for this anomaly family is generally
described as worst. The expected outcome, stated here in advance, is a
**fail**, most likely on checks 7, 8, and 14 (the same Sharpe-based checks
that failed `etf_momentum_v1`) — a near-miss or an outright miss on any of
these is a **fail**, enters the ledger as a fail, and triggers no
discussion of adjusting the gate.

**Check 9 (max drawdown) is also named as a likely failure, for a
distinct, structural reason**, not lumped in with the Sharpe checks above:
this design has no stop (Section 4), the 16 members are highly correlated
so entries cluster rather than diversify (Section 7), and the OOS window
contains February–March 2020 — a market-wide, correlated selloff of
exactly the kind that would fill all 5 slots together with nothing to
limit how far each can fall before an exit condition triggers. A
drawdown-check failure here would not be a surprise driven by
implementation error; it is the direct, foreseeable consequence of the
no-stop design already disclosed in Section 4.

**Next-open fills are expected to underperform the published literature
for a distinct reason** from the two above: most published close-fill
RSI(2) results enter and exit at the close the signal is observed on,
capturing whatever the overnight move is before the next session opens —
often a meaningful part of the reversal itself, since a sharp oversold
reading frequently rebounds first overnight or at the next open before
continuing intraday. This design's next-open, no-lookahead fills
(Section 4) forgo that overnight component by construction. This is
expected to cost real, measurable performance relative to published
close-fill backtests of the same rule — a conservatism this document
accepts deliberately (Section 4's no-lookahead requirement), not an
implementation shortfall to explain away if the result underperforms.

These are declared priors, not citations independently verified against
a specific paper by this document — recorded so a weak result on any of
checks 7, 8, 9, or 14 cannot later be framed as a surprise.

This file is append-immutable after its SHA-256 is recorded in its
manifest (`research/etf_reversal_v1_preregistration.manifest.json`). Any
rule change after freezing requires a new version or a separately hashed
amendment.

## 11. Required tests

RSI(2) against the hand-verified vector in Section 4 (both the RSI values
and the intermediate `avg_gain`/`avg_loss` series); entry/exit precedence
on the same bar (a symbol simultaneously satisfying an exit condition
while held, and being re-evaluated as a fresh entry candidate only after
that exit clears its slot — never both in the same day for the same
symbol); the exact `max_hold=5`, no-holidays worked example from Section
4 (Friday-close signal, Monday-open entry fill as day 1, `time_exit`
firing at the *following* Friday's close, exit fill the Monday after —
asserted as the literal expected dates, not just a day-count); `max_hold`
counting in trading days across a holiday (a holiday inside the hold
window must not extend or shorten the trading-day count); SMA200
boundary behavior (`close` exactly equal to `SMA200`, and the first day a
symbol has exactly 200 prior trading days available); slot ranking by
RSI(2) with an alphabetic tie-break (more entry signals than free slots
on one day, including a constructed exact RSI(2) tie); next-open fill
with no lookahead (a signal computed at T's close must use T+1's open
price); the missing-bar asymmetry (a missing entry-fill bar is skipped
and not retried; a missing exit-fill bar leaves the position open with
exit conditions re-checked at the next available close, not stuck or
force-closed early); `free_cash` sizing (`notional = min(equity/5,
free_cash)`, including the case where `free_cash <= 0` skips the
admission rather than sizing a negative or zero position, and the case
where same-day exit proceeds become available `free_cash` for that same
day's entries per the exit-before-entry ordering); the trading-day-only
equity series (no row for a weekend/holiday date); cost-model selection
using `stock_bps_per_leg`; deterministic repeated-run output; and a
static prohibition on importing any broker trading/order client.
