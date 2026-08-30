# ETF cross-sectional momentum v1 — frozen preregistration

Frozen: 2026-08-30. This document is a research specification, not evidence
that the strategy is profitable and not authorization to submit orders.

Version identifier: `etf_momentum_v1`.

## 1. Hypothesis

Cross-sectional momentum with an absolute-momentum filter, monthly rebalance.

1. **Ranking.** At each rebalance, compute each universe member's trailing
   total return over `lookback_months` months, using the skip-month
   convention: with `skip_last_month = 0`, the window ends at the most
   recently completed month-end; with `skip_last_month = 1`, the window
   ends one month earlier (excludes the most recently completed month
   entirely from the return calculation). This is the standard academic
   momentum convention (Jegadeesh–Titman skip-month; Asness–Moskowitz–
   Pedersen), included specifically to test whether the well-documented
   short-term-reversal contamination of naive 12-month momentum matters
   here. Rank the universe descending by this trailing return; ties break
   alphabetically by ticker. **BIL participates in this ranking pool as a
   regular candidate** — it is not excluded from being ranked/held on its
   own merits (e.g. a genuine flight-to-safety or high-short-rate regime
   can put BIL in the top `N` by its own trailing return), in addition to
   its separate role as the absolute-momentum filter's reference asset
   (item 3).
2. **Selection.** Hold the top `top_N` ranked members.
3. **Absolute-momentum filter.** For each of the `top_N` slots, compare
   that member's trailing return (same window, same `lookback_months`/
   `skip_last_month`) against BIL's trailing return over the identical
   window. If the slot's member scores below BIL, that slot holds BIL
   instead of the ranked member. This is evaluated per slot, not as a
   single universe-wide gate — a slot can independently fail into BIL
   while other slots stay in their ranked risk asset. (When BIL itself
   occupies a slot per item 1, this comparison is a no-op for that slot.)
4. **Weighting.** Equal-weight across the `top_N` slots — each held slot,
   including a BIL substitution, receives `1 / top_N` of current equity.
   Approved as the v1 default over inverse-vol-weighting: inverse-vol
   would require a volatility-lookback window that is not part of the
   frozen grid in Section 6, and Section 6 forbids any further tunable
   axis. Inverse-vol-weighting is v2-only — a distinct, separately
   preregistered `etf_momentum_v2`, never folded into v1 after the fact
   (Section 9).
5. **Rebalance timing.** Monthly, on the first NYSE trading day of each
   calendar month. Ranking is computed off the prior month's final close
   (no same-bar lookahead); the resulting target weights are transacted at
   the first trading day's close. This one-day gap between the ranking
   snapshot and the transaction price mirrors this project's existing
   "next bar" conventions (SLC's next-5-minute-bar entry;
   `crypto_xsec_momentum_v1`'s `daily_completed_bar_cutoff`).
6. **No intra-month stop.** Unlike `crypto_xsec_momentum_v1` (which has a
   fixed catastrophic ATR stop, monitored daily), this strategy has no
   stop-loss of any kind — positions close only at the next monthly
   rebalance, whether by falling out of the top `N`, being reassigned to
   BIL by the absolute filter, or a ranking change. This is a deliberate
   reading of the hypothesis as stated (pure rank–hold–rebalance, no
   mention of a stop) and a real divergence from the reused engine's
   position-management loop — see Section 7. It also means `pnl_r` has no
   stop-based denominator here; see Section 8's `pnl_r` resolution.

No additional entry indicator, chart pattern, or optimizer is part of v1.

## 2. Universe

Fixed, survivorship-free by construction (all 21 tickers named up front;
none added or removed based on results). The 11 SPDR sector ETFs, plus
SPY, QQQ, IWM, EFA, EEM, TLT, IEF, GLD, DBC, and BIL.

| Ticker | Role | First available daily bar (verified) |
|---|---|---|
| SPY | Broad market | 1993-01-29 |
| XLB | Sector: Materials | 1998-12-22 |
| XLC | Sector: Communication Services | 2018-06-19 |
| XLE | Sector: Energy | 1998-12-22 |
| XLF | Sector: Financials | 1998-12-22 |
| XLI | Sector: Industrials | 1998-12-22 |
| XLK | Sector: Technology | 1998-12-22 |
| XLP | Sector: Consumer Staples | 1998-12-22 |
| XLRE | Sector: Real Estate | 2015-10-08 |
| XLU | Sector: Utilities | 1998-12-22 |
| XLV | Sector: Health Care | 1998-12-22 |
| XLY | Sector: Consumer Discretionary | 1998-12-22 |
| QQQ | Nasdaq-100 | 1999-03-10 |
| IWM | Small cap | 2000-05-26 |
| EFA | Developed intl. equity | 2001-08-27 |
| EEM | Emerging market equity | 2003-04-14 |
| IEF | 7–10yr Treasury | 2002-07-30 |
| TLT | 20+yr Treasury | 2002-07-30 |
| GLD | Gold | 2004-11-18 |
| DBC | Broad commodities | 2006-02-06 |
| BIL | 1–3 month T-bill (filter reference + defensive asset) | 2007-05-30 |

Dates are each ticker's actual first returned daily bar (verified directly
against a live data pull, not taken from a published inception date, which
can differ by a few days from the first tradeable/quoted session). These
will be re-confirmed against the frozen snapshot itself (Section 4).

**Late-starting members (XLRE, XLC).** A member with insufficient history
for the grid cell's `lookback_months + skip_last_month` is excluded from
that month's ranking pool entirely — never forced to BIL, never given a
placeholder return. This reuses `crypto_xsec_momentum_v1`'s existing
`_trailing_return()` behavior unchanged: it already returns `None` when
the lookback bar is unavailable, and the ranking loop already skips `None`
results (`backtest/whole_bot_engine.py`, `_trailing_return`/rebalance
loop). Under the worst-case grid cell (`lookback_months=12`,
`skip_last_month=1`, needing 13 months of runway), XLRE becomes eligible
~2016-11 and XLC ~2019-07 — both well within the backtest window
(Section 3), so both are absent from ranking only in their own early
years, not for the whole test.

**BIL's own gap, resolved.** BIL is the filter's reference asset, not
just another ranked slot, so if it doesn't exist yet the absolute-momentum
filter has nothing to compare against, for every slot, universe-wide.
BIL's first bar is 2007-05-30, so a nominal window opening 2005-01-01 has
~2.4 years with no defensive asset defined. Resolved by setting the
backtest's effective start date to the first month-start with a full
`12 + 1 = 13`-month runway after BIL's first bar under the grid's own
worst-case lookback+skip combination — **2008-07-01** (Section 3).

## 3. Data

**Source: yfinance, `auto_adjust=True`** (split *and* dividend adjusted,
i.e. a genuine total-return series) — approved as a deliberate, stated
exception to every other backtest in this project (SLC, crypto), which use
Alpaca exclusively. Two verified reasons:

1. **Depth.** Alpaca's SIP daily-bar history on this account was checked
   directly and starts **2016-01-04** for SPY (and, by construction, no
   earlier for anything else) — a live query for January 2005 through
   January 2015 returned zero rows for every ticker tested. yfinance
   returns real daily bars back to each ticker's actual first session
   (Section 2's table), which is what makes a 2005-era window possible at
   all.
2. **Total return.** `utils/alpaca_bars.py::fetch_bars()` hardcodes
   `Adjustment.SPLIT` — split-adjusted only, no dividend adjustment. BIL's
   own price is nearly flat by construction (a T-bill fund); its entire
   return is the distributed yield. A split-only series would show BIL's
   trailing return as ~0% almost always, making the absolute-momentum
   filter nearly vacuous. Checked directly: BIL's raw close moved -0.08%
   over 2020–2023 while its dividend-adjusted close moved +6.70% over the
   same span (sum of distributions: $6.01/share) — confirming both that
   Alpaca's current path is unusable for this strategy as specified, and
   that yfinance's `auto_adjust=True` output is a real, materially
   different, verified-correct total-return series.

**Window: 2008-07-01 → 2026-08-01** (~18.1 years). The nominal request
(2005-01-01) is not achievable — Section 2's BIL gap — and this is the
resolved effective window, not a proposal.

**Frequency.** Daily bars for ranking/rebalance decisions; no intraday
data is used (there is no intraday exit rule — Section 1, item 6).

## 4. Data snapshot

Fetch each of the 21 tickers **once**, over the full effective window
(2008-07-01 → 2026-08-01) plus enough lead time to cover the deepest
grid lookback (`lookback_months=12` + `skip_last_month=1` = 13 months) —
i.e. fetch from 2007-06-01. Persist both the raw series (`auto_adjust=
False`, close + dividends columns) and the adjusted series (`auto_adjust=
True`) to parquet, one file pair per ticker, under
`backtest/data/etf_momentum_v1/`. Every downstream step — grid sweep,
selection, OOS evaluation, sensitivity neighbors, the final qualification
run — reads exclusively from this snapshot. No step re-fetches from
yfinance; a strategy run that cannot find its data in the snapshot fails
closed rather than silently fetching a fresh (and potentially different)
series.

Write a manifest (`backtest/data/etf_momentum_v1/manifest.json`)
recording, per file: relative path and SHA-256. Also record: the pinned
`yfinance.__version__` used for the fetch, the fetch timestamp (UTC), and
the requested date range per ticker. The manifest's own SHA-256 is
reported alongside the frozen document's SHA-256 (Section 10) and goes
into the evidence ledger entry for this strategy.

**Total-return canary, extended.** Before the snapshot is trusted for any
downstream use, verify the adjusted-vs-raw gap is real and correctly
sized for three tickers, not just BIL: **BIL, TLT, and XLU**. For each,
compute (adjusted total return − raw price return) over the snapshot
window and reconcile it against that ticker's own published distribution
history (yfinance's `.dividends` field, summed over the same window,
expressed as a fraction of the average price level) — the two should
agree to a reasonable tolerance (not to the last cent, since reinvestment
timing differs from a simple sum-of-dividends approximation, but the
adjusted/raw gap must be the same order of magnitude as the reconciled
distribution yield, for all three tickers). TLT and XLU are chosen
because they are meaningfully different distribution profiles from
BIL and from each other: TLT is a long-duration bond fund (dividend yield
comparable to or exceeding BIL's, but paid on a different schedule) and
XLU is an equity sector fund (a real but much smaller dividend yield,
tests that the adjustment isn't a bond-specific artifact).

## 5. Portfolio and costs

- Starting equity $100,000; cash-only; no leverage.
- Concentration control is `top_N` itself, by design — the existing
  20%-of-equity `max_position_pct` cap (used by every other strategy in
  this project) is **not** applied here. At `top_N=2`, each equal-weighted
  slot is 50% of equity; that concentration is the intended shape of a
  cross-sectional rotation strategy, with the absolute-momentum-to-BIL
  overlay as the risk control instead of a per-position size cap.
  `risk_per_trade`, `max_new_trades_per_day`, and `daily_loss_limit`
  (all used by SLC/crypto's per-symbol admission gates) do not apply to a
  monthly, fully-rebalanced portfolio and are not used.
- Costs per leg: reuse the existing `ResearchCost` models unchanged —
  zero 0 bps, baseline 5 bps, stressed 13 bps
  (`backtest/whole_bot_engine.py::COSTS`). These are the `stock_bps_per_leg`
  values, not `crypto_bps_per_leg` (35/50 bps) — see Section 7's engine-
  reuse note; `crypto_xsec_momentum_v1`'s cost helper is hardcoded to the
  crypto field and needs a one-line change to select `stock_bps_per_leg`
  for this strategy, since ETFs trade on-exchange with equity-like
  spreads, not crypto spreads.
- Report annualized turnover (position-months opened per year, and gross
  notional transacted per year as a fraction of average equity) alongside
  the standard trade count.

## 6. Grid (frozen now) and selection

Frozen grid, no other axis, ever:

- `lookback_months` ∈ {6, 9, 12}
- `skip_last_month` ∈ {0, 1}
- `top_N` ∈ {2, 3, 4}

18 cells. Selection is in-sample by baseline-cost (5 bps) Sharpe only —
the same method already used for `crypto_xsec_momentum_v1`'s own
`lookback_days` selection in `backtest/run_crypto_walkforward.py`
(run every candidate over the in-sample window, take the max-Sharpe cell,
freeze it, then evaluate once on out-of-sample without retuning).

**IS/OOS split:**
- In-sample: **2008-07-01 → 2019-12-31** (~11.5 years — includes the 2008
  GFC crash itself (Sept–Dec 2008) and its recovery, the 2011
  debt-ceiling selloff, and the Q4 2018 correction).
- Out-of-sample: **2020-01-01 → 2026-08-01** (~6.6 years — includes the
  2020 COVID crash, the 2022 rate-hike bear market, and the 2023–2025
  recovery). Long enough that even modest monthly turnover across 2–4
  slots should clear the existing `MIN_OOS_CLOSED_TRADES = 30` floor
  (`backtest/whole_bot_metrics.py`), but this is a plausibility argument,
  not a guarantee — confirmed, not assumed, when the run completes.

**Which window each check uses (Section 8's numbering):**
- Checks 1–13 run on the **full effective window** (2008-07-01 →
  2026-08-01), using the IS-selected cell evaluated end-to-end — these are
  the strategy's headline numbers, not IS-only or OOS-only figures.
- Checks 14–15 (walk-forward OOS) run on the **OOS window only**
  (2020-01-01 → 2026-08-01), comparing the IS-selected cell's OOS
  performance against the OOS benchmark.
- Check 16's neighbor cells are evaluated on the **IS window only**
  (2008-07-01 → 2019-12-31) — the same window the selection itself was
  made on, since the plateau check exists to validate that selection, not
  the full-window result.

## 7. Engine reuse

Reuses `backtest/whole_bot_engine.py::simulate_xsec_momentum_portfolio()`
(the `crypto_xsec_momentum_v1` engine) as its base — the same weekly-vs-
monthly rebalance loop shape, the same rank/select-top-N/exit-what-dropped-
out/enter-what's-new structure, and the same trade-dict aliasing that lets
`whole_bot_metrics.summarize_run()`/`qualify_strategy()` consume its
output unchanged (`_xsec_close_trade()`'s `status`/`exit_time`/`fill_price`
aliases, confirmed in source). Reused as-is: `_trailing_return()`'s
None-on-insufficient-data behavior (Section 2), the daily equity mark-to-
market loop, the rejected-candidate logging pattern.

Not reused — genuinely new code, not a parameter change on the existing
engine:
1. **Rebalance cadence.** Monthly/first-trading-day, not
   `rebalance_weekday`-based weekly.
2. **No stop-loss.** The existing engine's daily catastrophic-ATR-stop
   scan (step 1 of its loop) is dropped entirely — see Section 1, item 6.
3. **Absolute-momentum-to-BIL substitution**, replacing the existing
   engine's binary BTC-macro gate (`btc_macro_gate`/`_btc_macro_ok()`).
   Structurally different: the existing gate is universe-wide and binary;
   this is evaluated per slot and substitutes a specific asset (BIL)
   rather than going to cash/skipping the rebalance.
4. **Equal-weight sizing**, not the existing engine's
   `target_daily_vol_pct`-per-position vol-targeting.
5. **Cost field**: `stock_bps_per_leg`, not the hardcoded
   `crypto_bps_per_leg` in `_xsec_transaction_cost()` (Section 5).
6. **Data path**: reads exclusively from the Section 4 snapshot, not
   `fetch_daily_crypto_frames()`/Alpaca/live yfinance calls.

## 8. Qualification gate

All 16 checks from `backtest/whole_bot_metrics.py::qualify_strategy()`,
run with both `walk_forward` and `sensitivity` populated (the 13 base
checks plus both walk-forward OOS checks plus the sensitivity-plateau
check — this is the first strategy in this project to exercise all 16 in
a real run rather than a unit test). Thresholds, quoted verbatim from
source:

1. `closed_trades_at_least_100` — baseline closed trade count ≥ 100.
2. `data_coverage_at_least_95pct` — coverage rate ≥ 0.95.
3. `missing_outcomes_below_1pct` — missing-outcome rate < 0.01.
4. `baseline_expectancy_positive` — baseline net expectancy > 0.
5. `stressed_expectancy_positive` — stressed net expectancy > 0.
6. `baseline_profit_factor_at_least_1_15` — baseline profit factor ≥ 1.15.
7. `sharpe_at_least_1` — baseline annualized Sharpe ≥ 1.0.
8. `sharpe_beats_benchmark` — baseline Sharpe > benchmark Sharpe.
9. `max_drawdown_no_more_than_15pct` — max drawdown ≥ -0.15.
10. `recent_12m_positive` — most recent 12 months' net P&L > 0.
11. `positive_quarters_at_least_60pct` — positive-quarter fraction ≥ 0.60.
12. `bootstrap_lower_mean_r_positive` — bootstrap 95% lower mean R > 0.
13. `single_symbol_profit_no_more_than_25pct` — max single-symbol profit
    contribution ≤ 0.25.
14. `walk_forward_oos_sharpe_beats_benchmark` — OOS Sharpe > OOS benchmark
    Sharpe, AND OOS closed trades ≥ 30 (`MIN_OOS_CLOSED_TRADES`).
15. `walk_forward_oos_profit_factor_at_least_1_0` — OOS profit factor ≥
    1.0, same ≥30-trade floor.
16. `sensitivity_plateau_within_25pct_of_neighbor_median` — at least 4
    in-grid neighbor cells with ≥30 closed trades each
    (`MIN_PLATEAU_NEIGHBORS`, `MIN_NEIGHBOR_CLOSED_TRADES`), and their
    median Sharpe ≥ 75% of the selected cell's Sharpe
    (`PLATEAU_NEIGHBOR_MEDIAN_FRACTION`).

**Benchmark**: SPY buy-and-hold over the matching window, via the
existing `benchmark_summary()` helper — same convention as SLC and both
crypto strategies use SPY/BTC respectively as their passive benchmark.

**`pnl_r` resolution.** With no stop (Section 1, item 6), there is no
stop-distance to normalize by, unlike SLC/`crypto_xsec_momentum_v1` where
`pnl_r = net_pnl / (quantity * |entry - stop|)`. Defined instead as:

```
pnl_r = net_pnl / entry_notional      (entry_notional = quantity * entry_price)
```

for every trade this engine emits. Checked directly against
`backtest/whole_bot_metrics.py::summarize_run()`: check 12
(`bootstrap_lower_mean_r_positive`) consumes `pnl_r` directly (via
`bootstrap_95pct_lower_mean_r`, computed from the trades' `pnl_r` array).
Checks 4 and 5 (`baseline_expectancy_positive`/`stressed_expectancy_
positive`) consume `net_expectancy`, which `summarize_run()` computes as
the mean of `net_pnl` directly — **not** `pnl_r` — so they are unaffected
by this definition either way; noted here because the two checks were
originally expected to depend on it and, on inspection of the current
source, do not. `summarize_run()`/`qualify_strategy()` are otherwise
denominator-agnostic: they consume whatever float is in a trade's
`pnl_r` field with no assumption about what produced it. Section 10 adds a
test confirming nothing downstream still expects a stop-based
denominator.

**Checks needing an explicit resolution at this cadence:**

- **Check 3 (`missing_outcomes_below_1pct`) is structurally vacuous for
  this engine, and runs as specified anyway.**
  `simulate_xsec_momentum_portfolio()` hardcodes `"missing_outcomes": []`
  — every opened position is closed with a real price by construction
  (end-of-test forced close if nothing else), so this check is 0/N = 0%
  and trivially passes every run, unlike SLC where it can genuinely fail
  on missing intraday data. It is run as specified anyway (harmless,
  keeps the 16-check interface uniform across strategies); the ledger
  entry for this strategy marks it structurally vacuous for this engine
  rather than presenting the trivial pass as a meaningful result.
- **Check 16's "neighbor" is defined for this 3-dimensional grid as:** a
  neighbor is any grid cell one step away from the selected cell along
  **exactly one** of the three axes, holding the other two fixed — the
  direct 3-axis generalization of the existing 1-axis precedent (no prior
  strategy has exercised the plateau check on more than one varying axis:
  SLC's own sensitivity work checked a single ATR-period neighbor, and the
  check's own unit tests use a flat, dimension-agnostic neighbor list).
  Under this grid: `lookback_months=9` contributes 2 neighbors (6 and 12)
  vs. 1 for `lookback_months∈{6,12}`; `skip_last_month` always contributes
  exactly 1 (only 2 grid values exist); `top_N=3` contributes 2 vs. 1 for
  `top_N∈{2,4}`. A cell at a corner on *both* `lookback_months` and
  `top_N` has only 1+1+1=3 total neighbors — below
  `MIN_PLATEAU_NEIGHBORS=4` — and fails check 16 outright by construction,
  exactly as the existing check's own docstring intends for an edge/corner
  selection. **8 of the 18 cells are such corners**
  (`lookback_months ∈ {6,12}` × `top_N ∈ {2,4}` × `skip_last_month ∈
  {0,1}` = 2×2×2 = 8), and this outcome is accepted in advance: if
  selection lands on one of those 8, check 16 fails by construction and
  that is the recorded result, not grounds to revisit the neighbor
  definition or the grid after the fact.

## 9. One-shot rule

A single qualification run, after this document is approved and frozen.
The 18-cell grid sweep for in-sample selection (Section 6) is part of that
one run, not a separate exploratory step — cell selection happens
mechanically by max in-sample Sharpe, with no discretionary review of
intermediate per-cell results before the OOS/gate evaluation proceeds. The
result (pass or fail on the 16-check gate) enters the evidence ledger
exactly once. No post-hoc variants of this family — a different weighting
scheme, a different grid, a different universe, or a different window is a
new, separately preregistered version (`etf_momentum_v2` or later), never
a retroactive edit to this run or its interpretation. A failed result may
not be used to tune this version on the same evaluation period.

**Expected most-likely failure, stated in advance.** The literature prior
for published cross-sectional/time-series momentum strategies of this
general shape is an annualized Sharpe around 0.6–0.9 — below this
project's `sharpe_at_least_1` bar (check 7) of 1.0, which every other
strategy in this project is also held to. If the result lands in that
range, it is a **fail** on check 7, enters the ledger as a fail, and
triggers no discussion of lowering, adjusting, or special-casing the
threshold for this strategy. This is stated here, before the run, so a
near-miss on Sharpe specifically cannot be treated as a surprise or a
reason to revisit the gate after seeing the number.

## 10. Required tests

Before the qualification run: skip-month trailing-return calculation
(both `skip_last_month` values, verified against hand-computed cases);
absolute-momentum-to-BIL substitution (per-slot, not universe-wide;
correct when BIL itself would rank in the top N on its own trailing
return); late-starting-member exclusion from ranking (XLRE, XLC, and a
synthetic insufficient-history case); equal-weight sizing arithmetic;
no-stop position lifecycle (positions survive intra-month price action
unconditionally, close only at rebalance); `pnl_r = net_pnl /
entry_notional` computed correctly, plus a test that nothing in
`summarize_run()`/`qualify_strategy()`/the bootstrap helper still assumes
a stop-based denominator (i.e. that these consume `pnl_r` however it was
produced, with no reference to `stop` anywhere on this engine's path);
monthly first-trading-day timing including month-start holidays;
cost-model selection using `stock_bps_per_leg`; the snapshot's total-return
data path (Section 4's three-ticker canary: BIL, TLT, XLU, each
reconciled against published distributions); a test that no code path
outside the snapshot loader fetches from yfinance at run time; deterministic
repeated-run output; and a static prohibition on importing any broker
trading/order client, matching every other research module in this
project.

## 11. Freeze rule

This file is append-immutable after its SHA-256 is recorded in its
manifest (`research/etf_momentum_v1_preregistration.manifest.json`). Any
rule change after freezing requires a new version or a separately hashed
amendment. Failed results may not be used to tune this version on the same
evaluation period.
