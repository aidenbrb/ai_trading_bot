# stock_trend_momentum_v2 Preregistration

**Status: FROZEN.** This document is never edited after freezing. Any
future amendment is a **separate, dated, append-only document** (e.g.
`stock_trend_momentum_v2_preregistration_amendment_YYYY-MM-DD.md`) that
references this document's hash - never an in-place edit to this file.

**Frozen at (UTC)**: 2026-08-12T01:30:30Z
**Document hash**: computed and recorded by the freezing process immediately
after this file is written and before any test is run against it (see
Section 12 for the mechanism; the literal hash of *this specific file* is
necessarily produced after its own content is final, so it is written into
`research/stock_trend_momentum_v2_preregistration.manifest.json` alongside
this document rather than inside the document's own body).

This is a post-hoc-exploratory-informed but preregistered research plan.
It defines rules, thresholds, and an evaluation procedure **before** any of
them are implemented or tested. Nothing in this document is a result.

---

## 0. Why this document exists

`stock_trend_momentum_v1` failed qualification on its own merits. The
corrected, determinism-verified backtest
(`backtest/results/whole_bot/20260811_142144/`) and its follow-on
failure-analysis report
(`backtest/results/failure_analysis/20260811_142144_20260811_203512/report.md`)
showed a real but too-thin edge, stated precisely on the *same* fixed
1,227-trade cohort (report section 5(b), `zero_0bps` vs. `baseline_5bps`
rows): **+$0.7208/trade at zero cost** vs. **-$7.4527/trade at baseline
cost** - realistic transaction costs erase the edge. (The report's
*other* zero-cost figure, +$1.0878/trade, is the *path-dependent* view
over a smaller, 979-trade cohort - a different population, because cost
feeds into equity, then sizing, then admission. The two must never be
conflated; this document does not use the path-dependent zero-cost figure
for anything.)

Per the governing decision on this line of work: automatic stock entries
stay disabled, `stock_trend_momentum_v1`'s qualification thresholds are
not being loosened, and the next step is a genuinely new,
separately-versioned hypothesis - not tuning v1 against the same data it
already failed on. This document is that hypothesis, preregistered.

**This document defines rules only.** It contains no strategy code, and
nothing in it has been run. Implementing `stock_trend_momentum_v2`,
building its ablation/temporal-stability harness, and actually running any
of it are separate, later work, undertaken only after this document is
frozen (which it now is).

---

## 1. Versioning

`STOCK_STRATEGY_V2_VERSION = "stock_trend_momentum_v2"` - to be added to
`utils/strategy_signals.py` only when implementation begins. As of this
freeze, it is **not** added to that file, and it is **not** registered in
`utils/strategy_registry.py`. It has no evidence to report yet, so it has
no registry entry.

---

## 2. Baseline: v1's rule, restated in full

Everything below is v1's *actual, current* behavior, quoted/restated from
`utils/strategy_signals.py:67-151` and `backtest/whole_bot_engine.py`'s
`build_signal_calendar`, not paraphrased. v2 (Section 3) is defined as
this baseline plus exactly three additions - nothing here is removed or
loosened.

**Entry conditions** (`stock_trend_momentum_v1`, all must pass):
1. `close >= min_price`, where `min_price = 10.0` (the exact keyword
   argument `build_signal_calendar` passes).
2. Trend must be `UPTREND`. Exact definition
   (`utils.indicators.trend_label`): `close > sma_20 > sma_50 > sma_200`.
   Any other ordering, **or any missing/NaN input among
   close/sma_20/sma_50/sma_200**, resolves to `SIDEWAYS` - never treated
   as a pass by omission.
3. `50.0 <= rsi_14 <= 68.0` (`STOCK_RSI_LOW`/`STOCK_RSI_HIGH`).
4. `macd_hist > 0`.
5. `rel_volume >= 1.2` (`STOCK_MIN_REL_VOLUME`).
6. The ATR bracket (`utils/pricing.py::rounded_long_bracket`, stop at
   1.5x ATR below entry, target at 3.0x ATR above entry, `min_rr = 2.0`,
   tick size $0.01) must resolve without error.

**Conviction score** (computed only if all six conditions above pass):
base 65; **+8** if `rsi_14 >= 60`; **+8** if `rel_volume >= 1.5`; **+6**
if `macd_hist > close * 0.0003`; **+5** if
`(sma_20 - sma_50) / sma_50 > 0.01`; capped at 92.

**Candidate-inclusion floor** (`build_signal_calendar`, a *second*, separate
gate from the six conditions above): a candidate is added to that day's
list only if `decision.passed` **and** `(decision.conviction_score or 0)
>= 70`.

**Candidate ranking** (`build_signal_calendar`): each day's candidates are
sorted by descending conviction, then by symbol ascending -
`sort(key=lambda c: (-c.conviction, c.symbol))`. This determines which
candidates get that day's limited `max_new_trades_per_day`/`max_positions`
slots whenever more than one signal fires the same day.

**Universe**: `config/universe.py::UNIVERSE` (144 symbols including SPY;
confirmed identical to the corrected run's own `config.json` stock symbol
list). Pinned by file hash in Section 12, not reprinted symbol-by-symbol
here - the rule is "this file's list," not the list's contents frozen as
prose.

**Decision time**: 11:16 America/New_York, once per trading session
(`whole_bot_engine.py::DECISION_TIME_ET`).

**Date range for this study**: 2022-01-01 through 2026-08-09 (the corrected
run's own range).

**Portfolio configurations** (both evaluated):
- `current_1pct`: starting equity $100,000, risk 1.0% of equity per trade,
  max 5 open positions, max 2 new trades/day, max 20% of equity per
  position, 2% daily loss limit.
- `safe_0_25pct`: identical except risk 0.25% of equity per trade.

**Exit rules** (unchanged by this document): stop at the ATR-based stop
price; target at the ATR-based target price; `monitor_reversal` (exits at
the 30-minute monitor tick if trend has flipped to `DOWNTREND`, or
`macd_hist < 0` and `rsi_14 < 45`); `end_of_test` (marks to the last
available close if the backtest window ends before any other exit).

**Cost tiers** (`whole_bot_engine.py:49-52`, unchanged): `zero` (0bps/leg),
`baseline` (5bps/leg stock), `stressed` (13bps/leg stock).

---

## 3. The three preregistered changes

Everything in Section 2 carries over unchanged. `stock_trend_momentum_v2`
is v1 plus exactly these three additions, each independently toggleable
for the ablation study (Section 5). Nothing else about v1's rule changes.

### Delta1 - finite entry-order expiration

**Exact boundary.** Let `decision_day` be the NYSE session
(`utils/market_calendar.py::session_for`) on which a candidate was
approved - session 1. Using the existing
`trading_days_between(decision_day, decision_day + timedelta(days=30))`
(30 calendar days comfortably contains 6 sessions), `session_5` is that
list's 5th element (index 4). The exact expiration timestamp,
`expires_at`, is `session_for(session_5)["close"]` - session 5's own
regular-market close, including early closes (`session_for` already
returns each session's real close time; no separate early-close
adjustment is needed).

**A fill is valid only if `ts < expires_at`** - strict less-than, matching
this codebase's own existing convention for the identical boundary:
`_stock_regular_bars` already filters regular bars via
`index < session["close"]` (`whole_bot_engine.py:380-383`), because a
bar's timestamp marks the *start* of that minute, so a bar timestamped
exactly at close represents a minute that never actually traded.

**Enforcement location 1 - the outcome simulator itself.**
`simulate_order_outcome` precomputes a candidate's entire fill/exit
sequence once, scanning minute bars arbitrarily far into the future,
cached by `cache_key` - independent of `simulate_portfolio`'s day-by-day
loop. Removing an item from `active` on session 6 does not undo a fill
`simulate_order_outcome` already decided happened during session 6 itself,
before that day's 11:16 decision. The entry-fill search inside
`simulate_order_outcome` (both the vectorized `hit_positions` low-price
scan and the per-minute loop that follows it) must itself stop considering
bars once `ts >= expires_at`. If no fill occurred by then, the outcome is
a **new**, distinct status `"expired_unfilled"`, with `filled_at=None`
and `expires_at` carried on the outcome dict - never conflated with
`"unfilled_end"` (which means "still open at the *backtest's* end," a
different condition).

**Enforcement location 2 - the portfolio loop, for capital, not just
position count.** The existing `_occupies_slot()` capacity logic (from the
earlier zombie-position-cap fix) already correctly stops an
`expired_unfilled` item from occupying a `max_positions` slot, because
`filled_at=None`. But `simulate_portfolio` only ever returns
`reserved_notional` to `cash` inside its existing realization branch
(`if exit_time and exit_time <= decision: cash += reserved_notional +
net`), and an `expired_unfilled` item has no `exit_time` - so without a
further change, its cash would stay locked forever, reproducing on the
capital dimension exactly the artifact Delta1 exists to fix on the
position-count dimension. A **second**, parallel branch is required in the
same realization loop:

```python
elif outcome.get("expires_at") and outcome["expires_at"] <= decision:
    cash += item.reserved_notional   # full return - no net P&L, no cost
    # not added to realized_pnl/day_realized: nothing was gained or lost
    # item is dropped (not appended to `remaining`)
```

This is parallel to, not a replacement for, the existing exit branch.
`status` stays `"expired_unfilled"` in the recorded trade row - this
branch changes portfolio state, not the outcome's own label.
`_transaction_cost` already returns `0.0` for any non-`"closed"` status
(`whole_bot_engine.py:546`), so no change is needed there.

**Outcome-cache key must distinguish this policy.** `simulate_portfolio`'s
`cache_key` (`candidate.strategy_version, symbol, decision_time, entry,
stop, target`) does not encode whether Delta1's bounded fill-search was
applied. If an ablation harness shares one `outcome_cache` across variants
that differ only in Delta1, a candidate's outcome computed under
"unlimited" could be silently reused for a "5-session-expiration" variant
under the *same* cache key, corrupting results without error. The
implementation must either (a) include the expiration policy in
`cache_key` (e.g. an `expiration_sessions: int | None` field), or (b) use
a separate, non-shared `outcome_cache` per Delta1 setting. This risk is
specific to Delta1: Delta2 and Delta3 both filter which candidates are
*generated* (parallel to the existing conviction-score >= 70 floor), not
how a generated candidate's own outcome is computed, so neither needs to
be part of the cache key.

**Scope boundary.** Delta1 bounds only the *entry* fill search window.
Once a candidate fills before `expires_at`, its exit continues to be
governed by the existing stop/target/`monitor_reversal`/end-of-test logic
with no additional time limit - this delta is about unfilled resting
orders, not a maximum holding period for filled positions.

**Why 5 sessions**: a round, conventional swing-setup window (about one
calendar week) - **not** derived from the zombie orders' own observed
durations (110-1,193 days in the corrected run). Using those durations to
pick this number would be fitting the parameter to the exact data this
document must not cherry-pick from.

### Delta2 - ADX(14) >= 25 trend-strength filter

Defined entirely in terms of this codebase's own existing smoothing
primitive (`utils/indicators.py::atr`'s
`.ewm(alpha=1/period, min_periods=period, adjust=False)`,
`indicators.py:59-66`) rather than any external library, so no
implementation/library choice can introduce undisclosed variance:

1. `prev_high, prev_low, prev_close = high.shift(1), low.shift(1), close.shift(1)`.

   `tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)`
   - identical to `atr()`'s own TR formula (`indicators.py:61-65`).
   **First-bar treatment**: for the first bar, `prev_close` is NaN, so the
   second and third terms are NaN; `DataFrame.max(axis=1)` skips NaN by
   default, so `tr` reduces to `high - low` for that bar - inherited
   automatically from reusing `atr()`'s exact formula, not a separate
   decision.

   `up_move = high - prev_high`; `down_move = prev_low - low`.
   `np.where(...)` returns a plain NumPy array, not a pandas `Series` -
   `.ewm()` (step 2) does not exist on a bare array, so the result must be
   wrapped back into an indexed `Series` immediately:
   ```python
   plus_dm = pd.Series(
       np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
       index=high.index, dtype=float,
   )
   minus_dm = pd.Series(
       np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
       index=high.index, dtype=float,
   )
   ```
   **First-bar treatment**: `prev_high`/`prev_low` are NaN for the first
   bar, so `up_move`/`down_move` are NaN; any comparison against NaN is
   `False` in `np.where`'s condition, so both `plus_dm` and `minus_dm`
   resolve to `0.0` for that bar - not NaN. **Tie behavior**: if
   `up_move == down_move` exactly (both positive), neither `>` condition
   is true, so both `plus_dm` and `minus_dm` are `0.0` - strict inequality
   is required for either to register.

2. Each of `tr`, `plus_dm`, `minus_dm` is smoothed with the exact same
   `.ewm(alpha=1/14, min_periods=14, adjust=False)` call `atr()` already
   uses - internally consistent with how this codebase computes every
   other Wilder-style average, not a second, differently-seeded
   convention. **Missing hourly slots** (a gap, e.g. a trading halt) are
   handled exactly as `atr()`/`rsi()` already handle them today: `.ewm()`
   operates on whatever rows are present and silently skips gaps rather
   than rejecting the window - an existing, accepted property of this
   codebase's indicators, not a new decision for ADX.

3. `plus_di = 100 * smoothed_plus_dm / smoothed_tr.replace(0, np.nan)`,
   `minus_di = 100 * smoothed_minus_dm / smoothed_tr.replace(0, np.nan)`.
   **Division-by-zero**: the `.replace(0, np.nan)` on the denominator
   matches this codebase's own existing convention for the same problem
   (`relative_volume`'s `avg_vol.replace(0, np.nan)`, `bollinger_bands`'s
   `middle.replace(0, np.nan)` - `indicators.py:80,91`) - a zero-volatility
   window produces a clean NaN, never `inf` or a runtime warning.

4. `dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)` -
   same zero-denominator convention when both DI values are zero.

5. `adx_14 = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean()` -
   the same primitive applied a fourth time.

**Data source**: fully closed hourly bars only, gated through the exact
same `_snapshot()`/`completed_bar_cutoff()` discipline already applied to
every other indicator - never a partially-formed current bar.

**Structural minimum, derived exactly.** `smoothed_tr`, `smoothed_plus_dm`,
`smoothed_minus_dm` are three *parallel* `.ewm(min_periods=14)` calls -
each independently needs 14 observations of its own already-non-null
input (both are non-null from row 0 under the first-bar treatment above),
so each first becomes valid at row 13 (0-indexed) - the 14th observed row.
`plus_di`, `minus_di`, and `dx` are elementwise formulas with no windowing
of their own, so `dx` is also first valid at row 13 - null for rows 0-12,
non-null from row 13 onward. The *fourth*, sequentially-dependent
smoothing, `adx_14 = dx.ewm(min_periods=14).mean()`, needs 14 non-null
`dx` observations - rows 13-26 - so `adx_14` is first structurally valid
at row 26 (0-indexed), the **27th observed row**, assuming no
zero-denominator NaN propagates from step 3/4 above (a data-dependent
condition that can only push this later, never earlier).

**Enforcement is a two-step gate** (a bare NaN check alone would only
enforce the row-27 structural minimum, not a stricter policy - it could
permit ADX as early as row 27):
1. Reject unless at least **40 observed** hourly rows are available
   through the decision cutoff (the same completed-bar window `_snapshot`
   already uses) - an explicit row-count check, not inferred from NaN.
   Called "observed," not "consecutive," because gaps are defined above to
   be silently skipped, not rejected - if true wall-clock continuity is
   ever required, that would be a separate, additional continuity check
   this document does not include.
2. *Then*, reject unless `adx_14` is finite and `adx_14 >= 25`.

40 is a deliberate margin *above* the known structural minimum of 27 -
chosen because a stricter warm-up only ever produces *fewer* candidates
for a strategy whose whole premise is fewer, higher-quality signals, and
to absorb some data-dependent slack from the division-by-zero case without
needing to re-derive it precisely. (The existing SMA-200 requirement
inside the UPTREND check already implies well over 40 hourly observations
in the ordinary case - this explicit gate is not solving a warm-up problem
that doesn't otherwise exist; it is a deterministic, self-contained rule
for Delta2 specifically, so Delta2 never silently depends on what SMA-200's
warm-up happens to require elsewhere.)

**Scoped to Delta2-enabled variants only - `_snapshot()` itself is never
changed.** `_snapshot()` is shared by v1's stock signal, the entirely
unrelated crypto signal, and every ablation variant in this study,
including the ones with Delta2 off. Adding the row-count/finite/`>= 25`
checks to `_snapshot()`'s own generic `required` completeness tuple would
silently reject Delta2-*off* candidates over an indicator they never use,
contaminating the baseline-v1 and Delta1-alone/Delta3-alone ablation arms,
and would affect crypto, which has nothing to do with this preregistration.
Instead:
- `adx_14` **may be computed and stored** as an ordinary new column in the
  stock indicator frame (`indicator_frame()`), unconditionally, the same
  way `rsi_14`/`atr_14`/etc. already are - harmless by itself, since
  storing a column rejects nothing.
- The three-part gate above (row count, finite, `>= 25`) is checked
  **only inside the Delta2-enabled code path** - never inside
  `_snapshot()`.
- **Delta2-off variants** (baseline-v1-restated, Delta1-alone,
  Delta3-alone, Delta1+Delta3) **must reproduce v1's eligibility exactly**
  - they never read `adx_14` at all, so a candidate is never rejected
  because ADX happens to be unavailable, regardless of how much history
  exists.
- **Crypto behavior is unchanged** - `crypto_trend_momentum_v1` never
  references ADX, and this preregistration is stock-only.

**Threshold provenance, stated precisely**: ADX >= 25 as the
trend/no-trend cutoff is a widely used convention in technical-analysis
practice. This document does not claim it is Wilder's own originally
prescribed number - that would need a direct primary-source citation this
document does not have - only that it is a common, independently
established convention, not a value fit to this dataset.

### Delta3 - minimum expected-move-vs-cost filter

**Exact formula**, using the candidate's *final, rounded* entry and target
prices (i.e. `bracket["entry"]`/`bracket["target"]` after
`rounded_long_bracket()` - the actual prices that would be used for order
placement, not a pre-rounding ATR-multiple figure):

```
move_bps = (target - entry) / entry * 10_000
reject unless move_bps >= 50
```

50bps = 5x the baseline round-trip cost (`BASELINE_COST` is 5bps/leg =
10bps round trip, `whole_bot_engine.py:49-52`) - a general "edge should
clearly dominate cost" heuristic anchored to the cost model's own existing
constant, not to any observed v1 outcome.

---

## 4. Explicit transaction-cost assumptions

Reuses the existing `ZERO_COST`/`BASELINE_COST`/`STRESSED_COST` tiers
(`whole_bot_engine.py:49-52`) unchanged: 0 / 5bps / 13bps per leg, stock.
No new cost model is introduced. Delta3's threshold is pinned to the
baseline tier specifically.

---

## 5. Ablation matrix - diagnostic only, not a selection mechanism

Full 2^3 factorial across {Delta1, Delta2, Delta3}: 8 variants
(baseline-v1-restated, each single delta alone, each pair, and the full
combination). Each variant is run across both portfolios
(`current_1pct`/`safe_0_25pct`) and all three cost tiers - 48 simulation
slices total, all against already-cached price data. This isolates each
change's individual contribution and any interaction effects.

**The full three-change combination (Delta1+Delta2+Delta3, i.e.
`stock_trend_momentum_v2` as defined in Section 3) is the only variant
ever checked against the qualification bar (Section 8) or reported as a
registry candidate.** The other seven combinations are reported for
interpretability only - understanding *why* v2 did or didn't qualify. If
some other combination happens to look better than the full v2 on this
data, that is **not** grounds to substitute it as "the real v2" after the
fact - that is exactly the after-the-fact selection this preregistration
exists to prevent. Such a finding is only legitimate grounds for a *new*,
separately dated `v3` preregistration, chosen before looking at further
data (Section 11) - not a patch folded back into v2's own evaluation on
2022-2026.

---

## 6. Year-by-year temporal stability analysis

(Deliberately not called "walk-forward": no parameter is refit or
adjusted at any step, and "walk-forward" would overstate what this is.)

Five fixed calendar-year folds over the available data: 2022, 2023, 2024,
2025, 2026-partial (through 2026-08-09). The preregistered ruleset (the
full Delta1+Delta2+Delta3 combination, exactly as frozen in Section 3) is
fixed *once*, before fold 1 runs. No fold's result may feed back into
adjusting Delta1/Delta2/Delta3 for any fold, including earlier ones -
nothing is trained.

Each fold is reported individually (trade count, expectancy, win rate)
purely as a **diagnostic** - it shows whether performance is reasonably
consistent across time or concentrated in one lucky sub-period. **Only
the full-period aggregate is checked against the qualification gates**
(Section 8); no individual fold's pass/fail status is itself a
qualification criterion.

---

## 7. Why 2022-2026 is not a clean holdout

Stated explicitly, not glossed over: the author of this document has
already seen aggregate 2022-2026 statistics via the failure-analysis
report (yearly/quarterly breakdowns, regime and volatility clustering,
MFE/MAE by winner/loser). That means even a same-period temporal-stability
split carries hindsight risk in how Delta1-Delta3 were chosen, no matter
how the folds are drawn. The per-fold diagnostic in Section 6 is still
useful - it would catch a ruleset that only "works" in one lucky
sub-period - but it is not a substitute for genuinely unseen data. The
only real "untouched" test is Section 10's forward paper period.

---

## 8. Qualification bar - unchanged from v1

The full-period aggregate for the full Delta1+Delta2+Delta3 combination
(and *only* that combination, per Section 5) is checked against
`backtest/whole_bot_metrics.py::qualify_strategy` exactly as it exists
today. All 13 checks, stated precisely and verbatim:

1. Closed-trade count >= 100.
2. Data coverage >= 95%.
3. Missing-outcome rate < 1%.
4. Baseline net expectancy positive.
5. Stressed net expectancy positive.
6. Profit factor >= 1.15.
7. Sharpe >= 1.0.
8. Sharpe greater than the SPY benchmark's Sharpe.
9. Maximum drawdown <= 15%.
10. Trailing-12-month net P&L positive.
11. Positive quarters >= 60%.
12. Bootstrap 95%-lower-bound mean R positive.
13. Single-symbol profit concentration <= 25%.

This document does not propose changing any of these thresholds.

---

## 9. Deployment-scope guardrail

Stated as its own explicit, standalone clause: **even a full historical
pass under Section 8's unchanged bar does not, by itself, and cannot:**
**(a)** change `execution_eligible` for `stock_trend_momentum_v2` in
`utils/strategy_registry.py`, **(b)** enable any scheduled task that would
run it live, or **(c)** begin any automatic paper-money entries - not even
at the hard-capped 0.25% level described in Section 10. Every one of
those requires a separate, subsequently reviewed rollout plan; none is an
automatic consequence of this preregistration's own results.

No change to `utils/strategy_registry.py`, `run_pipeline.py`, any
scheduled task, or `nodes/execution_node.py` happens as part of this
preregistration or its eventual ablation/temporal-stability runs,
regardless of outcome.

---

## 10. Future forward-paper validation (described, not built)

Even a full historical qualification only makes v2 eligible for a real,
forward-only paper-trading shadow period, hard-capped at 0.25% risk per
trade, specifically because it is the only genuinely blind test - it
evaluates price action that has not happened yet as of this document's
freeze date. This connects directly to Section 7. Building this mechanism
(mirroring the three-state `historical_qualified`/`paper_forward_eligible`/
`execution_eligible` model scoped out separately during the zombie-bug
fix) is out of scope for this document - described here so the full
evidence chain is visible, not as work being started.

---

## 11. Anti-cherry-picking rule

The failure-analysis report may motivate *why* Delta1-Delta3 were chosen
(e.g. "losers' MAE runs close to full planned risk" as background
motivation for why a trend-strength filter is a reasonable hypothesis to
test at all) but **may not** be used to select a specific value because it
looked profitable in that same dataset (e.g. no "only trade the volatility
quartile that happened to be net-positive" filter, no threshold picked by
scanning which number produced the best 2022-2026 result). If the ablation
or temporal-stability results suggest a *new* rule not in this
preregistration, that becomes a candidate for a separately dated `v3`
preregistration, evaluated on data this document has not already examined
- not a patch folded back into v2's own evaluation on 2022-2026.

---

## 12. Provenance: source-file and cache hashes, frozen now

SHA256 hashes of the source files this preregistration's rules depend on,
computed at freeze time (2026-08-12T01:30:30Z):

| File | SHA256 |
|---|---|
| `utils/strategy_signals.py` | `92d6f6dd149b4ee75d7aa28ba8929438cb20a9360a4e37132d5b0911d9059151` |
| `utils/indicators.py` | `d244a858d49ba48de7e6b5f897300adabb5c815e13297f73c82fc1a7cdd4c47a` |
| `backtest/whole_bot_engine.py` | `59e62c21c249f35a0896992c3742d6b02b3badf85651b0aedb167bb4a92350df` |
| `backtest/whole_bot_metrics.py` | `64a5835a755c648d6b595e8d150fbbb781596cf0dc8d884554966d40d1fcc4ea` |
| `config/universe.py` | `b756bb8b34452619e22b6eeef8a49cfeeda0238847c4b41970ab1a7e7337f979` |
| `utils/market_calendar.py` | `e016e0eff92b9820bba99c6d174c7b3fb6534acca94f7edc14470e6ae4fdd964` |

`utils/market_calendar.py` is included because Delta1's expiration
timestamp depends on `session_for`/`trading_days_between`.

**Market-data cache, hashed at freeze time**:
`backtest/cache/bars_cache.db` SHA256 =
`2aa39029331c7dea6c620c8a23a43b70764fe40ad6761eda901da6775d83d841`
(identical to the hash recorded in the failure-analysis report's own
manifest, confirming no data has changed between that analysis and this
freeze).

**Why freeze-time, not only post-run**: recording a cache hash only after
the eventual run proves what that run used, but proves nothing about
whether it is the same dataset that existed when these rules were frozen.
Recording it here, now, gives something to check *against*.

**The eventual implementation/ablation/temporal-stability run's own
results manifest must include both this document's own hash (see the
accompanying `.manifest.json`) and a fresh hash of
`backtest/cache/bars_cache.db`, and must explicitly compare the fresh hash
against the one recorded above.**

**A mismatch invalidates the qualification run - it is not a caveat on an
otherwise-standing result.** If the fresh hash differs from the one frozen
here, the run must hard-stop *before* evaluating Section 8's qualification
bar, or, if it proceeds for exploratory/diagnostic purposes only, its
result must be recorded as `invalid`/`non-qualifying` outright - never
reported as a genuine pass with a footnote. A strategy cannot be said to
have qualified against data different from what this document froze. If
newer data is genuinely needed, that requires either a new, separately
dated preregistration, or a data-manifest amendment that is itself a
**separate, dated, append-only document** referencing this document's
hash - never an in-place edit to this file.

---

## Explicitly out of scope for this document

- Writing any code: no changes to `utils/strategy_signals.py`,
  `utils/indicators.py`, `backtest/whole_bot_engine.py`,
  `utils/strategy_registry.py`, `run_pipeline.py`, any node, or any test.
- Running any backtest, ablation slice, or temporal-stability fold.
- Any registry entry for `stock_trend_momentum_v2` - it does not exist in
  `utils/strategy_registry.py` until it has evidence to report, and even
  then only as rejected/pending, never `execution_eligible=True` directly.
