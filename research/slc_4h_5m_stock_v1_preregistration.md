# SLC 4-hour / 5-minute stock adaptation v1 — frozen preregistration

Frozen: 2026-08-12. This document is a research specification, not evidence
that the strategy is profitable and not authorization to submit orders.

## 1. Source and scope

Source video: https://youtu.be/ExvoIqNglOk

The source describes SLC: Structure, Level, Confirmation. It demonstrates the
method on Nasdaq-100 E-mini futures. This implementation adapts it to the
bot's fixed stock universe because the bot does not execute futures. The live
pipeline, strategy registry, execution node, schedulers, existing swing
strategy, crypto strategy, and ORB strategy must remain unchanged.

Version identifier: `slc_4h_5m_stock_v1`.

## 2. Rules stated or visually confirmed by the source

1. Use the 4-hour chart to classify the current day as uptrend, downtrend, or
   consolidation. Higher highs plus higher lows are an uptrend; lower highs
   plus lower lows are a downtrend; consolidation is not traded.
2. Use the 5-minute chart for entries. Take longs only in a 4-hour uptrend and
   shorts only in a 4-hour downtrend.
3. A supply/demand level is the full high-to-low range of the final candle
   immediately before a sharp move away. A level that price has crossed many
   times is invalid. A fresh level may be traded on its first return. A level
   cleanly broken once may be traded after price cleanly returns through it
   and retests it from the opposite side.
4. Confirmation uses TradingView Stochastic settings K length 5, K smoothing
   3, D smoothing 3, with 80 and 20 thresholds and closed-candle values. A
   short requires K above 80 at supply and a closed-candle cross back below
   80. The long rule is the exact mirror: K below 20 at demand and a cross
   back above 20.
5. Enter after confirmation, put the stop slightly beyond the level, and set
   the take-profit at exactly two times initial risk (2R).

No additional entry indicator, chart pattern, news condition, or optimizer is
part of v1.

## 3. Necessary automation conventions not specified by the source

These definitions make subjective visual language deterministic. They are
implementation conventions, not claims about words used by the creator. They
are frozen before testing and may not be tuned on v1's results.

### Sessions and bars

- Stocks only. Use split-adjusted Alpaca SIP historical data.
- Use NYSE regular trading hours and the exchange calendar, including holidays
  and early closes. Ignore premarket and after-hours bars.
- Construct session-anchored 4-hour bars from 5-minute bars: 09:30–13:30
  Eastern and 13:30–the official close. The shortened second bar is valid.
- At every 5-minute decision, structure uses only 4-hour bars whose end time is
  at or before that decision. A signal is acted on at the next 5-minute open.

### Structure

- A pivot high is higher than the highs of the two bars on each side; a pivot
  low is lower than the lows of the two bars on each side. Ties are not pivots.
- A pivot becomes known only after both right-side bars have closed.
- Uptrend requires the two latest confirmed pivot highs and pivot lows to be
  strictly rising. Downtrend requires both pairs to be strictly falling. All
  other states are consolidation and fail closed.

### Levels and an aggressive move

- ATR(14) on completed 5-minute bars is used only to quantify the video's
  subjective phrase "sharp move"; it is not an entry confirmation.
- A demand candidate is a base candle followed by three completed bars where
  at least two are bullish, the third closes above the base high, and the
  maximum high is at least 1.0 ATR above the base close.
- A supply candidate is the mirror: at least two following bars are bearish,
  the third closes below the base low, and the minimum low is at least 1.0 ATR
  below the base close.
- The level is the base candle's full low-to-high range and becomes active only
  after all three impulse bars close. Zero-width or non-finite levels are
  invalid. Levels expire after 20 NYSE sessions.
- Overlapping same-direction levels are retained; when simultaneous signals
  exist, use the most recently activated level, then greater impulse/ATR, then
  symbol alphabetically. One new trade per symbol per session is allowed.

### Fresh, reclaimed, and invalid levels

- A touch means a completed bar's `[low, high]` overlaps the level.
- A fresh level is eligible only on its first post-activation touch.
- A close beyond the far edge breaks the level: below demand or above supply.
- A once-broken demand becomes short-side supply only after a later close below
  its lower edge; a once-broken supply becomes long-side demand only after a
  later close above its upper edge. This is the video's broken-level reversal,
  expressed symmetrically. The next return to the level is its sole retest.
- Any subsequent close through the far edge, or any second retest, invalidates
  the level. This implements the video's instruction not to use chopped levels.

### Stochastic, entry, stop, and exit

- Raw K is `100 * (close - lowest_low_5) / (highest_high_5 - lowest_low_5)`.
  Smoothed K is SMA(3) of raw K and D is SMA(3) of smoothed K. A zero-range or
  non-finite window has no value and fails closed. D is recorded but not an
  extra gate because the source's demonstrated trigger uses K and 80/20.
- A short arms when a closed bar touching eligible supply has K > 80 and
  confirms when a later closed touching bar crosses from `>=80` to `<80`.
  A long mirrors this with K < 20 and a cross from `<=20` to `>20`.
- The entry is the next 5-minute bar's open. No new entry is allowed after
  15:30 Eastern. A missing next bar means no trade.
- "Slightly beyond" means `max($0.01, 0.10 * ATR14)` beyond the level at the
  confirmation bar. Non-positive risk rejects the trade. Target is exactly 2R.
- Outcomes use one-minute regular-session bars when available. A gap through a
  stop fills at that bar's open; otherwise stop/target fills at their level.
  If both occur in one minute, the stop wins (adverse ambiguity). Any open
  position is flattened at the final available minute of the official session.
  Missing outcome coverage remains missing, never a win, loss, or no-trigger.

## 4. Portfolio and costs

- Starting equity $100,000; cash-only; no leverage; maximum five positions;
  maximum two entries per day; one open position per symbol; maximum 20% of
  equity per position; stop approving entries after a 2% realized daily loss.
- Primary research risk is 0.25% of equity per trade. Report 1.0% sensitivity.
- Costs per leg: zero 0 bps, baseline 5 bps, stressed 13 bps.
- Shorts are simulated assuming shares are available to borrow. Results must
  disclose this unverified assumption and cannot authorize deployment until
  Alpaca paper-forward checks confirm shortability at every attempted entry.

## 5. Data period, reports, and limitations

- Intended full comparison: 2022-01-01 through 2026-08-09, all symbols in the
  bot's fixed stock universe. The report must state coverage and exclusions.
- The fixed present-day universe has survivorship/selection bias. Historical
  results are technical strategy before the live RSS news veto. Neither is to
  be silently treated as point-in-time information.
- Save immutable configuration, coverage, signals, trades, equity curves,
  quarterly/yearly summaries, missing outcomes, and qualification output.
- Research modules may import market-data clients but must not import any
  broker trading/order client. A cache-only mode must never fetch missing data.

## 6. Qualification and governance

Use the existing whole-bot qualification gates without weakening them: at
least 100 closed trades, at least 95% usable coverage, missing outcomes below
1%, positive baseline and stressed expectancy, baseline profit factor >=1.15,
annualized Sharpe >=1.0 and above SPY, drawdown no worse than 15%, positive
most-recent 12 months and at least 60% positive quarters, bootstrap 95% lower
mean R above zero, and no symbol above 25% of positive profit.

Passing this historical test does not enable execution. It permits a separate
paper-forward proposal at 0.25% risk for at least 90 calendar days and 30
closed trades. Until that proposal is separately reviewed, this version is
unregistered, research-only, and structurally unable to submit an order.

## 7. Required tests

Tests must cover pivot confirmation/no-lookahead, trend and consolidation,
session-anchored bars/DST/early close, exact Stochastic settings, mirrored
crosses, level activation, fresh/reclaimed/chopped states, next-bar entry,
15:30 cutoff, stop buffer, exact 2R target, gaps, adverse same-minute outcomes,
end-of-day flatten, portfolio gates, costs, missing data, deterministic output,
and a static prohibition on order-client imports.

## 8. Freeze rule

This file is append-immutable after its SHA-256 is recorded in its manifest.
Any rule change requires a new version or separately hashed amendment. Failed
results may not be used to tune this version on the same evaluation period.
