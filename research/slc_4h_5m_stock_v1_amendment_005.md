# SLC v1 preregistration amendment 005 — broker-valid price precision

Dated: 2026-08-13.

Parent document SHA-256:
`d453ae039c3d9145e986ff6cf27ce98af7418b7b5ea6da0d2ddb728e801c3e9d`.

Like amendment_004, this amendment documents a real operational deviation
made necessary by live broker order submission — it does not change the
signal, stop, or target *formulas* in the preregistration, but it changes
how their outputs are realized as broker-valid prices.

1. Tick size. Alpaca requires prices to be expressed at or above $1.00 in
   $0.01 increments, and below $1.00 in $0.0001 increments. Every price
   submitted to the broker is normalized to the tick implied by its own
   value.

2. Arithmetic. All tick normalization uses `decimal.Decimal`, never binary
   float rounding. Every incoming price converts via `Decimal(str(value))`,
   never `Decimal(value)` directly — the latter imports the exact binary
   floating-point representation's rounding noise before normalization
   ever runs, which would defeat the purpose of using `Decimal` at all.

3. Stop rounding, away from entry. A long stop rounds down to the tick; a
   short stop rounds up. This never lets tick rounding silently tighten
   protection closer to the entry than the frozen level-derived stop
   requires. The stop is computed from the level and ATR, independent of
   the entry price, exactly as the preregistration specifies.

4. Target rounding: two named alternatives, with this amendment explicitly
   selecting one.

   - Strict literal fidelity: reject any trade whose mathematically exact
     2R target cannot be represented at the permitted tick.
   - Practical fidelity (selected): round to the nearest valid tick that
     still achieves at least 2R, never less. A long target rounds up from
     the exact-2R value computed from the actual fill; a short target
     rounds down.

   Practical fidelity is selected because a valid tick achieving at least
   2R always exists at any realistic price, while strict fidelity would
   reject a meaningful fraction of trades for tick-precision reasons
   alone, unrelated to the strategy's signal logic. This is a real,
   disclosed deviation from literal "exactly 2R," not a rounding detail
   to gloss over: the submitted target is occasionally a whole tick above
   the mathematical 2R value. The submitted target is never labeled
   "exactly 2R" in any audit record, log, or report. It is recorded as
   the effective reward:risk, computed from the actual submitted broker
   prices — `(submitted_target − actual_fill) / (actual_fill − submitted_stop)`
   for a long, mirrored for a short — always at least 2.0, and always
   kept distinct from the frozen strategy's theoretical exact-2R
   definition.

5. Rounded-value validation, applied at two different points with two
   different consequences.

   - Pre-submission (using a fresh quote as the reference price, before
     any order exists): if the tick-rounded stop or target is zero,
     negative, non-finite, or directionally invalid against that
     reference price, there is nothing to flatten yet — the trade is
     skipped outright, and no submission is ever attempted.
   - Post-fill (using the actual fill price): if the tick-rounded stop or
     target is invalid against the real fill, a genuine open position
     exists. This is handled as a post-fill directional-validation
     failure — the position is marked protected_degraded and exited
     through the guarded emergency-exit path, never left open and never
     described as healthy.

6. Sizing and every post-fill risk check use the actual submitted
   (tick-rounded) stop, never the unrounded theoretical value from
   `utils/slc_signals.py`.

This amendment does not change the 0.25% risk, five-position, two-entry,
20% concentration, or 2% daily-loss rules in Section 4, and does not
change the stop or target formulas in Section 3 beyond the tick
normalization and validation described above.

This file is append-immutable after its SHA-256 is recorded.
