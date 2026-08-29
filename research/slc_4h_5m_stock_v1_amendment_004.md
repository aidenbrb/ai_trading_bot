# SLC v1 preregistration amendment 004 — live paper-forward execution deviations

Dated: 2026-08-12.

Parent document SHA-256:
`d453ae039c3d9145e986ff6cf27ce98af7418b7b5ea6da0d2ddb728e801c3e9d`.

Unlike amendments 001-003, this amendment documents real deviations from
the preregistration's stated rules, made necessary by live operation. It
does not change the signal logic itself (structure, zones, Stochastic
5/3/3, stop/target formulas), but it changes how two rules are realized
operationally, and it adds operational rules the preregistration did not
need to state for a historical backtest.

1. Data source. Section 5 specifies split-adjusted Alpaca SIP historical
   data. Live paper-forward decisions instead use Alpaca IEX real-time
   data, because SIP is not available in real time on this account tier.
   This is a genuine data-source substitution, disclosed here rather than
   silently absorbed.

2. Entry price. Section 3 defines entry as the next 5-minute bar's open,
   read from historical data. Live, the next bar has not happened yet at
   confirmation time, so live entries instead use the actual Alpaca fill
   price of a bracket order submitted for that next bar. The frozen stop
   formula (`max($0.01, 0.10 * ATR14)` beyond the level) is unchanged and
   is computed from the level, not from the entry price.

3. Exact live timing.
   - Preflight: 8:35 AM ET.
   - Entry cycles: 9:36 AM ET, every 5 minutes, through 3:31 PM ET (the
     cycle processing the bar closing at 3:30, the latest confirmation
     whose entry bar still opens at or before 3:30 per the frozen
     no-new-entry-after-15:30 rule). No entry-cycle invocations occur
     after 3:31 PM ET.
   - Early-close entry cutoff: the earlier of 3:30 PM ET or the official
     close minus 30 minutes.
   - Closeout: begins at the official session close minus 5 minutes,
     retries every minute, and reconciles after the close. This is the
     live realization of "flattened at the final available minute of the
     official session."

4. Data-freshness limits, not stated in the historical preregistration
   because a backtest has no real-time deadline:
   - An entry cycle starts exactly 1 minute after its target bar's close.
   - A single batched bar fetch is polled for at most 15 seconds; only the
     exact expected closed-bar timestamp is accepted.
   - An execution quote must be no more than 5 seconds old at the moment
     of order submission.
   - If either deadline is missed for a symbol/signal, nothing is
     submitted for it that cycle, and the miss is recorded as
     `stale_or_missing_data`.
   - A signal is never acted on after the fact via backfill: if a later
     cycle's gap-backfill reveals a confirmation that would have fired in
     a bar now in the past, it is logged as missed and discarded, never
     converted into a live order.

This amendment does not change the 0.25% risk, five-position, two-entry,
20% concentration, or 2% daily-loss rules in Section 4, and does not
change the exact-2R target rule beyond recalculating it from the actual
fill price per item 2 above.

This file is append-immutable after its SHA-256 is recorded.
