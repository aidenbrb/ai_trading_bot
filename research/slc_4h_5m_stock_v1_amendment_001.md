# SLC v1 preregistration amendment 001

Frozen before strategy implementation: 2026-08-12.

Parent document SHA-256:
`d453ae039c3d9145e986ff6cf27ce98af7418b7b5ea6da0d2ddb728e801c3e9d`.

This amendment resolves two wording/implementation ambiguities without using
any backtest result.

1. ATR(14) means the simple arithmetic mean of the latest 14 true ranges,
   where true range is the maximum of high-low, absolute(high-prior close),
   and absolute(low-prior close). It requires all 14 values.
2. The broken-level state is clarified as follows. A fresh supply breaks on a
   close above its high; it becomes eligible as supply again only after a later
   close below its low, followed by one retest from below. A fresh demand
   breaks on a close below its low; it becomes eligible as demand again only
   after a later close above its high, followed by one retest from above. The
   trade direction does not reverse. A further far-edge break or a second
   retest invalidates it.

This file is append-immutable after its SHA-256 is recorded.
