# SLC paper incident — 2026-08-19 TGT/MSFT reconciliation

## Scope and safety state

This is a paper-account incident. No real-money order was involved. After
the issue was reported, all three SLC Scheduled Tasks were disabled, both
execution switches were set false, and deployment status was transitioned
from `paper_active` to `suspended` (activation-event id
`7ebf7488-3837-4b5d-b2b3-3bbddcda692e`). A direct Alpaca read confirmed
zero positions and zero open orders before local reconciliation began.

The pre-reconciliation database is preserved at
`live_slc/backups/live_slc_pre_aug19_reconcile_20260819.db`.

## Broker-confirmed order outcomes

### TGT

- Entry order: `3550f40e-60f5-4c33-98dd-ff0e4b799dc8`
- Entry: 55 shares at $158.09
- Expected quote: $156.755
- Frozen stop: $151.85
- Post-fill monetary risk: $343.20
- Deterministic emergency exit: `63e30c16-a078-4428-8d13-d3d6931deb80`
- Exit: 55 shares at $157.790545 at 2026-08-19 13:36:57.658095 UTC
- Paper P&L before any unreported fees: -$16.470025

The emergency exit itself was correct. Fill slippage increased monetary
risk above 0.25% of current paper equity, so the existing fail-closed rule
required flattening. The defect occurred after that sell filled: Alpaca's
`get_open_position("TGT")` returned HTTP 404 / `position does not exist`,
which proves the symbol is flat, but the bot allowed the SDK exception to
escape instead of recording a confirmed-flat close.

### MSFT

- Entry order: `71cb959e-8a4c-40ae-9be0-4b14d480aee6`
- Entry: 45 shares at $482.767333
- Protective stop: `46c4d4e3-f0d7-455a-81b6-582a8c51f387`
- Stop fill: 45 shares at $481.208889 at 2026-08-19 13:45:10.521000 UTC
- Paper P&L before any unreported fees: -$70.129980

The broker bracket behaved correctly, but later cycles never reconciled a
locally-open position whose broker-side stop or target had already filled.
The local MSFT position therefore remained incorrectly marked open.

Combined broker-confirmed paper P&L: **-$86.600005**, before any fees that
Alpaca did not report through these order records.

## Corrections

1. A genuine Alpaca `APIError` backed by HTTP 404 is now recognized as
   confirmed absence. After a filled close order, `position does not exist`
   is treated as broker-confirmed flat; non-404/network failures remain
   ambiguous.
2. The close routine now retains and polls the order returned by a successful
   `submit_order` call. It no longer discards that response and immediately
   depends on a potentially lagging client-ID lookup.
3. Every cycle and closeout invocation now performs a read-only reconciliation
   pass before recovery/closeout action. A local position is closed only when:
   the broker position list confirms the symbol absent, exactly one known SLC
   stop/target/deterministic-exit order is filled, its quantity exactly matches,
   and a positive fill price exists.
4. Missing, conflicting, or quantity/direction-mismatched evidence marks the
   local position ambiguous and blocks entries. It is never guessed closed.
5. Position closure now persists the broker's actual exit timestamp, fill
   price, order id, exit reason, P&L, signal result, and idempotent session
   counters.

No SLC market-structure, zone, stochastic, entry, stop, target, ranking,
position-sizing, or numeric risk rule was changed.
