# SLC paper incident — NVDA — 2026-08-18

## Scope

This is an operational incident record for the first `paper_active`
session of `slc_4h_5m_stock_v1`. It changes no frozen signal, entry,
exit, ranking, sizing, or risk rule.

## Broker-confirmed facts

- Entry client order: `slc-3c32237cb7d591707c03-entry`
- Entry broker order: `2d7305a3-07a6-483e-a89a-37906e8450f6`
- Entry: buy 98 NVDA, fully filled at 221.399184 at
  2026-08-18 13:41:24.675438 UTC.
- Emergency-exit client order: `slc-040565ee91572c771432-exit`
- Exit broker order: `6a643f40-4f16-48e7-b9b3-1298580817d8`
- Exit: sell 98 NVDA, fully filled at 221.26 at
  2026-08-18 13:41:26.633670 UTC.
- Both bracket child orders were canceled with zero fills.
- Repeated read-only Alpaca checks after market close confirmed zero
  positions and zero open orders.
- Gross/net paper P&L recorded by this project (fees unavailable):
  -13.640032 dollars.

## Root cause

Several live execution paths used `str(order.status)` before comparing
with values such as `"filled"`. alpaca-py returns `OrderStatus` enum
objects for real broker responses, and `str(OrderStatus.FILLED)` is
`"OrderStatus.FILLED"`, not `"filled"`. Plain-string test doubles hid
the incompatibility.

The fully filled entry was therefore routed through the incomplete-fill
emergency-flatten branch. The emergency exit filled successfully, but
the same enum-conversion defect classified that fill as ambiguous. The
local database retained a `protected_degraded` position even though the
broker was flat, which correctly blocked every later entry that day.

## Evidence preserved

Before reconciliation:

- The formal session gate failed on non-flat local state and
  `protected_degraded_position_exists`.
- The paper-session audit failed on the `protected_degraded` position
  and the same system-wide block.
- The raw local entry/order/position records and broker order IDs were
  verified before any database correction.

Reconciliation must retain a critical `protected_degraded` audit event
and increment the session incident counter. The 2026-08-18 session must
remain a failed engineering/paper session even after its local position
is corrected to the broker-confirmed closed state.

