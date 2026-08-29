"""
Read-only Alpaca account readiness check for live_slc, meant to be run by
hand (e.g. Sunday night before a trading day) before enabling the
Scheduled Tasks: python -m live_slc.check_readiness

Deliberately NOT `run_slc_live.py --stage preflight` - that stage
bootstraps/backfills reducer state and detects splits, so it isn't
provably side-effect-free. This module makes exactly three read-only
broker calls and nothing else.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReadinessResult:
    passed: bool
    observed_account_id: str
    expected_account_id: str
    position_count: int
    open_order_count: int
    failures: list = field(default_factory=list)


def evaluate_readiness(client, expected_account_id: str) -> ReadinessResult:
    """Exactly three read-only broker calls: get_account(),
    get_all_positions(), get_orders(status=OPEN). Nothing else on
    `client` is ever referenced - callers/tests can prove this with a
    spy client that raises on anything else."""
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    account = client.get_account()
    observed_account_id = str(account.id)

    positions = client.get_all_positions() or []
    open_orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN)) or []

    failures: list = []
    if observed_account_id != expected_account_id:
        failures.append(
            f"observed account {observed_account_id!r} != expected {expected_account_id!r}"
        )
    if len(positions) != 0:
        failures.append(f"{len(positions)} open position(s) found, expected 0")
    if len(open_orders) != 0:
        failures.append(f"{len(open_orders)} open order(s) found, expected 0")

    return ReadinessResult(
        passed=not failures,
        observed_account_id=observed_account_id,
        expected_account_id=expected_account_id,
        position_count=len(positions),
        open_order_count=len(open_orders),
        failures=failures,
    )


def main() -> None:
    from live_slc.execution import get_alpaca_client
    from live_slc.settings import live_slc_settings

    client = get_alpaca_client()
    result = evaluate_readiness(client, live_slc_settings.SLC_EXPECTED_ACCOUNT_ID)

    print("live_slc readiness check")
    print(f"  observed account: {result.observed_account_id}")
    print(f"  expected account: {result.expected_account_id}")
    print(f"  open positions:   {result.position_count}")
    print(f"  open orders:      {result.open_order_count}")
    if result.passed:
        print("PASS - account matches expected, 0 positions, 0 open orders.")
        sys.exit(0)
    print(f"FAIL ({len(result.failures)} issue(s)):")
    for f in result.failures:
        print(f"  - {f}")
    sys.exit(1)


if __name__ == "__main__":
    main()
