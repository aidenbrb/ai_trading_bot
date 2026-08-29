"""Weekday stock / weekend crypto routing tests."""
from datetime import datetime
from zoneinfo import ZoneInfo

from utils.market_session import automatic_market, resolve_market

ET = ZoneInfo("America/New_York")


def test_saturday_routes_to_crypto():
    assert automatic_market(datetime(2026, 7, 25, 9, tzinfo=ET)) == "crypto"


def test_sunday_routes_to_crypto():
    assert automatic_market(datetime(2026, 7, 26, 9, tzinfo=ET)) == "crypto"


def test_monday_routes_to_stocks():
    assert automatic_market(datetime(2026, 7, 27, 9, tzinfo=ET)) == "stocks"


def test_naive_datetime_is_interpreted_as_eastern():
    assert automatic_market(datetime(2026, 7, 25, 23)) == "crypto"


def test_manual_market_overrides_schedule():
    saturday = datetime(2026, 7, 25, 9, tzinfo=ET)
    assert resolve_market("stocks", auto_enabled=True, now=saturday) == "stocks"


def test_disabled_auto_schedule_uses_all():
    saturday = datetime(2026, 7, 25, 9, tzinfo=ET)
    assert resolve_market("auto", auto_enabled=False, now=saturday) == "all"
