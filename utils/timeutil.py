"""
Time helpers.

utcnow() returns a *naive* UTC datetime, identical in value to the now-deprecated
datetime.datetime.utcnow(), but without the DeprecationWarning. Keeping it naive
(tzinfo=None) is deliberate: timestamps are stored naive in the SQLite DB and are
subtracted from each other and from DB-loaded values throughout the pipeline, so
introducing tz-aware datetimes here would raise "can't subtract offset-naive and
offset-aware datetimes".
"""
from __future__ import annotations

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Current UTC time as a naive datetime (tzinfo stripped)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)
