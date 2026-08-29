from datetime import date
from decimal import Decimal
from fractions import Fraction

import pandas as pd
import pytest

from live_slc.split_detection import (
    MIN_OVERLAP_BARS,
    SplitEvidence,
    closest_simple_ratio,
    corporate_action_split_evidence,
    price_ratio_split_evidence,
    reconcile_evidence,
)


def _bars(closes: list[float], *, start="2026-08-10 13:30:00") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(closes), freq="5min")
    return pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes, "volume": 100.0}, index=idx)


def test_closest_simple_ratio_finds_common_split_ratios():
    assert closest_simple_ratio(2.0) == Fraction(2, 1)
    assert closest_simple_ratio(0.25) == Fraction(1, 4)
    assert closest_simple_ratio(10.0) == Fraction(10, 1)
    assert closest_simple_ratio(1.25) == Fraction(5, 4)
    assert closest_simple_ratio(0.1) == Fraction(1, 10)


def test_closest_simple_ratio_rejects_near_unity():
    assert closest_simple_ratio(1.0) is None
    assert closest_simple_ratio(1.003) is None
    assert closest_simple_ratio(0.997) is None


def test_closest_simple_ratio_rejects_non_finite_and_non_positive():
    assert closest_simple_ratio(float("nan")) is None
    assert closest_simple_ratio(float("inf")) is None
    assert closest_simple_ratio(0.0) is None
    assert closest_simple_ratio(-2.0) is None


def test_price_ratio_split_evidence_detects_consistent_4_for_1_rescale():
    old_closes = [100.0, 101.0, 99.5, 102.0, 100.5, 103.0]
    cached = _bars(old_closes)
    fresh = _bars([c / 4.0 for c in old_closes])
    evidence = price_ratio_split_evidence("AAPL", cached, fresh)
    assert evidence is not None
    assert evidence.source == "price_ratio"
    assert evidence.scale_factor == Decimal(1) / Decimal(4)


def test_price_ratio_split_evidence_none_when_ratio_is_one():
    closes = [100.0, 101.0, 99.5, 102.0, 100.5, 103.0]
    cached = _bars(closes)
    fresh = _bars(closes)
    assert price_ratio_split_evidence("AAPL", cached, fresh) is None


def test_price_ratio_split_evidence_none_when_bars_disagree_on_ratio():
    """A real split rescales every overlapping bar by the SAME factor -
    inconsistent per-bar ratios must never be flagged as a split."""
    cached = _bars([100.0, 100.0, 100.0, 100.0, 100.0])
    fresh = _bars([50.0, 60.0, 40.0, 55.0, 45.0])  # wildly inconsistent ratios
    assert price_ratio_split_evidence("AAPL", cached, fresh) is None


def test_price_ratio_split_evidence_none_below_minimum_overlap():
    closes = [100.0, 50.0]  # only 2 bars, below MIN_OVERLAP_BARS
    assert len(closes) < MIN_OVERLAP_BARS
    cached = _bars([100.0, 100.0])
    fresh = _bars([50.0, 50.0])
    assert price_ratio_split_evidence("AAPL", cached, fresh) is None


def test_price_ratio_split_evidence_none_on_empty_or_no_overlap():
    cached = _bars([100.0] * 6)
    fresh = _bars([25.0] * 6, start="2026-08-11 13:30:00")  # disjoint timestamps
    assert price_ratio_split_evidence("AAPL", cached, fresh) is None
    assert price_ratio_split_evidence("AAPL", pd.DataFrame(), cached) is None


class _FakeSplit:
    def __init__(self, symbol, old_rate, new_rate):
        self.symbol = symbol
        self.old_rate = old_rate
        self.new_rate = new_rate


class _FakeResult:
    def __init__(self, data):
        self.data = data


class _FakeCorporateActionsClient:
    def __init__(self, data, *, raise_error=False):
        self._data = data
        self._raise_error = raise_error
        self.last_request = None

    def get_corporate_actions(self, request):
        self.last_request = request
        if self._raise_error:
            raise RuntimeError("simulated network failure")
        return _FakeResult(self._data)


def test_corporate_action_split_evidence_parses_forward_and_reverse_splits():
    client = _FakeCorporateActionsClient({
        "forward_splits": [_FakeSplit("AAPL", 1, 4)],
        "reverse_splits": [_FakeSplit("XYZ", 10, 1)],
    })
    result = corporate_action_split_evidence(client, ["AAPL", "XYZ", "MSFT"], today=date(2026, 8, 13))
    assert result["AAPL"].scale_factor == Decimal(1) / Decimal(4)
    assert result["AAPL"].source == "corporate_actions"
    assert result["XYZ"].scale_factor == Decimal(10) / Decimal(1)
    assert "MSFT" not in result


def test_corporate_action_split_evidence_never_raises_on_request_failure():
    client = _FakeCorporateActionsClient({}, raise_error=True)
    assert corporate_action_split_evidence(client, ["AAPL"]) == {}


def test_corporate_action_split_evidence_empty_symbols_never_calls_client():
    client = _FakeCorporateActionsClient({"forward_splits": [_FakeSplit("AAPL", 1, 4)]})
    assert corporate_action_split_evidence(client, []) == {}
    assert client.last_request is None


def test_reconcile_evidence_prefers_corporate_actions_when_sources_agree():
    corporate = SplitEvidence("AAPL", Decimal("0.25"), "corporate_actions", "d")
    price_ratio = SplitEvidence("AAPL", Decimal("0.2501"), "price_ratio", "d")
    evidence, conflicting = reconcile_evidence(corporate, price_ratio)
    assert evidence is corporate
    assert conflicting is False


def test_reconcile_evidence_flags_conflict_and_returns_no_evidence():
    corporate = SplitEvidence("AAPL", Decimal("0.25"), "corporate_actions", "d")
    price_ratio = SplitEvidence("AAPL", Decimal("0.5"), "price_ratio", "d")
    evidence, conflicting = reconcile_evidence(corporate, price_ratio)
    assert evidence is None
    assert conflicting is True


def test_reconcile_evidence_uses_whichever_single_source_is_available():
    price_ratio = SplitEvidence("AAPL", Decimal("0.25"), "price_ratio", "d")
    evidence, conflicting = reconcile_evidence(None, price_ratio)
    assert evidence is price_ratio
    assert conflicting is False


def test_reconcile_evidence_none_when_neither_source_has_evidence():
    assert reconcile_evidence(None, None) == (None, False)
