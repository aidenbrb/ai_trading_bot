"""
Split detection and ratio validation for live_slc's preflight stage
(rev. 11 Step 9).

Two independent, corroborating checks - both pure read/compute, neither
touches the DB nor calls any broker-mutating endpoint:

  1. Alpaca's corporate-actions feed (authoritative when reachable) -
     corporate_action_split_evidence().
  2. An overlap-window price-ratio heuristic (still works if the
     corporate-actions feed is unavailable or hasn't posted the action
     yet) - price_ratio_split_evidence(), generalized to ANY simple-
     fraction ratio via Fraction.limit_denominator() rather than a fixed
     {2, 3, 1.5, 4} allowlist, so an uncommon or reverse-split ratio
     (5:4, 1:10, ...) is detected the same way as a common one.

Both report `scale_factor` in the same convention: multiply an OLD-basis
(possibly stale) price by `scale_factor` to get the NEW (current) basis -
so the caller (run_slc_live.run_preflight()) can directly compare the two
sources and require agreement before ever touching cached state. This
module only ever answers "is there evidence of a split, and what ratio" -
it never decides what to do about it; the atomic rebuild-and-swap lives in
run_slc_live.py, next to the other SlcReducerState/SlcFiveMinBar writers.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from fractions import Fraction
from math import isfinite
from typing import Optional

import pandas as pd

PRICE_RATIO_TOLERANCE = 0.01  # 1% - tight enough that ordinary overnight
                               # gap volatility can never be mistaken for
                               # a split (a real split rescales EVERY
                               # overlapping bar by the identical factor;
                               # ordinary price action does not).
MAX_SIMPLE_DENOMINATOR = 20    # covers common ratios (2:1, 3:1, 1.5:1,
                                # 4:1) and uncommon/reverse ones (5:4,
                                # 1:10, 7:2) alike - no fixed allowlist.
NEAR_UNITY_TOLERANCE = 0.01    # a ratio this close to 1.0 is "no split",
                                # never flagged as a (degenerate) 1:1 one.
MIN_OVERLAP_BARS = 5           # below this, there isn't enough evidence
                                # to distinguish a split from noise.


@dataclass(frozen=True)
class SplitEvidence:
    symbol: str
    scale_factor: Decimal   # multiply an OLD-basis price by this for NEW-basis
    source: str              # "corporate_actions" | "price_ratio"
    detail: str               # human-readable - goes into the SlcAuditEvent payload


def closest_simple_ratio(
    observed_ratio: float, *, max_denominator: int = MAX_SIMPLE_DENOMINATOR,
    tolerance: float = PRICE_RATIO_TOLERANCE,
) -> Optional[Fraction]:
    """The best small-integer p/q approximation of `observed_ratio` within
    `tolerance` (relative), or None if no simple fraction fits that
    closely, or if `observed_ratio` is indistinguishable from 1.0 (not a
    split at all)."""
    if not isfinite(observed_ratio) or observed_ratio <= 0:
        return None
    if abs(observed_ratio - 1.0) <= NEAR_UNITY_TOLERANCE:
        return None
    approx = Fraction(observed_ratio).limit_denominator(max_denominator)
    if approx.numerator <= 0 or approx.denominator <= 0:
        return None
    approx_value = approx.numerator / approx.denominator
    if abs(approx_value - observed_ratio) / observed_ratio <= tolerance:
        return approx
    return None


def price_ratio_split_evidence(
    symbol: str, cached: pd.DataFrame, fresh: pd.DataFrame,
) -> Optional[SplitEvidence]:
    """Compare `cached` (already stored locally, possibly stale-basis)
    bars against `fresh` (just re-fetched for the SAME historical dates,
    current-basis) bars at their common timestamps. A genuine split
    rescales every overlapping bar by the identical factor - if the
    per-bar ratios disagree with each other beyond PRICE_RATIO_TOLERANCE,
    this is ordinary historical variation between two fetches (or a data
    hiccup), never a split, and must not be flagged."""
    if cached is None or fresh is None or cached.empty or fresh.empty:
        return None
    common_index = cached.index.intersection(fresh.index)
    if len(common_index) < MIN_OVERLAP_BARS:
        return None
    ratios = []
    for ts in common_index:
        old_close = float(cached.loc[ts, "close"])
        new_close = float(fresh.loc[ts, "close"])
        if old_close <= 0 or not isfinite(old_close) or not isfinite(new_close):
            continue
        ratios.append(new_close / old_close)
    if len(ratios) < MIN_OVERLAP_BARS:
        return None
    ratio_series = pd.Series(ratios)
    median_ratio = float(ratio_series.median())
    if median_ratio <= 0 or not isfinite(median_ratio):
        return None
    max_deviation = float((ratio_series - median_ratio).abs().max())
    if max_deviation / median_ratio > PRICE_RATIO_TOLERANCE:
        return None  # bars disagree on the ratio - not a uniform rescale
    fraction = closest_simple_ratio(median_ratio)
    if fraction is None:
        return None
    scale_factor = Decimal(fraction.numerator) / Decimal(fraction.denominator)
    return SplitEvidence(
        symbol=symbol, scale_factor=scale_factor, source="price_ratio",
        detail=(
            f"observed close ratio {median_ratio:.6f} ~= "
            f"{fraction.numerator}/{fraction.denominator} across "
            f"{len(ratios)} overlapping bars"
        ),
    )


def corporate_action_split_evidence(
    client, symbols: list[str], *, lookback_days: int = 5, today: Optional[date] = None,
) -> dict[str, SplitEvidence]:
    """One batched query across the whole universe for FORWARD_SPLIT and
    REVERSE_SPLIT actions in the last `lookback_days` days. `client` is an
    already-constructed alpaca.data.historical.corporate_actions.CorporateActionsClient
    (or a fake with the same .get_corporate_actions(request) -> object-
    with-.data interface) - injected, never constructed here, so this stays
    testable without a live connection (matches execution.py's client-
    injection convention). Returns {} (never raises) on any request
    failure - the caller falls back to the price-ratio check alone; a
    corporate-actions outage must never block preflight."""
    from alpaca.data.enums import CorporateActionsType
    from alpaca.data.requests import CorporateActionsRequest

    if not symbols:
        return {}
    end = today or date.today()
    start = end - timedelta(days=lookback_days)
    request = CorporateActionsRequest(
        symbols=list(symbols), start=start, end=end,
        types=[CorporateActionsType.FORWARD_SPLIT, CorporateActionsType.REVERSE_SPLIT],
    )
    try:
        result = client.get_corporate_actions(request)
    except Exception:
        return {}
    data = getattr(result, "data", None)
    if data is None and isinstance(result, dict):
        data = result.get("data", {})
    if not data:
        return {}

    evidence: dict[str, SplitEvidence] = {}
    for key in ("forward_splits", "reverse_splits"):
        for action in data.get(key, []):
            symbol = getattr(action, "symbol", None)
            old_rate = getattr(action, "old_rate", None)
            new_rate = getattr(action, "new_rate", None)
            if not symbol or not old_rate or not new_rate:
                continue
            # Alpaca convention: `old_rate` shares become `new_rate`
            # shares (e.g. a 4-for-1 forward split is old_rate=1,
            # new_rate=4) - price scales by old_rate/new_rate. Cross-
            # checked against price_ratio_split_evidence() by the caller
            # before ever being acted on, which also catches this
            # direction being wrong should Alpaca's field semantics ever
            # differ from this documented assumption.
            scale_factor = Decimal(str(old_rate)) / Decimal(str(new_rate))
            evidence[symbol] = SplitEvidence(
                symbol=symbol, scale_factor=scale_factor, source="corporate_actions",
                detail=f"{key} old_rate={old_rate} new_rate={new_rate}",
            )
    return evidence


def reconcile_evidence(
    corporate: Optional[SplitEvidence], price_ratio: Optional[SplitEvidence], *,
    agreement_tolerance: float = PRICE_RATIO_TOLERANCE,
) -> tuple[Optional[SplitEvidence], bool]:
    """Combine the two sources for one symbol. Returns (evidence_to_act_on,
    conflicting). Never returns evidence when the two sources disagree -
    that is a genuinely ambiguous state (fail closed: flagged for manual
    review by the caller, no automatic rebuild) rather than a guess at
    which source is right."""
    if corporate is None and price_ratio is None:
        return None, False
    if corporate is not None and price_ratio is not None:
        a, b = float(corporate.scale_factor), float(price_ratio.scale_factor)
        if abs(a - b) / max(a, b) > agreement_tolerance:
            return None, True
        return corporate, False  # both agree - corporate_actions is authoritative
    return corporate or price_ratio, False
