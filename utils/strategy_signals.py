"""Versioned, side-effect-free swing signal rules shared by live and research.

The functions in this module deliberately know nothing about SQLModel, news,
brokers, clocks, or account state.  A historical simulator can therefore call
the exact same technical rules as the live strategy nodes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Protocol

from utils.pricing import rounded_long_bracket


STOCK_STRATEGY_VERSION = "stock_trend_momentum_v1"
CRYPTO_STRATEGY_VERSION = "crypto_trend_momentum_v1"
STOCK_STRATEGY_V2_VERSION = "stock_trend_momentum_v2"

STOCK_V2_ADX_MIN_BARS_OBSERVED = 40
STOCK_V2_ADX_MIN = 25.0
STOCK_V2_MIN_MOVE_BPS = 50.0

STOCK_ATR_STOP_MULT = 1.5
STOCK_ATR_TARGET_MULT = 3.0
STOCK_MIN_REL_VOLUME = 1.2
STOCK_RSI_LOW = 50.0
STOCK_RSI_HIGH = 68.0

CRYPTO_ATR_STOP_MULT = 2.5
CRYPTO_ATR_TARGET_MULT = 5.0
CRYPTO_MIN_REL_VOLUME = 1.3
CRYPTO_RSI_LOW = 50.0
CRYPTO_RSI_HIGH = 75.0


class IndicatorLike(Protocol):
    trend: str | None
    rsi_14: float | None
    macd_hist: float | None
    rel_volume: float | None
    atr_14: float | None
    sma_20: float | None
    sma_50: float | None


@dataclass(frozen=True)
class SignalDecision:
    passed: bool
    reason: str
    strategy_version: str
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    rr: float | None = None
    conviction_score: int | None = None
    thesis: str | None = None

    def legacy_tuple(self) -> tuple[bool, str, dict]:
        """Return the shape historically consumed by the live nodes."""
        if not self.passed:
            return False, self.reason, {}
        return True, self.reason, {
            "entry": self.entry,
            "stop": self.stop,
            "target": self.target,
            "rr": self.rr,
            "conviction_score": self.conviction_score,
            "thesis": self.thesis,
        }


def stock_trend_momentum_v1(
    symbol: str,
    indicator: IndicatorLike,
    close: float,
    *,
    min_price: float,
    min_rr: float,
) -> SignalDecision:
    """Frozen stock v1 technical entry rule."""
    if close < min_price:
        return _reject(
            STOCK_STRATEGY_VERSION,
            f"Price ${close:.2f} below minimum ${min_price:.2f}",
        )
    if indicator.trend != "UPTREND":
        return _reject(
            STOCK_STRATEGY_VERSION,
            f"trend={indicator.trend} (need UPTREND: price>SMA20>SMA50>SMA200)",
        )

    rsi = indicator.rsi_14
    if rsi is None or not (STOCK_RSI_LOW <= rsi <= STOCK_RSI_HIGH):
        rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"
        return _reject(
            STOCK_STRATEGY_VERSION,
            f"RSI {rsi_str} outside {STOCK_RSI_LOW}-{STOCK_RSI_HIGH} momentum zone",
        )

    macd_h = indicator.macd_hist
    if macd_h is None or macd_h <= 0:
        mh_str = f"{macd_h:.4f}" if macd_h is not None else "N/A"
        return _reject(
            STOCK_STRATEGY_VERSION,
            f"MACD histogram {mh_str} <= 0 (no bullish momentum)",
        )

    rel_vol = indicator.rel_volume
    if rel_vol is None or rel_vol < STOCK_MIN_REL_VOLUME:
        rv_str = f"{rel_vol:.2f}x" if rel_vol is not None else "N/A"
        return _reject(
            STOCK_STRATEGY_VERSION,
            f"Relative volume {rv_str} < {STOCK_MIN_REL_VOLUME}x (low conviction)",
        )

    atr = indicator.atr_14 or (close * 0.015)
    try:
        bracket = rounded_long_bracket(
            close=close,
            atr=atr,
            stop_atr=STOCK_ATR_STOP_MULT,
            target_atr=STOCK_ATR_TARGET_MULT,
            min_rr=min_rr,
            tick_size=0.01,
        )
    except (TypeError, ValueError) as exc:
        return _reject(
            STOCK_STRATEGY_VERSION,
            f"Invalid ATR bracket; signal rejected ({exc})",
        )
    score = 65
    if rsi >= 60:
        score += 8
    if rel_vol >= 1.5:
        score += 8
    if macd_h > close * 0.0003:
        score += 6
    if (
        indicator.sma_20
        and indicator.sma_50
        and (indicator.sma_20 - indicator.sma_50) / indicator.sma_50 > 0.01
    ):
        score += 5
    score = min(score, 92)

    thesis = (
        f"Rule-based trend-momentum setup on {symbol}: "
        f"price in UPTREND (SMA-20 > SMA-50 > SMA-200), "
        f"RSI={rsi:.1f} in momentum zone ({STOCK_RSI_LOW}-{STOCK_RSI_HIGH}), "
        f"MACD histogram positive ({macd_h:.4f}), "
        f"volume {rel_vol:.1f}x average. "
        f"Stop {STOCK_ATR_STOP_MULT}x ATR at ${bracket['stop']:.2f}, "
        f"target {STOCK_ATR_TARGET_MULT}x ATR at ${bracket['target']:.2f}, "
        f"R:R={bracket['rr']:.2f}."
    )
    return _pass(STOCK_STRATEGY_VERSION, bracket, score, thesis)


def stock_trend_momentum_v2(
    symbol: str,
    indicator: IndicatorLike,
    close: float,
    *,
    min_price: float,
    min_rr: float,
    enable_adx_filter: bool,
    enable_cost_filter: bool,
) -> SignalDecision:
    """stock_trend_momentum_v1 plus optional Delta2 (ADX trend-strength) and
    Delta3 (minimum expected-move-vs-cost) filters, per the frozen
    stock_trend_momentum_v2 preregistration. Delta1 (finite entry-order
    expiration) is engine-level and has no signal-function parameter here.
    """
    decision = stock_trend_momentum_v1(symbol, indicator, close, min_price=min_price, min_rr=min_rr)
    if not decision.passed:
        return replace(decision, strategy_version=STOCK_STRATEGY_V2_VERSION)

    if enable_adx_filter:
        bars_observed = getattr(indicator, "bars_observed", 0)
        adx_14 = getattr(indicator, "adx_14", None)
        if (
            bars_observed < STOCK_V2_ADX_MIN_BARS_OBSERVED
            or adx_14 is None
            or not math.isfinite(adx_14)
            or adx_14 < STOCK_V2_ADX_MIN
        ):
            adx_str = f"{adx_14:.1f}" if adx_14 is not None else "N/A"
            return _reject(
                STOCK_STRATEGY_V2_VERSION,
                f"ADX {adx_str} (bars_observed={bars_observed}) fails "
                f">= {STOCK_V2_ADX_MIN} with >= {STOCK_V2_ADX_MIN_BARS_OBSERVED} observed rows (Delta2)",
            )

    if enable_cost_filter:
        move_bps = (decision.target - decision.entry) / decision.entry * 10_000
        if move_bps < STOCK_V2_MIN_MOVE_BPS:
            return _reject(
                STOCK_STRATEGY_V2_VERSION,
                f"target move {move_bps:.1f}bps < {STOCK_V2_MIN_MOVE_BPS:.0f}bps (Delta3)",
            )

    return replace(decision, strategy_version=STOCK_STRATEGY_V2_VERSION)


def crypto_trend_momentum_v1(
    symbol: str,
    indicator: IndicatorLike,
    close: float,
    *,
    min_rr: float,
) -> SignalDecision:
    """Frozen crypto v1 technical entry rule; the BTC macro gate is external."""
    if indicator.trend != "UPTREND":
        return _reject(
            CRYPTO_STRATEGY_VERSION,
            f"trend={indicator.trend} (need UPTREND: price>SMA20>SMA50>SMA200)",
        )

    rsi = indicator.rsi_14
    if rsi is None or not (CRYPTO_RSI_LOW <= rsi <= CRYPTO_RSI_HIGH):
        rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"
        return _reject(
            CRYPTO_STRATEGY_VERSION,
            f"RSI {rsi_str} outside {CRYPTO_RSI_LOW}-{CRYPTO_RSI_HIGH} momentum zone",
        )

    macd_h = indicator.macd_hist
    if macd_h is None or macd_h <= 0:
        mh_str = f"{macd_h:.6f}" if macd_h is not None else "N/A"
        return _reject(
            CRYPTO_STRATEGY_VERSION,
            f"MACD histogram {mh_str} <= 0 (bearish momentum)",
        )

    rel_vol = indicator.rel_volume
    if rel_vol is None or rel_vol < CRYPTO_MIN_REL_VOLUME:
        rv_str = f"{rel_vol:.2f}x" if rel_vol is not None else "N/A"
        return _reject(
            CRYPTO_STRATEGY_VERSION,
            f"Relative volume {rv_str} < {CRYPTO_MIN_REL_VOLUME}x (no volume confirmation)",
        )

    atr = indicator.atr_14 or (close * 0.025)
    try:
        bracket = rounded_long_bracket(
            close=close,
            atr=atr,
            stop_atr=CRYPTO_ATR_STOP_MULT,
            target_atr=CRYPTO_ATR_TARGET_MULT,
            min_rr=min_rr,
            tick_size=0.00000001,
        )
    except (TypeError, ValueError) as exc:
        return _reject(
            CRYPTO_STRATEGY_VERSION,
            f"Invalid ATR bracket; signal rejected ({exc})",
        )
    score = 65
    if rsi >= 62:
        score += 8
    if rel_vol >= 1.6:
        score += 8
    if macd_h > close * 0.0005:
        score += 6
    if (
        indicator.sma_20
        and indicator.sma_50
        and (indicator.sma_20 - indicator.sma_50) / indicator.sma_50 > 0.02
    ):
        score += 5
    score = min(score, 92)

    thesis = (
        f"Crypto trend-momentum setup on {symbol}: price in UPTREND (SMA-20 > SMA-50), "
        f"RSI={rsi:.1f} in momentum zone ({CRYPTO_RSI_LOW}-{CRYPTO_RSI_HIGH}), "
        f"MACD histogram={macd_h:.6f} (positive), "
        f"volume {rel_vol:.1f}x average (confirmed breakout). "
        f"ATR-based stop {CRYPTO_ATR_STOP_MULT}x ATR at {bracket['stop']:.4f}, "
        f"target {CRYPTO_ATR_TARGET_MULT}x ATR at {bracket['target']:.4f}, "
        f"R:R={bracket['rr']:.2f}."
    )
    return _pass(CRYPTO_STRATEGY_VERSION, bracket, score, thesis)


def _reject(version: str, reason: str) -> SignalDecision:
    return SignalDecision(False, reason, version)


def _pass(version: str, bracket: dict[str, float], score: int, thesis: str) -> SignalDecision:
    return SignalDecision(
        True,
        "All gates passed",
        version,
        entry=bracket["entry"],
        stop=bracket["stop"],
        target=bracket["target"],
        rr=bracket["rr"],
        conviction_score=score,
        thesis=thesis,
    )
