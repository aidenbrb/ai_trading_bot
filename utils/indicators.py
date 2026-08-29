"""
Pure-function technical indicator calculations.

All functions accept a pandas Series or DataFrame of OHLCV data and return
a numeric scalar or Series. No DB or side-effects here.

Indicators computed:
  - SMA 20, 50, 100, 200
  - RSI 14
  - MACD (12, 26, 9)
  - ATR 14
  - Bollinger Bands 20 (upper, middle, lower, width)
  - Average Volume 20 / Relative Volume
  - Trend label  (UPTREND / DOWNTREND / SIDEWAYS)
  - Momentum label (STRONG / NEUTRAL / WEAK / OVERBOUGHT)
"""
from __future__ import annotations

import pandas as pd
import numpy as np


# -- SMA -----------------------------------------------------------------------

def sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(window=period, min_periods=period).mean()


# -- RSI -----------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta  = close.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_l  = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# -- MACD ----------------------------------------------------------------------

def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast   = close.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal     = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram  = macd_line - signal
    return macd_line, signal, histogram


# -- ATR -----------------------------------------------------------------------

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


# -- ADX -----------------------------------------------------------------------

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index, dtype=float,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index, dtype=float,
    )

    smoothed_tr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    smoothed_plus_dm = plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    plus_di = 100 * smoothed_plus_dm / smoothed_tr.replace(0, np.nan)
    minus_di = 100 * smoothed_minus_dm / smoothed_tr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


# -- Bollinger Bands -----------------------------------------------------------

def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    middle  = close.rolling(window=period, min_periods=period).mean()
    std     = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper   = middle + std_dev * std
    lower   = middle - std_dev * std
    width   = (upper - lower) / middle.replace(0, np.nan)
    return upper, middle, lower, width


# -- Volume --------------------------------------------------------------------

def avg_volume(volume: pd.Series, period: int = 20) -> pd.Series:
    return volume.rolling(window=period, min_periods=period).mean()


def relative_volume(volume: pd.Series, avg_vol: pd.Series) -> pd.Series:
    return volume / avg_vol.replace(0, np.nan)


# -- Derived Labels ------------------------------------------------------------

def trend_label(close: float, sma20: float, sma50: float, sma200: float) -> str:
    """
    UPTREND   - price > SMA20 > SMA50 > SMA200
    DOWNTREND - price < SMA20 < SMA50 < SMA200
    SIDEWAYS  - anything else
    """
    if any(v is None or (isinstance(v, float) and np.isnan(v))
           for v in [close, sma20, sma50, sma200]):
        return "SIDEWAYS"
    if close > sma20 > sma50 > sma200:
        return "UPTREND"
    if close < sma20 < sma50 < sma200:
        return "DOWNTREND"
    return "SIDEWAYS"


def momentum_label(rsi_val: float, macd_hist_val: float) -> str:
    """
    OVERBOUGHT - RSI > 70
    STRONG     - RSI 55-70  AND MACD histogram positive
    WEAK       - RSI < 45   OR MACD histogram negative
    NEUTRAL    - everything else
    """
    if rsi_val is None or np.isnan(rsi_val):
        return "NEUTRAL"
    if rsi_val > 70:
        return "OVERBOUGHT"
    if rsi_val > 55 and (macd_hist_val is not None and not np.isnan(macd_hist_val) and macd_hist_val > 0):
        return "STRONG"
    if rsi_val < 45 or (macd_hist_val is not None and not np.isnan(macd_hist_val) and macd_hist_val < 0):
        return "WEAK"
    return "NEUTRAL"


# -- Single-row snapshot (used by indicator_node) ------------------------------

def compute_all(df: pd.DataFrame) -> dict:
    """
    Compute all indicators for a OHLCV DataFrame and return the LAST row's values.

    Parameters
    ----------
    df : pd.DataFrame
        Columns: open, high, low, close, volume  (indexed by date, ascending)

    Returns
    -------
    dict matching Indicator model fields (None where not enough history)
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    def _last(s: pd.Series):
        v = s.iloc[-1] if not s.empty else None
        return None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 4)

    s20  = sma(close, 20)
    s50  = sma(close, 50)
    s100 = sma(close, 100)
    s200 = sma(close, 200)

    rsi_s = rsi(close, 14)

    macd_l, macd_sig, macd_h = macd(close)
    atr_s                    = atr(high, low, close, 14)
    bb_up, bb_mid, bb_lo, bb_w = bollinger_bands(close, 20)
    avg_vol                  = avg_volume(volume, 20)
    rel_vol                  = relative_volume(volume, avg_vol)

    # Use the second-to-last bar for relative volume so that a partial intraday
    # bar (when the pipeline re-runs during market hours) doesn't kill the gate.
    # For swing trades the entry decision is based on yesterday's completed session.
    def _prev(s: pd.Series):
        if len(s) < 2:
            return _last(s)
        v = s.iloc[-2]
        return None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 4)

    last_close   = _last(close)
    last_sma20   = _last(s20)
    last_sma50   = _last(s50)
    last_sma200  = _last(s200)
    last_rsi     = _last(rsi_s)
    last_macd_h  = _last(macd_h)

    return {
        "sma_20":        last_sma20,
        "sma_50":        last_sma50,
        "sma_100":       _last(s100),
        "sma_200":       last_sma200,
        "rsi_14":        last_rsi,
        "macd_line":     _last(macd_l),
        "macd_signal":   _last(macd_sig),
        "macd_hist":     last_macd_h,
        "atr_14":        _last(atr_s),
        "bb_upper":      _last(bb_up),
        "bb_middle":     _last(bb_mid),
        "bb_lower":      _last(bb_lo),
        "bb_width":      _last(bb_w),
        "avg_volume_20": _last(avg_vol),
        "rel_volume":    _prev(rel_vol),
        "trend":         trend_label(last_close, last_sma20, last_sma50, last_sma200),
        "momentum":      momentum_label(last_rsi, last_macd_h),
    }
