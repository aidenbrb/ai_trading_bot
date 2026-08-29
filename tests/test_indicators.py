"""Tests for utils/indicators.py"""
import numpy as np
import pandas as pd
import pytest

from utils.indicators import (
    adx, atr, avg_volume, bollinger_bands, compute_all, macd,
    momentum_label, relative_volume, rsi, sma, trend_label,
)


def _make_df(n=250, seed=42, drift=0.1):
    """Synthetic OHLCV - uptrending by default."""
    np.random.seed(seed)
    close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5 + drift))
    high  = close + abs(np.random.randn(n) * 0.3)
    low   = close - abs(np.random.randn(n) * 0.3)
    vol   = pd.Series(np.random.randint(1_000_000, 5_000_000, n).astype(float))
    return pd.DataFrame({"close": close, "high": high, "low": low, "open": close, "volume": vol})


class TestSMA:
    def test_length_preserved(self):
        s = pd.Series(list(range(100)), dtype=float)
        result = sma(s, 20)
        assert len(result) == 100

    def test_nan_before_period(self):
        s = pd.Series(list(range(50)), dtype=float)
        result = sma(s, 20)
        assert result.iloc[:19].isna().all()
        assert not result.iloc[19:].isna().any()

    def test_value(self):
        s = pd.Series([1.0] * 30)
        result = sma(s, 10)
        assert round(result.iloc[-1], 6) == 1.0


class TestRSI:
    def test_range(self):
        s = pd.Series(np.random.randn(100).cumsum() + 100)
        result = rsi(s, 14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_overbought_series(self):
        # Mostly up (mean +1.5/bar), some down days so RSI is non-NaN
        np.random.seed(1)
        s = pd.Series(100 + np.cumsum(np.random.randn(200) + 1.5))
        result = rsi(s, 14).dropna()
        assert len(result) > 0 and result.iloc[-1] > 70

    def test_oversold_series(self):
        # Mostly down (mean -1.5/bar), some up days so RSI is non-NaN
        np.random.seed(1)
        s = pd.Series(500 + np.cumsum(np.random.randn(200) - 1.5))
        result = rsi(s, 14).dropna()
        assert len(result) > 0 and result.iloc[-1] < 30


class TestMACD:
    def test_returns_three_series(self):
        s = pd.Series(np.random.randn(100).cumsum() + 100)
        line, signal, hist = macd(s)
        assert len(line) == len(signal) == len(hist) == 100

    def test_histogram_is_line_minus_signal(self):
        s = pd.Series(np.random.randn(100).cumsum() + 100)
        line, signal, hist = macd(s)
        diff = (line - signal - hist).dropna().abs()
        assert (diff < 1e-10).all()


class TestATR:
    def test_positive(self):
        df = _make_df()
        result = atr(df["high"], df["low"], df["close"], 14)
        assert result.dropna().gt(0).all()


class TestADX:
    def test_row_27_structural_minimum(self):
        """smoothed_tr/plus_dm/minus_dm are three parallel min_periods=14
        ewm calls, first valid at row 13 (0-indexed); dx is elementwise
        so it is also first valid at row 13; the fourth, sequentially
        dependent ewm on dx needs 14 non-null dx observations (rows
        13-26), so adx_14 is first valid at row 26 - the 27th row."""
        df = _make_df(40, seed=7, drift=0.2)
        result = adx(df["high"], df["low"], df["close"], 14)
        assert result.iloc[:26].isna().all()
        assert result.iloc[26:].notna().all()

    def test_tie_moves_never_register(self):
        """up_move == down_move (both positive) every bar: neither
        plus_dm nor minus_dm registers, so plus_di/minus_di are always 0,
        dx's zero-denominator (plus_di+minus_di) is replaced with NaN,
        and adx_14 - a min_periods=14 ewm of an all-NaN dx series - never
        becomes non-null, even with far more than 27 rows."""
        n = 60
        high = pd.Series([100.0 + i for i in range(n)])
        low = pd.Series([50.0 - i for i in range(n)])
        close = (high + low) / 2
        result = adx(high, low, close, 14)
        assert result.isna().all()

    def test_flat_market_is_clean_nan_not_inf_or_warning(self):
        """A perfectly flat market: tr, plus_dm, and minus_dm are all
        zero, so smoothed_tr is zero and plus_di/minus_di's
        zero-denominator division is replaced with NaN (matching
        relative_volume/bollinger_bands' existing convention) - never inf
        and never a runtime warning."""
        n = 40
        high = pd.Series([100.0] * n)
        low = pd.Series([100.0] * n)
        close = pd.Series([100.0] * n)
        with np.errstate(all="raise"):
            result = adx(high, low, close, 14)
        assert result.isna().all()
        assert not np.isinf(result.dropna()).any()

    def test_hand_computed_values_period_2(self):
        """Hand-derived via the exact preregistered recursion (alpha=0.5,
        adjust=False, min_periods=2) for a 4-bar series - independent of
        the implementation, not a copy of its output."""
        high = pd.Series([10.0, 12.0, 11.0, 13.0])
        low = pd.Series([5.0, 6.0, 4.0, 7.0])
        close = pd.Series([7.0, 8.0, 6.0, 10.0])
        result = adx(high, low, close, period=2)
        assert result.iloc[0] is not None and pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx(66.66666666666667)
        assert result.iloc[3] == pytest.approx(54.76190476190476)

class TestBollingerBands:
    def test_upper_above_lower(self):
        df = _make_df()
        upper, mid, lower, width = bollinger_bands(df["close"], 20)
        valid = upper.dropna()
        assert (valid > lower.dropna()).all()

    def test_width_positive(self):
        df = _make_df()
        _, _, _, width = bollinger_bands(df["close"], 20)
        assert width.dropna().gt(0).all()


class TestRelativeVolume:
    def test_normal(self):
        vol = pd.Series([1_000_000.0] * 30)
        avg = avg_volume(vol, 20)
        result = relative_volume(vol, avg)
        assert abs(result.dropna().iloc[-1] - 1.0) < 0.01

    def test_surge(self):
        vol = pd.Series([1_000_000.0] * 29 + [5_000_000.0])
        avg = avg_volume(vol, 20)
        result = relative_volume(vol, avg)
        assert result.iloc[-1] > 4.0


class TestComputeAll:
    def test_all_keys_present(self):
        df = _make_df()
        result = compute_all(df)
        expected = [
            "sma_20", "sma_50", "sma_100", "sma_200",
            "rsi_14", "macd_line", "macd_signal", "macd_hist",
            "atr_14", "bb_upper", "bb_middle", "bb_lower", "bb_width",
            "avg_volume_20", "rel_volume", "trend", "momentum",
        ]
        for k in expected:
            assert k in result, f"Missing key: {k}"

    def test_no_none_on_full_history(self):
        df = _make_df(250)
        result = compute_all(df)
        for k, v in result.items():
            if k not in ("trend", "momentum"):
                assert v is not None, f"{k} is None with 250 bars"

    def test_rel_volume_uses_prev_bar(self):
        """Verify rel_volume uses yesterday's bar, not today's partial intraday."""
        df = _make_df(250)
        df_spiked = df.copy()
        df_spiked.loc[df_spiked.index[-1], "volume"] = 1.0  # tiny last bar (simulates partial intraday)
        result = compute_all(df_spiked)
        # rel_volume should reflect yesterday's normal volume, not today's 1-share
        assert result["rel_volume"] > 0.3, "rel_volume is using last bar instead of prev bar"

    def test_uptrend_detection(self):
        df = _make_df(250, drift=0.3)  # strong uptrend
        result = compute_all(df)
        assert result["trend"] == "UPTREND"

    def test_downtrend_detection(self):
        df = _make_df(250, drift=-0.3)
        result = compute_all(df)
        assert result["trend"] == "DOWNTREND"

    def test_short_history_returns_nones(self):
        df = _make_df(50)  # not enough bars for SMA-200
        result = compute_all(df)
        assert result["sma_200"] is None


class TestLabels:
    def test_trend_uptrend(self):
        assert trend_label(110, 105, 100, 90) == "UPTREND"

    def test_trend_downtrend(self):
        assert trend_label(80, 85, 90, 100) == "DOWNTREND"

    def test_trend_sideways(self):
        assert trend_label(100, 105, 95, 90) == "SIDEWAYS"

    def test_trend_none_values(self):
        assert trend_label(100, None, 95, 90) == "SIDEWAYS"

    def test_momentum_overbought(self):
        assert momentum_label(75, 0.5) == "OVERBOUGHT"

    def test_momentum_strong(self):
        assert momentum_label(60, 0.5) == "STRONG"

    def test_momentum_weak_rsi(self):
        assert momentum_label(40, 0.5) == "WEAK"

    def test_momentum_weak_macd(self):
        assert momentum_label(55, -0.1) == "WEAK"
