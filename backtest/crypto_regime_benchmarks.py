"""
Phase 3 Step 3 (addendum item 1): control benchmarks that isolate the BTC
20-day-SMA regime filter alone, with NO other entry logic (no RSI/MACD/
relvol gates, no ATR stop/target) - to test whether crypto_trend_daily_v1's
edge is actually coming from its own gates, or just from the regime filter
every variant already shares.

Two benchmarks:
  - BTC-only: 100% notional in BTC when its close > its own 20-day SMA
    (daily bars, a genuine 20-day SMA - not the hourly-mislabeled-as-daily
    one v1 uses), 0% (cash) otherwise.
  - BTC/ETH inverse-vol basket: same regime gate: when on, split notional
    across BTC and ETH inversely to each symbol's own 30-day realized vol
    (computed once, at entry - not re-weighted daily within a holding
    period, matching how a real systematic regime-follower would behave);
    when off, flat.

Deliberately separate from backtest/whole_bot_engine.py: this is a
control/benchmark, not a strategy variant, and has no signal gates, no
per-symbol admission limits, and no stop/target concept at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, time as dtime

import pandas as pd

from backtest.whole_bot_engine import (
    ResearchCost,
    _realized_vol,
    _xsec_bar,
    daily_completed_bar_cutoff,
)


def _regime_on(daily_ind: dict, day: date, regime_symbol: str = "BTC-USD") -> bool | None:
    """True if `regime_symbol`'s close > its own 20-day SMA at the
    completed prior-day bar. None if undeterminable (missing bar/SMA)."""
    frame = daily_ind.get(regime_symbol)
    if frame is None or frame.empty:
        return None
    cutoff = daily_completed_bar_cutoff(day)
    if cutoff not in frame.index:
        return None
    row = frame.loc[cutoff]
    if pd.isna(row.get("close")) or pd.isna(row.get("sma_20")):
        return None
    return bool(row["close"] > row["sma_20"])


@dataclass
class _BenchPosition:
    symbol: str
    entry_date: date
    entry_price: float
    quantity: float
    entry_notional: float


def simulate_regime_benchmark(
    daily_ind: dict,
    universe: list[str],
    start: date,
    end: date,
    cost: ResearchCost,
    *,
    inverse_vol: bool = False,
    vol_lookback_days: int = 30,
    starting_equity: float = 100_000.0,
    regime_symbol: str = "BTC-USD",
) -> dict:
    """Long `universe` (100% notional, equal-split if inverse_vol=False and
    len(universe)>1; inverse-vol-weighted if True) whenever `regime_symbol`
    is above its own 20-day SMA, flat otherwise. Weights are set once at
    entry and held fixed until the regime flips off - not re-balanced
    daily within a holding period."""
    cash = starting_equity
    open_positions: dict[str, _BenchPosition] = {}
    was_on = False
    trades: list[dict] = []
    daily_equity: list[dict] = []

    day = start
    while day <= end:
        regime = _regime_on(daily_ind, day, regime_symbol)
        if regime is None:
            regime = was_on  # hold last known state on a rare missing regime-symbol bar

        if regime and not was_on:
            equity_now = cash  # flat going in - cash IS equity at this instant
            bars = {}
            for sym in universe:
                bar = _xsec_bar(daily_ind.get(sym), day) if daily_ind.get(sym) is not None else None
                if bar is not None:
                    bars[sym] = bar
            if inverse_vol:
                vols = {}
                for sym in bars:
                    v = _realized_vol(daily_ind[sym], day, vol_lookback_days)
                    if v is not None:
                        vols[sym] = v
                inv = {s: 1.0 / v for s, v in vols.items()}
                total = sum(inv.values())
                weights = {s: w / total for s, w in inv.items()} if total else {}
            else:
                weights = {s: 1.0 / len(bars) for s in bars} if bars else {}

            for sym, w in weights.items():
                price = float(bars[sym]["close"])
                notional = equity_now * w
                quantity = notional / price
                cash -= notional
                open_positions[sym] = _BenchPosition(sym, day, price, quantity, notional)

        elif not regime and was_on:
            for sym, pos in list(open_positions.items()):
                bar = _xsec_bar(daily_ind.get(sym), day) if daily_ind.get(sym) is not None else None
                exit_price = float(bar["close"]) if bar is not None else pos.entry_price
                gross = pos.quantity * (exit_price - pos.entry_price)
                tx_cost = (pos.quantity * pos.entry_price + pos.quantity * exit_price) * cost.crypto_bps_per_leg / 10_000.0
                net = gross - tx_cost
                cash += pos.entry_notional + net
                trades.append({
                    "symbol": sym, "status": "closed", "entry_date": pos.entry_date, "exit_date": day,
                    "exit_time": day, "fill_price": pos.entry_price, "entry_price": pos.entry_price,
                    "exit_price": exit_price, "quantity": pos.quantity, "gross_pnl": gross,
                    "transaction_cost": tx_cost, "net_pnl": net,
                    "pnl_r": net / abs(pos.entry_notional) if pos.entry_notional else None,
                })
            open_positions = {}

        was_on = regime

        unrealized = 0.0
        for sym, pos in open_positions.items():
            bar = _xsec_bar(daily_ind.get(sym), day) if daily_ind.get(sym) is not None else None
            if bar is not None:
                unrealized += pos.quantity * (float(bar["close"]) - pos.entry_price)
        equity = cash + sum(p.entry_notional for p in open_positions.values()) + unrealized
        daily_equity.append({"date": day, "equity": equity})

        day += timedelta(days=1)

    final_equity = cash
    for sym, pos in open_positions.items():
        bar = _xsec_bar(daily_ind.get(sym), end) if daily_ind.get(sym) is not None else None
        exit_price = float(bar["close"]) if bar is not None else pos.entry_price
        gross = pos.quantity * (exit_price - pos.entry_price)
        tx_cost = (pos.quantity * pos.entry_price + pos.quantity * exit_price) * cost.crypto_bps_per_leg / 10_000.0
        net = gross - tx_cost
        final_equity += pos.entry_notional + net
        trades.append({
            "symbol": sym, "status": "closed", "entry_date": pos.entry_date, "exit_date": end,
            "exit_time": end, "fill_price": pos.entry_price, "entry_price": pos.entry_price,
            "exit_price": exit_price, "quantity": pos.quantity, "gross_pnl": gross,
            "transaction_cost": tx_cost, "net_pnl": net,
            "pnl_r": net / abs(pos.entry_notional) if pos.entry_notional else None,
        })
    if daily_equity:
        daily_equity[-1]["equity"] = final_equity

    label = "btc_only_regime" if universe == [regime_symbol] and not inverse_vol else "inverse_vol_basket_regime"
    return {
        "mode": label, "portfolio": "regime_benchmark", "cost_model": cost.name,
        "trades": trades, "rejected": [], "missing_outcomes": [],
        "daily_equity": daily_equity, "final_equity": final_equity,
    }
