"""
Standalone 5-min Opening Range Breakout (ORB) backtester for the day-trading
mode. Deliberately fully decoupled from trading.db / db/connection.py - a
broken backtest run can never touch the live pipeline's data. Its own local
bar cache lives at backtest/cache/bars_cache.db.

Reuses the exact same signal logic (utils/orb_signals.py), cost model
(utils/cost_model.py), and bar-fetching rules (utils/alpaca_bars.py) as the
live day-trading nodes, so backtested and live rules can never drift.
"""
