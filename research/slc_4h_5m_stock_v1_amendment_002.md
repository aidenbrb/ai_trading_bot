# SLC v1 preregistration amendment 002 — isolated cache

This amendment changes no signal, entry, exit, portfolio, cost, or
qualification rule.

SLC data acquisition and cache-only reads must use the dedicated
`backtest/cache/slc_bars_cache.db`, never the older whole-bot
`backtest/cache/bars_cache.db`. The first AAPL/SPY smoke dataset was migrated
to that isolated cache. All future SLC result manifests record the isolated
cache's pre/post hashes.

This file is append-immutable after its SHA-256 is recorded.
