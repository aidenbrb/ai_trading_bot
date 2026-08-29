# stock_trend_momentum_v2 cache-container recovery — 2026-08-12

This is an append-only provenance amendment. It changes no strategy rule,
market-data value, result, qualification gate, registry state, or execution
state.

After the completed v2 run, the first SLC research smoke test mistakenly used
the shared `backtest/cache/bars_cache.db` container. Immediately before that
fetch, the SLC smoke result recorded the still-valid frozen SHA-256
`2aa39029331c7dea6c620c8a23a43b70764fe40ad6761eda901da6775d83d841`.
The smoke appended exactly these records:

- fetched-range rowids 216318 through 216330 (13 rows);
- 32,867 bars in the previously nonexistent
  `research-stock-sip-5Minute` namespace; and
- 1,173 AAPL `research-stock-sip-1Minute` rows for the three closed intervals
  2025-06-02, 2025-06-05, and 2025-06-06, 13:30–20:00 UTC inclusive.

A read-only pre-delete audit established that no fetched-range row at or below
216317 covered any of those three AAPL minute windows, so the appended minute
rows did not replace data belonging to the frozen v2 dataset. All 34,040 bars
and all 13 range records were copied to the isolated
`backtest/cache/slc_bars_cache.db`, verified there, and then removed from the
whole-bot cache in one transaction. Post-removal checks found zero SLC
5-minute rows and a maximum fetched-range rowid of 216317.

SQLite does not restore a database file's prior byte layout after committed
inserts are deleted. Consequently, although the frozen dataset's logical rows
are restored, its container SHA-256 is now
`abfeee4cae5c4edda4b8c3323d33174d484d2fe7e22d1e97f43ab87776f40ef1`.
The preflight may accept this one exact recovery hash only when this amendment
also matches its embedded hash. Any other cache hash remains a hard failure.

The older completed v2 result is unchanged and retains its original pre/post
hash evidence. Future SLC acquisition uses only its isolated cache.
