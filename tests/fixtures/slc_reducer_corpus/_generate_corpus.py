"""
Generates the frozen SLC reducer-validation fixture corpus from the
existing isolated, cache-only SLC research cache. Never fetches - if the
requested window isn't already cached, this fails rather than reaching
out to Alpaca, so the corpus is reproducible from a specific cache state.

Re-run this only to deliberately expand or replace the corpus - each run
overwrites the fixture CSVs and the manifest, and both must be re-hashed
into live_slc/guardrails.py's GUARDRAILS_TIER2 baseline afterward.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from backtest.run_slc_backtest import _rth  # noqa: E402 - reused read-only, not modified
from backtest.slc_data import load_stock_bars  # noqa: E402
from utils.market_calendar import trading_days_between  # noqa: E402

FIXTURE_DIR = Path(__file__).parent
MANIFEST_PATH = REPO_ROOT / "research" / "slc_4h_5m_stock_v1_reducer_validation_corpus_manifest.json"

# Declared evaluation window: >20 trading sessions (level-expiration bound),
# spans a known NYSE early close (2024-11-29, day after Thanksgiving), and
# covers more than one visible price regime for each symbol.
EVAL_START = datetime(2024, 10, 1)
EVAL_END = datetime(2025, 1, 2)
# One extra real trailing 5-minute bar's worth of context beyond EVAL_END,
# solely so the frozen batch reference (which needs bar[i+1] to emit any
# confirmation on bar i) can evaluate the true last declared bar on equal
# footing with every other bar. Comparison itself stays scoped to
# confirmations within [EVAL_START, EVAL_END).
CONTEXT_END = datetime(2025, 1, 3)

SYMBOLS = ["AAPL", "AMD"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    frames, misses = load_stock_bars(SYMBOLS, EVAL_START, CONTEXT_END, minutes=5, mode="cache-only")
    if misses:
        raise RuntimeError(
            f"corpus window not fully cached (cache-only, no fetch attempted): {misses}"
        )
    days = trading_days_between(EVAL_START.date(), CONTEXT_END.date())
    fixtures: dict[str, dict] = {}
    for symbol in SYMBOLS:
        # RTH-filter exactly like run_slc_backtest.py's own _rth() does before
        # calling generate_signals() - the cache holds extended-hours bars
        # too, and those are never fed to the frozen reference in production,
        # so a fixture that included them would not be representative.
        frame = _rth(frames[symbol], days)
        if frame.empty:
            raise RuntimeError(f"no cached RTH bars for {symbol} in the requested window")
        path = FIXTURE_DIR / f"{symbol}_5min_2024-10-01_2025-01-03.csv"
        frame.to_csv(path, index=True, index_label="bar_time")
        fixtures[symbol] = {
            "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": _sha256_file(path),
            "bar_count": int(len(frame)),
            "first_bar": str(frame.index.min()),
            "last_bar": str(frame.index.max()),
        }

    manifest = {
        "strategy_version": "slc_4h_5m_stock_v1",
        "purpose": (
            "Frozen, extracted, immutable fixture corpus for reducer "
            "confirmation-output parity testing against the frozen batch "
            "generate_signals(). Written before live_slc/reducer.py was "
            "implemented, per rev. 6 Step 5."
        ),
        "source": "backtest/cache/slc_bars_cache.db (isolated SLC research cache, cache-only read, no fetch)",
        "declared_evaluation_window": {
            "start": EVAL_START.isoformat(),
            "end": EVAL_END.isoformat(),
        },
        "context_window_end": CONTEXT_END.isoformat(),
        "final_bar_handling": (
            "Each fixture includes one additional real 5-minute bar beyond "
            "the declared evaluation boundary (through context_window_end), "
            "solely so the frozen batch generate_signals() has the trailing "
            "bar it structurally requires. Comparison must stay scoped to "
            "confirmations whose entry/confirmation falls within "
            "declared_evaluation_window - the extra bar is data plumbing, "
            "never itself evaluated."
        ),
        "coverage_notes": (
            "Two symbols (AAPL, AMD), ~3 calendar months (~62 trading "
            "sessions - comfortably exceeds the 20-session level-expiration "
            "bound), includes the 2024-11-29 NYSE early close. This is a "
            "starting corpus, not a claim of exhaustive regime coverage; "
            "expand via a re-run of _generate_corpus.py with a wider window "
            "or more symbols if reducer parity testing needs more diversity "
            "than this provides."
        ),
        "fixtures": fixtures,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(fixtures)} fixture(s) and manifest at {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
