"""
Reducer confirmation-output parity and serialize/reload determinism, using
the frozen fixture corpus (research/slc_4h_5m_stock_v1_reducer_validation_corpus_manifest.json).
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from live_slc.reducer import (
    Confirmation,
    ReducerState,
    _parity_sort_key,
    confirmation_matches_signal,
    process_new_bar,
)
from utils.slc_signals import _frame, generate_signals

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "research" / "slc_4h_5m_stock_v1_reducer_validation_corpus_manifest.json"


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _load_fixture(symbol: str) -> pd.DataFrame:
    manifest = _load_manifest()
    fixture = manifest["fixtures"][symbol]
    path = REPO_ROOT / fixture["path"]
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return _frame(df)


def test_corpus_manifest_hashes_match_fixture_files():
    manifest = _load_manifest()
    import hashlib
    for symbol, fixture in manifest["fixtures"].items():
        path = REPO_ROOT / fixture["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == fixture["sha256"], f"{symbol} fixture file hash drifted from the frozen manifest"


def _reducer_key(c: Confirmation) -> tuple:
    return (str(c.confirmation_time), c.level_id, c.direction, c.structure,
             round(c.stochastic_k, 6), round(c.stop, 6))


@pytest.mark.parametrize("symbol", ["AAPL", "AMD"])
def test_reducer_matches_frozen_batch_reference_exactly(symbol):
    """Exact-equality parity (rev. 11 Step 13) across every field the two
    engines have in common - not the prior round(x, 6) comparison limited
    to 6 of 16 fields. See reducer.confirmation_matches_signal()'s
    docstring for exactly which 2 fields (stochastic_k/d) carry a tight
    tolerance, and why: empirically-verified float non-associativity in a
    rolling-window computation, not a logic divergence."""
    manifest = _load_manifest()
    eval_end = pd.Timestamp(manifest["declared_evaluation_window"]["end"])
    frame = _load_fixture(symbol)

    batch_signals = generate_signals(symbol, frame)
    batch_in_window = [s for s in batch_signals if s.confirmation_time < eval_end]

    state = ReducerState(symbol=symbol)
    reducer_confirmations = []
    for ts, row in frame.iterrows():
        state, confs = process_new_bar(state, row, ts)
        reducer_confirmations.extend(c for c in confs if c.confirmation_time < eval_end)

    batch_sorted = sorted(batch_in_window, key=_parity_sort_key)
    reducer_sorted = sorted(reducer_confirmations, key=_parity_sort_key)

    assert len(batch_sorted) > 0, "corpus produced zero batch signals - not a useful parity test"
    assert len(reducer_sorted) == len(batch_sorted)
    for reducer_confirmation, batch_signal in zip(reducer_sorted, batch_sorted):
        assert confirmation_matches_signal(reducer_confirmation, batch_signal), (
            f"{symbol} parity mismatch: reducer={reducer_confirmation} batch={batch_signal}"
        )


@pytest.mark.parametrize("symbol", ["AAPL"])
def test_reducer_serialize_reload_determinism(symbol):
    """The actual per-cycle operating model: every cycle is a fresh
    process reloading persisted state from live_slc.db. Splitting a run at
    an arbitrary point and reloading through JSON must produce identical
    subsequent confirmations to an uninterrupted run."""
    frame = _load_fixture(symbol)
    split = len(frame) // 2

    state_a = ReducerState(symbol=symbol)
    confs_a = []
    for ts, row in frame.iterrows():
        state_a, c = process_new_bar(state_a, row, ts)
        confs_a.extend(c)

    state_b = ReducerState(symbol=symbol)
    confs_b = []
    for ts, row in frame.iloc[:split].iterrows():
        state_b, c = process_new_bar(state_b, row, ts)
        confs_b.extend(c)
    reloaded = ReducerState.from_json(symbol, json.loads(json.dumps(state_b.to_json())))
    for ts, row in frame.iloc[split:].iterrows():
        reloaded, c = process_new_bar(reloaded, row, ts)
        confs_b.extend(c)

    assert sorted(_reducer_key(c) for c in confs_a) == sorted(_reducer_key(c) for c in confs_b)


def test_at_most_one_confirmation_per_symbol_per_session():
    """Cross-level dedup - the bug found and fixed during implementation."""
    frame = _load_fixture("AAPL")
    state = ReducerState(symbol="AAPL")
    confirmations = []
    for ts, row in frame.iterrows():
        state, confs = process_new_bar(state, row, ts)
        confirmations.extend(confs)
    days = [c.confirmation_time.date() for c in confirmations]
    assert len(days) == len(set(days)), "more than one confirmation emitted for the same session"
