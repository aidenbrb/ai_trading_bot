"""
Post-session dry-run verdict for live_slc: python -m live_slc.check_session_gate --date YYYY-MM-DD

Runs the actual `authorization.evaluate_dry_run_session_gate()` against
the real SlcSessionStat row for the given date - never a hand-rolled
subset of its checks - so every real criterion applies (cycle-completion
floor, coverage, guardrail/engine-parity/closeout checks, the
dry_run_proposal_count cap, zero currently-unresolved ambiguous states).

`synthetic_fixtures_passed` is never a human-asserted CLI flag: it is
computed here by independently replaying the frozen reducer validation
corpus (research/slc_4h_5m_stock_v1_reducer_validation_corpus_manifest.json)
through both the frozen batch generator and the live reducer, requiring
BOTH long and short signals for EACH symbol - not just a total signal
count, which alone would not prove both directions actually occurred.
Read-only throughout: no broker call, and never itself mutates
SlcDeploymentStatus.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date

import pandas as pd

from live_slc import authorization, reducer, run_slc_live
from live_slc.models import SlcSessionStat, get_live_slc_session
from utils.slc_signals import generate_signals


def verify_synthetic_fixtures() -> tuple:
    """Returns (passed, evidence_lines). evidence_lines is printed
    verbatim so the verdict's justification is auditable, not merely
    asserted. Fails closed (False, with an explicit evidence line) on a
    missing/tampered/unparseable corpus or a structurally malformed
    manifest - never an unstructured traceback out of main()."""
    evidence: list = []
    try:
        manifest = run_slc_live._load_reducer_corpus_manifest()
        window_end = pd.Timestamp(manifest["declared_evaluation_window"]["end"])
        all_ok = True
        for symbol in run_slc_live.REQUIRED_REDUCER_CORPUS_SYMBOLS:  # ("AAPL", "AMD")
            fixture = manifest["fixtures"][symbol]
            bars = run_slc_live._load_and_verify_corpus_fixture(symbol, fixture)

            result = reducer.check_engine_parity(symbol, bars, window_end=window_end)

            batch_signals = generate_signals(symbol, bars)  # generate_signals normalizes via _frame() internally
            in_window = [s for s in batch_signals if s.confirmation_time < window_end]
            long_count = sum(1 for s in in_window if s.direction == "long")
            short_count = sum(1 for s in in_window if s.direction == "short")

            symbol_ok = (
                result.matched
                and long_count > 0
                and short_count > 0
                and (long_count + short_count) == result.signal_count
            )
            evidence.append(
                f"  {symbol}: sha256={fixture['sha256'][:16]}... matched={result.matched} "
                f"long={long_count} short={short_count} "
                f"(reducer signal_count={result.signal_count}) -> {'OK' if symbol_ok else 'FAIL'}"
            )
            all_ok = all_ok and symbol_ok
        return all_ok, evidence
    except (run_slc_live.EngineParityCorpusInvalid, KeyError, ValueError, TypeError) as exc:
        return False, [f"corpus invalid: {exc}"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Session date, YYYY-MM-DD")
    args = parser.parse_args()

    fixtures_passed, evidence = verify_synthetic_fixtures()
    print("Synthetic fixture check (frozen reducer validation corpus):")
    for line in evidence:
        print(line)
    print(f"  -> synthetic_fixtures_passed = {fixtures_passed}")
    print()

    with get_live_slc_session() as session:
        stat = session.get(SlcSessionStat, date.fromisoformat(args.date))
        if stat is None:
            print(f"No SlcSessionStat row for {args.date}")
            sys.exit(1)
        failures = authorization.evaluate_dry_run_session_gate(
            stat, synthetic_fixtures_passed=fixtures_passed, session=session,
        )

    if failures:
        print(f"FAIL ({len(failures)} issue(s)):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS - dry-run session gate satisfied.")


if __name__ == "__main__":
    main()
