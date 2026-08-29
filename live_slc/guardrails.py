"""
Two-tier guardrail hashing, environment-drift checking, and the three-way
precondition gate (rev. 6 Step 2).

Duplicated from backtest/run_slc_backtest.py's hash pattern, not imported
from it - that module and its own baseline JSON stay fully untouched.

Tier 1 (bot-wide, must never drift): the same 9 files the SLC backtest
runner already guards.
Tier 2 (SLC's own scope, re-freezable via a new baseline + a new
SlcActivationEvent, never silent): the frozen signal-fidelity files this
package depends on, plus every live_slc/ execution file, the 3 scheduler
.bat scripts, and the shared hidden-window .vbs launcher those Scheduled
Tasks invoke (not a 4th distinct scheduler script - one shared launcher
for all three; note its own docstring caveat: hashing it protects its
bytes, not which .bat argument a given Scheduled Task passes it - that
argument lives in Windows Task Scheduler's own config, verified
separately, see research/slc_live_scheduled_task_config_20260817.json).

Gate scoping (rev. 6, corrected from rev. 5):
- assert_operational_preconditions(): preflight and every cycle's start,
  regardless of deployment status. Required for ANYTHING to run.
- assert_submission_preconditions(): brand-new entry submissions ONLY.
- assert_closeout_preconditions(): every risk-reducing action on an
  already-open position (target replacement, emergency flatten,
  reconciliation, and the closeout stage itself) - always reachable,
  independent of the other two gates.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

from live_slc import authorization, process_lock, rule_freeze
from live_slc.settings import live_slc_settings

REPO_ROOT = Path(__file__).resolve().parents[1]
TRADINGBOT_ROOT = REPO_ROOT.parent
LIVE_SLC_ROOT = Path(__file__).parent

DEPLOYMENT_BASELINE = REPO_ROOT / "research" / "slc_4h_5m_stock_v1_live_deployment_baseline_20260829.json"
EXPECTED_DEPLOYMENT_BASELINE_SHA256 = "78fb793721eb948801509fb18ead715501f63eb1746e16d0ea9f2ade941fd71b"

GUARDRAILS_TIER1 = {
    "../run_bot.bat": TRADINGBOT_ROOT / "run_bot.bat",
    "run_pipeline.py": REPO_ROOT / "run_pipeline.py",
    "utils/strategy_registry.py": REPO_ROOT / "utils" / "strategy_registry.py",
    "nodes/execution_node.py": REPO_ROOT / "nodes" / "execution_node.py",
    **{
        f"scripts/{path.name}": path
        for path in sorted((REPO_ROOT / "scripts").glob("*.bat"))
    },
}

# guardrails.py deliberately excludes itself, matching the precedent
# already set by backtest/run_slc_backtest.py's own GUARDRAILS dict (which
# also never hashes itself): the baseline's own SHA-256
# (EXPECTED_DEPLOYMENT_BASELINE_SHA256, hardcoded in this file) and the
# baseline's tier2 record of this file's hash would otherwise depend on
# each other circularly - this file's byte content can never be a stable
# input to its own recorded hash. A self-referential hash check is an
# inherent limitation of this pattern in general (the verifier can't fully
# verify itself); rule_freeze.py's independent hashing of the frozen
# strategy documents, and assert_closeout_preconditions() hashing
# execution.py/closeout.py (which do NOT have this circularity), are the
# mitigating layers.
_LIVE_SLC_MODULE_NAMES = [
    "__init__.py", "settings.py", "models.py", "rule_freeze.py",
    "process_lock.py", "bar_cache.py", "reducer.py",
    "ranking.py", "risk.py", "execution.py", "closeout.py",
    "authorization.py", "run_slc_live.py", "migrations.py",
    "split_detection.py", "check_schedule_health.py",
]

GUARDRAILS_TIER2 = {
    "utils/slc_signals.py": REPO_ROOT / "utils" / "slc_signals.py",
    "utils/market_calendar.py": REPO_ROOT / "utils" / "market_calendar.py",
    "config/universe.py": REPO_ROOT / "config" / "universe.py",
    "live_slc/requirements.lock": LIVE_SLC_ROOT / "requirements.lock",
    **{
        f"live_slc/{name}": LIVE_SLC_ROOT / name
        for name in _LIVE_SLC_MODULE_NAMES
    },
    **{
        f"scripts/slc_live/{name}": REPO_ROOT / "scripts" / "slc_live" / name
        for name in (
            "run_slc_preflight.bat", "run_slc_cycle.bat", "run_slc_closeout.bat",
            "run_slc_schedule_health.bat", "run_hidden.vbs",
        )
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tier_hashes(tier: dict[str, Path]) -> dict[str, str]:
    missing = [name for name, path in tier.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"guardrail file missing: {missing}")
    return {name: _sha256(path) for name, path in tier.items()}


def guardrail_hashes() -> dict[str, dict[str, str]]:
    return {"tier1": _tier_hashes(GUARDRAILS_TIER1), "tier2": _tier_hashes(GUARDRAILS_TIER2)}


def verify_deployment_baseline() -> dict:
    """Hard-stop on any drift in either tier since the frozen live baseline."""
    if EXPECTED_DEPLOYMENT_BASELINE_SHA256 is None:
        raise RuntimeError(
            "EXPECTED_DEPLOYMENT_BASELINE_SHA256 is not set - the live "
            "deployment baseline has not been frozen yet."
        )
    baseline_hash = _sha256(DEPLOYMENT_BASELINE)
    if baseline_hash != EXPECTED_DEPLOYMENT_BASELINE_SHA256:
        raise RuntimeError(
            "SLC live deployment baseline file hash mismatch: "
            f"expected={EXPECTED_DEPLOYMENT_BASELINE_SHA256} actual={baseline_hash}"
        )
    baseline = json.loads(DEPLOYMENT_BASELINE.read_text(encoding="utf-8"))
    expected = baseline.get("guardrails")
    current = guardrail_hashes()
    if expected != current:
        changed = []
        for tier in ("tier1", "tier2"):
            exp_tier = (expected or {}).get(tier, {})
            cur_tier = current.get(tier, {})
            changed.extend(
                f"{tier}:{name}"
                for name in sorted(set(exp_tier) | set(cur_tier))
                if exp_tier.get(name) != cur_tier.get(name)
            )
        raise RuntimeError(f"live_slc guardrail drift since frozen baseline: {changed}")
    return {
        "baseline_path": str(DEPLOYMENT_BASELINE.relative_to(REPO_ROOT)).replace("\\", "/"),
        "baseline_sha256": baseline_hash,
        "guardrails": current,
        "environment": baseline.get("environment", {}),
    }


def check_running_environment(expected_environment: dict) -> list[str]:
    """Compare the ACTUAL running interpreter/packages against the baseline's
    recorded environment - hashing requirements.lock alone can't detect an
    in-place `pip install` upgrade that never touched the lock file."""
    mismatches: list[str] = []
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    expected_python = expected_environment.get("python")
    if expected_python and actual_python != expected_python:
        mismatches.append(f"python {actual_python} != expected {expected_python}")
    for package, expected_version in expected_environment.get("packages", {}).items():
        try:
            actual_version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            mismatches.append(f"{package} not installed (expected {expected_version})")
            continue
        if actual_version != expected_version:
            mismatches.append(f"{package} {actual_version} != expected {expected_version}")
    return mismatches


# -- Three-way precondition gate -------------------------------------------

def assert_operational_preconditions(*, observed_account_id: str) -> dict:
    """Required for ANYTHING to run: preflight and every cycle (never
    closeout - see assert_closeout_preconditions(), which must stay
    reachable independent of this gate).

    Does NOT require paper_active status or either execution switch - a
    dry_run (or even not_authorized, for preflight/bar-ingestion purposes)
    must be able to reach this point. Returns the current deployment
    status plus the verified baseline for the caller to branch on.

    `observed_account_id` is required, not optional (rev. 11 Step 9): a
    dry_run session never reaches assert_submission_preconditions() (that
    gate is paper_active-only) and preflight never reaches
    assert_closeout_preconditions() (nothing to close yet) - without a
    check here, an entire one-session engineering dry-run could run
    against a misconfigured (wrong) Alpaca account, undetected, since
    neither of the other two gates would ever fire to catch it. The
    caller fetches this from its own broker client (a read-only
    client.get_account() call) before calling here - this module has no
    SDK dependency of its own, matching assert_submission_preconditions()/
    assert_closeout_preconditions()'s existing plain-string convention.
    """
    rule_freeze.verify_rule_freeze()
    baseline = verify_deployment_baseline()
    if not live_slc_settings.SLC_EXPECTED_ACCOUNT_ID:
        raise RuntimeError("SLC_EXPECTED_ACCOUNT_ID is not set")
    if observed_account_id != live_slc_settings.SLC_EXPECTED_ACCOUNT_ID:
        raise RuntimeError(
            f"operational preconditions failed: observed account {observed_account_id!r} != "
            f"expected {live_slc_settings.SLC_EXPECTED_ACCOUNT_ID!r}"
        )
    status_record = authorization.get_current_deployment_record()
    return {
        "status": status_record.status,
        "baseline": baseline,
        "status_record": status_record,
    }


def assert_submission_preconditions(operational: dict, *, observed_account_id: str,
                                     daily_loss_breached: bool) -> None:
    """Gates ONLY a brand-new bracket entry submission. Never call this for
    any action on a position that's already open - use
    assert_closeout_preconditions() for those instead."""
    status_record = operational["status_record"]
    if status_record.status != "paper_active":
        raise RuntimeError(f"new entries blocked: deployment status is {status_record.status!r}, not paper_active")
    failures = live_slc_settings.require_new_entry_env()
    if failures:
        raise RuntimeError(f"new entries blocked: {failures}")
    if observed_account_id != live_slc_settings.SLC_EXPECTED_ACCOUNT_ID:
        raise RuntimeError(
            f"new entries blocked: observed account {observed_account_id!r} != "
            f"expected {live_slc_settings.SLC_EXPECTED_ACCOUNT_ID!r}"
        )
    if status_record.observed_account_id and observed_account_id != status_record.observed_account_id:
        raise RuntimeError(
            "new entries blocked: observed account does not match the account "
            "pinned at the paper_active activation event"
        )
    if daily_loss_breached:
        raise RuntimeError("new entries blocked: account daily-loss halt is active")
    baseline_hash = operational["baseline"]["baseline_sha256"]
    if status_record.live_baseline_sha256 and baseline_hash != status_record.live_baseline_sha256:
        raise RuntimeError(
            "new entries blocked: current live baseline does not match the "
            "baseline pinned at the paper_active activation event"
        )
    proposal_path = REPO_ROOT / "research" / "slc_4h_5m_stock_v1_paper_forward_activation_proposal.md"
    if status_record.activation_proposal_sha256:
        if not proposal_path.is_file():
            raise RuntimeError("new entries blocked: activation proposal document is missing")
        if _sha256(proposal_path) != status_record.activation_proposal_sha256:
            raise RuntimeError(
                "new entries blocked: activation proposal document does not match "
                "the version pinned at the paper_active activation event"
            )
    env_mismatches = check_running_environment(operational["baseline"].get("environment", {}))
    if env_mismatches:
        raise RuntimeError(f"new entries blocked: running-environment drift: {env_mismatches}")


def assert_closeout_preconditions(*, observed_account_id: str) -> None:
    """The minimal gate for every risk-reducing action on an already-open
    position: target replacement, emergency cancel-and-flatten,
    reconciliation, and the dedicated closeout stage. Deliberately does
    NOT depend on deployment status, the daily-loss halt, or Tier-2
    signal-fidelity hashes - none of those bear on the ability to safely
    manage or exit a position that's already open. Still requires: broker
    credentials configured, the account-ID match, and execution.py's and
    closeout.py's file hashes intact (not this module's own hash - see the
    self-reference note on GUARDRAILS_TIER2 above; this function's own
    integrity is instead covered structurally, by keeping it small and
    reviewed, and by rule_freeze.py's independent hash of the frozen
    strategy documents)."""
    if not live_slc_settings.ALPACA_API_KEY or not live_slc_settings.ALPACA_SECRET_KEY:
        raise RuntimeError("closeout blocked: Alpaca credentials are not configured")
    if observed_account_id != live_slc_settings.SLC_EXPECTED_ACCOUNT_ID:
        raise RuntimeError(
            f"closeout blocked: observed account {observed_account_id!r} != "
            f"expected {live_slc_settings.SLC_EXPECTED_ACCOUNT_ID!r}"
        )
    minimal_files = {
        "live_slc/execution.py": LIVE_SLC_ROOT / "execution.py",
        "live_slc/closeout.py": LIVE_SLC_ROOT / "closeout.py",
    }
    missing = [name for name, path in minimal_files.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"closeout blocked: file missing: {missing}")
    # Compare against the same frozen baseline's tier2 record, scoped to
    # only these 3 files - drift elsewhere in tier2 must NOT block this path.
    if DEPLOYMENT_BASELINE.is_file() and EXPECTED_DEPLOYMENT_BASELINE_SHA256 is not None:
        baseline = json.loads(DEPLOYMENT_BASELINE.read_text(encoding="utf-8"))
        expected_tier2 = baseline.get("guardrails", {}).get("tier2", {})
        for name, path in minimal_files.items():
            expected_hash = expected_tier2.get(name)
            if expected_hash and _sha256(path) != expected_hash:
                raise RuntimeError(f"closeout blocked: {name} does not match its frozen hash")
