"""
Phase 6 Step 1: the human-only re-authorization signature gate.

Covers the required negative cases end to end (missing, invalid, tampered,
replayed, wrong-baseline, wrong-commit, and a directly-inserted row with no
signature at all), plus the payload/verify round trip in isolation. Every
signature here is produced by a disposable SOFTWARE ed25519 test key
(tests/_slc_reauth_helpers.py) through the real `ssh-keygen -Y sign` /
`-Y verify` mechanism - never the production hardware ed25519-sk path,
which this test suite cannot and must not exercise.
"""
import subprocess

import pytest

import live_slc.models as models
from live_slc import guardrails, reauth_signature
from live_slc.authorization import issue_reauth_nonce, record_transition
from live_slc.models import SlcActivationEvent, get_live_slc_session
from tests._slc_reauth_helpers import make_test_key, signed_reauth_kwargs


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "LIVE_SLC_DB_PATH", tmp_path / "test.db")
    models.init_live_slc_db()


@pytest.fixture
def signer(tmp_path, monkeypatch):
    key_path, allowed_signers_path, identity = make_test_key(tmp_path)
    monkeypatch.setattr(reauth_signature, "ALLOWED_SIGNERS_PATH", allowed_signers_path)
    return key_path, allowed_signers_path, identity


def _activate(**overrides):
    kwargs = dict(
        guardrail_baseline_sha256="a" * 64, activation_proposal_sha256="b" * 64,
        observed_account_id="acct-1",
    )
    kwargs.update(overrides)
    return kwargs


# -- payload / raw verify round trip (no DB) --------------------------------

def test_build_and_parse_payload_round_trips(tmp_path):
    payload = reauth_signature.build_payload(
        from_status="dry_run", to_status="paper_active",
        guardrail_baseline_sha256="a" * 64, activation_proposal_sha256="b" * 64,
        observed_account_id="acct-1", git_commit_sha="c" * 40,
        changed_guardrail_paths=["b.py", "a.py"], nonce="n1", timestamp_iso="2026-08-30T00:00:00+00:00",
    )
    fields = reauth_signature.parse_payload(payload)
    assert fields["guardrail_baseline_sha256"] == "a" * 64
    assert fields["changed_guardrail_paths"] == "a.py,b.py"  # sorted regardless of input order
    assert fields["nonce"] == "n1"


def test_verify_signature_true_for_a_genuine_signature(signer):
    key_path, allowed_signers_path, identity = signer
    payload = reauth_signature.build_payload(
        from_status="x", to_status="y", guardrail_baseline_sha256="a" * 64,
        activation_proposal_sha256=None, observed_account_id="acct-1",
        git_commit_sha="c" * 40, changed_guardrail_paths=[], nonce="n1",
        timestamp_iso="2026-08-30T00:00:00+00:00",
    )
    sig = reauth_signature.sign_for_test(payload, key_path)
    assert reauth_signature.verify_signature(
        payload, sig, signer_identity=identity, allowed_signers_path=allowed_signers_path,
    ) is True


def test_verify_signature_false_for_an_empty_allowed_signers_file(tmp_path):
    """The shipped production live_slc/allowed_signers is empty by design
    until the operator adds their real key - proves that state verifies
    nothing, not everything."""
    empty = tmp_path / "allowed_signers"
    empty.write_text("# empty\n", encoding="utf-8")
    assert reauth_signature.verify_signature(
        "anything", "garbage", signer_identity="whoever", allowed_signers_path=empty,
    ) is False


def test_verify_signature_false_when_allowed_signers_file_is_missing(tmp_path):
    assert reauth_signature.verify_signature(
        "anything", "garbage", signer_identity="whoever",
        allowed_signers_path=tmp_path / "does_not_exist",
    ) is False


# -- record_transition(): the required negative cases -----------------------

def test_missing_signature_rejected():
    record_transition("not_authorized", "dry_run", "start")
    with pytest.raises(RuntimeError, match="missing"):
        record_transition("dry_run", "paper_active", "activate", **_activate())


def test_invalid_signature_rejected(signer):
    key_path, allowed_signers_path, identity = signer
    record_transition("not_authorized", "dry_run", "start")
    sig_kwargs = signed_reauth_kwargs(
        key_path=key_path, identity=identity, from_status="dry_run", to_status="paper_active",
        guardrail_baseline_sha256="a" * 64, activation_proposal_sha256="b" * 64,
        observed_account_id="acct-1",
    )
    sig_kwargs["signature_blob"] = "not a real SSHSIG block at all"
    with pytest.raises(RuntimeError, match="signature verification failed"):
        record_transition("dry_run", "paper_active", "activate", **_activate(), **sig_kwargs)


def test_tampered_payload_rejected(signer):
    """The signature is genuine, but the payload text changed after
    signing (e.g. someone edited the baseline hash post-signature) - the
    cryptographic verify must fail even though every individual field
    still 'looks' plausible."""
    key_path, allowed_signers_path, identity = signer
    record_transition("not_authorized", "dry_run", "start")
    sig_kwargs = signed_reauth_kwargs(
        key_path=key_path, identity=identity, from_status="dry_run", to_status="paper_active",
        guardrail_baseline_sha256="a" * 64, activation_proposal_sha256="b" * 64,
        observed_account_id="acct-1",
    )
    tampered_payload = sig_kwargs["signed_payload"].replace("a" * 64, "f" * 64)
    sig_kwargs["signed_payload"] = tampered_payload
    with pytest.raises(RuntimeError, match="does not match the transition being recorded|signature verification failed"):
        record_transition(
            "dry_run", "paper_active", "activate",
            guardrail_baseline_sha256="f" * 64, activation_proposal_sha256="b" * 64,
            observed_account_id="acct-1", **{k: v for k, v in sig_kwargs.items() if k != "signed_payload"},
            signed_payload=tampered_payload,
        )


def test_replayed_nonce_rejected(signer):
    key_path, allowed_signers_path, identity = signer
    record_transition("not_authorized", "dry_run", "start")
    sig_kwargs = signed_reauth_kwargs(
        key_path=key_path, identity=identity, from_status="dry_run", to_status="paper_active",
        guardrail_baseline_sha256="a" * 64, activation_proposal_sha256="b" * 64,
        observed_account_id="acct-1",
    )
    record_transition("dry_run", "paper_active", "activate", **_activate(), **sig_kwargs)

    # Same nonce (and its still-valid signature) reused for a second,
    # different transition - must be rejected as a replay, not re-accepted.
    replay_kwargs = signed_reauth_kwargs(
        key_path=key_path, identity=identity, from_status="paper_active", to_status="paper_active",
        guardrail_baseline_sha256="d" * 64, activation_proposal_sha256="b" * 64,
        observed_account_id="acct-1", nonce=sig_kwargs["nonce"],
    )
    with pytest.raises(RuntimeError, match="already consumed|replay rejected"):
        record_transition(
            "paper_active", "paper_active", "second refreeze reusing the nonce",
            guardrail_baseline_sha256="d" * 64, activation_proposal_sha256="b" * 64,
            observed_account_id="acct-1", **replay_kwargs,
        )


def test_wrong_baseline_rejected(signer):
    """Payload was signed declaring one baseline hash, but a DIFFERENT
    hash is what's actually being recorded - must be rejected even though
    the signature itself is perfectly genuine for what it actually says."""
    key_path, allowed_signers_path, identity = signer
    record_transition("not_authorized", "dry_run", "start")
    sig_kwargs = signed_reauth_kwargs(
        key_path=key_path, identity=identity, from_status="dry_run", to_status="paper_active",
        guardrail_baseline_sha256="a" * 64, activation_proposal_sha256="b" * 64,
        observed_account_id="acct-1",
    )
    with pytest.raises(RuntimeError, match="does not match the transition being recorded"):
        record_transition(
            "dry_run", "paper_active", "activate",
            guardrail_baseline_sha256="z" * 64,  # different from what was signed
            activation_proposal_sha256="b" * 64, observed_account_id="acct-1", **sig_kwargs,
        )


def test_wrong_commit_rejected(signer):
    """Payload declares a git_commit_sha that isn't the actual current
    HEAD - must be rejected. Uses an obviously-fake but well-formed SHA
    rather than mocking git, so this exercises the real
    `git rev-parse HEAD` comparison."""
    key_path, allowed_signers_path, identity = signer
    record_transition("not_authorized", "dry_run", "start")
    sig_kwargs = signed_reauth_kwargs(
        key_path=key_path, identity=identity, from_status="dry_run", to_status="paper_active",
        guardrail_baseline_sha256="a" * 64, activation_proposal_sha256="b" * 64,
        observed_account_id="acct-1", git_commit_sha="0" * 40,
    )
    with pytest.raises(RuntimeError, match="does not match the actual current HEAD"):
        record_transition("dry_run", "paper_active", "activate", **_activate(), **sig_kwargs)


def test_valid_signature_is_accepted_and_recorded(signer):
    key_path, allowed_signers_path, identity = signer
    record_transition("not_authorized", "dry_run", "start")
    sig_kwargs = signed_reauth_kwargs(
        key_path=key_path, identity=identity, from_status="dry_run", to_status="paper_active",
        guardrail_baseline_sha256="a" * 64, activation_proposal_sha256="b" * 64,
        observed_account_id="acct-1",
    )
    event = record_transition("dry_run", "paper_active", "activate", **_activate(), **sig_kwargs)
    assert event.signature_blob == sig_kwargs["signature_blob"]
    assert event.signer_identity == identity
    with get_live_slc_session() as session:
        nonce_row = session.get(models.SlcReauthNonce, sig_kwargs["nonce"])
    assert nonce_row.consumed_at is not None
    assert nonce_row.consumed_by_event_id == event.id


def test_nonce_expired_beyond_24h_rejected(signer, monkeypatch):
    key_path, allowed_signers_path, identity = signer
    record_transition("not_authorized", "dry_run", "start")
    nonce = issue_reauth_nonce()
    import datetime as dt
    with get_live_slc_session() as session:
        row = session.get(models.SlcReauthNonce, nonce)
        row.created_at = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - dt.timedelta(hours=25)
        session.add(row)
    sig_kwargs = signed_reauth_kwargs(
        key_path=key_path, identity=identity, from_status="dry_run", to_status="paper_active",
        guardrail_baseline_sha256="a" * 64, activation_proposal_sha256="b" * 64,
        observed_account_id="acct-1", nonce=nonce,
    )
    with pytest.raises(RuntimeError, match="expired"):
        record_transition("dry_run", "paper_active", "activate", **_activate(), **sig_kwargs)


# -- guardrails.py read-time re-verification --------------------------------

def test_read_time_check_rejects_a_directly_inserted_unsigned_row():
    """Bypassing record_transition() entirely (a raw INSERT) must not
    fool assert_operational_preconditions()'s read-time check - it
    re-verifies the signature itself, not merely trusting that a row
    with the right baseline hash exists."""
    baseline_hash = "e" * 64
    with get_live_slc_session() as session:
        session.add(SlcActivationEvent(
            from_status="dry_run", to_status="paper_active", reason="direct DB insert, no signature",
            guardrail_baseline_sha256_at_transition=baseline_hash,
        ))
    with pytest.raises(RuntimeError, match="no signature-verified activation event"):
        guardrails.verify_baseline_is_signed(baseline_hash)


def test_read_time_check_rejects_a_row_with_a_tampered_stored_signature(signer):
    """A row that WAS signed at write time, but whose stored payload/
    signature has since been altered in the DB, must fail read-time
    re-verification - proving verification happens fresh on read, not
    just once at write and then trusted forever."""
    key_path, allowed_signers_path, identity = signer
    record_transition("not_authorized", "dry_run", "start")
    sig_kwargs = signed_reauth_kwargs(
        key_path=key_path, identity=identity, from_status="dry_run", to_status="paper_active",
        guardrail_baseline_sha256="a" * 64, activation_proposal_sha256="b" * 64,
        observed_account_id="acct-1",
    )
    record_transition("dry_run", "paper_active", "activate", **_activate(), **sig_kwargs)
    with get_live_slc_session() as session:
        from sqlmodel import select
        event = session.exec(
            select(SlcActivationEvent).where(SlcActivationEvent.to_status == "paper_active")
        ).first()
        event.signed_payload = event.signed_payload.replace("a" * 64, "f" * 64)
        session.add(event)
    with pytest.raises(RuntimeError, match="no signature-verified activation event"):
        guardrails.verify_baseline_is_signed("a" * 64)


def test_read_time_check_accepts_a_genuinely_valid_signed_baseline(signer):
    key_path, allowed_signers_path, identity = signer
    record_transition("not_authorized", "dry_run", "start")
    sig_kwargs = signed_reauth_kwargs(
        key_path=key_path, identity=identity, from_status="dry_run", to_status="paper_active",
        guardrail_baseline_sha256="a" * 64, activation_proposal_sha256="b" * 64,
        observed_account_id="acct-1",
    )
    record_transition("dry_run", "paper_active", "activate", **_activate(), **sig_kwargs)
    event = guardrails.verify_baseline_is_signed("a" * 64)
    assert event.guardrail_baseline_sha256_at_transition == "a" * 64


def test_read_time_check_rejects_a_legacy_pre_phase6_unsigned_event():
    """Every SlcActivationEvent recorded before Phase 6 (e.g. the real
    9f14af9 event) has signed_payload=None - this is exactly the state
    the read-time check must refuse to trust once armed, not a
    hypothetical."""
    baseline_hash = "9" * 64
    with get_live_slc_session() as session:
        session.add(SlcActivationEvent(
            from_status="paper_active", to_status="paper_active", reason="pre-Phase-6 legacy re-freeze",
            guardrail_baseline_sha256_at_transition=baseline_hash,
            operator_note="agent-executed per explicit user approval",
        ))
    with pytest.raises(RuntimeError, match="no signature-verified activation event"):
        guardrails.verify_baseline_is_signed(baseline_hash)
