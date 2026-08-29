"""Options research selection and sizing safety tests."""
from datetime import date, timedelta

from utils.options import OptionCandidate, candidate_rejection, select_candidate, size_long_option

TODAY = date(2026, 7, 24)


def _candidate(**overrides):
    values = dict(
        symbol="AAPL260904C00200000",
        underlying="AAPL",
        expiration_date=TODAY + timedelta(days=42),
        strike_price=200.0,
        bid=4.90,
        ask=5.10,
        open_interest=1000,
        delta=0.62,
        implied_volatility=0.30,
    )
    values.update(overrides)
    return OptionCandidate(**values)


def _reject(candidate):
    return candidate_rejection(
        candidate, as_of=TODAY, min_dte=30, max_dte=60,
        min_delta=0.55, max_delta=0.70, min_open_interest=500,
        max_spread_pct=0.08,
    )


def test_eligible_contract_passes():
    assert _reject(_candidate()) is None


def test_rejects_wide_spread():
    assert "spread" in _reject(_candidate(bid=4.0, ask=6.0))


def test_rejects_missing_delta():
    assert "delta unavailable" == _reject(_candidate(delta=None))


def test_rejects_low_open_interest():
    assert "open interest" in _reject(_candidate(open_interest=20))


def test_rejects_bad_dte():
    assert "DTE" in _reject(_candidate(expiration_date=TODAY + timedelta(days=7)))


def test_rejects_put_in_v1():
    assert "long calls" in _reject(_candidate(contract_type="put", delta=-0.62))


def test_selector_prefers_target_delta_then_spread():
    far_delta = _candidate(symbol="FAR", delta=0.69, bid=4.95, ask=5.05)
    target = _candidate(symbol="TARGET", delta=0.625, bid=4.85, ask=5.15)
    result = select_candidate(
        [far_delta, target], underlying_price=200, as_of=TODAY,
        min_dte=30, max_dte=60, min_delta=0.55, max_delta=0.70,
        min_open_interest=500, max_spread_pct=0.08,
    )
    assert result.symbol == "TARGET"


def test_sizing_uses_full_premium_as_max_loss():
    sizing = size_long_option(2.50, 100, 100_000, 0.0025, 0.02)
    assert sizing == {
        "contracts": 1,
        "premium_per_contract": 250.0,
        "total_premium": 250.0,
        "max_loss": 250.0,
    }


def test_sizing_refuses_unaffordable_contract():
    assert size_long_option(5.00, 100, 100_000, 0.0025, 0.02)["contracts"] == 0


def test_options_execution_flag_is_rejected(monkeypatch):
    from config.settings import Settings

    settings = Settings()
    monkeypatch.setattr(settings, "OPTIONS_EXECUTION_ENABLED", True)
    with __import__("pytest").raises(RuntimeError, match="not supported"):
        settings.validate(require_broker=False)
