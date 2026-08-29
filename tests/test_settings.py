"""Tests for config/settings.py - the Settings.validate() broker-credential gate."""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import Settings


def _settings(**overrides) -> Settings:
    s = Settings()
    s.LIVE_TRADING = False
    # Tests must not inherit the operator's active .env execution choices.
    s.CRYPTO_EXECUTION_ENABLED = False
    for key, value in overrides.items():
        setattr(s, key, value)
    return s


class TestRobinhoodDisabled:
    def test_requires_alpaca_creds(self):
        s = _settings(ROBINHOOD_ENABLED=False, ALPACA_API_KEY="", ALPACA_SECRET_KEY="")
        with pytest.raises(RuntimeError, match="ALPACA_API_KEY"):
            s.validate(require_broker=True)

    def test_passes_with_alpaca_creds(self):
        s = _settings(ROBINHOOD_ENABLED=False, ALPACA_API_KEY="key", ALPACA_SECRET_KEY="secret")
        s.validate(require_broker=True)   # should not raise

    def test_robinhood_creds_irrelevant(self):
        # Alpaca is the active broker, so missing Robinhood creds must not matter.
        s = _settings(ROBINHOOD_ENABLED=False, ALPACA_API_KEY="key", ALPACA_SECRET_KEY="secret",
                      ROBINHOOD_USERNAME="", ROBINHOOD_PASSWORD="")
        s.validate(require_broker=True)   # should not raise


class TestRobinhoodEnabled:
    def test_requires_robinhood_creds(self):
        s = _settings(ROBINHOOD_ENABLED=True, ROBINHOOD_USERNAME="", ROBINHOOD_PASSWORD="")
        with pytest.raises(RuntimeError, match="ROBINHOOD_USERNAME"):
            s.validate(require_broker=True)

    def test_passes_with_robinhood_creds(self):
        s = _settings(ROBINHOOD_ENABLED=True, ROBINHOOD_USERNAME="user", ROBINHOOD_PASSWORD="pw")
        s.validate(require_broker=True)   # should not raise

    def test_missing_alpaca_creds_irrelevant(self):
        # Robinhood is the active broker, so missing Alpaca creds must not matter.
        s = _settings(ROBINHOOD_ENABLED=True, ROBINHOOD_USERNAME="user", ROBINHOOD_PASSWORD="pw",
                      ALPACA_API_KEY="", ALPACA_SECRET_KEY="")
        s.validate(require_broker=True)   # should not raise


class TestLiveTradingStillBlocked:
    def test_blocks_regardless_of_broker(self):
        s = _settings(ROBINHOOD_ENABLED=True, ROBINHOOD_USERNAME="user", ROBINHOOD_PASSWORD="pw",
                      LIVE_TRADING=True)
        with pytest.raises(RuntimeError, match="LIVE_TRADING"):
            s.validate(require_broker=True)


class TestCryptoPaperSafety:
    def test_crypto_requires_paper_mode(self):
        s = _settings(
            CRYPTO_EXECUTION_ENABLED=True,
            ALPACA_PAPER_TRADE=False,
            ALPACA_API_KEY="key",
            ALPACA_SECRET_KEY="secret",
        )
        with pytest.raises(RuntimeError, match="ALPACA_PAPER_TRADE"):
            s.validate(require_broker=True)

    def test_crypto_and_robinhood_cannot_mix(self):
        s = _settings(
            CRYPTO_EXECUTION_ENABLED=True,
            ALPACA_PAPER_TRADE=True,
            ROBINHOOD_ENABLED=True,
            ROBINHOOD_USERNAME="user",
            ROBINHOOD_PASSWORD="pw",
        )
        with pytest.raises(RuntimeError, match="cannot both be true"):
            s.validate(require_broker=True)


class TestRequireBrokerFalse:
    def test_skips_broker_check_entirely(self):
        s = _settings(ROBINHOOD_ENABLED=True, ROBINHOOD_USERNAME="", ROBINHOOD_PASSWORD="",
                      ALPACA_API_KEY="", ALPACA_SECRET_KEY="")
        s.validate(require_broker=False)   # should not raise
