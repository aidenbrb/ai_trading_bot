"""
Settings for the live_slc paper-forward automation package.

Deliberately independent of config/settings.py: reads the existing Alpaca
credentials directly by env var name (same account, no second credential
set - see amendment_004 / rev. 6 Step 5), and every SLC_* switch defaults
off. config/settings.py is never imported here.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


class LiveSlcSettings:
    # -- Existing Alpaca paper credentials (shared account, not duplicated) --
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")

    # -- Preconditions this package re-reads live, never caches -------------
    ALPACA_PAPER_TRADE: bool = os.getenv("ALPACA_PAPER_TRADE", "true").lower() == "true"
    LIVE_TRADING: bool = os.getenv("LIVE_TRADING", "false").lower() == "true"
    ROBINHOOD_ENABLED: bool = os.getenv("ROBINHOOD_ENABLED", "false").lower() == "true"
    EXECUTION_ENABLED: bool = os.getenv("EXECUTION_ENABLED", "false").lower() == "true"

    # -- SLC-specific kill switch, default off -------------------------------
    SLC_PAPER_EXECUTION_ENABLED: bool = os.getenv("SLC_PAPER_EXECUTION_ENABLED", "false").lower() == "true"

    # -- Trust chain: the only account live_slc is ever allowed to act in ---
    SLC_EXPECTED_ACCOUNT_ID: str = os.getenv("SLC_EXPECTED_ACCOUNT_ID", "")

    def require_new_entry_env(self) -> list[str]:
        """Return the list of failed preconditions for a NEW entry only.

        Never used for closeout / management of an already-open position -
        those use a separate, minimal gate (live_slc.guardrails).
        """
        failures = []
        if not self.ALPACA_PAPER_TRADE:
            failures.append("ALPACA_PAPER_TRADE must be true")
        if self.LIVE_TRADING:
            failures.append("LIVE_TRADING must be false")
        if self.ROBINHOOD_ENABLED:
            failures.append("ROBINHOOD_ENABLED must be false")
        if not self.EXECUTION_ENABLED:
            failures.append("EXECUTION_ENABLED must be true")
        if not self.SLC_PAPER_EXECUTION_ENABLED:
            failures.append("SLC_PAPER_EXECUTION_ENABLED must be true")
        if not self.SLC_EXPECTED_ACCOUNT_ID:
            failures.append("SLC_EXPECTED_ACCOUNT_ID must be set")
        return failures


live_slc_settings = LiveSlcSettings()
