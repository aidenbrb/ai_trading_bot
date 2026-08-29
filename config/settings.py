"""
Central settings for ai_trading_bot.

All values are loaded from environment variables (or a .env file).
Copy .env.example to .env and fill in your values before running any node.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (ai_trading_bot/.env)
load_dotenv(Path(__file__).parent.parent / ".env")


class Settings:
    # -- Database --------------------------------------------------------------
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///trading.db")

    # -- Alpaca ----------------------------------------------------------------
    ALPACA_API_KEY: str    = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
    ALPACA_PAPER_TRADE: bool = os.getenv("ALPACA_PAPER_TRADE", "true").lower() == "true"

    # -- Safety (never change LIVE_TRADING without 3+ months paper validation) -
    EXECUTION_ENABLED: bool = os.getenv("EXECUTION_ENABLED", "false").lower() == "true"
    LIVE_TRADING: bool      = os.getenv("LIVE_TRADING", "false").lower() == "true"

    # -- Risk limits -----------------------------------------------------------
    MAX_RISK_PER_TRADE:    float = float(os.getenv("MAX_RISK_PER_TRADE", "0.01"))   # 1%
    MAX_OPEN_POSITIONS:    int   = int(os.getenv("MAX_OPEN_POSITIONS", "5"))
    MAX_NEW_TRADES_PER_DAY: int  = int(os.getenv("MAX_NEW_TRADES_PER_DAY", "2"))
    MIN_RISK_REWARD:       float = float(os.getenv("MIN_RISK_REWARD", "2.0"))
    DAILY_LOSS_LIMIT:      float = float(os.getenv("DAILY_LOSS_LIMIT", "0.02"))     # 2% of equity

    # Crypto orders use the same hardcoded Alpaca paper client as stocks.
    CRYPTO_EXECUTION_ENABLED: bool = os.getenv("CRYPTO_EXECUTION_ENABLED", "false").lower() == "true"

    # -- Automatic market schedule --------------------------------------------
    AUTO_MARKET_SCHEDULE: bool = os.getenv("AUTO_MARKET_SCHEDULE", "true").lower() == "true"
    WEEKEND_CRYPTO_RESEARCH: bool = os.getenv("WEEKEND_CRYPTO_RESEARCH", "true").lower() == "true"

    # -- Robinhood (stocks, LIVE account - no paper trading exists) -----------
    # When enabled, ALL stock order flow (execution_node + monitor_node) routes
    # to Robinhood instead of Alpaca paper. Crypto/Coinbase is unaffected.
    ROBINHOOD_USERNAME:     str  = os.getenv("ROBINHOOD_USERNAME", "")
    ROBINHOOD_PASSWORD:     str  = os.getenv("ROBINHOOD_PASSWORD", "")
    ROBINHOOD_TOTP_SECRET:  str  = os.getenv("ROBINHOOD_TOTP_SECRET", "")
    ROBINHOOD_ENABLED:      bool = os.getenv("ROBINHOOD_ENABLED", "false").lower() == "true"

    # -- Signal thresholds -----------------------------------------------------
    MIN_CONVICTION_SCORE: int = int(os.getenv("MIN_CONVICTION_SCORE", "70"))
    MIN_SETUP_QUALITY:    int = int(os.getenv("MIN_SETUP_QUALITY", "60"))

    # -- Data ------------------------------------------------------------------
    BAR_INTERVAL:     str  = os.getenv("BAR_INTERVAL", "1h")          # yfinance interval
    DOWNLOAD_DAYS:    int  = int(os.getenv("DOWNLOAD_DAYS", "720"))    # 2yr max for hourly
    MIN_HISTORY_BARS: int  = int(os.getenv("MIN_HISTORY_BARS", "200")) # bars needed for SMA 200
    MIN_PRICE:        float = float(os.getenv("MIN_PRICE", "10.0"))
    MIN_AVG_VOLUME:   int   = int(os.getenv("MIN_AVG_VOLUME", "1000000"))

    # -- Morning news gate -------------------------------------------------------
    NEWS_GATE_ENABLED:           bool  = os.getenv("NEWS_GATE_ENABLED", "true").lower() == "true"
    NEWS_MAX_AGE_HOURS:          float = float(os.getenv("NEWS_MAX_AGE_HOURS", "18"))
    NEWS_RSS_TIMEOUT_SECONDS:    int   = int(os.getenv("NEWS_RSS_TIMEOUT_SECONDS", "10"))
    NEWS_ITEM_MAX_AGE_HOURS:     float = float(os.getenv("NEWS_ITEM_MAX_AGE_HOURS", "72"))
    NEWS_VIX_HIGH_THRESHOLD:     float = float(os.getenv("NEWS_VIX_HIGH_THRESHOLD", "30"))
    NEWS_VIX_ELEVATED_THRESHOLD: float = float(os.getenv("NEWS_VIX_ELEVATED_THRESHOLD", "25"))
    NEWS_DATA_DIR:                str = os.getenv("NEWS_DATA_DIR", "")   # "" = repo-local default

    # -- Position monitoring -----------------------------------------------------
    MONITOR_ATR_TRAIL_MULT: float = float(os.getenv("MONITOR_ATR_TRAIL_MULT", "1.5"))

    # -- Options research (never submits orders) ---------------------------------
    OPTIONS_RESEARCH_ENABLED: bool = os.getenv("OPTIONS_RESEARCH_ENABLED", "false").lower() == "true"
    OPTIONS_EXECUTION_ENABLED: bool = os.getenv("OPTIONS_EXECUTION_ENABLED", "false").lower() == "true"
    OPTIONS_MIN_DTE: int = int(os.getenv("OPTIONS_MIN_DTE", "30"))
    OPTIONS_MAX_DTE: int = int(os.getenv("OPTIONS_MAX_DTE", "60"))
    OPTIONS_EXIT_DTE: int = int(os.getenv("OPTIONS_EXIT_DTE", "7"))
    OPTIONS_MIN_DELTA: float = float(os.getenv("OPTIONS_MIN_DELTA", "0.55"))
    OPTIONS_MAX_DELTA: float = float(os.getenv("OPTIONS_MAX_DELTA", "0.70"))
    OPTIONS_MIN_OPEN_INTEREST: int = int(os.getenv("OPTIONS_MIN_OPEN_INTEREST", "500"))
    OPTIONS_MAX_SPREAD_PCT: float = float(os.getenv("OPTIONS_MAX_SPREAD_PCT", "0.08"))
    OPTIONS_MAX_RISK_PER_TRADE: float = float(os.getenv("OPTIONS_MAX_RISK_PER_TRADE", "0.0025"))
    OPTIONS_MAX_PREMIUM_PCT: float = float(os.getenv("OPTIONS_MAX_PREMIUM_PCT", "0.02"))
    OPTIONS_RESEARCH_EQUITY: float = float(os.getenv("OPTIONS_RESEARCH_EQUITY", "100000"))

    # -- Broker connectivity ------------------------------------------------------
    RISK_REQUIRE_BROKER_CONNECTED: bool = os.getenv("RISK_REQUIRE_BROKER_CONNECTED", "true").lower() == "true"

    def validate(self, require_broker: bool | None = None) -> None:
        """
        Raise if critical settings are missing.

        require_broker - require Alpaca credentials. Defaults to EXECUTION_ENABLED,
                         because account state degrades to safe defaults when creds
                         are absent and live broker access is only truly needed to
                         submit real orders.
        """
        if require_broker is None:
            require_broker = self.EXECUTION_ENABLED

        if require_broker and self.ROBINHOOD_ENABLED:
            if not self.ROBINHOOD_USERNAME or not self.ROBINHOOD_PASSWORD:
                raise RuntimeError(
                    "ROBINHOOD_USERNAME and ROBINHOOD_PASSWORD must be set in "
                    "ai_trading_bot/.env before enabling EXECUTION_ENABLED with "
                    "ROBINHOOD_ENABLED=true."
                )
        elif require_broker and (not self.ALPACA_API_KEY or not self.ALPACA_SECRET_KEY):
            raise RuntimeError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in ai_trading_bot/.env "
                "before enabling EXECUTION_ENABLED."
            )
        if self.LIVE_TRADING:
            raise RuntimeError(
                "LIVE_TRADING=true is not yet supported. "
                "Complete 3+ months of paper trading validation first."
            )
        if self.OPTIONS_EXECUTION_ENABLED:
            raise RuntimeError(
                "OPTIONS_EXECUTION_ENABLED=true is not supported. "
                "Only options research mode has been implemented."
            )
        if self.CRYPTO_EXECUTION_ENABLED and not self.ALPACA_PAPER_TRADE:
            raise RuntimeError(
                "Crypto execution requires ALPACA_PAPER_TRADE=true. "
                "Real-money crypto execution is not supported."
            )
        if self.CRYPTO_EXECUTION_ENABLED and self.ROBINHOOD_ENABLED:
            raise RuntimeError(
                "CRYPTO_EXECUTION_ENABLED and ROBINHOOD_ENABLED cannot both be "
                "true because cross-broker portfolio risk aggregation is not implemented."
            )
        if not (0 < self.OPTIONS_MIN_DTE < self.OPTIONS_MAX_DTE):
            raise RuntimeError("Require 0 < OPTIONS_MIN_DTE < OPTIONS_MAX_DTE.")
        if not (0 < self.OPTIONS_MAX_RISK_PER_TRADE <= self.OPTIONS_MAX_PREMIUM_PCT <= 1):
            raise RuntimeError(
                "Require 0 < OPTIONS_MAX_RISK_PER_TRADE <= OPTIONS_MAX_PREMIUM_PCT <= 1."
            )

    def print_summary(self) -> None:
        print("=" * 50)
        print("  ai_trading_bot  settings")
        print("=" * 50)
        print(f"  Database:          {self.DATABASE_URL}")
        stock_broker = "Robinhood (LIVE)" if self.ROBINHOOD_ENABLED else "Alpaca (paper)"
        print(f"  Stock broker:      {stock_broker}")
        print(f"  Paper trading:     {self.ALPACA_PAPER_TRADE}")
        print(f"  Execution enabled: {self.EXECUTION_ENABLED}")
        print(f"  Live trading:      {self.LIVE_TRADING}")
        print(f"  Max risk/trade:    {self.MAX_RISK_PER_TRADE * 100:.1f}%")
        print(f"  Max positions:     {self.MAX_OPEN_POSITIONS}")
        print(f"  Daily loss limit:  {self.DAILY_LOSS_LIMIT * 100:.1f}%")
        print(f"  Auto schedule:     {self.AUTO_MARKET_SCHEDULE}")
        crypto_mode = (
            "Alpaca paper"
            if self.CRYPTO_EXECUTION_ENABLED else
            "disabled (research only)"
        )
        print(f"  Crypto execution:  {crypto_mode}")
        print(f"  News gate:         {'ENABLED' if self.NEWS_GATE_ENABLED else 'disabled'} "
              f"(max age {self.NEWS_MAX_AGE_HOURS:.0f}h)")
        print(f"  Broker required:   {self.RISK_REQUIRE_BROKER_CONNECTED}")
        options_mode = "RESEARCH" if self.OPTIONS_RESEARCH_ENABLED else "disabled"
        print(f"  Options:           {options_mode} (execution unavailable)")
        print("=" * 50)


settings = Settings()
