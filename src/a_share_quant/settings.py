"""Application settings loaded from environment variables and .env file.

All secrets are read from environment variables only.
The Settings class never creates .env files with secrets.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataProviderType(StrEnum):
    """Supported data providers."""

    MOCK = "mock"
    LOCAL_PARQUET = "local_parquet"
    JOINQUANT = "joinquant"
    TUSHARE = "tushare"
    AKSHARE = "akshare"


class BrokerProviderType(StrEnum):
    """Supported broker providers."""

    NONE = "none"
    CSV_MANUAL = "csv_manual"
    PAPER = "paper"
    QMT = "qmt"
    PTRADE = "ptrade"


class NotifyProviderType(StrEnum):
    """Supported notification providers."""

    CONSOLE = "console"
    WEBHOOK = "webhook"
    EMAIL = "email"


class Settings(BaseSettings):
    """Central application settings.

    Reads from environment variables and .env file.
    Secrets (tokens, passwords) are only read, never written or logged.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    app_env: str = "development"
    app_name: str = "a_share_quant_life"
    timezone: str = "Asia/Shanghai"
    log_level: str = "INFO"
    random_seed: int = 20260720
    dry_run: bool = True
    allow_live_trading: bool = False

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    data_root: Path = Path("./data")
    artifact_root: Path = Path("./artifacts")
    report_root: Path = Path("./reports")
    duckdb_path: Path = Path("./data/quant.duckdb")
    market_rules_path: Path = Path("./configs/market_rules_cn.yaml")

    # ------------------------------------------------------------------
    # Data provider
    # ------------------------------------------------------------------
    data_provider: DataProviderType = DataProviderType.MOCK
    data_start_date: str = "20100101"
    data_end_date: Optional[str] = None

    joinquant_username: Optional[str] = None
    joinquant_password: Optional[str] = None
    tushare_token: Optional[str] = None
    akshare_enabled: bool = False

    # ------------------------------------------------------------------
    # Research protocol
    # ------------------------------------------------------------------
    initial_capital: float = 100_000.0
    base_currency: str = "CNY"
    bar_frequency: str = "1d"
    signal_at: str = "close"
    execution_at: str = "next_open"
    benchmark_symbol: Optional[str] = None
    universe_profile: str = "liquid_etf_mvp"
    risk_profile: str = "paper_conservative"
    fee_profile: str = "cn_a_share_date_aware"

    # ------------------------------------------------------------------
    # Backtest controls
    # ------------------------------------------------------------------
    backtest_engine: str = "event_driven"
    strict_no_lookahead: bool = True
    conservative_same_bar_fill: bool = True
    simulate_t_plus_one: bool = True
    simulate_price_limits: bool = True
    simulate_suspensions: bool = True
    simulate_partial_fill: bool = False

    # ------------------------------------------------------------------
    # Broker / paper trading
    # ------------------------------------------------------------------
    broker_provider: BrokerProviderType = BrokerProviderType.PAPER
    broker_account_id: Optional[str] = None
    qmt_install_path: Optional[str] = None
    qmt_account_type: str = "STOCK"
    ptrade_endpoint: Optional[str] = None
    ptrade_account_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------
    notify_provider: NotifyProviderType = NotifyProviderType.CONSOLE
    notify_webhook_url: Optional[str] = None
    notify_email: Optional[str] = None

    # ------------------------------------------------------------------
    # Safety interlock
    # ------------------------------------------------------------------
    live_trading_acknowledgement: str = ""

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            msg = f"Invalid log level: {v}. Must be one of {valid_levels}"
            raise ValueError(msg)
        return upper

    @field_validator("initial_capital")
    @classmethod
    def validate_capital(cls, v: float) -> float:
        """Ensure capital is positive."""
        if v <= 0:
            msg = f"Initial capital must be positive, got {v}"
            raise ValueError(msg)
        return v

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def is_live_trading_allowed(self) -> bool:
        """Check if live trading is fully authorized.

        Live trading requires ALL of:
        - allow_live_trading = true
        - broker configured (not 'none')
        - broker_account_id set
        - live_trading_acknowledgement non-empty
        """
        return (
            self.allow_live_trading
            and self.broker_provider != BrokerProviderType.NONE
            and bool(self.broker_account_id)
            and bool(self.live_trading_acknowledgement)
        )

    def get_secret_status(self) -> dict[str, str]:
        """Return secret configuration status without exposing values.

        Used by `quant doctor` to show configured/not-configured.
        """
        return {
            "joinquant_username": "configured" if self.joinquant_username else "not configured",
            "joinquant_password": "configured" if self.joinquant_password else "not configured",
            "tushare_token": "configured" if self.tushare_token else "not configured",
            "broker_account_id": "configured" if self.broker_account_id else "not configured",
            "notify_webhook_url": "configured" if self.notify_webhook_url else "not configured",
        }


def get_settings() -> Settings:
    """Create and return application settings instance."""
    return Settings()
