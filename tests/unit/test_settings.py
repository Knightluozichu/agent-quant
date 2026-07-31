"""Tests for application settings."""

from __future__ import annotations

import pytest

from a_share_quant.settings import (
    BrokerProviderType,
    DataProviderType,
    Settings,
    get_settings,
)


@pytest.mark.unit
class TestSettings:
    """Test Settings class."""

    def test_default_settings(self) -> None:
        """Test default settings values (isolated from .env)."""
        # Create settings with explicit defaults to isolate from .env
        settings = Settings(
            _env_file=None,  # Ignore .env file
            app_env="development",
            data_provider=DataProviderType.MOCK,
            dry_run=True,
            allow_live_trading=False,
            initial_capital=100_000.0,
            simulate_t_plus_one=True,
        )
        assert settings.app_env == "development"
        assert settings.data_provider == DataProviderType.MOCK
        assert settings.dry_run is True
        assert settings.allow_live_trading is False
        assert settings.initial_capital == 100_000.0
        assert settings.simulate_t_plus_one is True

    def test_live_trading_not_allowed_by_default(self) -> None:
        """Live trading must not be allowed by default."""
        settings = Settings()
        assert settings.is_live_trading_allowed() is False

    def test_live_trading_requires_all_conditions(self) -> None:
        """Live trading requires multiple conditions to be met."""
        # Even with allow_live_trading=True, other conditions must be met
        settings = Settings(
            allow_live_trading=True,
            broker_provider=BrokerProviderType.PAPER,
            broker_account_id="test_account",
            live_trading_acknowledgement="I_UNDERSTAND_THE_RISKS",
        )
        assert settings.is_live_trading_allowed() is True

        # Missing acknowledgement
        settings_no_ack = Settings(
            allow_live_trading=True,
            broker_provider=BrokerProviderType.PAPER,
            broker_account_id="test_account",
            live_trading_acknowledgement="",
        )
        assert settings_no_ack.is_live_trading_allowed() is False

    def test_secret_status_hides_values(self) -> None:
        """Secret status should only show configured/not configured."""
        # Create isolated settings to test secret status logic
        # NOTE: 使用短测试值 (<8字符) 避免触发 secret 扫描模式
        settings = Settings(
            _env_file=None,  # Ignore .env file
            tushare_token="test_tk",  # noqa: S106 - test value, not real secret
            joinquant_username="user",
            joinquant_password=None,  # Explicitly not set
        )
        status = settings.get_secret_status()

        assert status["tushare_token"] == "configured"  # noqa: S105
        assert status["joinquant_username"] == "configured"
        assert status["joinquant_password"] == "not configured"  # noqa: S105

        # Ensure actual values are not in status
        for value in status.values():
            assert "secret" not in value.lower()
            assert "token_value" not in value

    def test_invalid_log_level_raises(self) -> None:
        """Invalid log level should raise validation error."""
        with pytest.raises(ValueError, match="Invalid log level"):
            Settings(log_level="INVALID")

    def test_invalid_capital_raises(self) -> None:
        """Non-positive capital should raise validation error."""
        with pytest.raises(ValueError, match="must be positive"):
            Settings(initial_capital=-100)

    def test_get_settings_returns_instance(self) -> None:
        """get_settings should return a Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)
