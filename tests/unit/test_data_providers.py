"""Tests for data providers."""

from __future__ import annotations

from datetime import date

import pytest

from a_share_quant.data.providers import MockProvider, get_data_provider
from a_share_quant.data.schemas import validate_daily_bar_df


@pytest.fixture
def mock_provider() -> MockProvider:
    """Create a mock provider."""
    return MockProvider(seed=20260720)


@pytest.mark.unit
class TestMockProvider:
    """Test MockProvider."""

    def test_name(self, mock_provider: MockProvider) -> None:
        assert mock_provider.name == "mock"

    def test_trading_calendar(self, mock_provider: MockProvider) -> None:
        cal = mock_provider.get_trading_calendar(
            start_date="20260101",
            end_date="20260131",
        )
        assert len(cal) > 0
        assert "trade_date" in cal.columns
        assert "is_open" in cal.columns
        # Weekends should be excluded
        for d in cal["trade_date"]:
            assert d.weekday() < 5

    def test_security_list(self, mock_provider: MockProvider) -> None:
        securities = mock_provider.get_security_list()
        assert len(securities) >= 5
        assert "symbol" in securities.columns
        assert "security_type" in securities.columns

    def test_security_list_filter(self, mock_provider: MockProvider) -> None:
        etfs = mock_provider.get_security_list(security_type="etf")
        assert len(etfs) == 3
        assert all(etfs["security_type"] == "etf")

    def test_security_info(self, mock_provider: MockProvider) -> None:
        info = mock_provider.get_security_info("510300.SSE")
        assert info is not None
        assert info.name == "沪深300ETF"
        assert info.security_type == "etf"

    def test_security_info_not_found(self, mock_provider: MockProvider) -> None:
        info = mock_provider.get_security_info("999999.SSE")
        assert info is None

    def test_daily_bars(self, mock_provider: MockProvider) -> None:
        bars = mock_provider.get_daily_bars(
            "510300.SSE",
            start_date="20260101",
            end_date="20260131",
        )
        assert len(bars) > 0
        assert "symbol" in bars.columns
        assert "trade_date" in bars.columns
        assert "open" in bars.columns
        assert "high" in bars.columns
        assert "low" in bars.columns
        assert "close" in bars.columns
        assert "volume" in bars.columns

    def test_daily_bars_ohlc_consistency(self, mock_provider: MockProvider) -> None:
        """High >= Low, High >= Open, High >= Close, etc."""
        bars = mock_provider.get_daily_bars(
            "510300.SSE",
            start_date="20260101",
            end_date="20260331",
        )
        assert (bars["high"] >= bars["low"]).all()
        assert (bars["high"] >= bars["open"]).all()
        assert (bars["high"] >= bars["close"]).all()
        assert (bars["low"] <= bars["open"]).all()
        assert (bars["low"] <= bars["close"]).all()

    def test_daily_bars_multiple_symbols(self, mock_provider: MockProvider) -> None:
        bars = mock_provider.get_daily_bars(
            ["510300.SSE", "510500.SSE"],
            start_date="20260101",
            end_date="20260131",
        )
        symbols = bars["symbol"].unique()
        assert len(symbols) == 2

    def test_daily_bars_deterministic(self) -> None:
        """Same seed should produce same data."""
        provider1 = MockProvider(seed=12345)
        provider2 = MockProvider(seed=12345)

        bars1 = provider1.get_daily_bars("510300.SSE", "20260101", "20260131")
        bars2 = provider2.get_daily_bars("510300.SSE", "20260101", "20260131")

        assert bars1.equals(bars2)

    def test_is_trading_day(self, mock_provider: MockProvider) -> None:
        # 2026-07-20 is a Monday
        assert mock_provider.is_trading_day("20260720")
        # 2026-07-19 is a Sunday
        assert not mock_provider.is_trading_day("20260719")

    def test_next_trading_day(self, mock_provider: MockProvider) -> None:
        # Friday -> Monday
        next_day = mock_provider.get_next_trading_day("20260717")  # Friday
        assert next_day == date(2026, 7, 20)  # Monday

    def test_prev_trading_day(self, mock_provider: MockProvider) -> None:
        # Monday -> Friday
        prev_day = mock_provider.get_prev_trading_day("20260720")  # Monday
        assert prev_day == date(2026, 7, 17)  # Friday

    def test_adjustment_factors(self, mock_provider: MockProvider) -> None:
        factors = mock_provider.get_adjustment_factors(
            "510300.SSE",
            "20260101",
            "20260131",
        )
        assert len(factors) > 0
        assert (factors["factor"] == 1.0).all()

    def test_suspension_info(self, mock_provider: MockProvider) -> None:
        suspensions = mock_provider.get_suspension_info(
            "510300.SSE",
            "20260101",
            "20260131",
        )
        assert len(suspensions) == 0  # Mock has no suspensions

    def test_industry_classification(self, mock_provider: MockProvider) -> None:
        industries = mock_provider.get_industry_classification()
        assert len(industries) > 0
        assert "industry_name" in industries.columns


@pytest.mark.unit
class TestDataProviderFactory:
    """Test data provider factory."""

    def test_get_mock_provider(self) -> None:
        provider = get_data_provider("mock")
        assert provider.name == "mock"

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            get_data_provider("unknown")


@pytest.mark.unit
class TestDataValidation:
    """Test data validation functions."""

    def test_validate_valid_df(self, mock_provider: MockProvider) -> None:
        bars = mock_provider.get_daily_bars("510300.SSE", "20260101", "20260131")
        errors = validate_daily_bar_df(bars)
        assert len(errors) == 0

    def test_validate_missing_columns(self) -> None:
        import pandas as pd

        df = pd.DataFrame({"symbol": ["510300.SSE"], "close": [100.0]})
        errors = validate_daily_bar_df(df)
        assert any("Missing columns" in e for e in errors)
