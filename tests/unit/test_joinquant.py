"""Tests for JoinQuant data provider."""

import importlib.util
from datetime import date

import pytest

from a_share_quant.data.providers.joinquant import (
    QuotaTracker,
    from_jq_symbol,
    to_jq_symbol,
)

_HAS_JQDATASDK = importlib.util.find_spec("jqdatasdk") is not None
requires_jqdatasdk = pytest.mark.skipif(
    not _HAS_JQDATASDK,
    reason="jqdatasdk not installed (uv sync --extra data-joinquant)",
)


class TestSymbolConversion:
    """Tests for symbol format conversion."""

    def test_to_jq_symbol_sse(self):
        assert to_jq_symbol("510300.SSE") == "510300.XSHG"
        assert to_jq_symbol("600519.SSE") == "600519.XSHG"

    def test_to_jq_symbol_szse(self):
        assert to_jq_symbol("000001.SZSE") == "000001.XSHE"
        assert to_jq_symbol("159915.SZSE") == "159915.XSHE"

    def test_from_jq_symbol_xshg(self):
        assert from_jq_symbol("510300.XSHG") == "510300.SSE"
        assert from_jq_symbol("600519.XSHG") == "600519.SSE"

    def test_from_jq_symbol_xshe(self):
        assert from_jq_symbol("000001.XSHE") == "000001.SZSE"
        assert from_jq_symbol("159915.XSHE") == "159915.SZSE"

    def test_roundtrip(self):
        symbols = ["510300.SSE", "000001.SZSE", "600519.SSE", "159915.SZSE"]
        for s in symbols:
            assert from_jq_symbol(to_jq_symbol(s)) == s


class TestQuotaTracker:
    """Tests for quota tracking."""

    def test_initial_state(self):
        tracker = QuotaTracker(daily_limit=500_000)
        assert tracker.remaining == 500_000
        assert tracker.used == 0

    def test_consume(self):
        tracker = QuotaTracker(daily_limit=1000)
        assert tracker.consume(100)
        assert tracker.used == 100
        assert tracker.remaining == 900

    def test_consume_exceeds_limit(self):
        tracker = QuotaTracker(daily_limit=100)
        assert tracker.consume(50)
        assert not tracker.consume(60)  # Would exceed
        assert tracker.used == 50  # Unchanged


class TestJoinQuantProviderIntegration:
    """Integration tests for JoinQuant provider.

    These tests require valid credentials in .env
    Mark with @pytest.mark.integration to skip in CI
    """

    @pytest.mark.integration
    @requires_jqdatasdk
    def test_authentication(self):
        """Test JQData authentication."""
        from a_share_quant.data.providers.joinquant import JoinQuantProvider

        provider = JoinQuantProvider()
        provider._ensure_auth()
        assert provider._authenticated

    @pytest.mark.integration
    @requires_jqdatasdk
    def test_get_trading_calendar(self):
        """Test trading calendar retrieval."""
        from a_share_quant.data.providers.joinquant import JoinQuantProvider

        provider = JoinQuantProvider()
        cal = provider.get_trading_calendar("SSE", date(2024, 1, 1), date(2024, 1, 31))

        assert len(cal) > 0
        assert "trade_date" in cal.columns

    @pytest.mark.integration
    @requires_jqdatasdk
    def test_get_daily_bars(self):
        """Test daily bars retrieval."""
        from a_share_quant.data.providers.joinquant import JoinQuantProvider

        provider = JoinQuantProvider()
        df = provider.get_daily_bars("510300.SSE", date(2024, 1, 1), date(2024, 1, 31))

        assert len(df) > 0
        assert "close" in df.columns

    @pytest.mark.integration
    @requires_jqdatasdk
    def test_quota_info(self):
        """Test quota info retrieval."""
        from a_share_quant.data.providers.joinquant import JoinQuantProvider

        provider = JoinQuantProvider()
        info = provider.get_quota_info()

        assert "tracked_remaining" in info
