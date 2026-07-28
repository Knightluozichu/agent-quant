"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest

from a_share_quant.settings import Settings


@pytest.fixture
def settings() -> Settings:
    """Create a settings instance with mock defaults."""
    return Settings(
        data_provider="mock",
        dry_run=True,
        allow_live_trading=False,
    )
