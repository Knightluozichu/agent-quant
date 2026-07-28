"""Tests for configuration loading and management."""

from __future__ import annotations

from pathlib import Path

import pytest

from a_share_quant.config import (
    ConfigLoader,
    load_market_rules_config,
    load_protocol_config,
    load_regimes_config,
    load_risk_config,
)


@pytest.fixture
def config_loader() -> ConfigLoader:
    """Create a config loader for the configs directory."""
    return ConfigLoader(Path(__file__).parent.parent.parent / "configs")


@pytest.mark.unit
class TestConfigLoader:
    """Test ConfigLoader class."""

    def test_load_protocol(self, config_loader: ConfigLoader) -> None:
        """Test loading protocol config."""
        config = config_loader.load("protocol.yaml")
        assert "protocol" in config
        assert config["protocol"]["version"] == "1.0"

    def test_load_market_rules(self, config_loader: ConfigLoader) -> None:
        """Test loading market rules config."""
        config = config_loader.load("market_rules_cn.yaml")
        assert "market_rules" in config
        assert config["market_rules"]["t_plus_one"]["enabled"] is True

    def test_load_risk(self, config_loader: ConfigLoader) -> None:
        """Test loading risk config."""
        config = config_loader.load("risk.yaml")
        assert "risk" in config
        assert "paper_conservative" in config["risk"]["profiles"]

    def test_load_regimes(self, config_loader: ConfigLoader) -> None:
        """Test loading regimes config."""
        config = config_loader.load("regimes.yaml")
        assert "regimes" in config
        assert len(config["regimes"]["states"]) == 9

    def test_load_all_strategies(self, config_loader: ConfigLoader) -> None:
        """Test loading all strategy configs."""
        strategies = config_loader.load_all_strategies()
        assert len(strategies) == 5
        assert "TREND_HOLD" in strategies
        assert "CASH_DEFENSE" in strategies

    def test_config_hash_stability(self, config_loader: ConfigLoader) -> None:
        """Test that config hash is stable for same content."""
        config1 = config_loader.load("protocol.yaml", use_cache=False)
        config2 = config_loader.load("protocol.yaml", use_cache=False)

        hash1 = ConfigLoader.compute_hash(config1)
        hash2 = ConfigLoader.compute_hash(config2)

        assert hash1 == hash2
        assert len(hash1) == 16

    def test_config_merge(self, config_loader: ConfigLoader) -> None:
        """Test config merging."""
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 10}, "e": 5}

        merged = config_loader.merge(base, override)

        assert merged["a"] == 1
        assert merged["b"]["c"] == 10  # Overridden
        assert merged["b"]["d"] == 3  # Preserved
        assert merged["e"] == 5  # Added

    def test_load_with_overrides(self, config_loader: ConfigLoader) -> None:
        """Test loading with overrides."""
        config = config_loader.load_with_overrides(
            "protocol.yaml",
            {"protocol": {"version": "2.0"}},
        )
        assert config["protocol"]["version"] == "2.0"

    def test_missing_config_raises(self, config_loader: ConfigLoader) -> None:
        """Test that missing config raises error."""
        with pytest.raises(FileNotFoundError):
            config_loader.load("nonexistent.yaml")


@pytest.mark.unit
class TestConfigFunctions:
    """Test config loading functions."""

    def test_load_protocol_config(self) -> None:
        """Test load_protocol_config function."""
        config = load_protocol_config()
        assert "protocol" in config

    def test_load_market_rules_config(self) -> None:
        """Test load_market_rules_config function."""
        config = load_market_rules_config()
        assert "market_rules" in config

    def test_load_risk_config(self) -> None:
        """Test load_risk_config function."""
        config = load_risk_config()
        assert "risk" in config

    def test_load_regimes_config(self) -> None:
        """Test load_regimes_config function."""
        config = load_regimes_config()
        assert "regimes" in config
