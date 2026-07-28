"""Configuration loading and management.

Provides:
- YAML config loading with validation
- Config merging and override support
- Config hashing for reproducibility
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class ConfigMeta(BaseModel):
    """Metadata about a loaded configuration."""

    source_path: Optional[str] = None
    config_hash: str = ""
    loaded_at: str = ""
    version: str = "1.0"


class ConfigLoader:
    """Load and manage YAML configurations.

    Supports:
    - Loading from file paths
    - Merging multiple configs
    - Override with dict
    - Hashing for reproducibility
    """

    def __init__(self, config_dir: Path | str = "configs") -> None:
        self.config_dir = Path(config_dir)
        self._cache: dict[str, dict[str, Any]] = {}

    def load(self, filename: str, use_cache: bool = True) -> dict[str, Any]:
        """Load a YAML config file.

        Args:
            filename: Config filename (relative to config_dir)
            use_cache: Whether to use cached version

        Returns:
            Parsed config dict
        """
        cache_key = str(filename)
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        path = self.config_dir / filename
        if not path.exists():
            msg = f"Config file not found: {path}"
            raise FileNotFoundError(msg)

        with path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        if use_cache:
            self._cache[cache_key] = config

        return config

    def load_all_strategies(self) -> dict[str, dict[str, Any]]:
        """Load all strategy configurations.

        Returns:
            Dict mapping strategy_id to config
        """
        strategies_dir = self.config_dir / "strategies"
        if not strategies_dir.exists():
            return {}

        strategies = {}
        for yaml_file in strategies_dir.glob("*.yaml"):
            config = self.load(f"strategies/{yaml_file.name}")
            strategy_id = config.get("strategy", {}).get("id", yaml_file.stem)
            strategies[strategy_id] = config

        return strategies

    def merge(
        self,
        base: dict[str, Any],
        override: dict[str, Any],
    ) -> dict[str, Any]:
        """Deep merge two config dicts.

        Args:
            base: Base configuration
            override: Override configuration (takes precedence)

        Returns:
            Merged configuration
        """
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self.merge(result[key], value)
            else:
                result[key] = value

        return result

    def load_with_overrides(
        self,
        filename: str,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Load config with optional overrides.

        Args:
            filename: Config filename
            overrides: Dict of overrides to apply

        Returns:
            Config with overrides applied
        """
        config = self.load(filename, use_cache=False)
        if overrides:
            config = self.merge(config, overrides)
        return config

    @staticmethod
    def compute_hash(config: dict[str, Any]) -> str:
        """Compute a stable hash for a config dict.

        The hash is computed from the JSON representation
        with sorted keys for stability.

        Args:
            config: Configuration dict

        Returns:
            SHA256 hash string (first 16 chars)
        """
        # Convert to JSON with sorted keys for stable hashing
        json_str = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def get_config_with_meta(
        self,
        filename: str,
        overrides: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], ConfigMeta]:
        """Load config and return with metadata.

        Args:
            filename: Config filename
            overrides: Optional overrides

        Returns:
            Tuple of (config, metadata)
        """
        from datetime import datetime, timezone

        config = self.load_with_overrides(filename, overrides)
        config_hash = self.compute_hash(config)

        meta = ConfigMeta(
            source_path=str(self.config_dir / filename),
            config_hash=config_hash,
            loaded_at=datetime.now(timezone.utc).isoformat(),
            version=config.get("version", "1.0"),
        )

        return config, meta

    def clear_cache(self) -> None:
        """Clear the config cache."""
        self._cache.clear()


# Global config loader instance
_default_loader: ConfigLoader | None = None


def get_config_loader(config_dir: Path | str = "configs") -> ConfigLoader:
    """Get or create the default config loader.

    Args:
        config_dir: Configuration directory path

    Returns:
        ConfigLoader instance
    """
    global _default_loader
    if _default_loader is None or _default_loader.config_dir != Path(config_dir):
        _default_loader = ConfigLoader(config_dir)
    return _default_loader


def load_protocol_config(config_dir: Path | str = "configs") -> dict[str, Any]:
    """Load the research protocol configuration."""
    loader = get_config_loader(config_dir)
    return loader.load("protocol.yaml")


def load_market_rules_config(config_dir: Path | str = "configs") -> dict[str, Any]:
    """Load the market rules configuration."""
    loader = get_config_loader(config_dir)
    return loader.load("market_rules_cn.yaml")


def load_risk_config(config_dir: Path | str = "configs") -> dict[str, Any]:
    """Load the risk management configuration."""
    loader = get_config_loader(config_dir)
    return loader.load("risk.yaml")


def load_regimes_config(config_dir: Path | str = "configs") -> dict[str, Any]:
    """Load the regime detection configuration."""
    loader = get_config_loader(config_dir)
    return loader.load("regimes.yaml")


def load_universe_config(config_dir: Path | str = "configs") -> dict[str, Any]:
    """Load the universe configuration."""
    loader = get_config_loader(config_dir)
    return loader.load("universe.yaml")


def load_strategy_config(
    strategy_id: str,
    config_dir: Path | str = "configs",
) -> dict[str, Any]:
    """Load a specific strategy configuration.

    Args:
        strategy_id: Strategy identifier (e.g., "TREND_HOLD")
        config_dir: Configuration directory path

    Returns:
        Strategy configuration dict
    """
    loader = get_config_loader(config_dir)
    # Convert strategy_id to filename
    filename = strategy_id.lower().replace("_", "_") + ".yaml"
    return loader.load(f"strategies/{filename}")
