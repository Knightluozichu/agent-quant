"""Market state domain models.

Implements the 9-state market model: Direction × Oscillation Level.

| Direction | Low Oscillation | Medium Oscillation | High Oscillation |
|-----------|-----------------|--------------------|------------------|
| UP        | Smooth Uptrend  | Staircase Uptrend  | Volatile Uptrend |
| FLAT      | Dead Water      | Narrow Range       | Wide Range       |
| DOWN      | Smooth Downtrend| Staircase Downtrend| Volatile Downtrend|
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class MarketDirection(StrEnum):
    """Market direction."""

    UP = "up"
    FLAT = "flat"
    DOWN = "down"


class OscillationLevel(StrEnum):
    """Oscillation/volatility level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MarketRegime(StrEnum):
    """The 9 market regimes (Direction × Oscillation).

    Each regime has specific characteristics and suitable strategies.
    """

    # UP direction
    SMOOTH_UPTREND = "smooth_uptrend"  # 平滑上涨
    STAIRCASE_UPTREND = "staircase_uptrend"  # 阶梯上涨
    VOLATILE_UPTREND = "volatile_uptrend"  # 震荡洗盘上涨

    # FLAT direction
    DEAD_WATER = "dead_water"  # 低波死水
    NARROW_RANGE = "narrow_range"  # 窄幅横盘
    WIDE_RANGE = "wide_range"  # 宽幅周期震荡

    # DOWN direction
    SMOOTH_DOWNTREND = "smooth_downtrend"  # 平滑下跌
    STAIRCASE_DOWNTREND = "staircase_downtrend"  # 阶梯下跌
    VOLATILE_DOWNTREND = "volatile_downtrend"  # 强反弹式震荡下跌

    # Unknown/transition
    UNKNOWN = "unknown"

    @classmethod
    def from_direction_oscillation(
        cls, direction: MarketDirection, oscillation: OscillationLevel
    ) -> MarketRegime:
        """Create regime from direction and oscillation level."""
        mapping = {
            (MarketDirection.UP, OscillationLevel.LOW): cls.SMOOTH_UPTREND,
            (MarketDirection.UP, OscillationLevel.MEDIUM): cls.STAIRCASE_UPTREND,
            (MarketDirection.UP, OscillationLevel.HIGH): cls.VOLATILE_UPTREND,
            (MarketDirection.FLAT, OscillationLevel.LOW): cls.DEAD_WATER,
            (MarketDirection.FLAT, OscillationLevel.MEDIUM): cls.NARROW_RANGE,
            (MarketDirection.FLAT, OscillationLevel.HIGH): cls.WIDE_RANGE,
            (MarketDirection.DOWN, OscillationLevel.LOW): cls.SMOOTH_DOWNTREND,
            (MarketDirection.DOWN, OscillationLevel.MEDIUM): cls.STAIRCASE_DOWNTREND,
            (MarketDirection.DOWN, OscillationLevel.HIGH): cls.VOLATILE_DOWNTREND,
        }
        return mapping.get((direction, oscillation), cls.UNKNOWN)

    @property
    def direction(self) -> MarketDirection:
        """Get the direction component."""
        if self in (
            MarketRegime.SMOOTH_UPTREND,
            MarketRegime.STAIRCASE_UPTREND,
            MarketRegime.VOLATILE_UPTREND,
        ):
            return MarketDirection.UP
        if self in (
            MarketRegime.DEAD_WATER,
            MarketRegime.NARROW_RANGE,
            MarketRegime.WIDE_RANGE,
        ):
            return MarketDirection.FLAT
        if self in (
            MarketRegime.SMOOTH_DOWNTREND,
            MarketRegime.STAIRCASE_DOWNTREND,
            MarketRegime.VOLATILE_DOWNTREND,
        ):
            return MarketDirection.DOWN
        return MarketDirection.FLAT  # Default for UNKNOWN

    @property
    def oscillation(self) -> OscillationLevel:
        """Get the oscillation component."""
        if self in (
            MarketRegime.SMOOTH_UPTREND,
            MarketRegime.DEAD_WATER,
            MarketRegime.SMOOTH_DOWNTREND,
        ):
            return OscillationLevel.LOW
        if self in (
            MarketRegime.STAIRCASE_UPTREND,
            MarketRegime.NARROW_RANGE,
            MarketRegime.STAIRCASE_DOWNTREND,
        ):
            return OscillationLevel.MEDIUM
        if self in (
            MarketRegime.VOLATILE_UPTREND,
            MarketRegime.WIDE_RANGE,
            MarketRegime.VOLATILE_DOWNTREND,
        ):
            return OscillationLevel.HIGH
        return OscillationLevel.MEDIUM  # Default for UNKNOWN

    @property
    def is_bullish(self) -> bool:
        """Check if regime is bullish."""
        return self.direction == MarketDirection.UP

    @property
    def is_bearish(self) -> bool:
        """Check if regime is bearish."""
        return self.direction == MarketDirection.DOWN

    @property
    def is_tradeable(self) -> bool:
        """Check if regime allows new positions (not dead water or smooth downtrend)."""
        return self not in (
            MarketRegime.DEAD_WATER,
            MarketRegime.SMOOTH_DOWNTREND,
            MarketRegime.UNKNOWN,
        )


class MarketState(BaseModel):
    """Current market state assessment.

    This is the output of the regime detection model, containing:
    - The detected regime and its probability distribution
    - Confidence level
    - Whether a transition was detected
    - Explanation of triggering features
    """

    model_config = {"frozen": True}

    as_of: datetime = Field(description="Timestamp of this assessment")
    direction: MarketDirection = Field(description="Market direction")
    oscillation: OscillationLevel = Field(description="Oscillation level")
    regime: MarketRegime = Field(description="The 9-state regime")
    probabilities: dict[str, float] = Field(
        default_factory=dict,
        description="Probability distribution over all 9 regimes",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the current regime assessment",
    )
    transition_detected: bool = Field(
        default=False,
        description="Whether a regime transition was detected",
    )
    previous_regime: Optional[MarketRegime] = Field(
        default=None,
        description="Previous regime if transition detected",
    )
    model_version: str = Field(
        default="rule_v1",
        description="Version of the regime detection model",
    )
    triggering_features: dict[str, float] = Field(
        default_factory=dict,
        description="Feature values that triggered this assessment",
    )
    explanation: str = Field(
        default="",
        description="Human-readable explanation",
    )

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(cls, v: dict[str, float]) -> dict[str, float]:
        """Ensure probabilities sum to approximately 1.0 if non-empty."""
        if v:
            total = sum(v.values())
            if abs(total - 1.0) > 0.01:
                msg = f"Probabilities must sum to 1.0, got {total}"
                raise ValueError(msg)
        return v

    @classmethod
    def create(
        cls,
        as_of: datetime,
        direction: MarketDirection,
        oscillation: OscillationLevel,
        confidence: float = 1.0,
        probabilities: dict[str, float] | None = None,
        model_version: str = "rule_v1",
        **kwargs: object,
    ) -> MarketState:
        """Create a MarketState with derived regime."""
        regime = MarketRegime.from_direction_oscillation(direction, oscillation)
        return cls(
            as_of=as_of,
            direction=direction,
            oscillation=oscillation,
            regime=regime,
            confidence=confidence,
            probabilities=probabilities or {},
            model_version=model_version,
            **kwargs,  # type: ignore[arg-type]
        )
