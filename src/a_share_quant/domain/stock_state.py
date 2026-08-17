"""Stock state domain models.

Represents the current state of an individual stock/ETF,
including its trading status, trend, and technical characteristics.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class TradingStatus(StrEnum):
    """Trading status of a security."""

    NORMAL = "normal"  # 正常交易
    SUSPENDED = "suspended"  # 停牌
    LIMIT_UP = "limit_up"  # 涨停
    LIMIT_DOWN = "limit_down"  # 跌停
    DELISTING = "delisting"  # 退市整理
    NEW_LISTING = "new_listing"  # 新股上市初期


class StockTrend(StrEnum):
    """Individual stock trend direction."""

    STRONG_UP = "strong_up"  # 强势上涨
    UP = "up"  # 上涨
    FLAT = "flat"  # 横盘
    DOWN = "down"  # 下跌
    STRONG_DOWN = "strong_down"  # 强势下跌


class StockState(BaseModel):
    """Current state of an individual stock/ETF.

    Captures trading status, trend, and key technical levels
    needed for strategy decisions.
    """

    model_config = {"frozen": True}

    symbol: str = Field(description="Symbol string like '600519.SSE'")
    as_of: datetime = Field(description="Timestamp of this assessment")

    # Trading status
    trading_status: TradingStatus = Field(
        default=TradingStatus.NORMAL,
        description="Current trading status",
    )
    is_tradeable: bool = Field(
        default=True,
        description="Whether the stock can be traded",
    )
    can_buy: bool = Field(
        default=True,
        description="Whether new buy orders can be placed",
    )
    can_sell: bool = Field(
        default=True,
        description="Whether existing positions can be sold",
    )

    # Trend
    trend: StockTrend = Field(
        default=StockTrend.FLAT,
        description="Individual stock trend",
    )
    trend_strength: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Trend strength from -1 (strong down) to 1 (strong up)",
    )

    # Price levels
    current_price: float = Field(description="Current/latest price")
    prev_close: Optional[float] = Field(
        default=None,
        description="Previous close price",
    )
    limit_up_price: Optional[float] = Field(
        default=None,
        description="Today's limit up price",
    )
    limit_down_price: Optional[float] = Field(
        default=None,
        description="Today's limit down price",
    )

    # Technical levels
    support_level: Optional[float] = Field(
        default=None,
        description="Key support level",
    )
    resistance_level: Optional[float] = Field(
        default=None,
        description="Key resistance level",
    )
    atr: Optional[float] = Field(
        default=None,
        description="Average True Range",
    )

    # Volume and liquidity
    volume_ratio: Optional[float] = Field(
        default=None,
        description="Volume ratio vs average",
    )
    turnover_rate: Optional[float] = Field(
        default=None,
        description="Turnover rate (换手率)",
    )
    liquidity_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Liquidity score from 0 (illiquid) to 1 (liquid)",
    )

    # Suspension info
    suspension_reason: Optional[str] = Field(
        default=None,
        description="Reason for suspension if applicable",
    )
    expected_resume_date: Optional[datetime] = Field(
        default=None,
        description="Expected resumption date if known",
    )

    # Days in current status
    days_suspended: int = Field(
        default=0,
        ge=0,
        description="Number of days suspended",
    )
    days_since_ipo: Optional[int] = Field(
        default=None,
        ge=0,
        description="Days since IPO (for new listing rules)",
    )

    @model_validator(mode="after")
    def set_tradeability(self) -> StockState:
        """Set tradeability flags based on trading status."""
        is_normal = self.trading_status == TradingStatus.NORMAL
        # Use object.__setattr__ since model is frozen
        object.__setattr__(self, "is_tradeable", is_normal)
        if not is_normal:
            object.__setattr__(self, "can_buy", False)
            object.__setattr__(self, "can_sell", False)
        return self

    def update_tradeability(self) -> StockState:
        """Return a new StockState with updated tradeability flags based on price limits."""
        can_buy = self.trading_status == TradingStatus.NORMAL and self.current_price < (
            self.limit_up_price or float("inf")
        )
        can_sell = self.trading_status == TradingStatus.NORMAL and self.current_price > (
            self.limit_down_price or 0
        )

        return self.model_copy(
            update={
                "can_buy": can_buy,
                "can_sell": can_sell,
            }
        )

    @property
    def is_at_limit_up(self) -> bool:
        """Check if stock is at limit up."""
        if self.limit_up_price is None:
            return False
        return abs(self.current_price - self.limit_up_price) < 0.01

    @property
    def is_at_limit_down(self) -> bool:
        """Check if stock is at limit down."""
        if self.limit_down_price is None:
            return False
        return abs(self.current_price - self.limit_down_price) < 0.01

    @property
    def is_suspended(self) -> bool:
        """Check if stock is suspended."""
        return self.trading_status == TradingStatus.SUSPENDED
