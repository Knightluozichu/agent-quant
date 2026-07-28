"""Trade plan domain models.

A TradePlan is the complete specification for a potential trade.
Any BUY signal without a complete exit plan is INVALID.

The TradePlan must include:
- Entry conditions and price range
- Stop loss (initial and trailing)
- Take profit targets
- Maximum holding period
- Regime invalidation rules
- Risk amount and expected reward
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from a_share_quant.domain.market_state import MarketState
from a_share_quant.domain.stock_state import StockState
from a_share_quant.domain.strategy_decision import StrategyId


class TradePlanStatus(StrEnum):
    """Status of a trade plan."""

    PENDING = "pending"  # 等待执行
    ACTIVE = "active"  # 已激活（持仓中）
    COMPLETED = "completed"  # 已完成
    CANCELLED = "cancelled"  # 已取消
    EXPIRED = "expired"  # 已过期
    REJECTED = "rejected"  # 被拒绝


class EntryTrigger(StrEnum):
    """Entry trigger types."""

    MARKET_OPEN = "market_open"  # 开盘价
    LIMIT_PRICE = "limit_price"  # 限价
    BREAKOUT = "breakout"  # 突破
    PULLBACK_CONFIRM = "pullback_confirm"  # 回调确认
    MEAN_REVERSION = "mean_reversion"  # 均值回归


class ExitReason(StrEnum):
    """Exit reason codes."""

    # Profit taking
    TAKE_PROFIT_1 = "take_profit_1"  # 第一止盈
    TAKE_PROFIT_2 = "take_profit_2"  # 第二止盈
    TRAILING_STOP = "trailing_stop"  # 移动止盈

    # Stop loss
    INITIAL_STOP = "initial_stop"  # 初始止损
    STRUCTURE_BREAK = "structure_break"  # 结构破位
    VOLATILITY_STOP = "volatility_stop"  # 波动率止损

    # Time-based
    TIME_STOP = "time_stop"  # 时间止损
    MAX_HOLDING = "max_holding"  # 最大持有期

    # Regime/strategy
    REGIME_INVALIDATION = "regime_invalidation"  # 状态失效
    STRATEGY_SWITCH = "strategy_switch"  # 策略切换

    # System
    CIRCUIT_BREAKER = "circuit_breaker"  # 熔断
    DATA_ERROR = "data_error"  # 数据异常
    MANUAL = "manual"  # 人工干预


class TrailingStopRule(BaseModel):
    """Trailing stop configuration."""

    model_config = {"frozen": True}

    enabled: bool = True
    atr_multiplier: float = Field(
        default=2.0,
        gt=0,
        description="ATR multiplier for trailing distance",
    )
    use_structure: bool = Field(
        default=True,
        description="Use structural highs/lows for trailing",
    )
    only_moves_up: bool = Field(
        default=True,
        description="Trailing stop can only move up (for long positions)",
    )
    activation_profit: float = Field(
        default=0.0,
        ge=0,
        description="Minimum profit before trailing activates (as ratio)",
    )


class TradePlan(BaseModel):
    """Complete trade plan with all exit conditions.

    INVARIANT: Any BUY signal without a complete TradePlan is INVALID.
    The TradePlan must specify stop loss, take profit, max holding,
    and invalidation conditions BEFORE entry.
    """

    model_config = {"frozen": True}

    # Identity
    trade_plan_id: str = Field(description="Unique trade plan identifier")
    symbol: str = Field(description="Symbol to trade")
    strategy_id: StrategyId = Field(description="Strategy that generated this plan")

    # Timing
    signal_time: datetime = Field(description="When the signal was generated")
    earliest_execution_time: datetime = Field(
        description="Earliest time to execute (T+1 open for A-shares)"
    )
    earliest_sell_time: datetime = Field(
        description="Earliest time to sell (T+1 after buy for A-shares)"
    )
    expiry_time: Optional[datetime] = Field(
        default=None,
        description="Plan expires if not executed by this time",
    )

    # State at signal time
    market_state: MarketState = Field(description="Market state at signal time")
    stock_state: StockState = Field(description="Stock state at signal time")

    # Versioning
    strategy_version: str = Field(default="v1")
    model_version: str = Field(default="v1")
    factor_snapshot_id: Optional[str] = Field(
        default=None,
        description="Reference to factor snapshot used",
    )

    # Entry specification
    entry_trigger: EntryTrigger = Field(description="How to trigger entry")
    entry_price_range: tuple[float, float] = Field(
        description="Acceptable entry price range (min, max)"
    )
    target_quantity: int = Field(
        gt=0,
        description="Target quantity in shares (must be multiple of lot size)",
    )
    target_weight: float = Field(
        ge=0,
        le=1,
        description="Target portfolio weight",
    )

    # Exit specification - STOP LOSS (required)
    initial_stop_price: float = Field(
        gt=0,
        description="Initial stop loss price (must be set before entry)",
    )
    stop_reason: str = Field(
        description="Why this stop level was chosen",
    )

    # Exit specification - TAKE PROFIT (required)
    take_profit_1: float = Field(
        gt=0,
        description="First take profit target price",
    )
    take_profit_1_ratio: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Ratio of position to close at TP1",
    )
    take_profit_2: Optional[float] = Field(
        default=None,
        gt=0,
        description="Second take profit target price",
    )
    trailing_stop_rule: TrailingStopRule = Field(
        default_factory=TrailingStopRule,
        description="Trailing stop configuration",
    )

    # Exit specification - TIME (required)
    max_holding_days: int = Field(
        gt=0,
        description="Maximum holding period in trading days",
    )

    # Exit specification - INVALIDATION (required)
    regime_invalidation_rule: str = Field(
        description="Conditions that invalidate this plan due to regime change",
    )
    signal_reset_rule: str = Field(
        default="new_signal_required",
        description="Rule for signal reset before re-entry",
    )
    cooldown_rule: str = Field(
        default="standard",
        description="Cooldown rule after exit",
    )

    # Risk specification
    estimated_risk_amount: float = Field(
        gt=0,
        description="Estimated risk amount in currency",
    )
    expected_reward: float = Field(
        description="Expected reward in currency",
    )
    expected_cost: float = Field(
        ge=0,
        description="Expected transaction costs",
    )
    reward_risk_ratio: float = Field(
        description="Expected reward / risk ratio",
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Confidence in this trade plan",
    )

    # Status
    status: TradePlanStatus = Field(
        default=TradePlanStatus.PENDING,
        description="Current status of the plan",
    )
    rejection_reason: Optional[str] = Field(
        default=None,
        description="Reason if plan was rejected",
    )

    @model_validator(mode="after")
    def validate_plan_completeness(self) -> TradePlan:
        """Validate that the plan has all required exit conditions."""
        # Stop loss must be below entry range for long positions
        entry_min = self.entry_price_range[0]
        if self.initial_stop_price >= entry_min:
            msg = (
                f"Stop price {self.initial_stop_price} must be below "
                f"entry range minimum {entry_min}"
            )
            raise ValueError(msg)

        # Take profit must be above entry range
        entry_max = self.entry_price_range[1]
        if self.take_profit_1 <= entry_max:
            msg = (
                f"Take profit 1 {self.take_profit_1} must be above "
                f"entry range maximum {entry_max}"
            )
            raise ValueError(msg)

        # Reward/risk ratio should be reasonable
        if self.reward_risk_ratio < 1.0:
            msg = f"Reward/risk ratio {self.reward_risk_ratio} should be >= 1.0"
            raise ValueError(msg)

        return self

    @property
    def risk_per_share(self) -> float:
        """Calculate risk per share."""
        entry_mid = (self.entry_price_range[0] + self.entry_price_range[1]) / 2
        return entry_mid - self.initial_stop_price

    @property
    def is_valid_for_execution(self) -> bool:
        """Check if plan is valid for execution."""
        return self.status == TradePlanStatus.PENDING
