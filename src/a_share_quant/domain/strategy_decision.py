"""Strategy decision domain models.

Represents the outcome of strategy routing and arbitration,
including which strategies were considered, selected, or rejected.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003  # pydantic 在运行时解析字段注解
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from a_share_quant.domain.market_state import (  # noqa: TC001  # pydantic 运行时解析
    MarketState,
)
from a_share_quant.domain.stock_state import (  # noqa: TC001  # pydantic 运行时解析
    StockState,
)

if TYPE_CHECKING:
    from a_share_quant.domain.market_state import MarketRegime


class StrategyId(StrEnum):
    """Strategy identifiers."""

    TREND_HOLD = "TREND_HOLD"  # 趋势持有
    PULLBACK_SWING = "PULLBACK_SWING"  # 回调波段
    RANGE_MEAN_REVERSION = "RANGE_MEAN_REVERSION"  # 区间均值回归
    BEAR_REBOUND = "BEAR_REBOUND"  # 熊市反弹
    CASH_DEFENSE = "CASH_DEFENSE"  # 现金防御


class RejectionReason(StrEnum):
    """Reasons why a strategy was rejected."""

    REGIME_MISMATCH = "regime_mismatch"  # 市场状态不匹配
    STOCK_STATE_INVALID = "stock_state_invalid"  # 个股状态不允许
    FREQUENCY_LIMIT = "frequency_limit"  # 频率限制
    COOLDOWN_ACTIVE = "cooldown_active"  # 冷却期
    RISK_BUDGET_EXCEEDED = "risk_budget_exceeded"  # 风险预算超限
    LIQUIDITY_INSUFFICIENT = "liquidity_insufficient"  # 流动性不足
    NO_SIGNAL = "no_signal"  # 无信号
    SIGNAL_RESET_REQUIRED = "signal_reset_required"  # 需要信号重置
    OWNERSHIP_CONFLICT = "ownership_conflict"  # 所有权冲突
    POSITION_LIMIT = "position_limit"  # 仓位限制
    CIRCUIT_BREAKER = "circuit_breaker"  # 熔断
    DATA_QUALITY = "data_quality"  # 数据质量问题
    STRATEGY_DISABLED = "strategy_disabled"  # 策略被禁用


class EligibilityResult(BaseModel):
    """Result of strategy eligibility check."""

    model_config = {"frozen": True}

    strategy_id: StrategyId
    eligible: bool
    rejection_reasons: list[RejectionReason] = Field(default_factory=list)
    score: float = Field(
        default=0.0,
        description="Eligibility score for ranking",
    )
    details: dict[str, str] = Field(
        default_factory=dict,
        description="Additional details about eligibility",
    )


class StrategyDecision(BaseModel):
    """Complete strategy routing and arbitration decision.

    Records the full decision process including:
    - Market and stock state at decision time
    - All strategies considered
    - Which strategy won and why
    - Which strategies were rejected and why
    """

    model_config = {"frozen": True}

    decision_id: str = Field(description="Unique decision identifier")
    as_of: datetime = Field(description="Decision timestamp")
    symbol: str = Field(description="Symbol this decision applies to")

    # State at decision time
    market_state: MarketState = Field(description="Market state at decision time")
    stock_state: StockState = Field(description="Stock state at decision time")

    # Strategy evaluation results
    enabled_strategies: list[StrategyId] = Field(
        default_factory=list,
        description="Strategies enabled for current regime",
    )
    evaluated_strategies: list[EligibilityResult] = Field(
        default_factory=list,
        description="All evaluated strategies with results",
    )
    winning_strategy: StrategyId | None = Field(
        default=None,
        description="The selected strategy (None means cash/no trade)",
    )
    rejected_strategies: dict[StrategyId, list[RejectionReason]] = Field(
        default_factory=dict,
        description="Rejected strategies with reasons",
    )

    # Versioning
    strategy_version: str = Field(
        default="v1",
        description="Strategy configuration version",
    )
    arbitration_version: str = Field(
        default="v1",
        description="Arbitration logic version",
    )

    # Explanation
    explanation: str = Field(
        default="",
        description="Human-readable explanation of the decision",
    )

    @property
    def regime(self) -> MarketRegime:
        """Get the market regime at decision time."""
        return self.market_state.regime

    @property
    def should_trade(self) -> bool:
        """Check if a trade should be made."""
        return (
            self.winning_strategy is not None and self.winning_strategy != StrategyId.CASH_DEFENSE
        )

    @property
    def is_cash_defense(self) -> bool:
        """Check if cash defense was selected."""
        return self.winning_strategy == StrategyId.CASH_DEFENSE

    def get_rejection_reasons(self, strategy_id: StrategyId) -> list[RejectionReason]:
        """Get rejection reasons for a specific strategy."""
        return self.rejected_strategies.get(strategy_id, [])
