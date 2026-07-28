"""Trade ledger domain models.

The TradeLedger is the complete record of a trade from signal to exit,
including all orders, fills, daily positions, and attribution.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from a_share_quant.domain.market_state import MarketRegime
from a_share_quant.domain.order import Fill, Order, Position
from a_share_quant.domain.strategy_decision import StrategyId
from a_share_quant.domain.trade_plan import ExitReason, TradePlan


class FailureReasonCode(BaseModel):
    """Failure reason code with module attribution."""

    model_config = {"frozen": True}

    code: str = Field(description="Failure reason code")
    description: str = Field(description="Human-readable description")
    responsible_module: str = Field(description="Module responsible for this failure")
    suggested_action: str = Field(default="", description="Suggested corrective action")


# Standard failure reason codes
FAILURE_REASONS: dict[str, FailureReasonCode] = {
    "REGIME_WRONG": FailureReasonCode(
        code="REGIME_WRONG",
        description="市场状态判断错误",
        responsible_module="regimes",
        suggested_action="检查状态识别模型和特征",
    ),
    "STOCK_STATE_WRONG": FailureReasonCode(
        code="STOCK_STATE_WRONG",
        description="个股状态判断错误",
        responsible_module="features",
        suggested_action="检查个股特征计算",
    ),
    "FALSE_BREAKOUT": FailureReasonCode(
        code="FALSE_BREAKOUT",
        description="假突破",
        responsible_module="strategies",
        suggested_action="增加突破确认条件",
    ),
    "FALLING_KNIFE": FailureReasonCode(
        code="FALLING_KNIFE",
        description="接飞刀（下跌中买入）",
        responsible_module="strategies",
        suggested_action="增加下跌过滤条件",
    ),
    "ENTRY_TOO_EARLY": FailureReasonCode(
        code="ENTRY_TOO_EARLY",
        description="入场过早",
        responsible_module="strategies",
        suggested_action="增加确认信号",
    ),
    "ENTRY_TOO_LATE": FailureReasonCode(
        code="ENTRY_TOO_LATE",
        description="入场过晚",
        responsible_module="strategies",
        suggested_action="优化入场时机",
    ),
    "STOP_TOO_TIGHT": FailureReasonCode(
        code="STOP_TOO_TIGHT",
        description="止损过紧",
        responsible_module="risk",
        suggested_action="调整止损距离",
    ),
    "STOP_TOO_WIDE": FailureReasonCode(
        code="STOP_TOO_WIDE",
        description="止损过宽",
        responsible_module="risk",
        suggested_action="调整止损距离",
    ),
    "EXIT_TOO_EARLY": FailureReasonCode(
        code="EXIT_TOO_EARLY",
        description="退出过早",
        responsible_module="strategies",
        suggested_action="优化止盈规则",
    ),
    "TIMEOUT": FailureReasonCode(
        code="TIMEOUT",
        description="时间止损",
        responsible_module="strategies",
        suggested_action="检查持有期设置",
    ),
    "COST_EROSION": FailureReasonCode(
        code="COST_EROSION",
        description="成本侵蚀收益",
        responsible_module="execution",
        suggested_action="减少交易频率或优化执行",
    ),
    "LIQUIDITY_FAILURE": FailureReasonCode(
        code="LIQUIDITY_FAILURE",
        description="流动性失败",
        responsible_module="execution",
        suggested_action="增加流动性过滤",
    ),
    "CORRELATION_SHOCK": FailureReasonCode(
        code="CORRELATION_SHOCK",
        description="相关性冲击",
        responsible_module="risk",
        suggested_action="降低相关性暴露",
    ),
    "DATA_QUALITY_FAILURE": FailureReasonCode(
        code="DATA_QUALITY_FAILURE",
        description="数据质量问题",
        responsible_module="data",
        suggested_action="检查数据源和质量规则",
    ),
    "MODEL_DRIFT": FailureReasonCode(
        code="MODEL_DRIFT",
        description="模型漂移",
        responsible_module="evolution",
        suggested_action="触发模型重新评估",
    ),
}


class AttributionBreakdown(BaseModel):
    """Breakdown of trade P&L attribution."""

    model_config = {"frozen": True}

    # Return components
    market_contribution: float = Field(
        default=0.0,
        description="市场贡献",
    )
    industry_contribution: float = Field(
        default=0.0,
        description="行业贡献",
    )
    factor_contribution: float = Field(
        default=0.0,
        description="因子贡献",
    )
    selection_alpha: float = Field(
        default=0.0,
        description="选股 Alpha",
    )
    timing_contribution: float = Field(
        default=0.0,
        description="入场择时贡献",
    )
    exit_contribution: float = Field(
        default=0.0,
        description="退出管理贡献",
    )

    # Cost components
    commission_cost: float = Field(
        default=0.0,
        description="佣金成本",
    )
    tax_cost: float = Field(
        default=0.0,
        description="税费成本",
    )
    slippage_cost: float = Field(
        default=0.0,
        description="滑点成本",
    )
    failed_execution_cost: float = Field(
        default=0.0,
        description="成交失败成本",
    )

    @property
    def total_cost(self) -> float:
        """Get total cost."""
        return (
            self.commission_cost
            + self.tax_cost
            + self.slippage_cost
            + self.failed_execution_cost
        )

    @property
    def gross_return(self) -> float:
        """Get gross return before costs."""
        return (
            self.market_contribution
            + self.industry_contribution
            + self.factor_contribution
            + self.selection_alpha
            + self.timing_contribution
            + self.exit_contribution
        )

    @property
    def net_return(self) -> float:
        """Get net return after costs."""
        return self.gross_return - self.total_cost


class TradeLedger(BaseModel):
    """Complete record of a trade from signal to exit.

    This is the master record that ties together:
    - The trade plan
    - All orders and fills
    - Daily position snapshots
    - Full attribution
    - Failure analysis
    """

    model_config = {"frozen": True}

    # Identity
    trade_id: str = Field(description="Unique trade identifier")
    trade_plan: TradePlan = Field(description="The trade plan")
    symbol: str = Field(description="Symbol traded")
    strategy_id: StrategyId = Field(description="Strategy that made this trade")

    # Orders and fills
    orders: list[Order] = Field(
        default_factory=list,
        description="All orders for this trade",
    )
    fills: list[Fill] = Field(
        default_factory=list,
        description="All fills for this trade",
    )
    position: Optional[Position] = Field(
        default=None,
        description="The position record",
    )

    # Timing
    signal_time: datetime = Field(description="Signal generation time")
    entry_time: Optional[datetime] = Field(
        default=None,
        description="Actual entry time",
    )
    exit_time: Optional[datetime] = Field(
        default=None,
        description="Actual exit time",
    )

    # P&L
    entry_price: Optional[float] = Field(
        default=None,
        description="Average entry price",
    )
    exit_price: Optional[float] = Field(
        default=None,
        description="Average exit price",
    )
    quantity: int = Field(
        default=0,
        description="Total quantity traded",
    )
    gross_pnl: float = Field(
        default=0.0,
        description="Gross P&L before costs",
    )
    net_pnl: float = Field(
        default=0.0,
        description="Net P&L after all costs",
    )
    total_fees: float = Field(
        default=0.0,
        description="Total fees paid",
    )

    # Risk metrics
    mfe: float = Field(
        default=0.0,
        description="Maximum Favorable Excursion",
    )
    mae: float = Field(
        default=0.0,
        description="Maximum Adverse Excursion",
    )

    # Benchmark comparison
    market_return: float = Field(
        default=0.0,
        description="同期市场收益",
    )
    industry_return: float = Field(
        default=0.0,
        description="同期行业收益",
    )
    excess_return: float = Field(
        default=0.0,
        description="超额收益",
    )

    # Exit
    exit_reason: Optional[ExitReason] = Field(
        default=None,
        description="Reason for exit",
    )
    failure_reason: Optional[str] = Field(
        default=None,
        description="Failure reason code if trade was a loss",
    )

    # Attribution
    attribution: Optional[AttributionBreakdown] = Field(
        default=None,
        description="Full attribution breakdown",
    )

    # Context
    entry_regime: Optional[MarketRegime] = Field(
        default=None,
        description="Market regime at entry",
    )
    exit_regime: Optional[MarketRegime] = Field(
        default=None,
        description="Market regime at exit",
    )
    holding_days: int = Field(
        default=0,
        description="Trading days held",
    )

    # Versioning
    model_version: str = Field(default="v1")
    data_snapshot_id: Optional[str] = Field(
        default=None,
        description="Data snapshot used",
    )

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.net_pnl > 0

    @property
    def is_closed(self) -> bool:
        """Check if trade is closed."""
        return self.exit_time is not None

    @property
    def return_pct(self) -> float:
        """Get return as percentage."""
        if self.entry_price is None or self.entry_price == 0:
            return 0.0
        return (self.net_pnl / (self.entry_price * self.quantity)) * 100 if self.quantity > 0 else 0.0

    def get_failure_reason_code(self) -> Optional[FailureReasonCode]:
        """Get the failure reason code object."""
        if self.failure_reason is None:
            return None
        return FAILURE_REASONS.get(self.failure_reason)
