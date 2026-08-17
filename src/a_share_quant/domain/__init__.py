"""Domain models for the A-share quant system.

This package contains all core domain objects:
- types: Basic value types (TradingDate, Symbol, Money, Currency)
- market_state: Market regime detection (9-state model)
- stock_state: Individual stock state
- strategy_decision: Strategy routing and arbitration
- trade_plan: Complete trade plan with exit conditions
- order: Orders, fills, and positions
- trade_ledger: Complete trade record with attribution
"""

from a_share_quant.domain.market_state import (
    MarketDirection,
    MarketRegime,
    MarketState,
    OscillationLevel,
)
from a_share_quant.domain.order import (
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    PositionStatus,
)
from a_share_quant.domain.stock_state import (
    StockState,
    StockTrend,
    TradingStatus,
)
from a_share_quant.domain.strategy_decision import (
    EligibilityResult,
    RejectionReason,
    StrategyDecision,
    StrategyId,
)
from a_share_quant.domain.trade_ledger import (
    AttributionBreakdown,
    FailureReasonCode,
    TradeLedger,
)
from a_share_quant.domain.trade_plan import (
    EntryTrigger,
    ExitReason,
    TradePlan,
    TradePlanStatus,
    TrailingStopRule,
)
from a_share_quant.domain.types import (
    Board,
    Currency,
    Exchange,
    Money,
    SecurityType,
    Symbol,
    TradingDate,
)

__all__ = [
    # Trade ledger
    "AttributionBreakdown",
    # Types
    "Board",
    "Currency",
    # Strategy decision
    "EligibilityResult",
    # Trade plan
    "EntryTrigger",
    "Exchange",
    "ExitReason",
    "FailureReasonCode",
    # Order
    "Fill",
    # Market state
    "MarketDirection",
    "MarketRegime",
    "MarketState",
    "Money",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "OscillationLevel",
    "Position",
    "PositionSide",
    "PositionStatus",
    "RejectionReason",
    "SecurityType",
    # Stock state
    "StockState",
    "StockTrend",
    "StrategyDecision",
    "StrategyId",
    "Symbol",
    "TradeLedger",
    "TradePlan",
    "TradePlanStatus",
    "TradingDate",
    "TradingStatus",
    "TrailingStopRule",
]
