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
    # Types
    "Board",
    "Currency",
    "Exchange",
    "Money",
    "SecurityType",
    "Symbol",
    "TradingDate",
    # Market state
    "MarketDirection",
    "MarketRegime",
    "MarketState",
    "OscillationLevel",
    # Stock state
    "StockState",
    "StockTrend",
    "TradingStatus",
    # Strategy decision
    "EligibilityResult",
    "RejectionReason",
    "StrategyDecision",
    "StrategyId",
    # Trade plan
    "EntryTrigger",
    "ExitReason",
    "TradePlan",
    "TradePlanStatus",
    "TrailingStopRule",
    # Order
    "Fill",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "Position",
    "PositionSide",
    "PositionStatus",
    # Trade ledger
    "AttributionBreakdown",
    "FailureReasonCode",
    "TradeLedger",
]
