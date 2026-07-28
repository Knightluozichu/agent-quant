"""Order, Fill, and Position domain models.

Strict separation of concerns:
- Signal → TradePlan → TargetPosition → OrderIntent → SubmittedOrder → Fill → Position → Exit

"产生买入信号" ≠ "已经买到"
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field

from a_share_quant.domain.strategy_decision import StrategyId
from a_share_quant.domain.trade_plan import ExitReason


class OrderSide(StrEnum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    """Order type."""

    MARKET = "market"  # 市价单
    LIMIT = "limit"  # 限价单


class OrderStatus(StrEnum):
    """Order status lifecycle."""

    CREATED = "created"  # 已创建
    SUBMITTED = "submitted"  # 已提交
    PARTIAL_FILLED = "partial_filled"  # 部分成交
    FILLED = "filled"  # 完全成交
    CANCELLED = "cancelled"  # 已取消
    REJECTED = "rejected"  # 被拒绝
    EXPIRED = "expired"  # 已过期


class Order(BaseModel):
    """An order to be submitted to the market.

    Represents the full lifecycle from creation to completion.
    """

    model_config = {"frozen": True}

    # Identity
    order_id: str = Field(description="Unique order identifier")
    trade_plan_id: str = Field(description="Associated trade plan")
    symbol: str = Field(description="Symbol to trade")

    # Order specification
    side: OrderSide = Field(description="Buy or sell")
    order_type: OrderType = Field(description="Market or limit")
    quantity: int = Field(gt=0, description="Order quantity in shares")
    limit_price: Optional[float] = Field(
        default=None,
        gt=0,
        description="Limit price (required for limit orders)",
    )

    # Timing
    created_at: datetime = Field(description="Order creation time")
    submitted_at: Optional[datetime] = Field(
        default=None,
        description="Order submission time",
    )
    filled_at: Optional[datetime] = Field(
        default=None,
        description="Order fill time",
    )
    cancelled_at: Optional[datetime] = Field(
        default=None,
        description="Order cancellation time",
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        description="Order expiry time",
    )

    # Status
    status: OrderStatus = Field(
        default=OrderStatus.CREATED,
        description="Current order status",
    )
    filled_quantity: int = Field(
        default=0,
        ge=0,
        description="Quantity filled so far",
    )
    average_fill_price: Optional[float] = Field(
        default=None,
        gt=0,
        description="Average fill price",
    )

    # Rejection info
    rejection_reason: Optional[str] = Field(
        default=None,
        description="Reason for rejection if applicable",
    )

    # Metadata
    strategy_id: StrategyId = Field(description="Strategy that generated this order")
    is_t_plus_one_sell: bool = Field(
        default=False,
        description="Whether this is a T+1 sell (selling shares bought today)",
    )

    @property
    def remaining_quantity(self) -> int:
        """Get remaining quantity to fill."""
        return self.quantity - self.filled_quantity

    @property
    def is_fully_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.filled_quantity >= self.quantity

    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in (
            OrderStatus.CREATED,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIAL_FILLED,
        )


class Fill(BaseModel):
    """A fill (execution) of an order.

    One order can have multiple fills (partial fills).
    """

    model_config = {"frozen": True}

    # Identity
    fill_id: str = Field(description="Unique fill identifier")
    order_id: str = Field(description="Associated order")
    symbol: str = Field(description="Symbol traded")

    # Fill details
    side: OrderSide = Field(description="Buy or sell")
    quantity: int = Field(gt=0, description="Filled quantity")
    price: float = Field(gt=0, description="Fill price")
    filled_at: datetime = Field(description="Fill timestamp")

    # Costs
    commission: float = Field(
        default=0.0,
        ge=0,
        description="Commission fee",
    )
    stamp_tax: float = Field(
        default=0.0,
        ge=0,
        description="Stamp tax (卖出印花税)",
    )
    transfer_fee: float = Field(
        default=0.0,
        ge=0,
        description="Transfer fee (过户费)",
    )
    handling_fee: float = Field(
        default=0.0,
        ge=0,
        description="Handling fee (经手费)",
    )
    slippage: float = Field(
        default=0.0,
        description="Slippage vs expected price",
    )

    @property
    def total_cost(self) -> float:
        """Get total transaction cost."""
        return self.commission + self.stamp_tax + self.transfer_fee + self.handling_fee

    @property
    def total_value(self) -> float:
        """Get total fill value (quantity * price)."""
        return self.quantity * self.price

    @property
    def net_value(self) -> float:
        """Get net value after costs."""
        if self.side == OrderSide.BUY:
            return -(self.total_value + self.total_cost)
        return self.total_value - self.total_cost


class PositionSide(StrEnum):
    """Position side."""

    LONG = "long"
    FLAT = "flat"  # No position


class PositionStatus(StrEnum):
    """Position status."""

    OPEN = "open"  # 持仓中
    CLOSED = "closed"  # 已平仓


class Position(BaseModel):
    """A position in a security.

    Tracks the current holding and its cost basis.
    """

    model_config = {"frozen": True}

    # Identity
    position_id: str = Field(description="Unique position identifier")
    symbol: str = Field(description="Symbol held")
    trade_plan_id: str = Field(description="Trade plan that opened this position")
    strategy_id: StrategyId = Field(description="Strategy owning this position")

    # Position details
    side: PositionSide = Field(
        default=PositionSide.LONG,
        description="Position side (only LONG for phase 1)",
    )
    quantity: int = Field(
        ge=0,
        description="Current quantity held",
    )
    sellable_quantity: int = Field(
        ge=0,
        description="Quantity that can be sold (T+1 rule)",
    )
    average_cost: float = Field(
        gt=0,
        description="Average cost per share including fees",
    )

    # Timing
    opened_at: datetime = Field(description="Position open time")
    closed_at: Optional[datetime] = Field(
        default=None,
        description="Position close time",
    )
    days_held: int = Field(
        default=0,
        ge=0,
        description="Trading days held",
    )

    # Status
    status: PositionStatus = Field(
        default=PositionStatus.OPEN,
        description="Position status",
    )

    # Exit tracking
    exit_reason: Optional[ExitReason] = Field(
        default=None,
        description="Reason for exit if closed",
    )

    # Price tracking
    entry_price: float = Field(gt=0, description="Entry price")
    current_price: Optional[float] = Field(
        default=None,
        description="Latest price",
    )
    highest_price: Optional[float] = Field(
        default=None,
        description="Highest price since entry (for trailing stop)",
    )
    lowest_price: Optional[float] = Field(
        default=None,
        description="Lowest price since entry (for MAE)",
    )
    current_stop_price: Optional[float] = Field(
        default=None,
        description="Current trailing stop price",
    )

    # P&L tracking
    realized_pnl: float = Field(
        default=0.0,
        description="Realized P&L from partial closes",
    )
    unrealized_pnl: Optional[float] = Field(
        default=None,
        description="Unrealized P&L at current price",
    )
    total_fees: float = Field(
        default=0.0,
        description="Total fees paid",
    )

    @property
    def market_value(self) -> float:
        """Get current market value."""
        if self.current_price is None:
            return 0.0
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        """Get total cost basis."""
        return self.quantity * self.average_cost

    @property
    def is_open(self) -> bool:
        """Check if position is open."""
        return self.status == PositionStatus.OPEN

    def update_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L at given price."""
        return (current_price - self.average_cost) * self.quantity
