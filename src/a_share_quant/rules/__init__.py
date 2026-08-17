"""Market rules engine for A-share trading.

Implements date-based rules that change over time:
- Fee schedules (stamp tax changes)
- Price limits (10% → 20% for ChiNext/STAR)
- T+1 settlement
- Order state machine
- Account and position ledger
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Optional

from a_share_quant.config import load_market_rules_config


# =============================================================================
# Fee Rules
# =============================================================================


@dataclass
class FeeSchedule:
    """Fee schedule for a specific date range."""

    effective_from: date
    stamp_tax_buy: float = 0.0
    stamp_tax_sell: float = 0.001
    commission: float = 0.00025
    min_commission: float = 5.0
    transfer_fee: float = 0.00001
    effective_until: Optional[date] = None

    def is_effective(self, on_date: date) -> bool:
        """Check if this schedule is effective on the given date."""
        if on_date < self.effective_from:
            return False
        if self.effective_until and on_date > self.effective_until:
            return False
        return True


class FeeCalculator:
    """Calculate trading fees based on date-effective rules."""

    def __init__(self, schedules: list[FeeSchedule] | None = None):
        if schedules is None:
            schedules = self._load_default_schedules()
        # Sort by effective_from descending for lookup
        self._schedules = sorted(schedules, key=lambda s: s.effective_from, reverse=True)

    @staticmethod
    def _load_default_schedules() -> list[FeeSchedule]:
        """Load fee schedules from config or use hardcoded defaults."""
        # Hardcoded defaults - always use these for consistency
        return [
            FeeSchedule(
                effective_from=date(2023, 8, 28),
                stamp_tax_buy=0.0,
                stamp_tax_sell=0.0005,  # Halved from 0.1% to 0.05%
                commission=0.00025,
                min_commission=5.0,
                transfer_fee=0.00001,
            ),
            FeeSchedule(
                effective_from=date(2008, 9, 19),
                effective_until=date(2023, 8, 27),
                stamp_tax_buy=0.0,
                stamp_tax_sell=0.001,  # 0.1%
                commission=0.00025,
                min_commission=5.0,
                transfer_fee=0.00001,
            ),
        ]

    def get_schedule(self, on_date: date) -> FeeSchedule:
        """Get the effective fee schedule for a date."""
        for schedule in self._schedules:
            if schedule.is_effective(on_date):
                return schedule
        # Return most recent if no match
        return (
            self._schedules[0] if self._schedules else FeeSchedule(effective_from=date(2000, 1, 1))
        )

    def calculate_buy_cost(self, amount: float, on_date: date) -> dict[str, float]:
        """Calculate total cost for a buy order."""
        schedule = self.get_schedule(on_date)

        commission = max(amount * schedule.commission, schedule.min_commission)
        stamp_tax = amount * schedule.stamp_tax_buy
        transfer_fee = amount * schedule.transfer_fee

        return {
            "commission": commission,
            "stamp_tax": stamp_tax,
            "transfer_fee": transfer_fee,
            "total": commission + stamp_tax + transfer_fee,
        }

    def calculate_sell_cost(self, amount: float, on_date: date) -> dict[str, float]:
        """Calculate total cost for a sell order."""
        schedule = self.get_schedule(on_date)

        commission = max(amount * schedule.commission, schedule.min_commission)
        stamp_tax = amount * schedule.stamp_tax_sell
        transfer_fee = amount * schedule.transfer_fee

        return {
            "commission": commission,
            "stamp_tax": stamp_tax,
            "transfer_fee": transfer_fee,
            "total": commission + stamp_tax + transfer_fee,
        }


# =============================================================================
# Price Limit Rules
# =============================================================================


@dataclass
class PriceLimitRule:
    """Price limit rule for a board."""

    board: str
    limit_pct: float
    effective_from: date
    effective_until: Optional[date] = None

    def is_effective(self, on_date: date) -> bool:
        if on_date < self.effective_from:
            return False
        if self.effective_until and on_date > self.effective_until:
            return False
        return True


class PriceLimitCalculator:
    """Calculate price limits based on board and date."""

    def __init__(self, rules: list[PriceLimitRule] | None = None):
        if rules is None:
            rules = self._load_default_rules()
        self._rules = rules

    @staticmethod
    def _load_default_rules() -> list[PriceLimitRule]:
        """Load price limit rules - use hardcoded defaults for consistency."""
        return [
            # Main board: 10% since 1996
            PriceLimitRule("MAIN", 0.10, date(1996, 12, 16)),
            # ChiNext: 20% after 2020-08-24 registration reform
            PriceLimitRule("CHINEXT", 0.20, date(2020, 8, 24)),
            # ChiNext: 10% before reform
            PriceLimitRule("CHINEXT", 0.10, date(2009, 10, 30), date(2020, 8, 23)),
            # STAR Market: 20% since launch
            PriceLimitRule("STAR", 0.20, date(2019, 7, 22)),
            # BSE: 30% since launch
            PriceLimitRule("BSE", 0.30, date(2021, 11, 15)),
        ]

    def get_limit_pct(self, board: str, on_date: date) -> float:
        """Get the price limit percentage for a board on a date."""
        for rule in self._rules:
            if rule.board == board and rule.is_effective(on_date):
                return rule.limit_pct
        return 0.10  # Default 10%

    def calculate_limit_prices(
        self,
        prev_close: float,
        board: str,
        on_date: date,
    ) -> tuple[float, float]:
        """Calculate upper and lower limit prices.

        Returns:
            (upper_limit, lower_limit)
        """
        limit_pct = self.get_limit_pct(board, on_date)
        upper = round(prev_close * (1 + limit_pct), 2)
        lower = round(prev_close * (1 - limit_pct), 2)
        return upper, lower

    def is_at_limit_up(self, price: float, prev_close: float, board: str, on_date: date) -> bool:
        """Check if price is at limit up."""
        upper, _ = self.calculate_limit_prices(prev_close, board, on_date)
        return abs(price - upper) < 0.01

    def is_at_limit_down(self, price: float, prev_close: float, board: str, on_date: date) -> bool:
        """Check if price is at limit down."""
        _, lower = self.calculate_limit_prices(prev_close, board, on_date)
        return abs(price - lower) < 0.01


# =============================================================================
# T+1 Settlement Rules
# =============================================================================


class SettlementRule:
    """T+1 settlement rule implementation."""

    def can_sell(self, buy_date: date, sell_date: date) -> bool:
        """Check if shares bought on buy_date can be sold on sell_date.

        T+1 rule: shares bought today can only be sold from next trading day.
        """
        return sell_date > buy_date

    def get_sellable_quantity(
        self,
        total_quantity: int,
        buy_date: date,
        current_date: date,
    ) -> int:
        """Get quantity that can be sold today.

        For T+1: if bought today, 0 sellable; otherwise full quantity.
        """
        if self.can_sell(buy_date, current_date):
            return total_quantity
        return 0


# =============================================================================
# Order State Machine
# =============================================================================


class OrderState(str, Enum):
    """Order lifecycle states."""

    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    PARTIAL_FILLED = "PARTIAL_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


# Valid state transitions
VALID_TRANSITIONS: dict[OrderState, set[OrderState]] = {
    OrderState.CREATED: {OrderState.SUBMITTED, OrderState.CANCELLED},
    OrderState.SUBMITTED: {
        OrderState.PARTIAL_FILLED,
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.REJECTED,
        OrderState.EXPIRED,
    },
    OrderState.PARTIAL_FILLED: {
        OrderState.PARTIAL_FILLED,
        OrderState.FILLED,
        OrderState.CANCELLED,
    },
    OrderState.FILLED: set(),  # Terminal
    OrderState.CANCELLED: set(),  # Terminal
    OrderState.REJECTED: set(),  # Terminal
    OrderState.EXPIRED: set(),  # Terminal
}


@dataclass
class OrderStateMachine:
    """State machine for order lifecycle."""

    state: OrderState = OrderState.CREATED
    history: list[tuple[OrderState, OrderState, str]] = field(default_factory=list)

    def can_transition(self, to_state: OrderState) -> bool:
        """Check if transition to state is valid."""
        return to_state in VALID_TRANSITIONS.get(self.state, set())

    def transition(self, to_state: OrderState, reason: str = "") -> bool:
        """Attempt to transition to a new state.

        Returns True if transition was successful.
        """
        if not self.can_transition(to_state):
            return False

        old_state = self.state
        self.state = to_state
        self.history.append((old_state, to_state, reason))
        return True

    @property
    def is_terminal(self) -> bool:
        """Check if order is in a terminal state."""
        return self.state in {
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
        }


# =============================================================================
# Account and Position Ledger
# =============================================================================


@dataclass
class CashAccount:
    """Cash account tracking."""

    balance: float = 0.0
    frozen: float = 0.0
    history: list[dict] = field(default_factory=list)

    @property
    def available(self) -> float:
        """Available cash for trading."""
        return self.balance - self.frozen

    def deposit(self, amount: float, on_date: date, memo: str = "") -> None:
        """Deposit cash."""
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self.balance += amount
        self.history.append(
            {
                "date": on_date,
                "type": "deposit",
                "amount": amount,
                "balance_after": self.balance,
                "memo": memo,
            }
        )

    def withdraw(self, amount: float, on_date: date, memo: str = "") -> None:
        """Withdraw cash."""
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self.available:
            raise ValueError(f"Insufficient available balance: {self.available}")
        self.balance -= amount
        self.history.append(
            {
                "date": on_date,
                "type": "withdraw",
                "amount": -amount,
                "balance_after": self.balance,
                "memo": memo,
            }
        )

    def freeze(self, amount: float) -> None:
        """Freeze cash for pending orders."""
        if amount > self.available:
            raise ValueError(f"Cannot freeze more than available: {self.available}")
        self.frozen += amount

    def unfreeze(self, amount: float) -> None:
        """Unfreeze cash."""
        self.frozen = max(0, self.frozen - amount)

    def apply_trade(self, amount: float, fee: float, side: str, on_date: date) -> None:
        """Apply a trade to the account."""
        if side == "BUY":
            total_cost = amount + fee
            self.balance -= total_cost
            self.unfreeze(amount)  # Release frozen amount
        else:  # SELL
            net_proceeds = amount - fee
            self.balance += net_proceeds

        self.history.append(
            {
                "date": on_date,
                "type": f"trade_{side.lower()}",
                "amount": -amount - fee if side == "BUY" else amount - fee,
                "fee": fee,
                "balance_after": self.balance,
            }
        )


@dataclass
class PositionLot:
    """A single lot of a position (for T+1 tracking)."""

    quantity: int
    cost_price: float
    buy_date: date
    sellable: bool = False  # Becomes True after T+1


@dataclass
class PositionLedger:
    """Position tracking with T+1 support."""

    positions: dict[str, list[PositionLot]] = field(default_factory=dict)
    _settlement = SettlementRule()

    def add_position(self, symbol: str, quantity: int, price: float, on_date: date) -> None:
        """Add a new position lot."""
        if symbol not in self.positions:
            self.positions[symbol] = []
        self.positions[symbol].append(
            PositionLot(
                quantity=quantity,
                cost_price=price,
                buy_date=on_date,
                sellable=False,  # T+1: not sellable on buy day
            )
        )

    def update_settlement(self, current_date: date) -> None:
        """Update sellable status based on T+1 rule."""
        for symbol, lots in self.positions.items():
            for lot in lots:
                lot.sellable = self._settlement.can_sell(lot.buy_date, current_date)

    def get_total_quantity(self, symbol: str) -> int:
        """Get total position quantity."""
        return sum(lot.quantity for lot in self.positions.get(symbol, []))

    def get_sellable_quantity(self, symbol: str) -> int:
        """Get sellable quantity (T+1 compliant)."""
        return sum(lot.quantity for lot in self.positions.get(symbol, []) if lot.sellable)

    def get_average_cost(self, symbol: str) -> float:
        """Get average cost price."""
        lots = self.positions.get(symbol, [])
        if not lots:
            return 0.0
        total_cost = sum(lot.quantity * lot.cost_price for lot in lots)
        total_qty = sum(lot.quantity for lot in lots)
        return total_cost / total_qty if total_qty > 0 else 0.0

    def reduce_position(self, symbol: str, quantity: int, on_date: date) -> float:
        """Reduce position by selling. Returns realized cost basis.

        Uses FIFO for lot selection.
        """
        lots = self.positions.get(symbol, [])
        if not lots:
            raise ValueError(f"No position for {symbol}")

        sellable = self.get_sellable_quantity(symbol)
        if quantity > sellable:
            raise ValueError(f"Cannot sell {quantity}, only {sellable} sellable (T+1)")

        remaining = quantity
        realized_cost = 0.0

        # Sell from oldest lots first (FIFO)
        for lot in sorted(lots, key=lambda l: l.buy_date):
            if not lot.sellable or remaining <= 0:
                continue
            sell_qty = min(lot.quantity, remaining)
            realized_cost += sell_qty * lot.cost_price
            lot.quantity -= sell_qty
            remaining -= sell_qty

        # Remove empty lots
        self.positions[symbol] = [l for l in lots if l.quantity > 0]
        if not self.positions[symbol]:
            del self.positions[symbol]

        return realized_cost

    def get_all_positions(self) -> dict[str, dict]:
        """Get summary of all positions."""
        result = {}
        for symbol in self.positions:
            result[symbol] = {
                "quantity": self.get_total_quantity(symbol),
                "sellable": self.get_sellable_quantity(symbol),
                "avg_cost": self.get_average_cost(symbol),
            }
        return result
