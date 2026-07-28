"""Paper trading broker.

Simulates real trading without actual money:
1. PaperBroker: Simulated order execution
2. Daily scheduler: Generate signals at close
3. Position tracking with T+1
4. Reconciliation
5. Restart recovery
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from a_share_quant.rules import CashAccount, PositionLedger, FeeCalculator


# =============================================================================
# Paper Orders
# =============================================================================

@dataclass
class PaperOrder:
    """A paper trading order."""

    order_id: str
    symbol: str
    side: str  # BUY, SELL
    quantity: int
    order_type: str = "MARKET"
    limit_price: Optional[float] = None
    status: str = "PENDING"  # PENDING, FILLED, CANCELLED, REJECTED
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    fill_price: Optional[float] = None
    fill_quantity: int = 0
    commission: float = 0.0
    reject_reason: str = ""
    strategy_name: str = ""

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "limit_price": self.limit_price,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "fill_price": self.fill_price,
            "fill_quantity": self.fill_quantity,
            "commission": self.commission,
            "reject_reason": self.reject_reason,
            "strategy_name": self.strategy_name,
        }


# =============================================================================
# Paper Broker
# =============================================================================

class PaperBroker:
    """Simulated broker for paper trading.

    Features:
    - Simulated order execution at next open
    - T+1 settlement
    - Fee calculation
    - Position tracking
    - State persistence for restart recovery
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        state_file: Path | str = "paper_state.json",
    ):
        self._account = CashAccount(balance=initial_capital)
        self._positions = PositionLedger()
        self._fee_calc = FeeCalculator()
        self._state_file = Path(state_file)
        self._orders: list[PaperOrder] = []
        self._order_counter = 0
        self._current_date: Optional[date] = None

        # Try to restore state
        self._restore_state()

    @property
    def cash(self) -> float:
        return self._account.balance

    @property
    def available_cash(self) -> float:
        return self._account.available

    def get_positions(self) -> dict[str, dict]:
        """Get all positions."""
        return self._positions.get_all_positions()

    def get_position(self, symbol: str) -> Optional[dict]:
        """Get position for a symbol."""
        positions = self._positions.positions.get(symbol)
        if not positions:
            return None
        return {
            "quantity": self._positions.get_total_quantity(symbol),
            "sellable": self._positions.get_sellable_quantity(symbol),
            "avg_cost": self._positions.get_average_cost(symbol),
        }

    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        strategy_name: str = "",
    ) -> PaperOrder:
        """Submit a new order."""
        self._order_counter += 1
        order = PaperOrder(
            order_id=f"P{self._order_counter:06d}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            strategy_name=strategy_name,
        )

        # Validate order
        if side == "BUY":
            if limit_price:
                required = limit_price * quantity * 1.001  # Include estimated fees
            else:
                required = quantity * 100  # Estimate
            if required > self.available_cash:
                order.status = "REJECTED"
                order.reject_reason = "Insufficient funds"
                self._orders.append(order)
                return order

        elif side == "SELL":
            sellable = self._positions.get_sellable_quantity(symbol)
            if quantity > sellable:
                order.status = "REJECTED"
                order.reject_reason = f"T+1: only {sellable} sellable"
                self._orders.append(order)
                return order

        order.status = "PENDING"
        self._orders.append(order)
        return order

    def execute_pending_orders(
        self,
        market_data: dict[str, dict],
        trade_date: date,
    ) -> list[PaperOrder]:
        """Execute pending orders at market open.

        Args:
            market_data: {symbol: {open, high, low, close, volume}}
            trade_date: Current trading date
        """
        self._current_date = trade_date
        self._positions.update_settlement(trade_date)

        filled_orders = []

        for order in self._orders:
            if order.status != "PENDING":
                continue

            if order.symbol not in market_data:
                order.status = "REJECTED"
                order.reject_reason = "No market data"
                continue

            bar = market_data[order.symbol]
            fill_price = bar["open"]  # Execute at open

            # Check limit price
            if order.order_type == "LIMIT" and order.limit_price:
                if order.side == "BUY" and fill_price > order.limit_price:
                    continue  # Price too high, keep pending
                if order.side == "SELL" and fill_price < order.limit_price:
                    continue  # Price too low, keep pending

            # Execute
            amount = fill_price * order.quantity
            if order.side == "BUY":
                costs = self._fee_calc.calculate_buy_cost(amount, trade_date)
                total_cost = amount + costs["total"]

                if total_cost > self.available_cash:
                    order.status = "REJECTED"
                    order.reject_reason = "Insufficient funds at execution"
                    continue

                self._account.apply_trade(amount, costs["total"], "BUY", trade_date)
                self._positions.add_position(
                    order.symbol, order.quantity, fill_price, trade_date
                )
            else:  # SELL
                costs = self._fee_calc.calculate_sell_cost(amount, trade_date)
                self._positions.reduce_position(order.symbol, order.quantity, trade_date)
                self._account.apply_trade(amount, costs["total"], "SELL", trade_date)

            order.status = "FILLED"
            order.filled_at = datetime.now()
            order.fill_price = fill_price
            order.fill_quantity = order.quantity
            order.commission = costs["total"]
            filled_orders.append(order)

        self._save_state()
        return filled_orders

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        for order in self._orders:
            if order.order_id == order_id and order.status == "PENDING":
                order.status = "CANCELLED"
                return True
        return False

    def get_orders(self, status: Optional[str] = None) -> list[PaperOrder]:
        """Get orders, optionally filtered by status."""
        if status:
            return [o for o in self._orders if o.status == status]
        return self._orders.copy()

    def get_equity(self, market_data: dict[str, dict]) -> float:
        """Calculate total equity."""
        equity = self._account.balance
        for symbol, lots in self._positions.positions.items():
            if symbol in market_data:
                qty = sum(lot.quantity for lot in lots)
                equity += qty * market_data[symbol]["close"]
        return equity

    # -------------------------------------------------------------------------
    # State Persistence
    # -------------------------------------------------------------------------

    def _save_state(self) -> None:
        """Save state to file for restart recovery."""
        state = {
            "cash_balance": self._account.balance,
            "cash_frozen": self._account.frozen,
            "order_counter": self._order_counter,
            "current_date": self._current_date.isoformat() if self._current_date else None,
            "positions": {},
            "orders": [o.to_dict() for o in self._orders[-100:]],  # Keep last 100
        }

        for symbol, lots in self._positions.positions.items():
            state["positions"][symbol] = [
                {
                    "quantity": lot.quantity,
                    "cost_price": lot.cost_price,
                    "buy_date": lot.buy_date.isoformat(),
                    "sellable": lot.sellable,
                }
                for lot in lots
            ]

        self._state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _restore_state(self) -> None:
        """Restore state from file."""
        if not self._state_file.exists():
            return

        try:
            state = json.loads(self._state_file.read_text(encoding="utf-8"))

            self._account.balance = state.get("cash_balance", self._account.balance)
            self._account.frozen = state.get("cash_frozen", 0)
            self._order_counter = state.get("order_counter", 0)

            if state.get("current_date"):
                self._current_date = date.fromisoformat(state["current_date"])

            # Restore positions
            from a_share_quant.rules import PositionLot
            for symbol, lots_data in state.get("positions", {}).items():
                self._positions.positions[symbol] = [
                    PositionLot(
                        quantity=lot["quantity"],
                        cost_price=lot["cost_price"],
                        buy_date=date.fromisoformat(lot["buy_date"]),
                        sellable=lot["sellable"],
                    )
                    for lot in lots_data
                ]

        except Exception:
            pass  # Start fresh if restore fails


# =============================================================================
# Daily Scheduler
# =============================================================================

@dataclass
class DailySignal:
    """Signal generated for daily execution."""

    trade_date: date
    symbol: str
    action: str  # BUY, SELL, HOLD
    quantity: int
    strategy_name: str
    reason: str
    confidence: float = 0.0


class DailyScheduler:
    """Schedule daily signal generation and execution.

    Workflow:
    1. At close: Generate signals for next day
    2. At next open: Execute signals
    3. Record results
    """

    def __init__(self, broker: PaperBroker):
        self._broker = broker
        self._pending_signals: list[DailySignal] = []
        self._execution_log: list[dict] = []

    def generate_signals(
        self,
        signals: list[DailySignal],
        trade_date: date,
    ) -> None:
        """Queue signals for next day execution."""
        self._pending_signals = [s for s in signals if s.action != "HOLD"]

    def execute_signals(
        self,
        market_data: dict[str, dict],
        trade_date: date,
    ) -> list[PaperOrder]:
        """Execute pending signals."""
        orders = []

        for signal in self._pending_signals:
            order = self._broker.submit_order(
                symbol=signal.symbol,
                side=signal.action,
                quantity=signal.quantity,
                strategy_name=signal.strategy_name,
            )
            orders.append(order)

        # Execute at open
        filled = self._broker.execute_pending_orders(market_data, trade_date)

        # Log execution
        self._execution_log.append({
            "date": trade_date.isoformat(),
            "signals": len(self._pending_signals),
            "orders": len(orders),
            "filled": len(filled),
        })

        self._pending_signals = []
        return filled

    def get_execution_log(self) -> list[dict]:
        return self._execution_log.copy()
