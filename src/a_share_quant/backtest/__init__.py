"""Event-driven backtest engine.

Core loop:
1. Load data for date range
2. For each trading day:
   a. Update market state (regime detection)
   b. Update stock states
   c. Generate strategy decisions
   d. Create trade plans with exit rules
   e. Execute orders (with A-share rules)
   f. Update positions and account
   g. Record metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional, Protocol

import pandas as pd

from a_share_quant.data.providers.base import DataProvider
from a_share_quant.rules import (
    CashAccount,
    FeeCalculator,
    PositionLedger,
    PriceLimitCalculator,
    SettlementRule,
)


# =============================================================================
# Events
# =============================================================================


@dataclass
class MarketEvent:
    """New market data available."""

    trade_date: date


@dataclass
class SignalEvent:
    """Strategy generated a signal."""

    trade_date: date
    symbol: str
    action: str  # BUY, SELL, HOLD
    strength: float
    strategy_name: str
    metadata: dict = field(default_factory=dict)


@dataclass
class OrderEvent:
    """Order to be executed."""

    trade_date: date
    symbol: str
    side: str  # BUY, SELL
    quantity: int
    order_type: str = "MARKET"  # MARKET, LIMIT
    limit_price: Optional[float] = None
    strategy_name: str = ""


@dataclass
class FillEvent:
    """Order was filled."""

    trade_date: date
    symbol: str
    side: str
    quantity: int
    fill_price: float
    commission: float
    stamp_tax: float
    transfer_fee: float


# =============================================================================
# Execution Simulator
# =============================================================================


class ExecutionSimulator:
    """Simulate order execution with A-share rules.

    Rules:
    - T+1: Cannot sell shares bought today
    - Price limits: May not fill at limit up/down
    - Fees: Commission, stamp tax, transfer fee
    - Lot size: Must be multiples of 100 (except ETFs)
    """

    def __init__(
        self,
        fee_calculator: FeeCalculator | None = None,
        price_limit_calculator: PriceLimitCalculator | None = None,
    ):
        self._fee_calc = fee_calculator or FeeCalculator()
        self._price_limit_calc = price_limit_calculator or PriceLimitCalculator()
        self._settlement = SettlementRule()

    def simulate_fill(
        self,
        order: OrderEvent,
        bar: dict,  # OHLCV for the day
        prev_close: float,
        board: str,
    ) -> Optional[FillEvent]:
        """Simulate order execution.

        Returns FillEvent if order can be filled, None otherwise.
        """
        # Check price limits
        upper_limit, lower_limit = self._price_limit_calc.calculate_limit_prices(
            prev_close, board, order.trade_date
        )

        # Determine fill price
        if order.order_type == "LIMIT" and order.limit_price:
            fill_price = order.limit_price
            # Check if limit price is achievable
            if order.side == "BUY" and fill_price < bar["low"]:
                return None  # Price too low, no fill
            if order.side == "SELL" and fill_price > bar["high"]:
                return None  # Price too high, no fill
        else:
            # Market order: use open price (conservative)
            fill_price = bar["open"]

        # Check if at limit up (no sellers) or limit down (no buyers)
        is_limit_up = abs(bar["close"] - upper_limit) < 0.01
        is_limit_down = abs(bar["close"] - lower_limit) < 0.01

        if order.side == "BUY" and is_limit_up:
            # At limit up, may not be able to buy
            if bar["open"] >= upper_limit:
                return None  # Opened at limit up, no sellers

        if order.side == "SELL" and is_limit_down:
            # At limit down, may not be able to sell
            if bar["open"] <= lower_limit:
                return None  # Opened at limit down, no buyers

        # Calculate fees
        amount = fill_price * order.quantity
        if order.side == "BUY":
            costs = self._fee_calc.calculate_buy_cost(amount, order.trade_date)
        else:
            costs = self._fee_calc.calculate_sell_cost(amount, order.trade_date)

        return FillEvent(
            trade_date=order.trade_date,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=costs["commission"],
            stamp_tax=costs["stamp_tax"],
            transfer_fee=costs["transfer_fee"],
        )


# =============================================================================
# Performance Metrics
# =============================================================================


@dataclass
class PerformanceMetrics:
    """Performance metrics for a backtest."""

    total_return: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_holding_days: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "avg_holding_days": self.avg_holding_days,
        }


class MetricsCalculator:
    """Calculate performance metrics from equity curve."""

    def calculate(
        self,
        equity_curve: pd.Series,
        trades: list[dict],
        risk_free_rate: float = 0.02,
    ) -> PerformanceMetrics:
        """Calculate all metrics."""
        if len(equity_curve) < 2:
            return PerformanceMetrics()

        returns = equity_curve.pct_change().dropna()

        # Total return
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

        # Annual return (assuming 252 trading days)
        n_days = len(equity_curve)
        annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0

        # Volatility (annualized)
        volatility = returns.std() * (252**0.5) if len(returns) > 1 else 0

        # Sharpe ratio
        excess_return = annual_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0

        # Max drawdown
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0

        # Trade statistics
        winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in trades if t.get("pnl", 0) < 0]

        win_rate = len(winning_trades) / len(trades) if trades else 0

        gross_profit = sum(t.get("pnl", 0) for t in winning_trades)
        gross_loss = abs(sum(t.get("pnl", 0) for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_holding = sum(t.get("holding_days", 0) for t in trades) / len(trades) if trades else 0

        return PerformanceMetrics(
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            avg_holding_days=avg_holding,
        )


# =============================================================================
# Backtest Engine
# =============================================================================


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    start_date: date
    end_date: date
    initial_capital: float = 1_000_000.0
    symbols: list[str] = field(default_factory=list)
    commission_rate: float = 0.00025
    slippage: float = 0.0
    benchmark: str = "510300.SSE"


@dataclass
class BacktestResult:
    """Result of a backtest run."""

    config: BacktestConfig
    equity_curve: pd.Series
    trades: list[dict]
    metrics: PerformanceMetrics
    daily_returns: pd.Series
    positions_history: list[dict]


class BacktestEngine:
    """Event-driven backtest engine."""

    def __init__(
        self,
        data_provider: DataProvider,
        config: BacktestConfig,
    ):
        self._provider = data_provider
        self._config = config
        self._executor = ExecutionSimulator()
        self._metrics_calc = MetricsCalculator()

        # State
        self._account = CashAccount(balance=config.initial_capital)
        self._positions = PositionLedger()
        self._equity_history: list[tuple[date, float]] = []
        self._trades: list[dict] = []
        self._daily_data: dict[str, pd.DataFrame] = {}

    def _load_data(self) -> None:
        """Load all required data."""
        for symbol in self._config.symbols:
            df = self._provider.get_daily_bars(
                symbol,
                self._config.start_date,
                self._config.end_date,
            )
            if not df.empty:
                self._daily_data[symbol] = df

    def _get_trading_days(self) -> list[date]:
        """Get list of trading days in range."""
        try:
            cal = self._provider.get_trading_calendar(
                "SSE",
                self._config.start_date,
                self._config.end_date,
            )
            if not cal.empty and "trade_date" in cal.columns:
                return sorted(cal["trade_date"].tolist())
        except Exception:
            pass

        # Fallback: extract from data
        all_dates = set()
        for df in self._daily_data.values():
            if "trade_date" in df.columns:
                all_dates.update(df["trade_date"].tolist())
        return sorted(all_dates)

    def _get_bar(self, symbol: str, trade_date: date) -> Optional[dict]:
        """Get OHLCV bar for symbol on date."""
        if symbol not in self._daily_data:
            return None
        df = self._daily_data[symbol]
        row = df[df["trade_date"] == trade_date]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    def _calculate_equity(self, trade_date: date) -> float:
        """Calculate total equity (cash + positions market value)."""
        equity = self._account.balance

        for symbol, lots in self._positions.positions.items():
            bar = self._get_bar(symbol, trade_date)
            if bar:
                qty = sum(lot.quantity for lot in lots)
                equity += qty * bar["close"]

        return equity

    def run(
        self,
        strategy_fn: Optional[Callable[[date, dict, dict], list[OrderEvent]]] = None,
    ) -> BacktestResult:
        """Run the backtest.

        Args:
            strategy_fn: Optional strategy function that takes
                (trade_date, market_data, positions) and returns orders.
        """
        self._load_data()
        trading_days = self._get_trading_days()

        for trade_date in trading_days:
            # Update settlement (T+1)
            self._positions.update_settlement(trade_date)

            # Get market data for today
            market_data = {}
            for symbol in self._config.symbols:
                bar = self._get_bar(symbol, trade_date)
                if bar:
                    market_data[symbol] = bar

            # Generate orders from strategy
            orders = []
            if strategy_fn:
                positions_info = self._positions.get_all_positions()
                orders = strategy_fn(trade_date, market_data, positions_info)

            # Execute orders
            for order in orders:
                self._execute_order(order, market_data)

            # Record equity
            equity = self._calculate_equity(trade_date)
            self._equity_history.append((trade_date, equity))

        # Build results
        equity_series = pd.Series(
            [e[1] for e in self._equity_history],
            index=[e[0] for e in self._equity_history],
        )
        daily_returns = equity_series.pct_change().dropna()

        metrics = self._metrics_calc.calculate(equity_series, self._trades)

        return BacktestResult(
            config=self._config,
            equity_curve=equity_series,
            trades=self._trades,
            metrics=metrics,
            daily_returns=daily_returns,
            positions_history=[],
        )

    def _execute_order(self, order: OrderEvent, market_data: dict) -> None:
        """Execute a single order."""
        if order.symbol not in market_data:
            return

        bar = market_data[order.symbol]

        # Get previous close for price limit calculation
        # For simplicity, use open as approximation
        prev_close = bar.get("pre_close", bar["open"])

        # Determine board (simplified)
        board = "MAIN"
        if order.symbol.startswith("30"):
            board = "CHINEXT"
        elif order.symbol.startswith("68"):
            board = "STAR"

        # Check T+1 for sells
        if order.side == "SELL":
            sellable = self._positions.get_sellable_quantity(order.symbol)
            if order.quantity > sellable:
                return  # Cannot sell more than sellable

        # Simulate fill
        fill = self._executor.simulate_fill(order, bar, prev_close, board)
        if fill is None:
            return

        # Apply fill
        amount = fill.fill_price * fill.quantity
        total_fee = fill.commission + fill.stamp_tax + fill.transfer_fee

        if order.side == "BUY":
            # Check cash availability
            if amount + total_fee > self._account.available:
                return  # Insufficient funds
            self._account.apply_trade(amount, total_fee, "BUY", order.trade_date)
            self._positions.add_position(
                order.symbol, fill.quantity, fill.fill_price, order.trade_date
            )
        else:  # SELL
            self._positions.reduce_position(order.symbol, fill.quantity, order.trade_date)
            self._account.apply_trade(amount, total_fee, "SELL", order.trade_date)

        # Record trade
        self._trades.append(
            {
                "date": order.trade_date,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": fill.quantity,
                "price": fill.fill_price,
                "commission": fill.commission,
                "stamp_tax": fill.stamp_tax,
                "transfer_fee": fill.transfer_fee,
                "strategy": order.strategy_name,
            }
        )
