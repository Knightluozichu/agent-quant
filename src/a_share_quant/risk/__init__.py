"""Risk management framework.

Core risk controls:
1. Position sizing (Kelly criterion, fixed fractional)
2. Portfolio-level limits
3. Drawdown control
4. Correlation limits
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# Position Sizing
# =============================================================================

@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    symbol: str
    quantity: int
    amount: float
    pct_of_equity: float
    method: str
    reason: str


class PositionSizer:
    """Calculate position sizes based on risk parameters."""

    def __init__(
        self,
        max_position_pct: float = 0.20,  # Max 20% per position
        max_total_exposure: float = 0.80,  # Max 80% total exposure
        risk_per_trade: float = 0.02,  # Risk 2% per trade
        lot_size: int = 100,  # A-share lot size
    ):
        self.max_position_pct = max_position_pct
        self.max_total_exposure = max_total_exposure
        self.risk_per_trade = risk_per_trade
        self.lot_size = lot_size

    def fixed_fractional(
        self,
        equity: float,
        price: float,
        stop_loss_price: float,
        current_exposure: float = 0.0,
    ) -> PositionSizeResult:
        """Fixed fractional position sizing.

        Risk a fixed percentage of equity per trade.
        """
        # Risk amount
        risk_amount = equity * self.risk_per_trade

        # Risk per share
        risk_per_share = abs(price - stop_loss_price)
        if risk_per_share <= 0:
            return PositionSizeResult(
                symbol="", quantity=0, amount=0, pct_of_equity=0,
                method="fixed_fractional", reason="Invalid stop loss"
            )

        # Position size based on risk
        quantity = int(risk_amount / risk_per_share)

        # Apply max position limit
        max_amount = equity * self.max_position_pct
        max_quantity = int(max_amount / price)
        quantity = min(quantity, max_quantity)

        # Apply exposure limit
        available_exposure = equity * self.max_total_exposure - current_exposure
        max_by_exposure = int(available_exposure / price)
        quantity = min(quantity, max_by_exposure)

        # Round to lot size
        quantity = (quantity // self.lot_size) * self.lot_size

        amount = quantity * price
        pct = amount / equity if equity > 0 else 0

        return PositionSizeResult(
            symbol="",
            quantity=quantity,
            amount=amount,
            pct_of_equity=pct,
            method="fixed_fractional",
            reason=f"Risk {self.risk_per_trade:.1%} of equity",
        )

    def kelly_criterion(
        self,
        equity: float,
        price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        kelly_fraction: float = 0.5,  # Half-Kelly for safety
    ) -> PositionSizeResult:
        """Kelly criterion position sizing.

        Uses historical win rate and payoff ratio.
        """
        if avg_loss <= 0 or win_rate <= 0:
            return PositionSizeResult(
                symbol="", quantity=0, amount=0, pct_of_equity=0,
                method="kelly", reason="Invalid parameters"
            )

        # Kelly formula: f = (bp - q) / b
        # b = avg_win / avg_loss (payoff ratio)
        # p = win_rate
        # q = 1 - p
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p

        kelly_pct = (b * p - q) / b
        kelly_pct = max(0, kelly_pct)  # Never negative

        # Apply fraction (half-Kelly)
        position_pct = kelly_pct * kelly_fraction

        # Apply max position limit
        position_pct = min(position_pct, self.max_position_pct)

        amount = equity * position_pct
        quantity = int(amount / price)
        quantity = (quantity // self.lot_size) * self.lot_size
        amount = quantity * price

        return PositionSizeResult(
            symbol="",
            quantity=quantity,
            amount=amount,
            pct_of_equity=amount / equity if equity > 0 else 0,
            method="kelly",
            reason=f"Kelly={kelly_pct:.1%}, using {kelly_fraction:.0%} Kelly",
        )


# =============================================================================
# Portfolio Risk Limits
# =============================================================================

@dataclass
class RiskLimits:
    """Portfolio-level risk limits."""
    max_drawdown: float = 0.15  # Max 15% drawdown
    max_daily_loss: float = 0.05  # Max 5% daily loss
    max_positions: int = 10
    max_sector_exposure: float = 0.40  # Max 40% per sector
    max_correlation: float = 0.70  # Max correlation between positions
    min_cash_reserve: float = 0.10  # Keep 10% cash


@dataclass
class RiskCheckResult:
    """Result of a risk check."""
    passed: bool
    rule_id: str
    message: str
    current_value: float = 0.0
    limit_value: float = 0.0


class PortfolioRiskManager:
    """Manage portfolio-level risk."""

    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()
        self._peak_equity: float = 0.0
        self._daily_start_equity: float = 0.0
        self._trade_date: Optional[date] = None

    def update_state(self, equity: float, trade_date: date) -> None:
        """Update risk manager state."""
        if self._trade_date != trade_date:
            # New day
            self._daily_start_equity = equity
            self._trade_date = trade_date

        self._peak_equity = max(self._peak_equity, equity)

    def check_drawdown(self, equity: float) -> RiskCheckResult:
        """Check if drawdown limit is breached."""
        if self._peak_equity <= 0:
            return RiskCheckResult(True, "DRAWDOWN", "No peak equity set")

        drawdown = (self._peak_equity - equity) / self._peak_equity
        passed = drawdown <= self.limits.max_drawdown

        return RiskCheckResult(
            passed=passed,
            rule_id="DRAWDOWN",
            message=f"Drawdown {drawdown:.1%} {'within' if passed else 'exceeds'} limit {self.limits.max_drawdown:.1%}",
            current_value=drawdown,
            limit_value=self.limits.max_drawdown,
        )

    def check_daily_loss(self, equity: float) -> RiskCheckResult:
        """Check if daily loss limit is breached."""
        if self._daily_start_equity <= 0:
            return RiskCheckResult(True, "DAILY_LOSS", "No daily start equity set")

        daily_return = (equity - self._daily_start_equity) / self._daily_start_equity
        passed = daily_return >= -self.limits.max_daily_loss

        return RiskCheckResult(
            passed=passed,
            rule_id="DAILY_LOSS",
            message=f"Daily return {daily_return:.1%} {'within' if passed else 'exceeds'} limit -{self.limits.max_daily_loss:.1%}",
            current_value=daily_return,
            limit_value=-self.limits.max_daily_loss,
        )

    def check_position_count(self, current_positions: int) -> RiskCheckResult:
        """Check if position count limit is breached."""
        passed = current_positions < self.limits.max_positions
        return RiskCheckResult(
            passed=passed,
            rule_id="POSITION_COUNT",
            message=f"Positions {current_positions} {'within' if passed else 'exceeds'} limit {self.limits.max_positions}",
            current_value=current_positions,
            limit_value=self.limits.max_positions,
        )

    def check_cash_reserve(self, equity: float, cash: float) -> RiskCheckResult:
        """Check if cash reserve is maintained."""
        cash_pct = cash / equity if equity > 0 else 0
        passed = cash_pct >= self.limits.min_cash_reserve
        return RiskCheckResult(
            passed=passed,
            rule_id="CASH_RESERVE",
            message=f"Cash {cash_pct:.1%} {'above' if passed else 'below'} minimum {self.limits.min_cash_reserve:.1%}",
            current_value=cash_pct,
            limit_value=self.limits.min_cash_reserve,
        )

    def can_open_position(
        self,
        equity: float,
        cash: float,
        current_positions: int,
    ) -> tuple[bool, list[RiskCheckResult]]:
        """Check if a new position can be opened."""
        checks = [
            self.check_drawdown(equity),
            self.check_daily_loss(equity),
            self.check_position_count(current_positions),
            self.check_cash_reserve(equity, cash),
        ]

        all_passed = all(c.passed for c in checks)
        return all_passed, checks


# =============================================================================
# Drawdown Control
# =============================================================================

class DrawdownController:
    """Control trading based on drawdown levels.

    Implements progressive de-risking as drawdown increases.
    """

    def __init__(
        self,
        warning_level: float = 0.05,  # 5% drawdown: reduce size
        critical_level: float = 0.10,  # 10% drawdown: halt new positions
        halt_level: float = 0.15,  # 15% drawdown: exit all
    ):
        self.warning_level = warning_level
        self.critical_level = critical_level
        self.halt_level = halt_level

    def get_exposure_multiplier(self, current_drawdown: float) -> float:
        """Get exposure multiplier based on drawdown.

        Returns:
            1.0 = full exposure
            0.5 = half exposure
            0.0 = no new positions
        """
        if current_drawdown >= self.halt_level:
            return 0.0
        elif current_drawdown >= self.critical_level:
            return 0.0  # No new positions
        elif current_drawdown >= self.warning_level:
            # Linear reduction from 1.0 to 0.5
            progress = (current_drawdown - self.warning_level) / (self.critical_level - self.warning_level)
            return 1.0 - 0.5 * progress
        else:
            return 1.0

    def should_halt_trading(self, current_drawdown: float) -> bool:
        """Check if trading should be halted."""
        return current_drawdown >= self.halt_level

    def should_exit_all(self, current_drawdown: float) -> bool:
        """Check if all positions should be exited."""
        return current_drawdown >= self.halt_level
