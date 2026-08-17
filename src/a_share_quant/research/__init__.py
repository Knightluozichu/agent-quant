"""Research protocol for rigorous backtesting.

Implements:
1. Time splitting (train/validation/test)
2. Walk-forward validation
3. Baseline strategies (buy-and-hold, cash, random)
4. Cost stress testing
5. Parameter perturbation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# Time Splitting
# =============================================================================


@dataclass
class TimeSplit:
    """A time period split for research."""

    name: str
    start_date: date
    end_date: date

    @property
    def days(self) -> int:
        return (self.end_date - self.start_date).days

    def contains(self, d: date) -> bool:
        return self.start_date <= d <= self.end_date


@dataclass
class ResearchProtocol:
    """Research protocol with time splits.

    Standard split:
    - Train: 60% for strategy development
    - Validation: 20% for parameter tuning
    - Test: 20% locked for final evaluation (NEVER touch during development)
    """

    full_start: date
    full_end: date
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    test_ratio: float = 0.20

    def __post_init__(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

    @property
    def total_days(self) -> int:
        return (self.full_end - self.full_start).days

    def get_splits(self) -> dict[str, TimeSplit]:
        """Get train/validation/test splits."""
        total = self.total_days
        train_days = int(total * self.train_ratio)
        val_days = int(total * self.val_ratio)

        train_end = self.full_start + timedelta(days=train_days)
        val_end = train_end + timedelta(days=val_days)

        return {
            "train": TimeSplit("train", self.full_start, train_end),
            "validation": TimeSplit("validation", train_end + timedelta(days=1), val_end),
            "test": TimeSplit("test", val_end + timedelta(days=1), self.full_end),
        }

    def get_train_period(self) -> TimeSplit:
        return self.get_splits()["train"]

    def get_validation_period(self) -> TimeSplit:
        return self.get_splits()["validation"]

    def get_test_period(self) -> TimeSplit:
        """Get locked test period. NEVER use for development."""
        return self.get_splits()["test"]


# =============================================================================
# Walk-Forward Validation
# =============================================================================


@dataclass
class WalkForwardWindow:
    """A single walk-forward window."""

    fold: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date


class WalkForwardValidator:
    """Walk-forward validation framework.

    Splits data into rolling windows:
    - Train on historical window
    - Test on subsequent period
    - Roll forward and repeat
    """

    def __init__(
        self,
        train_days: int = 252,  # 1 year training
        test_days: int = 63,  # 3 months testing
        step_days: int = 21,  # Step forward 1 month
    ):
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days

    def generate_windows(
        self,
        start_date: date,
        end_date: date,
    ) -> list[WalkForwardWindow]:
        """Generate walk-forward windows."""
        windows = []
        fold = 0

        current_train_start = start_date
        while True:
            train_end = current_train_start + timedelta(days=self.train_days)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.test_days)

            # Check if we have enough data
            if test_end > end_date:
                break

            windows.append(
                WalkForwardWindow(
                    fold=fold,
                    train_start=current_train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )

            fold += 1
            current_train_start += timedelta(days=self.step_days)

        return windows

    def validate_no_overlap(self, windows: list[WalkForwardWindow]) -> bool:
        """Verify test windows don't overlap with training."""
        for w in windows:
            if w.test_start <= w.train_end:
                return False
        return True


# =============================================================================
# Baseline Strategies
# =============================================================================


class BaselineStrategy:
    """Base class for baseline strategies."""

    name: str = "BASELINE"

    def generate_returns(
        self,
        prices: pd.Series,
        initial_capital: float = 1_000_000,
    ) -> pd.Series:
        """Generate returns for the baseline."""
        raise NotImplementedError


class BuyAndHoldBaseline(BaselineStrategy):
    """Buy and hold benchmark."""

    name = "BUY_AND_HOLD"

    def generate_returns(
        self,
        prices: pd.Series,
        initial_capital: float = 1_000_000,
    ) -> pd.Series:
        """Buy at start, hold until end."""
        if len(prices) == 0:
            return pd.Series(dtype=float)

        # Normalize to initial capital
        return prices / prices.iloc[0] * initial_capital


class CashBaseline(BaselineStrategy):
    """Cash (risk-free) benchmark."""

    name = "CASH"

    def __init__(self, annual_rate: float = 0.02):
        self.annual_rate = annual_rate

    def generate_returns(
        self,
        prices: pd.Series,
        initial_capital: float = 1_000_000,
    ) -> pd.Series:
        """Earn risk-free rate."""
        n_days = len(prices)
        daily_rate = self.annual_rate / 252
        returns = [initial_capital * (1 + daily_rate) ** i for i in range(n_days)]
        return pd.Series(returns, index=prices.index)


class RandomBaseline(BaselineStrategy):
    """Random trading benchmark."""

    name = "RANDOM"

    def __init__(self, seed: int = 42, trade_probability: float = 0.1):
        self.seed = seed
        self.trade_probability = trade_probability

    def generate_returns(
        self,
        prices: pd.Series,
        initial_capital: float = 1_000_000,
    ) -> pd.Series:
        """Random entry/exit."""
        rng = np.random.default_rng(self.seed)
        n = len(prices)

        equity = [initial_capital]
        position = 0  # 0 = no position, 1 = long
        entry_price = 0.0

        for i in range(1, n):
            price = prices.iloc[i]
            prev_price = prices.iloc[i - 1]

            if position == 0:
                # Random entry
                if rng.random() < self.trade_probability:
                    position = 1
                    entry_price = prev_price
                equity.append(equity[-1])
            else:
                # Update equity with price change
                daily_return = (price - prev_price) / prev_price
                equity.append(equity[-1] * (1 + daily_return))

                # Random exit
                if rng.random() < self.trade_probability:
                    position = 0

        return pd.Series(equity, index=prices.index)


# =============================================================================
# Cost Stress Testing
# =============================================================================


@dataclass
class CostStressResult:
    """Result of cost stress test."""

    base_sharpe: float
    stressed_sharpe: float
    sharpe_decay: float
    base_return: float
    stressed_return: float
    cost_multiplier: float


class CostStressTester:
    """Test strategy robustness to increased costs."""

    def __init__(self, multipliers: list[float] | None = None):
        self.multipliers = multipliers or [1.0, 1.5, 2.0, 3.0]

    def stress_test(
        self,
        base_returns: pd.Series,
        base_costs: float,
    ) -> list[CostStressResult]:
        """Run cost stress test."""
        results = []

        base_sharpe = self._calculate_sharpe(base_returns)
        base_total = (1 + base_returns).prod() - 1

        for mult in self.multipliers:
            # Increase costs
            additional_cost = base_costs * (mult - 1) / 252  # Daily cost
            stressed_returns = base_returns - additional_cost

            stressed_sharpe = self._calculate_sharpe(stressed_returns)
            stressed_total = (1 + stressed_returns).prod() - 1

            results.append(
                CostStressResult(
                    base_sharpe=base_sharpe,
                    stressed_sharpe=stressed_sharpe,
                    sharpe_decay=(base_sharpe - stressed_sharpe) / base_sharpe
                    if base_sharpe != 0
                    else 0,
                    base_return=base_total,
                    stressed_return=stressed_total,
                    cost_multiplier=mult,
                )
            )

        return results

    def _calculate_sharpe(self, returns: pd.Series, rf: float = 0.02) -> float:
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        excess = returns.mean() - rf / 252
        return excess / returns.std() * np.sqrt(252)


# =============================================================================
# Parameter Perturbation
# =============================================================================


@dataclass
class PerturbationResult:
    """Result of parameter perturbation test."""

    param_name: str
    base_value: float
    perturbed_values: list[float]
    base_metric: float
    perturbed_metrics: list[float]
    sensitivity: float  # Std of metrics / base metric


class ParameterPerturbation:
    """Test strategy robustness to parameter changes."""

    def __init__(self, perturbation_pct: float = 0.20, n_samples: int = 5):
        self.perturbation_pct = perturbation_pct
        self.n_samples = n_samples

    def generate_perturbations(self, base_value: float) -> list[float]:
        """Generate perturbed parameter values."""
        delta = base_value * self.perturbation_pct
        perturbations = []
        for i in range(self.n_samples):
            # Uniform distribution around base
            offset = (i / (self.n_samples - 1) - 0.5) * 2 * delta if self.n_samples > 1 else 0
            perturbations.append(base_value + offset)
        return perturbations

    def calculate_sensitivity(
        self,
        base_metric: float,
        perturbed_metrics: list[float],
    ) -> float:
        """Calculate parameter sensitivity."""
        if base_metric == 0 or len(perturbed_metrics) < 2:
            return 0.0
        return np.std(perturbed_metrics) / abs(base_metric)
