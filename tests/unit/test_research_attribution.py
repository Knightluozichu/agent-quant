"""Tests for research protocol and attribution."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from a_share_quant.research import (
    BuyAndHoldBaseline,
    CashBaseline,
    CostStressTester,
    ParameterPerturbation,
    RandomBaseline,
    ResearchProtocol,
    WalkForwardValidator,
)
from a_share_quant.attribution import (
    AttributionEngine,
    FailureAnalysis,
    analyze_failures,
)


class TestResearchProtocol:
    """Tests for research protocol."""

    def test_time_splits(self):
        protocol = ResearchProtocol(
            full_start=date(2020, 1, 1),
            full_end=date(2024, 12, 31),
        )
        splits = protocol.get_splits()

        assert "train" in splits
        assert "validation" in splits
        assert "test" in splits

        # Check ratios roughly
        total = protocol.total_days
        train = splits["train"]
        assert train.days == pytest.approx(total * 0.6, rel=0.05)

    def test_invalid_ratios(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            ResearchProtocol(
                full_start=date(2020, 1, 1),
                full_end=date(2024, 12, 31),
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3,  # Sums to 1.1
            )

    def test_test_period_locked(self):
        protocol = ResearchProtocol(
            full_start=date(2020, 1, 1),
            full_end=date(2024, 12, 31),
        )
        test = protocol.get_test_period()
        assert test.name == "test"
        assert test.end_date == date(2024, 12, 31)


class TestWalkForwardValidator:
    """Tests for walk-forward validation."""

    def test_generate_windows(self):
        wf = WalkForwardValidator(
            train_days=252,
            test_days=63,
            step_days=21,
        )
        windows = wf.generate_windows(date(2020, 1, 1), date(2024, 12, 31))

        assert len(windows) > 0
        assert wf.validate_no_overlap(windows)

    def test_window_structure(self):
        wf = WalkForwardValidator(train_days=100, test_days=20, step_days=10)
        windows = wf.generate_windows(date(2020, 1, 1), date(2021, 12, 31))

        for w in windows:
            assert w.test_start > w.train_end
            assert (w.train_end - w.train_start).days == 100


class TestBaselineStrategies:
    """Tests for baseline strategies."""

    def test_buy_and_hold(self):
        prices = pd.Series([100, 110, 105, 115, 120])
        baseline = BuyAndHoldBaseline()
        equity = baseline.generate_returns(prices, 1_000_000)

        assert equity.iloc[0] == 1_000_000
        assert equity.iloc[-1] == 1_200_000  # 20% gain

    def test_cash_baseline(self):
        prices = pd.Series([100] * 252)  # 1 year
        baseline = CashBaseline(annual_rate=0.02)
        equity = baseline.generate_returns(prices, 1_000_000)

        # Should earn ~2% annually
        final = equity.iloc[-1]
        assert final > 1_000_000
        assert final < 1_030_000  # Less than 3%

    def test_random_baseline(self):
        prices = pd.Series([100 + i for i in range(100)])
        baseline = RandomBaseline(seed=42)
        equity = baseline.generate_returns(prices, 1_000_000)

        assert len(equity) == len(prices)
        assert equity.iloc[0] == 1_000_000


class TestCostStressTester:
    """Tests for cost stress testing."""

    def test_stress_test(self):
        returns = pd.Series([0.01, -0.005, 0.008, 0.002, -0.003])
        tester = CostStressTester(multipliers=[1.0, 2.0, 3.0])
        results = tester.stress_test(returns, base_costs=0.001)

        assert len(results) == 3
        # Higher costs should reduce Sharpe
        assert results[2].stressed_sharpe <= results[0].stressed_sharpe


class TestParameterPerturbation:
    """Tests for parameter perturbation."""

    def test_generate_perturbations(self):
        perturb = ParameterPerturbation(perturbation_pct=0.20, n_samples=5)
        values = perturb.generate_perturbations(100)

        assert len(values) == 5
        assert min(values) == pytest.approx(80)
        assert max(values) == pytest.approx(120)

    def test_sensitivity(self):
        perturb = ParameterPerturbation()
        sensitivity = perturb.calculate_sensitivity(
            base_metric=1.0,
            perturbed_metrics=[0.9, 1.0, 1.1, 0.95, 1.05],
        )
        assert sensitivity > 0


class TestAttributionEngine:
    """Tests for attribution engine."""

    def test_attribute_trade(self):
        engine = AttributionEngine()
        attr = engine.attribute_trade(
            entry_price=100,
            exit_price=110,
            quantity=1000,
            entry_date=date(2024, 1, 1),
            exit_date=date(2024, 1, 15),
            costs=50,
        )

        assert attr.gross_pnl == 10000  # (110-100) * 1000
        assert attr.net_pnl == 9950  # 10000 - 50
        assert attr.costs == 50

    def test_mfe_mae(self):
        engine = AttributionEngine()
        prices = pd.Series([100, 105, 115, 108, 110])  # Max 115, min 100
        attr = engine.attribute_trade(
            entry_price=100,
            exit_price=110,
            quantity=1000,
            entry_date=date(2024, 1, 1),
            exit_date=date(2024, 1, 5),
            costs=0,
            prices_during_hold=prices,
        )

        assert attr.mfe == 15000  # (115-100) * 1000
        assert attr.mae == 0  # Never went below entry
        assert attr.mfe_capture == pytest.approx(10000 / 15000, rel=0.01)


class TestFailureAnalysis:
    """Tests for failure analysis."""

    def test_analyze_failures(self):
        trades = [
            {"pnl": 100, "exit_reason": "TAKE_PROFIT"},
            {"pnl": -50, "exit_reason": "STOP_LOSS"},
            {"pnl": -30, "exit_reason": "STOP_LOSS"},
            {"pnl": 80, "exit_reason": "TIME_EXIT"},
            {"pnl": -20, "exit_reason": "REGIME_CHANGE"},
        ]
        analysis = analyze_failures(trades)

        assert analysis.total_trades == 5
        assert analysis.winning_trades == 2
        assert analysis.losing_trades == 3
        assert analysis.win_rate == pytest.approx(0.4)
        assert analysis.failure_reasons["STOP_LOSS"] == 2
        assert analysis.top_failure_reason == "STOP_LOSS"

    def test_empty_trades(self):
        analysis = analyze_failures([])
        assert analysis.total_trades == 0
        assert analysis.win_rate == 0
