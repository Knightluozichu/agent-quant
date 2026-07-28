"""Tests for risk management and evolution."""

from datetime import date, datetime

import pytest

from a_share_quant.risk import (
    DrawdownController,
    PositionSizer,
    PortfolioRiskManager,
    RiskLimits,
)
from a_share_quant.evolution import (
    EvolutionManager,
    StrategyPerformance,
    StrategyStatus,
    StrategyVersion,
)


class TestPositionSizer:
    """Tests for position sizing."""

    def test_fixed_fractional(self):
        sizer = PositionSizer(risk_per_trade=0.02)
        result = sizer.fixed_fractional(
            equity=1_000_000,
            price=10.0,
            stop_loss_price=9.5,  # 5% stop
        )
        # Risk 2% = 20,000, risk per share = 0.5
        # Quantity = 20,000 / 0.5 = 40,000
        # But max position is 20% = 200,000 / 10 = 20,000
        assert result.quantity <= 20000
        assert result.quantity % 100 == 0  # Lot size

    def test_max_position_limit(self):
        sizer = PositionSizer(max_position_pct=0.10)
        result = sizer.fixed_fractional(
            equity=1_000_000,
            price=10.0,
            stop_loss_price=9.0,
        )
        assert result.amount <= 100_000  # 10% of equity

    def test_kelly_criterion(self):
        sizer = PositionSizer()
        result = sizer.kelly_criterion(
            equity=1_000_000,
            price=10.0,
            win_rate=0.6,
            avg_win=0.10,
            avg_loss=0.05,
        )
        assert result.quantity > 0
        assert result.method == "kelly"

    def test_kelly_negative_edge(self):
        """Kelly should return 0 for negative edge."""
        sizer = PositionSizer()
        result = sizer.kelly_criterion(
            equity=1_000_000,
            price=10.0,
            win_rate=0.3,  # Low win rate
            avg_win=0.05,
            avg_loss=0.10,  # High loss
        )
        assert result.quantity == 0


class TestPortfolioRiskManager:
    """Tests for portfolio risk management."""

    def test_drawdown_check(self):
        rm = PortfolioRiskManager(RiskLimits(max_drawdown=0.15))
        rm.update_state(1_000_000, date(2024, 1, 1))  # Peak
        rm.update_state(900_000, date(2024, 1, 15))  # 10% drawdown

        result = rm.check_drawdown(900_000)
        assert result.passed  # 10% < 15%

    def test_drawdown_breach(self):
        rm = PortfolioRiskManager(RiskLimits(max_drawdown=0.10))
        rm.update_state(1_000_000, date(2024, 1, 1))
        rm.update_state(850_000, date(2024, 1, 15))  # 15% drawdown

        result = rm.check_drawdown(850_000)
        assert not result.passed  # 15% > 10%

    def test_position_count(self):
        rm = PortfolioRiskManager(RiskLimits(max_positions=5))
        result = rm.check_position_count(3)
        assert result.passed

        result = rm.check_position_count(5)
        assert not result.passed

    def test_can_open_position(self):
        rm = PortfolioRiskManager()
        rm.update_state(1_000_000, date(2024, 1, 1))

        can_open, checks = rm.can_open_position(
            equity=1_000_000,
            cash=500_000,
            current_positions=3,
        )
        assert can_open


class TestDrawdownController:
    """Tests for drawdown control."""

    def test_full_exposure(self):
        ctrl = DrawdownController()
        assert ctrl.get_exposure_multiplier(0.02) == 1.0

    def test_reduced_exposure(self):
        ctrl = DrawdownController(warning_level=0.05, critical_level=0.10)
        multiplier = ctrl.get_exposure_multiplier(0.075)  # Midway
        assert 0.5 < multiplier < 1.0

    def test_no_new_positions(self):
        ctrl = DrawdownController(critical_level=0.10)
        assert ctrl.get_exposure_multiplier(0.12) == 0.0

    def test_halt_trading(self):
        ctrl = DrawdownController(halt_level=0.15)
        assert ctrl.should_halt_trading(0.16)
        assert not ctrl.should_halt_trading(0.14)


class TestEvolutionManager:
    """Tests for champion/challenger evolution."""

    def test_register_champion(self):
        em = EvolutionManager()
        champion = StrategyVersion(
            strategy_id="TREND_HOLD",
            version=1,
            name="Trend Hold v1",
            params={"ma_period": 20},
        )
        em.register_champion(champion)

        assert champion.status == StrategyStatus.CHAMPION
        assert em.get_champion("TREND_HOLD") == champion

    def test_add_challenger(self):
        em = EvolutionManager()
        champion = StrategyVersion(
            strategy_id="TREND_HOLD",
            version=1,
            name="Trend Hold v1",
            params={"ma_period": 20},
        )
        em.register_champion(champion)

        challenger = StrategyVersion(
            strategy_id="TREND_HOLD",
            version=2,
            name="Trend Hold v2",
            params={"ma_period": 30},
            parent_version=1,
        )
        assert em.add_challenger(challenger)
        assert len(em.get_challengers("TREND_HOLD")) == 1

    def test_max_challengers(self):
        em = EvolutionManager(max_challengers=2)
        champion = StrategyVersion(
            strategy_id="TEST", version=1, name="Test", params={}
        )
        em.register_champion(champion)

        for i in range(3):
            challenger = StrategyVersion(
                strategy_id="TEST", version=i+2, name=f"Test v{i+2}", params={}
            )
            result = em.add_challenger(challenger)
            if i < 2:
                assert result
            else:
                assert not result  # Max reached

    def test_promotion(self):
        em = EvolutionManager(min_sample_size=10, promotion_threshold=0.10)

        champion = StrategyVersion(
            strategy_id="TREND_HOLD", version=1, name="v1", params={}
        )
        champion.performance = StrategyPerformance(
            total_return=0.10, sharpe_ratio=1.0, sample_size=50
        )
        em.register_champion(champion)

        challenger = StrategyVersion(
            strategy_id="TREND_HOLD", version=2, name="v2", params={}
        )
        challenger.performance = StrategyPerformance(
            total_return=0.25, sharpe_ratio=1.5, sample_size=50
        )
        em.add_challenger(challenger)

        should_promote, reason = em.evaluate_promotion("TREND_HOLD")
        assert should_promote

        assert em.promote_challenger("TREND_HOLD", 2, reason)
        assert em.get_champion("TREND_HOLD").version == 2
        assert champion.status == StrategyStatus.RETIRED

    def test_config_hash(self):
        v1 = StrategyVersion(
            strategy_id="TEST", version=1, name="Test",
            params={"a": 1, "b": 2}
        )
        v2 = StrategyVersion(
            strategy_id="TEST", version=2, name="Test",
            params={"b": 2, "a": 1}  # Same params, different order
        )
        assert v1.config_hash == v2.config_hash

    def test_status_report(self):
        em = EvolutionManager()
        champion = StrategyVersion(
            strategy_id="TEST", version=1, name="Test", params={}
        )
        em.register_champion(champion)

        report = em.get_status_report()
        assert "TEST" in report["champions"]
        assert report["retired_count"] == 0
