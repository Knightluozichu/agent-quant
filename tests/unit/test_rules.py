"""Tests for market rules engine."""

from datetime import date

import pytest

from a_share_quant.rules import (
    CashAccount,
    FeeCalculator,
    OrderState,
    OrderStateMachine,
    PositionLedger,
    PriceLimitCalculator,
    SettlementRule,
)


class TestFeeCalculator:
    """Tests for fee calculation."""

    def test_buy_cost(self):
        calc = FeeCalculator()
        costs = calc.calculate_buy_cost(100000, date(2024, 1, 15))
        assert costs["stamp_tax"] == 0  # No stamp tax on buy
        assert costs["commission"] >= 5  # Min commission
        assert costs["total"] > 0

    def test_sell_cost_after_2023(self):
        """Stamp tax halved after 2023-08-28."""
        calc = FeeCalculator()
        costs = calc.calculate_sell_cost(100000, date(2024, 1, 15))
        assert costs["stamp_tax"] == pytest.approx(50, rel=0.01)  # 0.05%

    def test_sell_cost_before_2023(self):
        """Full stamp tax before 2023-08-28."""
        calc = FeeCalculator()
        costs = calc.calculate_sell_cost(100000, date(2023, 1, 15))
        assert costs["stamp_tax"] == pytest.approx(100, rel=0.01)  # 0.1%

    def test_min_commission(self):
        """Small trades should have minimum commission."""
        calc = FeeCalculator()
        costs = calc.calculate_buy_cost(1000, date(2024, 1, 15))
        assert costs["commission"] == 5.0  # Min commission


class TestPriceLimitCalculator:
    """Tests for price limit calculation."""

    def test_main_board_limit(self):
        calc = PriceLimitCalculator()
        limit = calc.get_limit_pct("MAIN", date(2024, 1, 15))
        assert limit == 0.10

    def test_chinext_limit_after_2020(self):
        calc = PriceLimitCalculator()
        limit = calc.get_limit_pct("CHINEXT", date(2024, 1, 15))
        assert limit == 0.20

    def test_chinext_limit_before_2020(self):
        calc = PriceLimitCalculator()
        limit = calc.get_limit_pct("CHINEXT", date(2019, 1, 15))
        assert limit == 0.10

    def test_star_board_limit(self):
        calc = PriceLimitCalculator()
        limit = calc.get_limit_pct("STAR", date(2024, 1, 15))
        assert limit == 0.20

    def test_calculate_limit_prices(self):
        calc = PriceLimitCalculator()
        upper, lower = calc.calculate_limit_prices(10.0, "MAIN", date(2024, 1, 15))
        assert upper == pytest.approx(11.0)
        assert lower == pytest.approx(9.0)


class TestSettlementRule:
    """Tests for T+1 settlement."""

    def test_cannot_sell_same_day(self):
        rule = SettlementRule()
        assert not rule.can_sell(date(2024, 1, 15), date(2024, 1, 15))

    def test_can_sell_next_day(self):
        rule = SettlementRule()
        assert rule.can_sell(date(2024, 1, 15), date(2024, 1, 16))

    def test_sellable_quantity(self):
        rule = SettlementRule()
        # Bought today
        assert rule.get_sellable_quantity(1000, date(2024, 1, 15), date(2024, 1, 15)) == 0
        # Bought yesterday
        assert rule.get_sellable_quantity(1000, date(2024, 1, 15), date(2024, 1, 16)) == 1000


class TestOrderStateMachine:
    """Tests for order state machine."""

    def test_initial_state(self):
        sm = OrderStateMachine()
        assert sm.state == OrderState.CREATED

    def test_valid_transition(self):
        sm = OrderStateMachine()
        assert sm.transition(OrderState.SUBMITTED)
        assert sm.state == OrderState.SUBMITTED

    def test_invalid_transition(self):
        sm = OrderStateMachine()
        assert not sm.transition(OrderState.FILLED)  # Can't go directly to FILLED
        assert sm.state == OrderState.CREATED

    def test_terminal_state(self):
        sm = OrderStateMachine()
        sm.transition(OrderState.SUBMITTED)
        sm.transition(OrderState.FILLED)
        assert sm.is_terminal
        assert not sm.transition(OrderState.CANCELLED)  # Can't transition from terminal

    def test_history_tracking(self):
        sm = OrderStateMachine()
        sm.transition(OrderState.SUBMITTED, "sent to exchange")
        sm.transition(OrderState.FILLED, "fully executed")
        assert len(sm.history) == 2


class TestCashAccount:
    """Tests for cash account."""

    def test_deposit(self):
        acc = CashAccount()
        acc.deposit(100000, date(2024, 1, 1))
        assert acc.balance == 100000
        assert acc.available == 100000

    def test_withdraw(self):
        acc = CashAccount(balance=100000)
        acc.withdraw(50000, date(2024, 1, 1))
        assert acc.balance == 50000

    def test_withdraw_insufficient(self):
        acc = CashAccount(balance=10000)
        with pytest.raises(ValueError, match="Insufficient"):
            acc.withdraw(50000, date(2024, 1, 1))

    def test_freeze(self):
        acc = CashAccount(balance=100000)
        acc.freeze(30000)
        assert acc.available == 70000
        assert acc.frozen == 30000

    def test_apply_buy_trade(self):
        acc = CashAccount(balance=100000)
        acc.freeze(50000)
        acc.apply_trade(50000, 15, "BUY", date(2024, 1, 15))
        assert acc.balance == pytest.approx(49985)  # 100000 - 50000 - 15


class TestPositionLedger:
    """Tests for position ledger with T+1."""

    def test_add_position(self):
        ledger = PositionLedger()
        ledger.add_position("510300.SSE", 1000, 4.5, date(2024, 1, 15))
        assert ledger.get_total_quantity("510300.SSE") == 1000

    def test_t_plus_1_not_sellable_same_day(self):
        ledger = PositionLedger()
        ledger.add_position("510300.SSE", 1000, 4.5, date(2024, 1, 15))
        ledger.update_settlement(date(2024, 1, 15))
        assert ledger.get_sellable_quantity("510300.SSE") == 0

    def test_t_plus_1_sellable_next_day(self):
        ledger = PositionLedger()
        ledger.add_position("510300.SSE", 1000, 4.5, date(2024, 1, 15))
        ledger.update_settlement(date(2024, 1, 16))
        assert ledger.get_sellable_quantity("510300.SSE") == 1000

    def test_reduce_position(self):
        ledger = PositionLedger()
        ledger.add_position("510300.SSE", 1000, 4.5, date(2024, 1, 15))
        ledger.update_settlement(date(2024, 1, 16))
        cost = ledger.reduce_position("510300.SSE", 500, date(2024, 1, 16))
        assert cost == pytest.approx(2250)  # 500 * 4.5
        assert ledger.get_total_quantity("510300.SSE") == 500

    def test_cannot_sell_more_than_sellable(self):
        ledger = PositionLedger()
        ledger.add_position("510300.SSE", 1000, 4.5, date(2024, 1, 15))
        ledger.update_settlement(date(2024, 1, 15))  # Same day
        with pytest.raises(ValueError, match="T\\+1"):
            ledger.reduce_position("510300.SSE", 500, date(2024, 1, 15))

    def test_average_cost(self):
        ledger = PositionLedger()
        ledger.add_position("510300.SSE", 1000, 4.0, date(2024, 1, 15))
        ledger.add_position("510300.SSE", 1000, 5.0, date(2024, 1, 16))
        assert ledger.get_average_cost("510300.SSE") == pytest.approx(4.5)
