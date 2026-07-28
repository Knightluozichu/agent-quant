"""Tests for domain models."""

from __future__ import annotations

from datetime import datetime

import pytest

from a_share_quant.domain import (
    Board,
    Currency,
    Exchange,
    MarketDirection,
    MarketRegime,
    MarketState,
    Money,
    OscillationLevel,
    StockState,
    Symbol,
    TradingDate,
    TradingStatus,
)


@pytest.mark.unit
class TestTradingDate:
    """Test TradingDate value type."""

    def test_from_string_yyyymmdd(self) -> None:
        td = TradingDate("20260720")
        assert td.to_str() == "20260720"

    def test_from_string_iso(self) -> None:
        td = TradingDate("2026-07-20")
        assert td.to_str() == "20260720"

    def test_comparison(self) -> None:
        td1 = TradingDate("20260720")
        td2 = TradingDate("20260721")
        assert td1 < td2
        assert td2 > td1
        assert td1 == TradingDate("2026-07-20")


@pytest.mark.unit
class TestSymbol:
    """Test Symbol value type."""

    def test_from_string(self) -> None:
        sym = Symbol.from_string("600519.SSE")
        assert sym.code == "600519"
        assert sym.exchange == Exchange.SSE

    def test_board_inference_main(self) -> None:
        sym = Symbol.from_string("600519.SSE")
        assert sym.board == Board.MAIN

    def test_board_inference_gem(self) -> None:
        sym = Symbol.from_string("300750.SZSE")
        assert sym.board == Board.GEM

    def test_board_inference_star(self) -> None:
        sym = Symbol.from_string("688981.SSE")
        assert sym.board == Board.STAR

    def test_to_string(self) -> None:
        sym = Symbol(code="600519", exchange=Exchange.SSE)
        assert sym.to_string() == "600519.SSE"


@pytest.mark.unit
class TestMoney:
    """Test Money value type."""

    def test_addition(self) -> None:
        m1 = Money(100.50)
        m2 = Money(50.25)
        result = m1 + m2
        assert float(result.amount) == 150.75

    def test_currency_mismatch_raises(self) -> None:
        m1 = Money(100, Currency.CNY)
        m2 = Money(50, Currency.USD)
        with pytest.raises(ValueError, match="Cannot add"):
            _ = m1 + m2


@pytest.mark.unit
class TestMarketRegime:
    """Test MarketRegime enum."""

    def test_from_direction_oscillation(self) -> None:
        regime = MarketRegime.from_direction_oscillation(
            MarketDirection.UP, OscillationLevel.LOW
        )
        assert regime == MarketRegime.SMOOTH_UPTREND

    def test_all_nine_regimes(self) -> None:
        """Verify all 9 regimes can be created."""
        regimes = []
        for direction in MarketDirection:
            for oscillation in OscillationLevel:
                regime = MarketRegime.from_direction_oscillation(direction, oscillation)
                regimes.append(regime)
        assert len(regimes) == 9
        assert len(set(regimes)) == 9  # All unique

    def test_is_bullish(self) -> None:
        assert MarketRegime.SMOOTH_UPTREND.is_bullish
        assert not MarketRegime.SMOOTH_DOWNTREND.is_bullish

    def test_is_tradeable(self) -> None:
        assert MarketRegime.SMOOTH_UPTREND.is_tradeable
        assert not MarketRegime.DEAD_WATER.is_tradeable
        assert not MarketRegime.SMOOTH_DOWNTREND.is_tradeable


@pytest.mark.unit
class TestMarketState:
    """Test MarketState model."""

    def test_create(self) -> None:
        state = MarketState.create(
            as_of=datetime(2026, 7, 20, 15, 0),
            direction=MarketDirection.UP,
            oscillation=OscillationLevel.LOW,
            confidence=0.85,
        )
        assert state.regime == MarketRegime.SMOOTH_UPTREND
        assert state.confidence == 0.85

    def test_probabilities_validation(self) -> None:
        """Probabilities must sum to 1.0."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            MarketState(
                as_of=datetime(2026, 7, 20),
                direction=MarketDirection.UP,
                oscillation=OscillationLevel.LOW,
                regime=MarketRegime.SMOOTH_UPTREND,
                confidence=0.8,
                probabilities={"up": 0.5, "down": 0.3},  # Sums to 0.8
            )


@pytest.mark.unit
class TestStockState:
    """Test StockState model."""

    def test_create(self) -> None:
        state = StockState(
            symbol="600519.SSE",
            as_of=datetime(2026, 7, 20, 15, 0),
            current_price=1800.0,
        )
        assert state.is_tradeable
        assert state.can_buy
        assert state.can_sell

    def test_suspended_stock(self) -> None:
        state = StockState(
            symbol="600519.SSE",
            as_of=datetime(2026, 7, 20, 15, 0),
            current_price=1800.0,
            trading_status=TradingStatus.SUSPENDED,
        )
        assert state.is_suspended
        assert not state.is_tradeable

    def test_limit_up(self) -> None:
        state = StockState(
            symbol="600519.SSE",
            as_of=datetime(2026, 7, 20, 15, 0),
            current_price=1980.0,
            limit_up_price=1980.0,
        )
        assert state.is_at_limit_up
