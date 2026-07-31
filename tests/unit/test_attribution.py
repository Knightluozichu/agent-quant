"""Tests for the 七星V3 P&L attribution engine (P3-E3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from a_share_quant.attribution.engine import (
    AttributionEngine,
    AttributionReport,
    TradeAttribution,
    generate_html_report,
)

if TYPE_CHECKING:
    from pathlib import Path

CODE = "518880"


# =============================================================================
# Helpers
# =============================================================================


def _make_ohlc(
    code: str, records: list[tuple[float, float, float]], start: str = "2026-01-01"
) -> pd.DataFrame:
    """Build a per-code OHLC DataFrame from (close, high, low) tuples."""
    dates = pd.date_range(start, periods=len(records), freq="B")
    closes = [r[0] for r in records]
    highs = [r[1] for r in records]
    lows = [r[2] for r in records]
    return pd.DataFrame(
        {
            "trade_date": dates,
            "open": closes,
            "close": closes,
            "high": highs,
            "low": lows,
            "volume": 10000,
            "symbol": code,
        }
    )


def _flat_records(n: int = 30, close: float = 90.0) -> list[tuple[float, float, float]]:
    return [(close, close * 1.01, close * 0.99)] * n


def _d(df: pd.DataFrame, i: int) -> str:
    return str(df["trade_date"].iloc[i].date())


def _eq_curve(
    df: pd.DataFrame, idxs: list[int], equities: list[float], code: str = CODE
) -> pd.DataFrame:
    """Build an equity_curve from row indices and equity values."""
    return pd.DataFrame(
        {
            "trade_date": [df["trade_date"].iloc[i] for i in idxs],
            "equity": equities,
            "holding": [code] * len(idxs),
        }
    )


def _trade(i: int, df: pd.DataFrame, action: str, shares: int, price: float,
           fee: float = 0.0, code: str = CODE, slippage: float | None = None) -> dict:
    """Build a single trade_log entry."""
    entry: dict = {
        "date": _d(df, i),
        "action": action,
        "code": code,
        "shares": shares,
        "price": price,
        "fee": fee,
    }
    if slippage is not None:
        entry["slippage"] = slippage
    return entry


# =============================================================================
# Factor contribution
# =============================================================================


class TestFactorContribution:
    """Factor contribution: 10d vs 20d momentum signal-weighted P&L."""

    def test_factor_split_known_inputs(self):
        # close[0]=100/1.3 -> mom_20d=0.3 ; close[10]=100/1.2 -> mom_10d=0.2
        # denom=0.5 -> w10=0.4, w20=0.6 ; gross=1000 -> 400 / 600
        recs = _flat_records()
        recs[0] = (100 / 1.3, 101.0, 99.0)
        recs[10] = (100 / 1.2, 101.0, 99.0)
        recs[20] = (100.0, 101.0, 99.0)  # entry
        recs[25] = (110.0, 111.0, 109.0)  # exit
        df = _make_ohlc(CODE, recs)
        data = {CODE: df}

        trade_log = [
            _trade(20, df, "buy", 100, 100.0, fee=10.0),
            _trade(25, df, "sell", 100, 110.0, fee=10.0),
        ]
        eq = _eq_curve(df, [20, 25], [100000.0, 110000.0])
        report = AttributionEngine().analyze(trade_log, eq, data)

        assert report.factor_returns["momentum_10d"] == pytest.approx(400.0, rel=1e-6)
        assert report.factor_returns["momentum_20d"] == pytest.approx(600.0, rel=1e-6)

    def test_factor_returns_sum_to_gross_pnl(self):
        # The two factors fully decompose gross P&L (weights sum to 1).
        recs = _flat_records()
        recs[0] = (100 / 1.3, 101.0, 99.0)
        recs[10] = (100 / 1.2, 101.0, 99.0)
        recs[20] = (100.0, 101.0, 99.0)
        recs[25] = (110.0, 111.0, 109.0)
        df = _make_ohlc(CODE, recs)
        data = {CODE: df}
        trade_log = [
            _trade(20, df, "buy", 100, 100.0),
            _trade(25, df, "sell", 100, 110.0),
        ]
        eq = _eq_curve(df, [20, 25], [100000.0, 110000.0])
        report = AttributionEngine().analyze(trade_log, eq, data)

        gross = sum(t.gross_pnl for t in report.trades)
        assert sum(report.factor_returns.values()) == pytest.approx(gross, rel=1e-9)
        assert gross == pytest.approx(1000.0)

    def test_zero_momentum_splits_evenly(self):
        # When both signals are zero, P&L is split 50/50.
        recs = [(100.0, 101.0, 99.0)] * 30  # flat -> momentum 0
        df = _make_ohlc(CODE, recs)
        data = {CODE: df}
        trade_log = [
            _trade(20, df, "buy", 100, 100.0),
            _trade(25, df, "sell", 100, 110.0),
        ]
        eq = _eq_curve(df, [20, 25], [100000.0, 110000.0])
        report = AttributionEngine().analyze(trade_log, eq, data)

        assert report.factor_returns["momentum_10d"] == pytest.approx(500.0)
        assert report.factor_returns["momentum_20d"] == pytest.approx(500.0)


# =============================================================================
# MFE / MAE
# =============================================================================


class TestMfeMae:
    """Maximum favorable / adverse excursion per trade."""

    def test_mfe_mae_known_window(self):
        recs = _flat_records()
        # holding window idx 20..25 ; max high 115, min low 95
        recs[20] = (100.0, 102.0, 99.0)
        recs[21] = (103.0, 104.0, 95.0)   # adverse low
        recs[22] = (108.0, 115.0, 107.0)  # favorable high
        recs[23] = (106.0, 109.0, 105.0)
        recs[24] = (109.0, 111.0, 108.0)
        recs[25] = (110.0, 112.0, 109.0)
        df = _make_ohlc(CODE, recs)
        data = {CODE: df}

        trade_log = [
            _trade(20, df, "buy", 100, 100.0),
            _trade(25, df, "sell", 100, 110.0),
        ]
        eq = _eq_curve(df, [20, 25], [100000.0, 110000.0])
        report = AttributionEngine().analyze(trade_log, eq, data)

        assert len(report.trades) == 1
        trade = report.trades[0]
        # MFE = (115 - 100) * 100 = 1500 ; MAE = (100 - 95) * 100 = 500
        assert trade.mfe == pytest.approx(1500.0)
        assert trade.mae == pytest.approx(500.0)
        assert trade.holding_days > 0

    def test_mfe_mae_never_adverse(self):
        # Price only goes up -> MAE = 0.
        recs = _flat_records()
        for i in range(20, 26):
            recs[i] = (100.0 + i, 100.0 + i + 1, 100.0 + i - 0.5)
        df = _make_ohlc(CODE, recs)
        data = {CODE: df}
        trade_log = [
            _trade(20, df, "buy", 100, 100.0),
            _trade(25, df, "sell", 100, 125.0),
        ]
        eq = _eq_curve(df, [20, 25], [100000.0, 125000.0])
        report = AttributionEngine().analyze(trade_log, eq, data)

        trade = report.trades[0]
        assert trade.mae == pytest.approx(0.0)
        assert trade.mfe > 0


# =============================================================================
# Cost attribution
# =============================================================================


class TestCostAttribution:
    """Cost drag = fees + slippage."""

    def test_estimated_slippage_from_close(self):
        # buy: price=100, close=100 -> slip 0 ; sell: price=110, close=109 -> slip 100
        recs = _flat_records()
        recs[20] = (100.0, 101.0, 99.0)
        recs[25] = (109.0, 110.0, 108.0)
        df = _make_ohlc(CODE, recs)
        data = {CODE: df}

        trade_log = [
            _trade(20, df, "buy", 100, 100.0, fee=25.0),
            _trade(25, df, "sell", 100, 110.0, fee=25.0),
        ]
        eq = _eq_curve(df, [20, 25], [100000.0, 110000.0])
        report = AttributionEngine().analyze(trade_log, eq, data)

        # fees 25+25 + slippage 0+100 = 150
        assert report.cost_drag == pytest.approx(150.0)

    def test_explicit_slippage_field(self):
        recs = _flat_records()
        recs[20] = (100.0, 101.0, 99.0)
        recs[25] = (110.0, 111.0, 109.0)
        df = _make_ohlc(CODE, recs)
        data = {CODE: df}

        trade_log = [
            _trade(20, df, "buy", 100, 100.0, fee=25.0, slippage=5.0),
            _trade(25, df, "sell", 100, 110.0, fee=25.0, slippage=8.0),
        ]
        eq = _eq_curve(df, [20, 25], [100000.0, 110000.0])
        report = AttributionEngine().analyze(trade_log, eq, data)

        # fees 25+25 + explicit slippage 5+8 = 63
        assert report.cost_drag == pytest.approx(63.0)

    def test_trade_net_pnl_subtracts_costs(self):
        recs = _flat_records()
        recs[20] = (100.0, 101.0, 99.0)
        recs[25] = (110.0, 111.0, 109.0)
        df = _make_ohlc(CODE, recs)
        data = {CODE: df}
        trade_log = [
            _trade(20, df, "buy", 100, 100.0, fee=10.0),
            _trade(25, df, "sell", 100, 110.0, fee=10.0),
        ]
        eq = _eq_curve(df, [20, 25], [100000.0, 110000.0])
        report = AttributionEngine().analyze(trade_log, eq, data)

        trade = report.trades[0]
        # gross 1000 - costs 20 = 980
        assert trade.gross_pnl == pytest.approx(1000.0)
        assert trade.costs == pytest.approx(20.0)
        assert trade.pnl == pytest.approx(980.0)


# =============================================================================
# Timing contribution
# =============================================================================


class TestTimingContribution:
    """Rebalance vs buy-and-hold excess return."""

    def test_timing_beats_buy_and_hold(self):
        # Buy-hold holds declining A (100 -> 90, -10%); strategy equity +10%.
        df = _make_ohlc("A", [(100.0, 101.0, 99.0), (95.0, 96.0, 94.0), (90.0, 91.0, 89.0)])
        data = {"A": df}
        trade_log = [
            _trade(0, df, "buy", 1000, 100.0, code="A"),
            _trade(2, df, "sell", 1000, 90.0, code="A"),
        ]
        eq = _eq_curve(df, [0, 1, 2], [100000.0, 105000.0, 110000.0], code="A")
        report = AttributionEngine().analyze(trade_log, eq, data)

        assert report.buy_hold_return == pytest.approx(-0.10, rel=1e-6)
        assert report.timing_return == pytest.approx(0.20, rel=1e-6)


# =============================================================================
# Report structure & edge cases
# =============================================================================


class TestReportStructure:
    def test_report_has_required_fields(self):
        recs = _flat_records()
        recs[20] = (100.0, 101.0, 99.0)
        recs[25] = (110.0, 111.0, 109.0)
        df = _make_ohlc(CODE, recs)
        data = {CODE: df}
        trade_log = [
            _trade(20, df, "buy", 100, 100.0, fee=10.0),
            _trade(25, df, "sell", 100, 110.0, fee=10.0),
        ]
        eq = _eq_curve(df, [20, 25], [100000.0, 110000.0])
        report = AttributionEngine().analyze(trade_log, eq, data)

        assert isinstance(report, AttributionReport)
        assert set(report.factor_returns) == {"momentum_10d", "momentum_20d"}
        assert isinstance(report.timing_return, float)
        assert isinstance(report.cost_drag, float)
        assert len(report.trades) == 1
        assert isinstance(report.trades[0], TradeAttribution)
        # TradeAttribution spec fields
        t = report.trades[0]
        for attr in ("entry_date", "exit_date", "code", "pnl", "mfe", "mae", "holding_days"):
            assert hasattr(t, attr)
        # monthly_summary shape
        assert all(set(v) == {"return", "factor_contrib", "timing", "cost"}
                       for v in report.monthly_summary.values())

    def test_empty_trade_log(self):
        recs = _flat_records()
        df = _make_ohlc(CODE, recs)
        data = {CODE: df}
        eq = _eq_curve(df, [20, 25], [100000.0, 100000.0])
        report = AttributionEngine().analyze([], eq, data)

        assert report.trades == []
        assert report.n_trades == 0
        assert report.cost_drag == 0.0
        assert report.factor_returns == {"momentum_10d": 0.0, "momentum_20d": 0.0}
        assert report.win_rate == 0.0

    def test_invalid_momentum_window_raises(self):
        with pytest.raises(ValueError, match="positive"):
            AttributionEngine(momentum_short=0)


# =============================================================================
# HTML report generation
# =============================================================================


class TestHtmlReport:
    def test_html_file_created_and_non_empty(self, tmp_path: Path):
        recs = _flat_records()
        recs[0] = (100 / 1.3, 101.0, 99.0)
        recs[10] = (100 / 1.2, 101.0, 99.0)
        recs[20] = (100.0, 102.0, 99.0)
        recs[21] = (103.0, 104.0, 95.0)
        recs[22] = (108.0, 115.0, 107.0)
        recs[25] = (110.0, 112.0, 109.0)
        df = _make_ohlc(CODE, recs)
        data = {CODE: df}
        trade_log = [
            _trade(20, df, "buy", 100, 100.0, fee=25.0),
            _trade(25, df, "sell", 100, 110.0, fee=25.0),
        ]
        eq = _eq_curve(df, [20, 25], [100000.0, 110000.0])
        report = AttributionEngine().analyze(trade_log, eq, data)

        out = tmp_path / "attribution_report.html"
        result = generate_html_report(report, out)

        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0
        content = out.read_text(encoding="utf-8")
        assert "七星V3" in content
        assert "因子贡献" in content
        assert "MFE" in content
