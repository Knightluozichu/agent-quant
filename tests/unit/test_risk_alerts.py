"""Tests for the risk alert module (P3-E4, NON-auto-trading).

CRITICAL: these tests verify that the module only ever produces *advisory*
alerts and suggestions. No trade is ever auto-executed. Every suggestion must
contain the manual-confirmation disclaimer "此为建议, 需人工确认后手动执行".
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from a_share_quant.risk.alerts import (
    HUMAN_CONFIRM_DISCLAIMER,
    RiskAlert,
    RiskAlertLevel,
    RiskMonitor,
    generate_suggestion,
    send_alert_bark,
)
from a_share_quant.risk.checker import RiskChecker


# ---------------------------------------------------------------------------
# Fixture: stub out Bark push so tests never hit the network or print.
# ---------------------------------------------------------------------------
@pytest.fixture
def pushed_alerts(monkeypatch) -> list[RiskAlert]:
    """Replace send_alert_bark with a recorder and return the call list."""
    calls: list[RiskAlert] = []

    def _fake_send(alert: RiskAlert, bark_config: Any = None) -> bool:
        calls.append(alert)
        return True

    monkeypatch.setattr("a_share_quant.risk.alerts.send_alert_bark", _fake_send)
    return calls


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class TestRiskAlertLevel:
    def test_three_levels_exist(self):
        assert RiskAlertLevel.INFO.value == "INFO"
        assert RiskAlertLevel.WARNING.value == "WARNING"
        assert RiskAlertLevel.CRITICAL.value == "CRITICAL"

    def test_levels_are_distinct(self):
        assert RiskAlertLevel.INFO != RiskAlertLevel.WARNING
        assert RiskAlertLevel.WARNING != RiskAlertLevel.CRITICAL


class TestRiskAlertModel:
    def test_default_timestamp_is_timezone_aware(self):
        alert = RiskAlert(
            level=RiskAlertLevel.INFO,
            category="x",
            message="m",
            metric_value=0.0,
            threshold=1.0,
        )
        assert alert.timestamp.tzinfo is not None

    def test_suggestion_defaults_empty(self):
        alert = RiskAlert(
            level=RiskAlertLevel.INFO,
            category="x",
            message="m",
            metric_value=0.0,
            threshold=1.0,
        )
        assert alert.suggestion == ""
        assert alert.pending_action is None


# ---------------------------------------------------------------------------
# Intraday drawdown
# ---------------------------------------------------------------------------
class TestIntradayDrawdown:
    def test_warning_when_exceeds_5pct(self, pushed_alerts):
        monitor = RiskMonitor()
        # 94000 vs 100000 peak -> 6% drawdown > 5%
        alert = monitor.check_intraday_drawdown(current_value=94000.0, peak_value=100000.0)

        assert alert is not None
        assert alert.level is RiskAlertLevel.WARNING
        assert alert.category == "intraday_drawdown"
        assert alert.metric_value == pytest.approx(0.06, rel=1e-6)
        assert alert.threshold == pytest.approx(0.05)
        assert "人工确认" in alert.suggestion
        # Bark push triggered for WARNING
        assert alert in pushed_alerts

    def test_exactly_at_threshold_no_alert(self, pushed_alerts):
        monitor = RiskMonitor()
        # exactly 5% drawdown -> not strictly greater -> no alert
        alert = monitor.check_intraday_drawdown(current_value=95000.0, peak_value=100000.0)
        assert alert is None
        assert pushed_alerts == []

    def test_no_alert_within_threshold(self, pushed_alerts):
        monitor = RiskMonitor()
        # 3% drawdown < 5%
        alert = monitor.check_intraday_drawdown(current_value=97000.0, peak_value=100000.0)
        assert alert is None
        assert pushed_alerts == []

    def test_invalid_peak_returns_none(self, pushed_alerts):
        monitor = RiskMonitor()
        assert monitor.check_intraday_drawdown(100.0, 0.0) is None
        assert monitor.check_intraday_drawdown(100.0, -1.0) is None


# ---------------------------------------------------------------------------
# Weekly drawdown
# ---------------------------------------------------------------------------
class TestWeeklyDrawdown:
    def test_warning_with_reduction_suggestion(self, pushed_alerts):
        monitor = RiskMonitor()
        # -4% weekly return -> 4% drawdown > 3%
        alert = monitor.check_weekly_drawdown(weekly_return=-0.04)

        assert alert is not None
        assert alert.level is RiskAlertLevel.WARNING
        assert alert.category == "weekly_drawdown"
        assert alert.pending_action == "reduce_position"
        assert alert.metric_value == pytest.approx(0.04, rel=1e-6)
        assert alert.threshold == pytest.approx(0.03)
        # reduction suggestion
        assert "减仓" in alert.suggestion
        assert "人工确认" in alert.suggestion
        # pushed
        assert alert in pushed_alerts

    def test_no_alert_within_threshold(self, pushed_alerts):
        monitor = RiskMonitor()
        # -2% < 3% threshold
        assert monitor.check_weekly_drawdown(-0.02) is None
        # positive return -> no drawdown
        assert monitor.check_weekly_drawdown(0.01) is None
        assert pushed_alerts == []


# ---------------------------------------------------------------------------
# Monthly drawdown
# ---------------------------------------------------------------------------
class TestMonthlyDrawdown:
    def test_critical_with_all_to_cash_suggestion(self, pushed_alerts):
        monitor = RiskMonitor()
        # -10% monthly return -> 10% drawdown > 8%
        alert = monitor.check_monthly_drawdown(monthly_return=-0.10)

        assert alert is not None
        assert alert.level is RiskAlertLevel.CRITICAL
        assert alert.category == "monthly_drawdown"
        assert alert.pending_action == "all_to_cash"
        assert alert.metric_value == pytest.approx(0.10, rel=1e-6)
        assert alert.threshold == pytest.approx(0.08)
        # all-to-cash suggestion
        assert "清仓" in alert.suggestion
        assert "现金" in alert.suggestion
        assert "人工确认" in alert.suggestion
        # urgent push
        assert alert in pushed_alerts

    def test_no_alert_within_threshold(self, pushed_alerts):
        monitor = RiskMonitor()
        # -7% < 8%
        assert monitor.check_monthly_drawdown(-0.07) is None
        assert monitor.check_monthly_drawdown(0.05) is None
        assert pushed_alerts == []


# ---------------------------------------------------------------------------
# Single position exposure
# ---------------------------------------------------------------------------
class TestSinglePositionExposure:
    def test_info_when_below_80pct(self, pushed_alerts):
        monitor = RiskMonitor()
        # 70000 / 100000 = 70% < 80%
        alert = monitor.check_single_position_exposure(holding_value=70000.0, total_value=100000.0)

        assert alert is not None
        assert alert.level is RiskAlertLevel.INFO
        assert alert.category == "single_position_exposure"
        assert alert.metric_value == pytest.approx(0.70, rel=1e-6)
        assert alert.threshold == pytest.approx(0.80)
        assert "人工确认" in alert.suggestion
        # INFO is not pushed by default
        assert alert not in pushed_alerts

    def test_no_alert_when_above_80pct(self, pushed_alerts):
        monitor = RiskMonitor()
        # 90% >= 80%
        assert monitor.check_single_position_exposure(90000.0, 100000.0) is None
        assert pushed_alerts == []

    def test_no_alert_when_no_holding(self, pushed_alerts):
        monitor = RiskMonitor()
        # fully cash -> DEFENSE state, no alert
        assert monitor.check_single_position_exposure(0.0, 100000.0) is None
        assert pushed_alerts == []

    def test_no_alert_when_total_zero(self, pushed_alerts):
        monitor = RiskMonitor()
        assert monitor.check_single_position_exposure(1000.0, 0.0) is None


# ---------------------------------------------------------------------------
# generate_suggestion
# ---------------------------------------------------------------------------
class TestGenerateSuggestion:
    @pytest.mark.parametrize(
        ("category", "metric_value", "threshold"),
        [
            ("intraday_drawdown", 0.06, 0.05),
            ("weekly_drawdown", 0.04, 0.03),
            ("monthly_drawdown", 0.10, 0.08),
            ("single_position_exposure", 0.70, 0.80),
            ("unknown_category", 1.0, 0.5),
        ],
    )
    def test_every_category_has_disclaimer(self, category, metric_value, threshold):
        alert = RiskAlert(
            level=RiskAlertLevel.INFO,
            category=category,
            message="x",
            suggestion="",
            metric_value=metric_value,
            threshold=threshold,
        )
        suggestion = generate_suggestion(alert)
        assert "人工确认" in suggestion
        assert HUMAN_CONFIRM_DISCLAIMER in suggestion

    def test_intraday_suggestion_mentions_drawdown(self):
        alert = RiskAlert(
            level=RiskAlertLevel.WARNING,
            category="intraday_drawdown",
            message="x",
            metric_value=0.06,
            threshold=0.05,
        )
        assert "日内回撤" in generate_suggestion(alert)

    def test_monthly_suggestion_mentions_all_to_cash(self):
        alert = RiskAlert(
            level=RiskAlertLevel.CRITICAL,
            category="monthly_drawdown",
            message="x",
            metric_value=0.10,
            threshold=0.08,
        )
        s = generate_suggestion(alert)
        assert "清仓" in s
        assert "现金" in s


# ---------------------------------------------------------------------------
# send_alert_bark
# ---------------------------------------------------------------------------
class TestSendAlertBark:
    def test_returns_false_when_notify_unavailable(self, monkeypatch):
        # Force `import notify` to fail inside send_alert_bark.
        monkeypatch.setitem(sys.modules, "notify", None)
        alert = RiskAlert(
            level=RiskAlertLevel.WARNING,
            category="intraday_drawdown",
            message="m",
            suggestion="s",
            metric_value=0.06,
            threshold=0.05,
        )
        assert send_alert_bark(alert) is False

    def test_never_raises_on_missing_config(self):
        alert = RiskAlert(
            level=RiskAlertLevel.INFO,
            category="x",
            message="m",
            suggestion="s",
            metric_value=0.0,
            threshold=1.0,
        )
        # No bark_config supplied and notify likely unavailable in test env.
        assert send_alert_bark(alert, bark_config=None) is False


# ---------------------------------------------------------------------------
# RiskChecker (integration)
# ---------------------------------------------------------------------------
class TestRiskChecker:
    def test_check_all_triggers_intraday_and_exposure(self, pushed_alerts):
        state = {
            "initial_capital": 100000.0,
            "cash": 50000.0,
            "holding": "518880",
            "shares": 1000,
            "entry_price": 5.0,
            "trade_log": [],
        }
        # price 4.0 -> holding 4000, total 54000
        # intraday drawdown from 100000 = 46% > 5% -> WARNING
        # exposure 4000/54000 ~ 7.4% < 80% -> INFO
        realtime = {"518880": 4.0}
        checker = RiskChecker()

        alerts = checker.check_all(state, data={}, realtime_prices=realtime)

        categories = {a.category for a in alerts}
        assert "intraday_drawdown" in categories
        assert "single_position_exposure" in categories
        # weekly/monthly skipped because trade_log is empty
        assert "weekly_drawdown" not in categories
        assert "monthly_drawdown" not in categories

    def test_check_all_no_alerts_when_healthy(self, pushed_alerts):
        state = {
            "initial_capital": 100000.0,
            "cash": 5000.0,
            "holding": "518880",
            "shares": 1000,
            "entry_price": 5.0,
            "trade_log": [],
        }
        # price 100 -> holding 100000, total 105000
        # drawdown vs 100000 peak = negative -> no alert
        # exposure 100000/105000 ~ 95% >= 80% -> no alert
        realtime = {"518880": 100.0}
        checker = RiskChecker()

        alerts = checker.check_all(state, data={}, realtime_prices=realtime)
        assert alerts == []

    def test_check_all_empty_state_is_safe(self, pushed_alerts):
        # Defensive: an empty/minimal state must not raise.
        checker = RiskChecker()
        alerts = checker.check_all(
            state={"initial_capital": 100000.0, "cash": 100000.0},
            data={},
            realtime_prices={},
        )
        # No holding -> no intraday drawdown (peak==current==capital), no exposure alert
        assert alerts == []
