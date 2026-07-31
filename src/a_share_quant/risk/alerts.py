"""Risk alert module (NON-auto-trading) — P3-E4 风控告警-非自动交易.

CRITICAL GUARANTEE
------------------
This module NEVER auto-executes trades. ``ALLOW_LIVE_TRADING`` is ``false``.
All outputs are *alerts* and *suggestions* only. Every suggestion explicitly
states: "此为建议, 需人工确认后手动执行".

The module exposes:
- :class:`RiskAlertLevel`: severity enum (INFO / WARNING / CRITICAL)
- :class:`RiskAlert`: pydantic model describing one advisory alert
- :class:`RiskMonitor`: checks risk metrics and generates (and pushes) alerts
- :func:`generate_suggestion`: derive a human-readable suggestion from an alert
- :func:`send_alert_bark`: push an alert via Bark (delegates to ``scripts/notify.py``)

Thresholds follow the 七星V3 strategy risk policy:

  - intraday drawdown  > 5%  → WARNING,  push Bark alert
  - weekly drawdown    > 3%  → WARNING,  push alert + pending reduction suggestion
  - monthly drawdown   > 8%  → CRITICAL, push urgent alert + pending all-to-cash suggestion
  - single position exposure < 80% → INFO concentration alert
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

_log = logging.getLogger(__name__)

#: Disclaimer appended to every suggestion — emphasises manual confirmation.
#: Every suggestion string MUST contain this text.
HUMAN_CONFIRM_DISCLAIMER = "此为建议, 需人工确认后手动执行"


class RiskAlertLevel(StrEnum):
    """Severity levels for risk alerts.

    Ordered by severity: ``INFO < WARNING < CRITICAL``.
    """

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class RiskAlert(BaseModel):
    """A single risk alert (advisory only, never an auto order).

    Attributes:
        level: Severity level.
        category: Risk category, e.g. ``"intraday_drawdown"``.
        message: Human-readable description of the risk condition.
        suggestion: Human-readable suggested action (NOT an auto order).
            Always contains :data:`HUMAN_CONFIRM_DISCLAIMER`.
        timestamp: When the alert was generated (UTC, timezone-aware).
        metric_value: The observed metric value that triggered the alert.
        threshold: The threshold the metric crossed.
        pending_action: Optional machine-readable pending action tag, e.g.
            ``"reduce_position"`` / ``"all_to_cash"``. This is a *suggestion*
            only — it is never executed automatically.
    """

    # NOTE: intentionally mutable (not frozen) so generate_suggestion() can be
    # applied after construction via ``alert.suggestion = generate_suggestion(alert)``.
    model_config = {"extra": "forbid"}

    level: RiskAlertLevel
    category: str
    message: str
    suggestion: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metric_value: float
    threshold: float
    pending_action: str | None = None


# ---------------------------------------------------------------------------
# Bark level / sound mapping
# ---------------------------------------------------------------------------
_BARK_LEVEL_MAP: dict[RiskAlertLevel, str] = {
    RiskAlertLevel.INFO: "active",
    RiskAlertLevel.WARNING: "timeSensitive",  # 突破勿扰
    RiskAlertLevel.CRITICAL: "timeSensitive",
}
_BARK_SOUND_MAP: dict[RiskAlertLevel, str] = {
    RiskAlertLevel.INFO: "minuet",
    RiskAlertLevel.WARNING: "alarm",
    RiskAlertLevel.CRITICAL: "alarm",
}


def generate_suggestion(alert: RiskAlert) -> str:
    """Generate a human-readable suggestion string from an alert.

    This is a *suggestion*, NOT an auto order. The returned string always
    contains :data:`HUMAN_CONFIRM_DISCLAIMER` ("此为建议, 需人工确认后手动执行").

    Args:
        alert: The alert to derive a suggestion for. The function inspects
            ``alert.category``, ``alert.metric_value`` and ``alert.threshold``.

    Returns:
        A suggestion string ending with the manual-confirmation disclaimer.
    """
    category = alert.category
    mv = alert.metric_value
    th = alert.threshold
    if category == "intraday_drawdown":
        action = f"日内回撤 {mv:.2%} 超过阈值 {th:.2%}, 建议关注持仓、暂不追加, 必要时人工减仓"
    elif category == "weekly_drawdown":
        action = f"周回撤 {mv:.2%} 超过阈值 {th:.2%}, 建议减仓至半仓观望, 等待趋势企稳后再行动"
    elif category == "monthly_drawdown":
        action = f"月回撤 {mv:.2%} 超过阈值 {th:.2%}, 建议全部清仓、转为现金防御"
    elif category == "single_position_exposure":
        action = f"单标的集中度 {mv:.2%} 低于阈值 {th:.2%}, 持仓偏分散, 请核对是否为预期状态"
    else:
        action = f"指标值 {mv:.4f} 触发阈值 {th:.4f}, 请人工核查"
    return f"{action}。{HUMAN_CONFIRM_DISCLAIMER}"


def send_alert_bark(
    alert: RiskAlert,
    bark_config: dict[str, str] | None = None,
) -> bool:
    """Push a Bark notification for the given alert.

    Delegates to ``scripts/notify.py::push_bark``. If the notify module is not
    importable (e.g. the scripts directory is not on ``sys.path``) the call is
    a no-op and returns ``False``. This function never raises.

    Args:
        alert: The alert to push.
        bark_config: Optional overrides, supports keys:

            - ``"bark_key"``: device key (exposed via ``BARK_KEY`` env var for
              ``notify.push_bark``; restored afterwards).
            - ``"group"``: Bark group name (default ``"七星V3"``).
            - ``"sound"``: Bark sound (default chosen by alert level).

    Returns:
        ``True`` if the push succeeded, ``False`` otherwise.
    """
    cfg = bark_config or {}
    bark_level = _BARK_LEVEL_MAP.get(alert.level, "active")
    sound = cfg.get("sound") or _BARK_SOUND_MAP.get(alert.level, "minuet")
    group = cfg.get("group", "七星V3")
    title = f"[{alert.level.value}] {alert.category}"
    body = f"{alert.message}\n建议: {alert.suggestion}"

    # If a bark_key is supplied, expose it to notify.push_bark via env var
    # (notify.push_bark reads BARK_KEY env first, then its config file).
    # The previous value is always restored in the ``finally`` block.
    key_override = cfg.get("bark_key")
    old_env = os.environ.get("BARK_KEY")
    if key_override:
        os.environ["BARK_KEY"] = key_override
    try:
        try:
            import notify  # type: ignore[import-not-found]
        except ImportError:
            _log.warning("notify 模块不可用, 跳过 Bark 推送: %s", alert.category)
            return False
        return bool(
            notify.push_bark(
                title=title,
                body=body,
                level=bark_level,
                sound=sound,
                group=group,
            )
        )
    finally:
        if key_override:
            if old_env is None:
                os.environ.pop("BARK_KEY", None)
            else:
                os.environ["BARK_KEY"] = old_env


class RiskMonitor:
    """Check risk metrics and generate advisory alerts.

    All checks return an optional :class:`RiskAlert`. When a threshold is
    breached the monitor constructs the alert, attaches a suggestion via
    :func:`generate_suggestion`, pushes it via :func:`send_alert_bark` (for
    WARNING/CRITICAL levels) and returns it.

    IMPORTANT: This monitor only produces *suggestions*. It never places
    orders. ``ALLOW_LIVE_TRADING`` remains ``false``.
    """

    def __init__(
        self,
        intraday_drawdown_threshold: float = 0.05,
        weekly_drawdown_threshold: float = 0.03,
        monthly_drawdown_threshold: float = 0.08,
        min_single_position_exposure: float = 0.80,
        bark_config: dict[str, str] | None = None,
    ) -> None:
        self.intraday_drawdown_threshold = intraday_drawdown_threshold
        self.weekly_drawdown_threshold = weekly_drawdown_threshold
        self.monthly_drawdown_threshold = monthly_drawdown_threshold
        self.min_single_position_exposure = min_single_position_exposure
        self.bark_config = bark_config

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------
    def _build_alert(
        self,
        level: RiskAlertLevel,
        category: str,
        message: str,
        metric_value: float,
        threshold: float,
        pending_action: str | None = None,
        push: bool = True,
    ) -> RiskAlert:
        """Build an alert with a generated suggestion and optionally push it."""
        alert = RiskAlert(
            level=level,
            category=category,
            message=message,
            suggestion="",
            metric_value=metric_value,
            threshold=threshold,
            pending_action=pending_action,
        )
        alert.suggestion = generate_suggestion(alert)
        if push:
            send_alert_bark(alert, self.bark_config)
        return alert

    # ------------------------------------------------------------------
    # Public checks
    # ------------------------------------------------------------------
    def check_intraday_drawdown(
        self,
        current_value: float,
        peak_value: float,
    ) -> RiskAlert | None:
        """Check intraday drawdown. ``> 5%`` → WARNING + Bark push.

        Args:
            current_value: Current account value.
            peak_value: Intraday peak value (watermark).

        Returns:
            A WARNING alert if drawdown exceeds the threshold, else ``None``.
        """
        if peak_value <= 0:
            return None
        drawdown = (peak_value - current_value) / peak_value
        if drawdown <= self.intraday_drawdown_threshold:
            return None
        return self._build_alert(
            level=RiskAlertLevel.WARNING,
            category="intraday_drawdown",
            message=(f"日内回撤 {drawdown:.2%} 超过阈值 {self.intraday_drawdown_threshold:.2%}"),
            metric_value=drawdown,
            threshold=self.intraday_drawdown_threshold,
        )

    def check_weekly_drawdown(self, weekly_return: float) -> RiskAlert | None:
        """Check weekly drawdown. ``> 3%`` → WARNING + Bark push + reduction suggestion.

        Args:
            weekly_return: Signed weekly return (e.g. ``-0.04`` for -4%).
                A positive return produces no drawdown alert.

        Returns:
            A WARNING alert (with ``pending_action="reduce_position"``) if the
            weekly drawdown exceeds the threshold, else ``None``.
        """
        drawdown = -weekly_return if weekly_return < 0 else 0.0
        if drawdown <= self.weekly_drawdown_threshold:
            return None
        return self._build_alert(
            level=RiskAlertLevel.WARNING,
            category="weekly_drawdown",
            message=(
                f"周回撤 {drawdown:.2%} (收益 {weekly_return:.2%}) 超过阈值 "
                f"{self.weekly_drawdown_threshold:.2%}"
            ),
            metric_value=drawdown,
            threshold=self.weekly_drawdown_threshold,
            pending_action="reduce_position",
        )

    def check_monthly_drawdown(self, monthly_return: float) -> RiskAlert | None:
        """Check monthly drawdown. ``> 8%`` → CRITICAL + urgent push + all-to-cash suggestion.

        Args:
            monthly_return: Signed monthly return (e.g. ``-0.10`` for -10%).
                A positive return produces no drawdown alert.

        Returns:
            A CRITICAL alert (with ``pending_action="all_to_cash"``) if the
            monthly drawdown exceeds the threshold, else ``None``.
        """
        drawdown = -monthly_return if monthly_return < 0 else 0.0
        if drawdown <= self.monthly_drawdown_threshold:
            return None
        return self._build_alert(
            level=RiskAlertLevel.CRITICAL,
            category="monthly_drawdown",
            message=(
                f"月回撤 {drawdown:.2%} (收益 {monthly_return:.2%}) 超过阈值 "
                f"{self.monthly_drawdown_threshold:.2%}"
            ),
            metric_value=drawdown,
            threshold=self.monthly_drawdown_threshold,
            pending_action="all_to_cash",
        )

    def check_single_position_exposure(
        self,
        holding_value: float,
        total_value: float,
    ) -> RiskAlert | None:
        """Check single-position concentration. ``< 80%`` → INFO alert (not pushed).

        For the 七星V3 single-ETF strategy the portfolio is normally
        concentrated in one holding. If the holding drops below 80% of total
        account value an INFO nudge is produced. A fully-cash position
        (``holding_value <= 0``) is the intentional DEFENSE state and does not
        trigger an alert.

        Args:
            holding_value: Market value of the single holding.
            total_value: Total account value (cash + holding).

        Returns:
            An INFO alert if exposure is below the threshold, else ``None``.
        """
        if total_value <= 0 or holding_value <= 0:
            return None
        exposure = holding_value / total_value
        if exposure >= self.min_single_position_exposure:
            return None
        return self._build_alert(
            level=RiskAlertLevel.INFO,
            category="single_position_exposure",
            message=(
                f"单标的集中度 {exposure:.2%} 低于阈值 {self.min_single_position_exposure:.2%}"
            ),
            metric_value=exposure,
            threshold=self.min_single_position_exposure,
            # INFO level: low severity, not pushed by default.
            push=False,
        )
