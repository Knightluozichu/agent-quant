"""Risk checker — runs all risk checks against live state (NON-auto-trading).

P3-E4 风控告警-非自动交易

The :class:`RiskChecker` takes the live account state (``state.json``) and
current market data and produces a list of :class:`RiskAlert` objects. It can
be called from a cron job or a web endpoint. It produces ONLY alerts and
suggestions; it never executes trades. ``ALLOW_LIVE_TRADING`` is ``false``.

State format (from ``data/live/state.json``)::

    {
      "initial_capital": 100000.0,
      "cash": 50000.0,
      "holding": "518880",
      "shares": 1000,
      "entry_price": 5.0,
      "trade_log": [...]
    }
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from a_share_quant.risk.alerts import RiskAlert, RiskMonitor

#: Project timezone (matches settings.timezone). The checker resolves "today"
#: in Shanghai time so that evening runs align with the A-share trading day.
_SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


class RiskChecker:
    """Run all risk checks against state and market data.

    The checker reconstructs an equity curve from ``state["trade_log"]`` and
    the historical ``data`` (parquet DataFrames) to compute weekly and monthly
    returns. If history is insufficient those checks are skipped gracefully.

    Args:
        monitor: Optional :class:`RiskMonitor` (with thresholds / bark config).
            A default monitor is created if not supplied.
    """

    #: Trading-day lookback for the weekly drawdown check.
    WEEKLY_LOOKBACK = 5
    #: Trading-day lookback for the monthly drawdown check (~1 month).
    MONTHLY_LOOKBACK = 21

    def __init__(self, monitor: RiskMonitor | None = None) -> None:
        self.monitor = monitor or RiskMonitor()

    def check_all(
        self,
        state: dict[str, Any],
        data: dict[str, pd.DataFrame],
        realtime_prices: dict[str, float],
    ) -> list[RiskAlert]:
        """Run all risk checks and return the generated alerts.

        Args:
            state: Live account state (the ``state.json`` contents).
            data: Historical market data ``{code: DataFrame}`` with at least a
                ``trade_date`` column and a ``close`` column.
            realtime_prices: Current prices ``{code: price}`` for the holdings.

        Returns:
            A list of :class:`RiskAlert` (possibly empty). Alerts are also
            pushed via Bark by the underlying :class:`RiskMonitor` depending on
            severity.
        """
        alerts: list[RiskAlert] = []

        holding = state.get("holding")
        shares = int(state.get("shares", 0) or 0)
        cash = float(state.get("cash", 0) or 0)
        initial_capital = float(state.get("initial_capital", 0) or 0)

        # Resolve the current price of the holding (realtime first, else last close)
        current_price = self._resolve_price(holding, data, realtime_prices)
        holding_value = (shares * current_price) if (holding and current_price > 0) else 0.0
        current_value = cash + holding_value

        # Peak watermark: prefer a tracked peak_value, else fall back to initial capital.
        peak_value = float(state.get("peak_value") or initial_capital)

        # 1. Intraday drawdown (WARNING + Bark push when > 5%)
        alert = self.monitor.check_intraday_drawdown(current_value, peak_value)
        if alert is not None:
            alerts.append(alert)

        # 2 & 3. Weekly / monthly drawdown from the reconstructed equity curve
        weekly_ret, monthly_ret = self._compute_period_returns(state, data, current_value)
        if weekly_ret is not None:
            alert = self.monitor.check_weekly_drawdown(weekly_ret)
            if alert is not None:
                alerts.append(alert)
        if monthly_ret is not None:
            alert = self.monitor.check_monthly_drawdown(monthly_ret)
            if alert is not None:
                alerts.append(alert)

        # 4. Single-position concentration (INFO when < 80%)
        alert = self.monitor.check_single_position_exposure(holding_value, current_value)
        if alert is not None:
            alerts.append(alert)

        return alerts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_price(
        holding: str | None,
        data: dict[str, pd.DataFrame],
        realtime_prices: dict[str, float],
    ) -> float:
        """Resolve the current price of ``holding``: realtime first, else close."""
        if not holding:
            return 0.0
        if holding in realtime_prices and realtime_prices[holding] > 0:
            return float(realtime_prices[holding])
        if holding in data:
            df = data[holding]
            if len(df) > 0:
                return float(df.iloc[-1]["close"])
        return 0.0

    @staticmethod
    def _price_on(data: dict[str, pd.DataFrame], code: str, td: date) -> float:
        """Close price of ``code`` on trading date ``td`` (0.0 if missing)."""
        if code not in data:
            return 0.0
        df = data[code]
        dates = pd.to_datetime(df["trade_date"]).dt.date
        sub = df[dates == td]
        if sub.empty:
            return 0.0
        return float(sub.iloc[0]["close"])

    @staticmethod
    def _common_trading_dates(data: dict[str, pd.DataFrame]) -> list[date]:
        """Sorted list of trading dates common to all symbols in ``data``."""
        common: set[date] | None = None
        for df in data.values():
            dates = set(pd.to_datetime(df["trade_date"]).dt.date.tolist())
            common = dates if common is None else (common & dates)
        if not common:
            return []
        return sorted(common)

    def _compute_period_returns(
        self,
        state: dict[str, Any],
        data: dict[str, pd.DataFrame],
        current_value: float,
    ) -> tuple[float | None, float | None]:
        """Compute weekly and monthly returns from the reconstructed equity curve.

        Returns ``(weekly_return, monthly_return)``. Either may be ``None`` when
        there is insufficient history.
        """
        series = self._reconstruct_equity(state, data)

        # Replace/append today's value using the realtime-derived current_value.
        # Resolve "today" in Shanghai time (A-share trading day alignment).
        today = datetime.now(tz=_SHANGHAI_TZ).date()
        if series and series[-1][0] >= today:
            series[-1] = (today, current_value)
        else:
            series.append((today, current_value))

        weekly_ret: float | None = None
        monthly_ret: float | None = None

        need_weekly = self.WEEKLY_LOOKBACK + 1
        need_monthly = self.MONTHLY_LOOKBACK + 1
        if len(series) >= need_weekly:
            past = series[-need_weekly][1]
            if past > 0:
                weekly_ret = current_value / past - 1.0
        if len(series) >= need_monthly:
            past = series[-need_monthly][1]
            if past > 0:
                monthly_ret = current_value / past - 1.0
        return weekly_ret, monthly_ret

    def _reconstruct_equity(
        self,
        state: dict[str, Any],
        data: dict[str, pd.DataFrame],
    ) -> list[tuple[date, float]]:
        """Reconstruct the historical account-value curve from the trade log.

        Replays ``state["trade_log"]`` over the common trading dates and marks
        each open position to market using ``data`` closes. Returns an empty
        list if the trade log or trading dates are unavailable.
        """
        trade_log = state.get("trade_log") or []
        if not trade_log:
            return []
        trading_dates = self._common_trading_dates(data)
        if not trading_dates:
            return []

        start_date = str(trade_log[0].get("date", ""))
        trades_by_date: dict[str, list[dict[str, Any]]] = {}
        for t in trade_log:
            d = str(t.get("date", ""))
            trades_by_date.setdefault(d, []).append(t)

        cash = float(state.get("initial_capital", 0) or 0)
        holding: str | None = None
        shares = 0
        series: list[tuple[date, float]] = []

        for td in trading_dates:
            td_str = str(td)
            if td_str < start_date:
                continue
            for t in trades_by_date.get(td_str, []):
                action = t.get("action")
                amount = float(t.get("amount", 0) or 0)
                if action == "sell":
                    cash += amount
                    holding = None
                    shares = 0
                elif action == "buy":
                    cash -= amount
                    holding = t.get("code")
                    shares = int(t.get("shares", 0) or 0)
            value = cash
            if holding:
                price = self._price_on(data, holding, td)
                if price > 0:
                    value += shares * price
            series.append((td, value))
        return series
