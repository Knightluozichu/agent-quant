"""P&L attribution engine for the 七星V3 strategy (P3-E3: 盈亏归因引擎).

Decomposes strategy returns into four complementary lenses:

  1. **Factor contribution** — splits gross trade P&L between the 10-day and
     20-day momentum signals that 七星V3 uses for selection (weighted by the
     relative signal strength at entry, consistent with the strategy's equal
     0.5/0.5 weighting).
  2. **Timing contribution** — excess return of the rebalanced strategy over a
     buy-and-hold baseline that holds the initial portfolio to the end.
  3. **Cost attribution** — total drag from fees plus slippage.
  4. **MFE / MAE** — Maximum Favorable / Adverse Excursion per round-trip trade.

The engine consumes the ``trade_log`` / ``equity_curve`` / ``data`` artifacts
produced by the 七星V3 backtest and returns a structured :class:`AttributionReport`
that can be rendered to a self-contained HTML page.

trade_log entry format (either ``date`` or ``execution_date`` is accepted)::

    {"date": "2026-01-15", "action": "buy", "code": "518880",
     "shares": 100, "price": 5.0, "fee": 25}

equity_curve columns: ``trade_date, equity, holding``.
data: ``{code: DataFrame}`` with columns ``trade_date, open, high, low, close``.
"""

from __future__ import annotations

from collections import deque
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field

FACTOR_MOMENTUM_10D = "momentum_10d"
FACTOR_MOMENTUM_20D = "momentum_20d"

_EPS = 1e-9


# =============================================================================
# Pydantic models
# =============================================================================


class TradeAttribution(BaseModel):
    """Attribution for a single completed (round-trip) trade."""

    entry_date: date
    exit_date: date
    code: str
    pnl: float = 0.0  # net realized P&L (currency) = gross_pnl - costs
    mfe: float = 0.0  # max favorable excursion (currency, >= 0)
    mae: float = 0.0  # max adverse excursion (currency, >= 0)
    holding_days: int = 0

    # supporting detail
    shares: float = 0.0
    entry_price: float = 0.0
    exit_price: float = 0.0
    gross_pnl: float = 0.0
    costs: float = 0.0
    factor_10d: float = 0.0  # gross P&L attributed to 10d momentum
    factor_20d: float = 0.0  # gross P&L attributed to 20d momentum


class AttributionReport(BaseModel):
    """Full P&L attribution report for 七星V3."""

    factor_returns: dict[str, float] = Field(default_factory=dict)
    timing_return: float = 0.0
    cost_drag: float = 0.0
    trades: list[TradeAttribution] = Field(default_factory=list)
    monthly_summary: dict[str, dict[str, float]] = Field(default_factory=dict)

    # headline statistics (convenience for reporting)
    total_return: float = 0.0
    buy_hold_return: float = 0.0
    total_pnl: float = 0.0
    n_trades: int = 0
    n_winning: int = 0
    win_rate: float = 0.0
    period_start: date | None = None
    period_end: date | None = None


# =============================================================================
# Engine
# =============================================================================


class AttributionEngine:
    """Decompose 七星V3 strategy returns into factor / timing / cost / MFE-MAE.

    The engine is lookahead-safe: every ``as_of`` lookup only uses data on or
    before the reference date (nearest-available history).
    """

    def __init__(self, momentum_short: int = 10, momentum_long: int = 20) -> None:
        if momentum_short <= 0 or momentum_long <= 0:
            raise ValueError("momentum windows must be positive")
        self.momentum_short = momentum_short
        self.momentum_long = momentum_long
        self._df_cache: dict[str, pd.DataFrame] = {}

    # -------------------------------------------------------------------------
    # public API
    # -------------------------------------------------------------------------

    def analyze(
        self,
        trade_log: list[dict[str, Any]],
        equity_curve: pd.DataFrame,
        data: dict[str, pd.DataFrame],
    ) -> AttributionReport:
        """Run the full attribution decomposition.

        Args:
            trade_log: chronological list of trade dicts.
            equity_curve: DataFrame with ``trade_date, equity, holding``.
            data: per-code OHLC DataFrames.

        Returns:
            A populated :class:`AttributionReport`.
        """
        eq = self._prepare_equity(equity_curve)
        valid = self._valid_trades(trade_log)

        trades = self._match_trades(valid, data)
        factor_returns = self._factor_returns(trades)
        cost_drag = self._compute_cost_drag(valid, data)
        timing_return, buy_hold_return = self._compute_timing(valid, eq, data)
        monthly_summary = self._compute_monthly_summary(trades, valid, eq, data)

        total_pnl = sum(t.pnl for t in trades)
        n_winning = sum(1 for t in trades if t.pnl > 0)
        n_trades = len(trades)
        win_rate = n_winning / n_trades if n_trades else 0.0

        return AttributionReport(
            factor_returns=factor_returns,
            timing_return=timing_return,
            cost_drag=cost_drag,
            trades=trades,
            monthly_summary=monthly_summary,
            total_return=self._series_return(eq["equity"]) if not eq.empty else 0.0,
            buy_hold_return=buy_hold_return,
            total_pnl=total_pnl,
            n_trades=n_trades,
            n_winning=n_winning,
            win_rate=win_rate,
            period_start=eq["trade_date"].iloc[0].to_pydatetime().date() if not eq.empty else None,
            period_end=eq["trade_date"].iloc[-1].to_pydatetime().date() if not eq.empty else None,
        )

    # -------------------------------------------------------------------------
    # trade matching (FIFO)
    # -------------------------------------------------------------------------

    def _match_trades(
        self,
        trade_log: list[dict[str, Any]],
        data: dict[str, pd.DataFrame],
    ) -> list[TradeAttribution]:
        ordered = sorted(trade_log, key=lambda t: self._to_date(self._trade_date_of(t)))
        open_lots: dict[str, deque[dict[str, Any]]] = {}
        trades: list[TradeAttribution] = []

        for t in ordered:
            action = str(t.get("action", "")).lower()
            code = str(t.get("code", ""))
            if not code or action not in ("buy", "sell"):
                continue
            exit_date = self._to_date(self._trade_date_of(t))
            shares = float(t.get("shares", 0) or 0)
            price = float(t.get("price", 0) or 0)
            fee = float(t.get("fee", 0) or 0)
            if shares <= 0:
                continue

            if action == "buy":
                open_lots.setdefault(code, deque()).append(
                    {"date": exit_date, "shares": shares, "price": price, "fee": fee}
                )
                continue

            # sell: consume oldest lots (FIFO)
            lots = open_lots.get(code)
            if not lots:
                continue
            remaining = shares
            sell_fee_per_share = fee / shares
            while remaining > _EPS and lots:
                lot = lots[0]
                matched = min(remaining, float(lot["shares"]))
                lot_shares = float(lot["shares"])
                buy_fee_per_share = float(lot["fee"]) / lot_shares if lot_shares else 0.0
                entry_price = float(lot["price"])
                gross = (price - entry_price) * matched
                costs = (buy_fee_per_share + sell_fee_per_share) * matched
                net = gross - costs
                mfe, mae = self._compute_mfe_mae(
                    data, code, self._to_date(lot["date"]), exit_date, entry_price, matched
                )
                f10, f20 = self._factor_split(data, code, self._to_date(lot["date"]), gross)
                trades.append(
                    TradeAttribution(
                        entry_date=self._to_date(lot["date"]),
                        exit_date=exit_date,
                        code=code,
                        pnl=net,
                        gross_pnl=gross,
                        costs=costs,
                        mfe=mfe,
                        mae=mae,
                        holding_days=(exit_date - self._to_date(lot["date"])).days,
                        shares=matched,
                        entry_price=entry_price,
                        exit_price=price,
                        factor_10d=f10,
                        factor_20d=f20,
                    )
                )
                lot["shares"] = lot_shares - matched
                lot["fee"] = float(lot["fee"]) - buy_fee_per_share * matched
                remaining -= matched
                if float(lot["shares"]) <= _EPS:
                    lots.popleft()
        return trades

    # -------------------------------------------------------------------------
    # factor contribution
    # -------------------------------------------------------------------------

    @staticmethod
    def _factor_returns(trades: list[TradeAttribution]) -> dict[str, float]:
        return {
            FACTOR_MOMENTUM_10D: sum(t.factor_10d for t in trades),
            FACTOR_MOMENTUM_20D: sum(t.factor_20d for t in trades),
        }

    def _factor_split(
        self,
        data: dict[str, pd.DataFrame],
        code: str,
        entry_date: date,
        gross_pnl: float,
    ) -> tuple[float, float]:
        """Split gross P&L by relative momentum signal strength at entry.

        七星V3 weights the two momentum factors equally (0.5/0.5); because the
        equal weight cancels in the ratio, the contribution share reduces to
        ``|mom_10d| / (|mom_10d| + |mom_20d|)``. When both signals are zero the
        P&L is split evenly.
        """
        mom_10d = self._momentum(data, code, entry_date, self.momentum_short)
        mom_20d = self._momentum(data, code, entry_date, self.momentum_long)
        denom = abs(mom_10d) + abs(mom_20d)
        if denom <= 0.0:
            w10 = w20 = 0.5
        else:
            w10 = abs(mom_10d) / denom
            w20 = abs(mom_20d) / denom
        return gross_pnl * w10, gross_pnl * w20

    def _momentum(
        self,
        data: dict[str, pd.DataFrame],
        code: str,
        as_of: date,
        period: int,
    ) -> float:
        df = self._norm_df(code, data)
        if df is None:
            return 0.0
        ts = pd.Timestamp(as_of)
        n = int((df.index <= ts).sum())
        if n <= period:  # need at least period+1 observations up to as_of
            return 0.0
        pos = n - 1
        close = df["close"]
        base = float(close.iloc[pos - period])
        if base == 0.0:
            return 0.0
        cur = float(close.iloc[pos])
        return (cur - base) / base

    # -------------------------------------------------------------------------
    # MFE / MAE
    # -------------------------------------------------------------------------

    def _compute_mfe_mae(
        self,
        data: dict[str, pd.DataFrame],
        code: str,
        entry_date: date,
        exit_date: date,
        entry_price: float,
        shares: float,
    ) -> tuple[float, float]:
        df = self._norm_df(code, data)
        if df is None:
            return 0.0, 0.0
        ets = pd.Timestamp(entry_date)
        xts = pd.Timestamp(exit_date)
        window = df[(df.index >= ets) & (df.index <= xts)]
        if window.empty:
            return 0.0, 0.0
        if "high" in window.columns and "low" in window.columns:
            max_high = float(window["high"].max())
            min_low = float(window["low"].min())
        else:
            max_high = float(window["close"].max())
            min_low = float(window["close"].min())
        mfe = max(0.0, (max_high - entry_price) * shares)
        mae = max(0.0, (entry_price - min_low) * shares)
        return mfe, mae

    # -------------------------------------------------------------------------
    # cost attribution
    # -------------------------------------------------------------------------

    def _compute_cost_drag(
        self,
        trade_log: list[dict[str, Any]],
        data: dict[str, pd.DataFrame],
    ) -> float:
        """Total cost drag = fees + slippage (positive magnitude)."""
        total_fee = sum(float(t.get("fee", 0) or 0) for t in trade_log)
        total_slip = 0.0
        for t in trade_log:
            explicit = t.get("slippage")
            if explicit is not None:
                total_slip += float(explicit)
                continue
            total_slip += self._estimate_slippage(t, data)
        return total_fee + total_slip

    def _estimate_slippage(self, t: dict[str, Any], data: dict[str, pd.DataFrame]) -> float:
        code = str(t.get("code", ""))
        d = self._to_date(self._trade_date_of(t))
        price = float(t.get("price", 0) or 0)
        shares = float(t.get("shares", 0) or 0)
        close = self._close_on(data, code, d)
        if close is None or shares <= 0:
            return 0.0
        return abs(price - close) * shares

    # -------------------------------------------------------------------------
    # timing contribution
    # -------------------------------------------------------------------------

    def _compute_timing(
        self,
        trade_log: list[dict[str, Any]],
        eq: pd.DataFrame,
        data: dict[str, pd.DataFrame],
    ) -> tuple[float, float]:
        """Return (timing_return, buy_hold_return).

        timing_return = strategy_return - buy_hold_return, both measured over
        the window [first_buy_date, last_equity_date].
        """
        if eq.empty:
            return 0.0, 0.0
        buys = [t for t in trade_log if str(t.get("action", "")).lower() == "buy"]
        if not buys:
            return self._series_return(eq["equity"]), 0.0

        first_buy = min(self._to_date(self._trade_date_of(t)) for t in buys)
        eq_win = eq[eq["trade_date"] >= pd.Timestamp(first_buy)]
        strat_ret = self._series_return(eq_win["equity"]) if not eq_win.empty else 0.0

        initial = [t for t in buys if self._to_date(self._trade_date_of(t)) == first_buy]
        initial_cost = sum(
            float(t.get("shares", 0)) * float(t.get("price", 0)) + float(t.get("fee", 0))
            for t in initial
        )
        last_ts = eq["trade_date"].iloc[-1]
        final_value = 0.0
        for t in initial:
            close = self._close_on(data, str(t.get("code", "")), last_ts)
            if close is not None:
                final_value += float(t.get("shares", 0)) * close
        bh_ret = final_value / initial_cost - 1 if initial_cost > 0 else 0.0
        return strat_ret - bh_ret, bh_ret

    # -------------------------------------------------------------------------
    # monthly summary
    # -------------------------------------------------------------------------

    def _compute_monthly_summary(
        self,
        trades: list[TradeAttribution],
        trade_log: list[dict[str, Any]],
        eq: pd.DataFrame,
        data: dict[str, pd.DataFrame],
    ) -> dict[str, dict[str, float]]:
        if eq.empty:
            return {}
        dates = eq["trade_date"]
        months = dates.dt.strftime("%Y-%m")
        strat_monthly = self._monthly_returns(eq["equity"], months)
        bh_monthly = self._buy_hold_monthly(trade_log, eq, data)

        factor_by_month: dict[str, float] = {}
        for t in trades:
            m = t.exit_date.strftime("%Y-%m")
            factor_by_month[m] = factor_by_month.get(m, 0.0) + t.factor_10d + t.factor_20d

        cost_by_month: dict[str, float] = {}
        for entry in trade_log:
            m = self._to_date(self._trade_date_of(entry)).strftime("%Y-%m")
            fee = float(entry.get("fee", 0) or 0)
            slip = (
                float(entry["slippage"])
                if entry.get("slippage") is not None
                else self._estimate_slippage(entry, data)
            )
            cost_by_month[m] = cost_by_month.get(m, 0.0) + fee + slip

        all_months = sorted(
            set(strat_monthly) | set(bh_monthly) | set(factor_by_month) | set(cost_by_month)
        )
        summary: dict[str, dict[str, float]] = {}
        for m in all_months:
            strat = strat_monthly.get(m, 0.0)
            bh = bh_monthly.get(m, 0.0)
            summary[m] = {
                "return": strat,
                "factor_contrib": factor_by_month.get(m, 0.0),
                "timing": strat - bh,
                "cost": cost_by_month.get(m, 0.0),
            }
        return summary

    def _buy_hold_monthly(
        self,
        trade_log: list[dict[str, Any]],
        eq: pd.DataFrame,
        data: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        buys = [t for t in trade_log if str(t.get("action", "")).lower() == "buy"]
        if not buys:
            return {}
        first_buy = min(self._to_date(self._trade_date_of(t)) for t in buys)
        initial = [t for t in buys if self._to_date(self._trade_date_of(t)) == first_buy]
        initial_cost = sum(
            float(t.get("shares", 0)) * float(t.get("price", 0)) + float(t.get("fee", 0))
            for t in initial
        )
        fb_ts = pd.Timestamp(first_buy)
        bh_vals: list[float] = []
        for d in eq["trade_date"]:
            if d < fb_ts:
                bh_vals.append(initial_cost)
                continue
            val = 0.0
            for t in initial:
                close = self._close_on(data, str(t.get("code", "")), d)
                if close is not None:
                    val += float(t.get("shares", 0)) * close
            bh_vals.append(val if val > 0 else initial_cost)
        bh = pd.Series(bh_vals)
        return self._monthly_returns(bh, eq["trade_date"].dt.strftime("%Y-%m"))

    @staticmethod
    def _monthly_returns(values: pd.Series, months: pd.Series) -> dict[str, float]:
        if len(values) == 0:
            return {}
        out: dict[str, float] = {}
        prev_last: float | None = None
        for m in sorted(months.unique()):
            mask = months == m
            grp = values[mask]
            last = float(grp.iloc[-1])
            if prev_last is None:
                first = float(grp.iloc[0])
                out[m] = last / first - 1 if first != 0 else 0.0
            else:
                out[m] = last / prev_last - 1 if prev_last != 0 else 0.0
            prev_last = last
        return out

    # -------------------------------------------------------------------------
    # helpers
    # -------------------------------------------------------------------------

    def _valid_trades(self, trade_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for t in trade_log:
            action = str(t.get("action", "")).lower()
            if action not in ("buy", "sell"):
                continue
            if self._trade_date_of(t) in (None, ""):
                continue
            shares = float(t.get("shares", 0) or 0)
            if shares <= 0:
                continue
            out.append(t)
        return out

    @staticmethod
    def _trade_date_of(t: dict[str, Any]) -> Any:
        for key in ("date", "execution_date", "trade_date"):
            if t.get(key) is not None:
                return t[key]
        return ""

    @staticmethod
    def _to_date(d: Any) -> date:
        if isinstance(d, datetime):
            return d.date()
        if isinstance(d, date):
            return d
        if d is None or d == "":
            raise ValueError("missing trade date")
        return pd.Timestamp(d).date()

    @staticmethod
    def _series_return(s: pd.Series) -> float:
        if s is None or len(s) == 0:
            return 0.0
        first = float(s.iloc[0])
        last = float(s.iloc[-1])
        if first == 0:
            return 0.0
        return last / first - 1

    @staticmethod
    def _prepare_equity(equity_curve: pd.DataFrame) -> pd.DataFrame:
        if equity_curve is None or equity_curve.empty:
            return pd.DataFrame(columns=["trade_date", "equity", "holding"])
        eq = equity_curve.copy()
        eq["trade_date"] = pd.to_datetime(eq["trade_date"])
        return eq.sort_values("trade_date").reset_index(drop=True)

    def _norm_df(self, code: str, data: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
        if code in self._df_cache:
            return self._df_cache[code]
        if code not in data:
            return None
        df = data[code].copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values("trade_date").drop_duplicates("trade_date").set_index("trade_date")
        self._df_cache[code] = df
        return df

    def _close_on(self, data: dict[str, pd.DataFrame], code: str, query_date: Any) -> float | None:
        df = self._norm_df(code, data)
        if df is None:
            return None
        ts = pd.Timestamp(query_date)
        sub = df["close"][df.index <= ts]
        if sub.empty:
            return None
        return float(sub.iloc[-1])


# =============================================================================
# HTML report
# =============================================================================


def _fmt_pct(x: Any) -> str:
    return f"{float(x) * 100:.2f}%"


def _fmt_money(x: Any) -> str:
    v = float(x)
    sign = "-" if v < 0 else ""
    return f"{sign}¥{abs(v):,.2f}"


def _fmt_num(x: Any) -> str:
    return f"{float(x):,.4f}"


def generate_html_report(report: AttributionReport, output_path: Path) -> Path:
    """Render an :class:`AttributionReport` to a self-contained HTML file.

    Uses the Jinja2 template at ``templates/monthly_report.html.j2`` next to
    this module. No external CSS/JS dependencies.
    """
    template_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["fmt_pct"] = _fmt_pct
    env.filters["fmt_money"] = _fmt_money
    env.filters["fmt_num"] = _fmt_num

    template = env.get_template("monthly_report.html.j2")
    factor_max = max((abs(v) for v in report.factor_returns.values()), default=0.0)
    html = template.render(report=report, factor_max=factor_max)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
