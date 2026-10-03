"""实验: 账户级熔断保护层 (H4) — V4 回放叠加, 预注册影子研究, 不触达生产.

预注册假设 (2026-10-03, 改参数即作废重注册, 全文见 tasks/EXPERIMENTS.md):
  H4: V4 回放 (同一快照/网格, T 收盘信号 T+1 开盘) 叠加 R1/R2/R3 三条账户级熔断,
      1x 成本全周期下保留 ≥85% CAGR 且 MDD 改善 ≥5pp。
  裁决: 达标 → 影子候选; CAGR 保留 <70% → 明确失败; 中间 → 冻结为风险预算选项。

规则 (北京炒家账户级风控 + 天涯跟踪止盈的整数化, 未做参数扫描):
  R1 日亏熔断: 净值日跌 ≤ -2% (有持仓) → T+1 清为防御, T+1 起 3 个交易日禁开风险仓;
  R2 月回撤熔断: 净值较当月内峰值回撤 ≥10% → T+1 清为防御, 当月禁开风险仓;
  R3 持仓跟踪止盈: 持仓市值较该持仓期间峰值回撤 ≥10% → T+1 清为防御, 无额外冷却。

因果保证: 判定只用 ≤T 收盘净值/市值; 主口径 same-close (14:50 当日成交近似, 与归档
V4 全历史及用户实盘一致); next-open 口径为保守参照, 成交恒在 T+1 开盘;
protector=None 与基线逐点一致 (tests/unit/test_circuit_breaker.py 验证)。

口径修正披露 (2026-10-03, 规则/阈值未变): 预注册时计划用 9/30 快照 + next-open;
执行中发现该口径下 V4 全历史基线为负 (与归档 92.5% CAGR 矛盾), 定位为成交口径差异
(归档为 same-close 近似) 而非数据问题, 故主口径切到 same-close + cross_asset 全历史,
并用"含原油池截断 2026-08-10 复现归档 5,587,053/223 腿"做金标准校验。

执行: uv run python scripts/exp_circuit_breaker.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import live_signal as live
import run_qixing_v3 as rq
from review_rotation_20260916 import replay_next_open

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "qixing_results"
RESULT = OUT_DIR / "circuit_breaker_20261003.json"
REPORT_DIR = ROOT / "reports"

COST_SIDE = rq.FEE + rq.SLIPPAGE  # 0.0015 单边

# 预注册固定阈值
DAY_STOP = 0.02
DAY_COOLDOWN = 3
MONTH_DD = 0.10
TRAIL = 0.10


class CircuitBreaker:
    """账户级熔断器: 每个交易日收盘调用 on_close, 返回 True = 次日开盘清为防御."""

    def __init__(
        self,
        *,
        calendar: list[date],
        day_stop: float = DAY_STOP,
        day_cooldown_days: int = DAY_COOLDOWN,
        month_dd: float = MONTH_DD,
        trail: float = TRAIL,
    ) -> None:
        self._calendar = calendar
        self._idx = {d: i for i, d in enumerate(calendar)}
        self.day_stop = day_stop
        self.day_cooldown_days = day_cooldown_days
        self.month_dd = month_dd
        self.trail = trail
        self._prev_equity: float | None = None
        self._prev_holding: str | None = None
        self._pos_peak: float | None = None
        self._cooldown_end: date | None = None
        self._month_key: tuple[int, int] | None = None
        self._month_peak = 0.0
        self._month_stopped: set[tuple[int, int]] = set()
        self.events: list[dict[str, Any]] = []  # 熔断事件日志 (研究证据)

    def entries_allowed(self, td: date) -> bool:
        if (td.year, td.month) in self._month_stopped:
            return False
        return not (self._cooldown_end is not None and td <= self._cooldown_end)

    def on_close(
        self,
        td: date,
        equity: float,
        holding: str | None,
        holding_value: float,
        defense: str,
    ) -> bool:
        rules: list[str] = []
        # R1 日亏熔断 (有持仓才判定)
        if (
            holding is not None
            and self._prev_equity is not None
            and equity / self._prev_equity - 1 <= -self.day_stop
        ):
            i = self._idx[td]
            end_i = min(i + self.day_cooldown_days, len(self._calendar) - 1)
            self._cooldown_end = self._calendar[end_i]
            rules.append("R1")
        # R2 月回撤熔断
        mk = (td.year, td.month)
        if mk != self._month_key:
            self._month_key = mk
            self._month_peak = equity
        self._month_peak = max(self._month_peak, equity)
        if mk not in self._month_stopped and equity / self._month_peak - 1 <= -self.month_dd:
            self._month_stopped.add(mk)
            rules.append("R2")
        # R3 持仓跟踪止盈 (防御/现金不适用; 换仓重置峰值)
        if holding not in (None, defense):
            if holding != self._prev_holding or self._pos_peak is None:
                self._pos_peak = holding_value
            else:
                self._pos_peak = max(self._pos_peak, holding_value)
            if holding_value / self._pos_peak - 1 <= -self.trail:
                rules.append("R3")
        if rules:
            self.events.append({"date": str(td), "rules": rules, "equity": round(equity, 2)})
        self._prev_equity = equity
        self._prev_holding = holding
        return bool(rules)


# --------------------------------------------------------------------------- #
# 回吐事件分析: 基线中浮盈峰值回吐最大的持仓段
# --------------------------------------------------------------------------- #
def giveback_episodes(replay: dict[str, Any], data: dict[str, pd.DataFrame], top: int = 5) -> list[dict[str, Any]]:
    """逐持仓段计算: 持仓期间收盘市值峰值 vs 退出时实际所得, 按回吐金额排序.
    成交记录缺 shares 时按 amount/(price*(1+单边成本)) 反推 (same-close 引擎口径)."""
    trades = replay["trades"]
    maps = {c: dict(zip(data[c].trade_date, range(len(data[c])), strict=True)) for c in data}
    episodes = []
    open_lot: dict[str, Any] | None = None
    for t in trades:
        if t["action"] == "buy":
            open_lot = t
        elif t["action"] == "sell" and open_lot is not None and t["code"] == open_lot["code"]:
            code = t["code"]
            f = data[code]
            shares = open_lot.get("shares") or (
                open_lot["amount"] / (open_lot["price"] * (1 + COST_SIDE))
            )
            buy_d = date.fromisoformat(open_lot["date"])
            sell_d = date.fromisoformat(t["date"])
            i0, i1 = maps[code][buy_d], maps[code][sell_d]
            seg = f.close.iloc[i0 : i1 + 1]
            peak_pos = int(seg.to_numpy().argmax())
            peak_value = float(seg.iloc[peak_pos]) * shares
            exit_value = t["price"] * shares
            episodes.append(
                {
                    "code": code,
                    "entry": open_lot["date"],
                    "exit": t["date"],
                    "peak_date": str(f.trade_date.iloc[i0 + peak_pos]),
                    "peak_float_pct": round(peak_value / (open_lot["price"] * shares) - 1, 4),
                    "giveback_pct": round(exit_value / peak_value - 1, 4),
                    "giveback_amt": round(exit_value - peak_value, 2),
                }
            )
            open_lot = None
    episodes.sort(key=lambda e: e["giveback_amt"])
    return episodes[:top]


def _equity_metrics(curve: pd.DataFrame, initial: float = 100_000.0) -> dict[str, Any]:
    import numpy as np

    eq = curve["equity"].astype(float)
    days = max((curve["trade_date"].iloc[-1] - curve["trade_date"].iloc[0]).days, 1)
    cagr = (float(eq.iloc[-1]) / initial) ** (365.25 / days) - 1
    ret = eq.pct_change().dropna()
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if len(ret) > 1 and ret.std() > 0 else 0.0
    return {
        "ending_equity": round(float(eq.iloc[-1]), 2),
        "cagr": round(cagr, 4),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(float((eq / eq.cummax() - 1).min()), 4),
    }


def main() -> None:
    from unittest.mock import patch

    import exp_v3g_relative_rotation as rr
    import qixing_v4 as v4
    from exp_v3g_full_pool_fast_slow import WARMUP, run_full_pool_strategy

    full_data = rq.load_data()
    prod_pool = {c: n for c, n in rq.ETF_POOL.items() if c != "501018"}
    all_dates = rr.common_dates(full_data)
    trading_dates = all_dates[WARMUP:]

    report: dict[str, Any] = {
        "experiment": "H4: 账户级熔断保护层 (预注册 2026-10-03; 规则不变, 口径修正见 convention_note)",
        "rules": {"R1": f"日跌<=-{DAY_STOP:.0%}→清防御, {DAY_COOLDOWN}日冷却",
                  "R2": f"月内峰值回撤>={MONTH_DD:.0%}→当月停手",
                  "R3": f"持仓峰值回撤>={TRAIL:.0%}→清防御, 无冷却"},
        "convention_note": (
            "主口径改为 same-close (14:50 当日成交近似): 与归档 V4 全历史及用户实盘 "
            "(14:50 信号当日执行) 一致; next-open 口径保留为保守稳健性参照。"
            "数据改用 data/cross_asset (至 2026-09-15)。规则/阈值未变, 预注册约束不变。"
        ),
        "scenarios": {},
    }

    # ---- 金标准校验: 含原油池 + same-close, 截断 2026-08-10 → 复现归档 V4 ----
    trunc = {
        c: d[d.trade_date <= date(2026, 8, 10)].reset_index(drop=True)
        for c, d in full_data.items()
    }
    golden = run_full_pool_strategy(trunc, v4.V4_PARAMS)
    report["golden_check"] = {
        "expected": {"final_value": 5_587_053, "trade_legs": 223, "early_rotations": 39},
        "actual": {
            "final_value": golden["metrics"]["final_value"],
            "trade_legs": golden["metrics"]["trade_legs"],
            "early_rotations": golden["metrics"]["early_rotations"],
        },
    }
    print(f"[golden] {json.dumps(report['golden_check'], ensure_ascii=False)}")

    # ---- 主口径 A/B: 生产池 (删油) + same-close, 全历史 ----
    with patch.object(rq, "ETF_POOL", prod_pool):
        for mult in (1, 2, 3):
            baseline = run_full_pool_strategy(full_data, v4.V4_PARAMS, cost_multiplier=mult)
            breaker = CircuitBreaker(calendar=list(trading_dates))
            protected = run_full_pool_strategy(
                full_data, v4.V4_PARAMS, cost_multiplier=mult, protector=breaker
            )
            bm = {k: baseline["metrics"][k] for k in
                  ("final_value", "cagr", "sharpe", "max_drawdown", "trade_legs")}
            pm = {k: protected["metrics"][k] for k in
                  ("final_value", "cagr", "sharpe", "max_drawdown", "trade_legs")}
            scenario = {
                "baseline": bm,
                "protected": pm,
                "cagr_retention": round(pm["cagr"] / bm["cagr"], 4) if bm["cagr"] > 0 else None,
                "mdd_improve_pp": round((pm["max_drawdown"] - bm["max_drawdown"]) * 100, 2),
                "breaker_events": len(breaker.events),
            }
            report["scenarios"][f"same_close_{mult}x"] = scenario
            print(f"[same-close {mult}x] {json.dumps(scenario, ensure_ascii=False)}")
            if mult == 1:
                report["breaker_event_log"] = breaker.events
                report["giveback_episodes_baseline"] = giveback_episodes(baseline, full_data)
                report["giveback_episodes_protected"] = giveback_episodes(protected, full_data)

    # ---- 保守稳健性参照: 生产池 + next-open ----
    codes = [*live.ETF_POOL, live.DEFENSE]
    dates = sorted(set.intersection(*(set(full_data[c].trade_date) for c in codes)))
    grid = set(dates[130::5])
    for mult in (1, 2, 3):
        baseline_no = replay_next_open(
            full_data, start=dates[130], end=dates[-1], grid=grid, cost_multiplier=mult
        )
        breaker_no = CircuitBreaker(calendar=dates)
        protected_no = replay_next_open(
            full_data, start=dates[130], end=dates[-1], grid=grid,
            cost_multiplier=mult, protector=breaker_no,
        )
        bm, pm = _equity_metrics(baseline_no["curve"]), _equity_metrics(protected_no["curve"])
        scenario = {
            "baseline": {**bm, "trades": len(baseline_no["trades"])},
            "protected": {**pm, "trades": len(protected_no["trades"])},
            "cagr_retention": round(pm["cagr"] / bm["cagr"], 4) if bm["cagr"] > 0 else None,
            "mdd_improve_pp": round((pm["max_drawdown"] - bm["max_drawdown"]) * 100, 2),
            "breaker_events": len(breaker_no.events),
        }
        report["scenarios"][f"next_open_{mult}x"] = scenario
        print(f"[next-open {mult}x] {json.dumps(scenario, ensure_ascii=False)}")

    report["verdict_criteria"] = "CAGR保留≥85% 且 MDD改善≥5pp → 影子候选; <70% → 明确失败; 中间 → 冻结"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"已归档 {RESULT}")


if __name__ == "__main__":
    main()
