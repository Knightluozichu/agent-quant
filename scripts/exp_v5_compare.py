"""实验: V5 (V4 + RSRS 择时过滤) 与 V3-G / V4 同口径全量对比 — 预注册 H5, 不触达生产.

预注册 (2026-10-03, 改参数即作废重注册, 全文见 tasks/EXPERIMENTS.md):
  V5 = V4 (V4_PARAMS 原样) + RSRS 滞回择时过滤 (参数与 H3 完全一致: N=18, M=600,
  z 阈值 ±0.7, 滞回, 初始允许持有; 零新拟合)。T 收盘对 V4 选定 target 的自身
  高低价序列算 z; z<-0.7 强制防御 (非调仓日视同紧急退出), z>+0.7 放行, 中间滞回。
  裁决: CAGR 保留 ≥85% 且 MDD 改善 ≥5pp → 影子候选; <70% → 明确失败; 中间 → 冻结。

口径: same-close (14:50 当日成交近似, 与实盘一致), 生产池 (删 501018),
cross_asset 全历史, 1x/2x/3x 成本; 对照 V3-G (disabled) 与 V4 同数据同跑。

执行: uv run python scripts/exp_v5_compare.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qixing_v4 as v4
import run_qixing_v3 as rq
from exp_rsrs_momentum import decide_risk_state, rsrs_beta_series, rsrs_z
from exp_v3g_full_pool_fast_slow import run_full_pool_strategy
from qixing_v4 import FullPoolParams

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "qixing_results"
RESULT = OUT_DIR / "v5_compare_20261003.json"

RSRS_N, RSRS_M, RSRS_THR = 18, 600, 0.7  # 与 H3 预注册一致


class RsrsTargetFilter:
    """RSRS 滞回择时过滤器: 对 V4 最终 target 施加风险开/关."""

    def __init__(
        self,
        *,
        z_series: dict[str, pd.Series],
        defense: str,
        thr: float = RSRS_THR,
    ) -> None:
        self._z = z_series
        self._defense = defense
        self._thr = thr
        self._risk_on = True  # 预注册: 初始允许持有
        self.blocks = 0

    @classmethod
    def from_data(
        cls, data: dict[str, pd.DataFrame], codes: list[str], defense: str
    ) -> RsrsTargetFilter:
        """由行情数据预计算各代码 z 序列 (每个 z 只用 ≤T 数据, 逐日取用)."""
        z_series: dict[str, pd.Series] = {}
        for c in codes:
            beta = rsrs_beta_series(data[c].high.to_numpy(), data[c].low.to_numpy(), n=RSRS_N)
            z = rsrs_z(beta, m=RSRS_M)
            z.index = data[c].trade_date.to_numpy()
            z_series[c] = z
        return cls(z_series=z_series, defense=defense)

    def __call__(self, td: date, target: str, holding: str | None) -> str:
        if target == self._defense:
            return target
        z_val = float("nan")
        series = self._z.get(target)
        if series is not None and td in series.index:
            z_val = float(series.loc[td])
        self._risk_on = decide_risk_state(z_val, self._risk_on, self._thr)
        if not self._risk_on:
            self.blocks += 1
            return self._defense
        return target


def yearly_returns(curve: pd.DataFrame, initial: float = 100_000.0) -> dict[str, float]:
    """日权益曲线 → 逐年收益 (首年自 initial 起, 非完整年度如实保留)."""
    df = curve.copy()
    df["year"] = pd.to_datetime(df["trade_date"]).dt.year
    year_end = df.groupby("year")["equity"].last()
    out: dict[str, float] = {}
    prev = initial
    for y, eq in year_end.items():
        out[str(y)] = round(float(eq) / prev - 1, 4)
        prev = float(eq)
    return out


def main() -> None:
    full_data = rq.load_data()
    prod_pool = {c: n for c, n in rq.ETF_POOL.items() if c != "501018"}
    risk_codes = list(prod_pool)

    strategies: dict[str, tuple[Any, Any]] = {
        "V3-G": (FullPoolParams.disabled(), None),
        "V4": (v4.V4_PARAMS, None),
        "V5": (
            v4.V4_PARAMS,
            RsrsTargetFilter.from_data(full_data, risk_codes, rq.DEFENSE),
        ),
    }

    report: dict[str, Any] = {
        "experiment": "H5: V5 (V4+RSRS过滤) vs V4 vs V3-G (预注册 2026-10-03)",
        "params": {"rsrs_n": RSRS_N, "rsrs_m": RSRS_M, "rsrs_thr": RSRS_THR,
                   "v4": "V4_PARAMS 原样", "v3g": "FullPoolParams.disabled()"},
        "convention": "same-close (14:50 当日成交近似), 生产池(删501018), cross_asset全历史",
        "results": {},
    }
    table_rows: list[dict[str, Any]] = []
    with patch.object(rq, "ETF_POOL", prod_pool):
        for name, (params, tfilter) in strategies.items():
            per_mult: dict[str, Any] = {}
            for mult in (1, 2, 3):
                tf = tfilter if mult == 1 else (
                    RsrsTargetFilter.from_data(full_data, risk_codes, rq.DEFENSE)
                    if name == "V5" else None
                )
                res = run_full_pool_strategy(
                    full_data, params, cost_multiplier=mult, target_filter=tf
                )
                m = res["metrics"]
                per_mult[f"{mult}x"] = {
                    "final_value": round(m["final_value"], 2),
                    "total_return": round(m["total_return"], 4),
                    "cagr": round(m["cagr"], 4),
                    "sharpe": round(m["sharpe"], 3),
                    "max_drawdown": round(m["max_drawdown"], 4),
                    "trade_legs": m["trade_legs"],
                }
                if mult == 1:
                    per_mult["yearly_returns"] = yearly_returns(res["equity_curve"])
                    per_mult["span"] = [str(res["equity_curve"]["trade_date"].iloc[0])[:10],
                                        str(res["equity_curve"]["trade_date"].iloc[-1])[:10]]
                    if name == "V5":
                        per_mult["rsrs_blocks"] = tfilter.blocks if tfilter else 0
            report["results"][name] = per_mult
            row = {"strategy": name, **per_mult["1x"]}
            table_rows.append(row)
            print(f"[{name}] {json.dumps(per_mult['1x'], ensure_ascii=False)}")

    report["verdict_criteria"] = "V5 vs V4: CAGR保留≥85% 且 MDD改善≥5pp → 影子候选; <70% → 明确失败; 中间 → 冻结"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"已归档 {RESULT}")


if __name__ == "__main__":
    main()
