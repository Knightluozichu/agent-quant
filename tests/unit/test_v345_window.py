"""V3-G/V4/V5 样本外窗口 (2026-07-01 → 最新) 单元测试.

对应: scripts/exp_v345_window_20261003.py 与 run_full_pool_strategy 的 start 参数.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))

import qixing_v4 as v4
import run_qixing_v3 as rq
from exp_v3g_full_pool_fast_slow import run_full_pool_strategy
from test_circuit_breaker import _cal, _synthetic_pool


def _pool() -> dict[str, pd.DataFrame]:
    data = _synthetic_pool()
    days = _cal(date(2025, 1, 2), 200)
    for extra in (set(rq.ETF_POOL) | {rq.DEFENSE}) - set(data):
        flat = [(d, 50.0, 50.02, 49.98, 50.0, 1_000_000.0) for d in days]
        data[extra] = pd.DataFrame(
            flat, columns=["trade_date", "open", "high", "low", "close", "volume"]
        )
    return data


@pytest.mark.unit
def test_start_window_fresh_cash_and_first_date():
    data = _pool()
    start = _cal(date(2025, 1, 2), 200)[170]
    res = run_full_pool_strategy(data, v4.V4_PARAMS, start=start)
    curve = res["equity_curve"]
    assert pd.Timestamp(curve["trade_date"].iloc[0]).date() == start
    # 全新现金起步: 首日权益 = 10 万 (未成交) 或成交后近似 10 万 (仅费用损耗)
    assert float(curve["equity"].iloc[0]) == pytest.approx(100_000.0, rel=0.01)
    # 窗口前的日子不得出现在曲线里
    assert pd.Timestamp(curve["trade_date"].iloc[-1]).date() > start


@pytest.mark.unit
def test_start_none_is_full_history_baseline():
    data = _pool()
    a = run_full_pool_strategy(data, v4.V4_PARAMS)
    b = run_full_pool_strategy(data, v4.V4_PARAMS, start=None)
    assert a["equity_curve"]["equity"].tolist() == b["equity_curve"]["equity"].tolist()
    assert a["trades"] == b["trades"]


@pytest.mark.unit
def test_window_grid_stays_production_aligned():
    """窗口运行的调仓网格必须锚定全序列 (dates[130::5]), 不因起点平移."""
    data = _pool()
    days = _cal(date(2025, 1, 2), 200)
    start = days[173]  # 非网格对齐起点 (130+43)
    res = run_full_pool_strategy(data, v4.V4_PARAMS, start=start)
    expected_grid = {d for d in set(days[130::5]) if d >= start}
    trade_days = {date.fromisoformat(t["date"]) for t in res["trades"]}
    # 成交只允许发生在生产网格日 (紧急/熔断类除外, 合成数据无此类事件)
    assert trade_days <= expected_grid
