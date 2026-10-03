"""V5 (V4 + RSRS 择时过滤) 单元测试.

注入点机制用 stub filter 验证; RSRS 过滤器的滞回/放行/强制防御用可控 z 序列验证;
z 序列本身的因果性由 tests/unit/test_rsrs_momentum.py 覆盖 (复用 erm 原语).
对应被测模块: scripts/exp_v5_compare.py; 预注册: tasks/EXPERIMENTS.md 2026-10-03 H5.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))

import exp_v5_compare as ev5
import live_signal as live
import qixing_v4 as v4
import run_qixing_v3 as rq
from exp_v3g_full_pool_fast_slow import run_full_pool_strategy
from test_circuit_breaker import _cal, _synthetic_pool

DEFENSE = live.DEFENSE


# --------------------------------------------------------------------------- #
# RsrsTargetFilter 滞回语义
# --------------------------------------------------------------------------- #
def _filter_with_z(z_by_code: dict[str, list[float]], days: list[date]) -> ev5.RsrsTargetFilter:
    z_series = {
        code: pd.Series(vals, index=days[: len(vals)]) for code, vals in z_by_code.items()
    }
    return ev5.RsrsTargetFilter(z_series=z_series, defense=DEFENSE, thr=0.7)


@pytest.mark.unit
def test_filter_forces_defense_below_threshold():
    days = _cal(date(2026, 6, 1), 6)
    f = _filter_with_z({"518880": [0.1, -0.8, 0.0, 0.0, 0.0, 0.0]}, days)
    assert f(days[0], "518880", None) == "518880"   # z=0.1 中间区, 初始允许
    assert f(days[1], "518880", "518880") == DEFENSE  # z=-0.8 强制防御
    assert f(days[2], "518880", DEFENSE) == DEFENSE   # 滞回: 中间区保持关


@pytest.mark.unit
def test_filter_reopens_above_threshold():
    days = _cal(date(2026, 6, 1), 5)
    f = _filter_with_z({"518880": [-0.9, 0.0, 0.8, 0.2]}, days)
    assert f(days[0], "518880", None) == DEFENSE    # 触发关
    assert f(days[1], "518880", DEFENSE) == DEFENSE  # 滞回保持
    assert f(days[2], "518880", DEFENSE) == "518880"  # z=0.8 重新放行
    assert f(days[3], "518880", "518880") == "518880"


@pytest.mark.unit
def test_filter_passes_defense_and_missing_z():
    days = _cal(date(2026, 6, 1), 3)
    f = _filter_with_z({}, days)
    assert f(days[0], DEFENSE, None) == DEFENSE       # 防御 target 不干预
    assert f(days[1], "518880", DEFENSE) == "518880"  # z 缺失 = 中性, 沿用初始允许


# --------------------------------------------------------------------------- #
# 引擎 target_filter 注入点
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_engine_target_filter_forces_defense_same_day():
    """stub filter 在指定日强制防御 → same-close 口径当日成交."""
    data = _synthetic_pool()
    days = _cal(date(2025, 1, 2), 200)
    for extra in (set(rq.ETF_POOL) | {rq.DEFENSE}) - set(data):
        flat = [(d, 50.0, 50.02, 49.98, 50.0, 1_000_000.0) for d in days]
        data[extra] = pd.DataFrame(
            flat, columns=["trade_date", "open", "high", "low", "close", "volume"]
        )
    force_day = days[170]

    def stub_filter(td: date, target: str, holding: str | None) -> str:
        return DEFENSE if td == force_day else target

    res = run_full_pool_strategy(data, v4.V4_PARAMS, target_filter=stub_filter)
    sells = [t for t in res["trades"] if t["action"] == "sell" and t["code"] == "518880"]
    assert sells, "强制防御必须卖出持仓"
    first = min(sells, key=lambda t: t["date"])
    assert first["date"] == str(force_day)  # same-close 当日成交
    assert first["price"] == pytest.approx(float(data["518880"].close.iloc[170]), rel=1e-6)
    # 卖出同日必须买入防御
    buys_def = [t for t in res["trades"] if t["action"] == "buy" and t["code"] == DEFENSE]
    assert any(t["date"] == str(force_day) for t in buys_def)


@pytest.mark.unit
def test_engine_target_filter_none_is_baseline():
    data = _synthetic_pool()
    days = _cal(date(2025, 1, 2), 200)
    for extra in (set(rq.ETF_POOL) | {rq.DEFENSE}) - set(data):
        flat = [(d, 50.0, 50.02, 49.98, 50.0, 1_000_000.0) for d in days]
        data[extra] = pd.DataFrame(
            flat, columns=["trade_date", "open", "high", "low", "close", "volume"]
        )
    a = run_full_pool_strategy(data, v4.V4_PARAMS)
    b = run_full_pool_strategy(data, v4.V4_PARAMS, target_filter=None)
    assert a["equity_curve"]["equity"].tolist() == b["equity_curve"]["equity"].tolist()
