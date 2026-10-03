"""账户级熔断保护层 (H4) 单元测试.

规则触发用合成净值序列验证; 因果性用"熔断信号日不得成交、T+1 开盘成交"验证;
基线回归用 protector=None 与不传 protector 等价验证.
对应被测模块: scripts/exp_circuit_breaker.py
预注册: tasks/EXPERIMENTS.md 2026-10-03 H4 (R1/R2/R3).
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

import exp_circuit_breaker as ecb
import live_signal as live
from review_rotation_20260916 import replay_next_open

DEFENSE = live.DEFENSE


def _cal(start: date, n: int) -> list[date]:
    """n 个连续工作日."""
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


# --------------------------------------------------------------------------- #
# R1 日亏熔断
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_r1_triggers_on_2pct_daily_loss():
    cal = _cal(date(2026, 6, 1), 10)
    cb = ecb.CircuitBreaker(calendar=cal)
    assert cb.on_close(cal[0], 100_000.0, "518880", 100_000.0, DEFENSE) is False
    assert cb.on_close(cal[1], 97_400.0, "518880", 97_400.0, DEFENSE) is True  # -2.6%
    # T+1 起 3 个交易日禁开风险仓
    assert cb.entries_allowed(cal[2]) is False
    assert cb.entries_allowed(cal[3]) is False
    assert cb.entries_allowed(cal[4]) is False
    assert cb.entries_allowed(cal[5]) is True


@pytest.mark.unit
def test_r1_ignores_small_loss_and_cash():
    cal = _cal(date(2026, 6, 1), 5)
    cb = ecb.CircuitBreaker(calendar=cal)
    cb.on_close(cal[0], 100_000.0, "518880", 100_000.0, DEFENSE)
    assert cb.on_close(cal[1], 98_500.0, "518880", 98_500.0, DEFENSE) is False  # -1.5%
    # 全现金 (无持仓) 不触发
    assert cb.on_close(cal[2], 98_500.0, None, 0.0, DEFENSE) is False


# --------------------------------------------------------------------------- #
# R2 月回撤熔断
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_r2_month_drawdown_stops_rest_of_month():
    cal = _cal(date(2026, 6, 1), 30)
    cb = ecb.CircuitBreaker(calendar=cal)
    cb.on_close(cal[0], 100_000.0, "518880", 100_000.0, DEFENSE)  # 月内峰值
    assert cb.on_close(cal[5], 89_900.0, "518880", 89_900.0, DEFENSE) is True  # -10.1%
    # 当月剩余交易日全部禁开 (R1 冷却 3 天之外仍被 R2 压住)
    assert cb.entries_allowed(cal[9]) is False
    assert cb.entries_allowed(cal[15]) is False
    # 跨月解除, 峰值重置
    next_month = next(d for d in cal if d.month == 7)
    assert cb.entries_allowed(next_month) is True


@pytest.mark.unit
def test_r2_ignores_shallow_drawdown():
    # 每日 -1% 渐进阴跌到月内 -9%: 不触发 R1 (单日未达 -2%), 也不触发 R2 (未达 -10%)
    cal = _cal(date(2026, 6, 1), 12)
    cb = ecb.CircuitBreaker(calendar=cal)
    equity = 100_000.0
    assert cb.on_close(cal[0], equity, "518880", equity, DEFENSE) is False
    for d in cal[1:10]:
        equity *= 0.99
        assert cb.on_close(d, equity, "518880", equity, DEFENSE) is False
    assert equity / 100_000.0 - 1 == pytest.approx(-0.0865, abs=1e-3)  # 累计约 -8.6%
    assert cb.entries_allowed(cal[11]) is True


# --------------------------------------------------------------------------- #
# R3 持仓跟踪止盈
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_r3_trailing_giveback_triggers():
    # 现金 90 万缓冲, 使净值波动远小于持仓波动 → 隔离 R3, 不连带 R1/R2
    cal = _cal(date(2026, 6, 1), 10)
    cb = ecb.CircuitBreaker(calendar=cal)
    cb.on_close(cal[0], 1_000_000.0, "518880", 100_000.0, DEFENSE)
    cb.on_close(cal[1], 1_010_000.0, "518880", 110_000.0, DEFENSE)  # 持仓峰值
    assert cb.on_close(cal[2], 998_900.0, "518880", 98_900.0, DEFENSE) is True  # 峰值 -10.1%
    # R3 无月封: 净值仅 -1.1%, R2 不得触发
    assert (cal[2].year, cal[2].month) not in cb._month_stopped


@pytest.mark.unit
def test_r3_peak_resets_on_holding_change():
    cal = _cal(date(2026, 6, 1), 10)
    cb = ecb.CircuitBreaker(calendar=cal)
    cb.on_close(cal[0], 1_000_000.0, "518880", 100_000.0, DEFENSE)
    cb.on_close(cal[1], 1_020_000.0, "518880", 120_000.0, DEFENSE)
    # 换仓到 159915, 新持仓以首日市值为基准, 不许拿旧峰值判
    assert cb.on_close(cal[2], 1_018_000.0, "159915", 118_000.0, DEFENSE) is False
    assert cb.on_close(cal[3], 1_007_000.0, "159915", 107_000.0, DEFENSE) is False  # -9.3%
    assert cb.on_close(cal[4], 1_005_000.0, "159915", 105_000.0, DEFENSE) is True  # -11%


@pytest.mark.unit
def test_r3_ignores_defense_holding():
    cal = _cal(date(2026, 6, 1), 5)
    cb = ecb.CircuitBreaker(calendar=cal)
    cb.on_close(cal[0], 100_000.0, DEFENSE, 100_000.0, DEFENSE)
    assert cb.on_close(cal[1], 99_900.0, DEFENSE, 99_900.0, DEFENSE) is False


# --------------------------------------------------------------------------- #
# 回放集成: 熔断 T+1 开盘成交 (因果), protector=None 基线回归
# --------------------------------------------------------------------------- #
def _synthetic_pool(n: int = 200) -> dict[str, pd.DataFrame]:
    """518880 稳定领涨 (V4 应持有), 第 160 天单日 -2.5% (触发 R1, 不触发 V4 -3% 门);
    其余标的阴跌或走平."""
    days = _cal(date(2025, 1, 2), n)
    out = {}
    codes = dict.fromkeys(live.ETF_POOL, -0.0005)
    codes["518880"] = 0.003
    for code, drift in codes.items():
        closes = [10.0]
        for i in range(1, n):
            step = closes[-1] * (1 + drift)
            if code == "518880" and i == 160:
                step = closes[-1] * 0.975  # -2.5%
            closes.append(step)
        rows = []
        prev = closes[0]
        for d, c in zip(days, closes, strict=True):
            rows.append((d, prev, max(prev, c) * 1.001, min(prev, c) * 0.999, c, 1_000_000.0))
            prev = c
        out[code] = pd.DataFrame(
            rows, columns=["trade_date", "open", "high", "low", "close", "volume"]
        )
    flat = [(d, 100.0, 100.05, 99.95, 100.0, 1_000_000.0) for d in days]
    out[DEFENSE] = pd.DataFrame(
        flat, columns=["trade_date", "open", "high", "low", "close", "volume"]
    )
    return out


@pytest.mark.unit
def test_breaker_flatten_fills_next_open_not_signal_day():
    data = _synthetic_pool()
    dates = sorted(set.intersection(*(set(f.trade_date) for f in data.values())))
    start = dates[130]
    grid = set(dates[130::5])
    crash_day = dates[160]
    next_day = dates[161]

    protected = replay_next_open(
        data, start=start, end=dates[-1], grid=grid,
        protector=ecb.CircuitBreaker(calendar=dates),
    )
    sells = [t for t in protected["trades"] if t["action"] == "sell" and t["code"] == "518880"]
    assert sells, "保护版必须卖出 518880"
    first_sell = min(sells, key=lambda t: t["date"])
    assert first_sell["signal_date"] == str(crash_day)  # 熔断信号日 = 崩盘日收盘
    assert first_sell["date"] == str(next_day)          # 成交 = 次日
    assert first_sell["price"] == pytest.approx(
        float(data["518880"].open.iloc[161])
    )  # 用开盘价

    # 基线 (无熔断): V4 自身规则不该在同一时点卖出 (-2.5% 不触发 -3% 门)
    baseline = replay_next_open(data, start=start, end=dates[-1], grid=grid)
    base_sells = [
        t for t in baseline["trades"]
        if t["action"] == "sell" and t["code"] == "518880" and t["date"] <= str(dates[165])
    ]
    assert not base_sells, "基线在 R1 触发窗口内不应卖出 (差异证明熔断生效)"


@pytest.mark.unit
def test_protector_none_is_exact_baseline():
    data = _synthetic_pool()
    dates = sorted(set.intersection(*(set(f.trade_date) for f in data.values())))
    kw = {"start": dates[130], "end": dates[-1], "grid": set(dates[130::5])}
    a = replay_next_open(data, **kw)
    b = replay_next_open(data, **kw, protector=None)
    assert a["curve"]["equity"].tolist() == b["curve"]["equity"].tolist()
    assert a["trades"] == b["trades"]


# --------------------------------------------------------------------------- #
# same-close 主口径: 金标准复现 + 熔断当日成交约定
# --------------------------------------------------------------------------- #
CROSS_ASSET = Path(__file__).parent.parent.parent / "data" / "cross_asset"


@pytest.mark.unit
@pytest.mark.skipif(not CROSS_ASSET.exists(), reason="cross_asset 数据不在本地")
def test_same_close_engine_reproduces_archived_v4():
    """含原油池 + same-close + 截断 2026-08-10 必须复现归档 V4 全历史
    (5,587,053 / 223 腿 / 39 次提前轮动, tasks/EXPERIMENTS.md 2026-08-11)."""
    import qixing_v4 as v4
    import run_qixing_v3 as rq
    from exp_v3g_full_pool_fast_slow import run_full_pool_strategy

    data = rq.load_data()
    trunc = {
        c: d[d.trade_date <= date(2026, 8, 10)].reset_index(drop=True)
        for c, d in data.items()
    }
    res = run_full_pool_strategy(trunc, v4.V4_PARAMS)
    assert res["metrics"]["final_value"] == pytest.approx(5_587_053, rel=0.01)
    assert res["metrics"]["trade_legs"] == 223
    assert res["metrics"]["early_rotations"] == 39


@pytest.mark.unit
def test_same_close_breaker_flattens_same_day():
    """same-close 口径约定: 熔断 T 收盘判定 → 当日 ~14:50 近似价成交 (与基线同口径).
    与 next-open 口径 (T+1 开盘) 的差异必须显式体现在测试里."""
    import qixing_v4 as v4
    import run_qixing_v3 as rq
    from exp_v3g_full_pool_fast_slow import run_full_pool_strategy

    data = _synthetic_pool()
    # rq.load_data 形状对齐: same-close 引擎吃 rq 全池 (含 501018), 补齐缺失代码
    days = _cal(date(2025, 1, 2), 200)
    for extra in (set(rq.ETF_POOL) | {rq.DEFENSE}) - set(data):
        flat = [(d, 50.0, 50.02, 49.98, 50.0, 1_000_000.0) for d in days]
        data[extra] = pd.DataFrame(
            flat, columns=["trade_date", "open", "high", "low", "close", "volume"]
        )
    all_dates = sorted(set.intersection(*(set(f.trade_date) for f in data.values())))
    crash_day = days[160]

    protected = run_full_pool_strategy(
        data, v4.V4_PARAMS, protector=ecb.CircuitBreaker(calendar=all_dates[130:])
    )
    sells = [t for t in protected["trades"] if t["action"] == "sell" and t["code"] == "518880"]
    assert sells, "保护版必须卖出 518880"
    first_sell = min(sells, key=lambda t: t["date"])
    assert first_sell["date"] == str(crash_day)  # same-close: 当日成交
    assert first_sell["price"] == pytest.approx(
        float(data["518880"].close.iloc[160]), rel=1e-6
    )
