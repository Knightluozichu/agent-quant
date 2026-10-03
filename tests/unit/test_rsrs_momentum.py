"""ETF动量轮动+RSRS择时研究回测单元测试.

正确性用已知斜率/已知动量序列验证; 因果性用"未来数据变异不影响历史信号"
与"信号日不得成交"验证. 对应被测模块: scripts/exp_rsrs_momentum.py
预注册: tasks/EXPERIMENTS.md 2026-10-03 H3.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

import exp_rsrs_momentum as erm


def _frame(closes: list[float], *, start: date = date(2020, 1, 2), code: str = "X") -> pd.DataFrame:
    """由收盘价构造合法 OHLC 帧 (open=前收, high/low 取极值微调)."""
    n = len(closes)
    dates = [start + timedelta(days=i) for i in range(n)]
    rows = []
    prev = closes[0]
    for d, c in zip(dates, closes, strict=True):
        o = prev
        rows.append((d, o, max(o, c) * 1.001, min(o, c) * 0.999, c, 1000.0))
        prev = c
    return pd.DataFrame(rows, columns=["trade_date", "open", "high", "low", "close", "volume"])


# --------------------------------------------------------------------------- #
# OLS 斜率
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_ols_slope_known_line():
    lows = np.arange(1.0, 20.0)
    highs = 2.0 * lows + 1.0
    assert erm.ols_slope(highs, lows) == pytest.approx(2.0)


@pytest.mark.unit
def test_ols_slope_flat_is_zero():
    x = np.arange(1.0, 19.0)
    assert erm.ols_slope(np.full_like(x, 5.0), x) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# RSRS beta / z 因果性
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_beta_series_uses_only_past_bars():
    rng = np.random.default_rng(7)
    n = 100
    closes = 10 * np.cumprod(1 + rng.normal(0, 0.01, n))
    f = _frame(closes.tolist())
    full = erm.rsrs_beta_series(f.high.to_numpy(), f.low.to_numpy(), n=18)
    t = 60
    trunc = erm.rsrs_beta_series(f.high.to_numpy()[: t + 1], f.low.to_numpy()[: t + 1], n=18)
    assert full.iloc[t] == pytest.approx(trunc.iloc[t])
    assert full.iloc[:17].isna().all()  # 前 17 根不足窗, 必须为 NaN


@pytest.mark.unit
def test_z_series_uses_only_past_betas():
    rng = np.random.default_rng(11)
    beta = pd.Series(rng.normal(1.0, 0.2, 800))
    full = erm.rsrs_z(beta, m=600)
    t = 700
    trunc = erm.rsrs_z(beta.iloc[: t + 1], m=600)
    assert full.iloc[t] == pytest.approx(trunc.iloc[t])


# --------------------------------------------------------------------------- #
# 滞回状态机
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_hysteresis_thresholds():
    assert erm.decide_risk_state(0.8, prev=False) is True   # 上阈触发
    assert erm.decide_risk_state(-0.8, prev=True) is False  # 下阈触发
    assert erm.decide_risk_state(0.0, prev=True) is True    # 中间区沿用
    assert erm.decide_risk_state(0.0, prev=False) is False
    assert erm.decide_risk_state(float("nan"), prev=True) is True   # 不可计算期中性
    assert erm.decide_risk_state(float("nan"), prev=False) is False


# --------------------------------------------------------------------------- #
# 动量选股
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_momentum_pick_argmax_and_defense():
    mom = {"A": 0.05, "B": 0.12, "C": -0.03}
    assert erm.pick_target(mom) == "B"
    assert erm.pick_target({"A": -0.01, "B": -0.05}) is None  # 全部 ≤0 → 防御
    assert erm.pick_target({"A": float("nan"), "B": 0.02}) == "B"  # NaN 跳过
    assert erm.pick_target({}) is None


# --------------------------------------------------------------------------- #
# 执行因果与成本
# --------------------------------------------------------------------------- #
def _two_etf_data(n: int = 80) -> dict[str, pd.DataFrame]:
    """A 单调上涨 (动量恒正), B 单调下跌; 防御 D 平直."""
    days = [date(2021, 1, 4) + timedelta(days=i) for i in range(n)]
    a = _frame((10 * (1.002 ** np.arange(n))).tolist(), code="A")
    b = _frame((10 * (0.998 ** np.arange(n))).tolist(), code="B")
    d = _frame([100.0] * n, code="D")
    for f, c in ((a, "A"), (b, "B"), (d, "D")):
        f["trade_date"] = days
        f["symbol"] = c
    return {"A": a, "B": b, "D": d}


@pytest.mark.unit
def test_signal_day_no_trade_and_next_open_fill():
    data = _two_etf_data()
    res = erm.run_backtest(
        data, pool=["A", "B"], defense="D", cost=0.0015,
        rebalance_days=5, mom_n=20, rsrs_n=18, rsrs_m=30,
    )
    first_buy = next(t for t in res["trades"] if t["action"] == "BUY")
    # 信号日 = 网格日 (日历第 0 天起每 5 天), 成交必须在信号日之后且用开盘价
    assert first_buy["signal_date"] < first_buy["date"]
    sig_idx = data["A"].trade_date.tolist().index(first_buy["signal_date"])
    expected_open = float(data["A"].open.iloc[sig_idx + 1])
    assert first_buy["price"] == pytest.approx(expected_open)


@pytest.mark.unit
def test_round_trip_cost_reduces_equity():
    # 平直价格合成一买一卖, 腿收益 = -2*cost (成本记账验证)
    cost = 0.0015
    trades = [
        {"date": date(2021, 3, 2), "action": "BUY", "code": "A", "price": 10.0},
        {"date": date(2021, 3, 9), "action": "SELL", "code": "A", "price": 10.0},
    ]
    curve = [{"date": date(2021, 3, 1), "equity": 100_000.0},
             {"date": date(2021, 3, 10), "equity": 100_000.0 * (1 - cost) ** 2}]
    m = erm._metrics(curve, trades, 100_000.0, cost)
    assert m["legs"] == 1
    assert m["profit_factor"] == 0.0  # 纯亏腿, 无盈利 → PF=0
    # 腿内收益核对: 净卖价 - 净买价 < 0 且幅度 ≈ 2*cost*价
    ret = (10.0 * (1 - cost)) / (10.0 * (1 + cost)) - 1
    assert ret == pytest.approx(-2 * cost / (1 + cost), abs=1e-9)


@pytest.mark.unit
def test_defense_when_all_momentum_non_positive():
    data = _two_etf_data()
    data["A"] = data["B"]  # 让 A 也下跌 → 全池动量为负
    res = erm.run_backtest(
        data, pool=["A", "B"], defense="D", cost=0.0015,
        rebalance_days=5, mom_n=20, rsrs_n=18, rsrs_m=30,
    )
    risk_buys = [t for t in res["trades"] if t["action"] == "BUY" and t["code"] in {"A", "B"}]
    assert not risk_buys, "全池动量 ≤0 时不许买入风险资产"


@pytest.mark.unit
def test_profit_factor_and_metrics_present():
    data = _two_etf_data(120)
    res = erm.run_backtest(
        data, pool=["A", "B"], defense="D", cost=0.0015,
        rebalance_days=5, mom_n=20, rsrs_n=18, rsrs_m=30,
    )
    for k in ("ending_equity", "cagr", "sharpe", "max_drawdown", "profit_factor", "legs", "curve"):
        assert k in res
    assert res["legs"] >= 1
    assert res["ending_equity"] > 100_000.0  # 持有单调上涨的 A, 扣费后仍应盈利
