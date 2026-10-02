"""结构交易 (隔日/缠论) 研究回测单元测试.

几何正确性用已知结构的手工序列验证; 因果性用"信号日不得成交"验证.
对应被测模块: scripts/exp_structure_trade.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

import exp_structure_trade as est


def _bars(rows: list[tuple[float, float, float, float]]) -> list[est.Bar]:
    """rows: (open, high, low, close)"""
    return [est.Bar(i, o, h, lo, c) for i, (o, h, lo, c) in enumerate(rows)]


# --------------------------------------------------------------------------- #
# 包含关系
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_inclusion_merge_up_takes_higher_high_and_higher_low():
    # 方向向上 (bar0→bar1 升高), bar2 被 bar1 包含 → 合并取高高/高低
    bars = _bars([
        (10, 11, 9, 10.5),
        (10.5, 12, 10, 11.5),
        (11, 11.5, 10.2, 11),  # 被前一根包含 (高12>11.5, 低10<10.2)
    ])
    merged = est.merge_inclusion(bars)
    assert len(merged) == 2
    assert merged[-1].high == pytest.approx(12)
    assert merged[-1].low == pytest.approx(10.2)


@pytest.mark.unit
def test_inclusion_merge_down_takes_lower_low_and_lower_high():
    bars = _bars([
        (10, 11, 9, 9.5),
        (9.5, 10, 8, 8.5),   # 方向向下
        (9, 9.8, 8.4, 9),    # 被前一根包含 (高10>9.8, 低8<8.4)
    ])
    merged = est.merge_inclusion(bars)
    assert len(merged) == 2
    assert merged[-1].high == pytest.approx(9.8)
    assert merged[-1].low == pytest.approx(8)


# --------------------------------------------------------------------------- #
# 分型
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_fractal_top_and_bottom_detected_on_merged_bars():
    bars = _bars([
        (10, 11, 9, 10),
        (11, 13, 10.5, 12),   # 顶候选 (高最高且低也最高)
        (11, 12, 10.2, 11),
        (10, 11, 8, 9),
        (9, 9.2, 7.5, 8),     # 底候选 (低最低且高也最低)
        (8.5, 9.5, 8.2, 9),
    ])
    merged = est.merge_inclusion(bars)
    fr = est.find_fractals(merged)
    tops = [f for f in fr if f.kind == "top"]
    bottoms = [f for f in fr if f.kind == "bottom"]
    assert len(tops) == 1 and tops[0].idx == 1
    assert len(bottoms) >= 1


# --------------------------------------------------------------------------- #
# 笔
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_bi_requires_min_bars_between_extremes():
    # 顶底之间不足 3 根独立 K 线 → 不成笔
    bars = _bars([
        (10, 11, 9, 10),
        (11, 13, 10.5, 12),   # 顶
        (10, 11, 9.5, 10),    # 底 (与顶之间没有独立K线)
        (10.5, 12, 10, 11.5),
    ])
    merged = est.merge_inclusion(bars)
    fr = est.find_fractals(merged)
    bi = est.build_bi(merged, fr)
    assert bi == []


@pytest.mark.unit
def test_bi_connects_alternating_fractals():
    # 明确 zigzag: 底(1) → 顶(5) → 底(9), 极值间隔均为 4 (中间 3 根)
    bars = _bars([
        (11, 12, 10.5, 11),      # 0
        (10, 11, 9, 10),         # 1 底
        (10, 11.5, 9.8, 11),     # 2
        (11, 12.5, 10.8, 12),    # 3
        (11.8, 13.5, 11.5, 13),  # 4
        (12.5, 14, 12, 13.5),    # 5 顶
        (11.5, 12.2, 10.5, 11),  # 6
        (10.5, 11, 9.5, 10),     # 7
        (9.8, 10.2, 9, 9.5),     # 8
        (9.5, 10, 8.5, 9),       # 9 底
        (9.5, 11, 9.2, 10.5),    # 10
        (10.5, 12.5, 10.2, 12),  # 11
    ])
    merged = est.merge_inclusion(bars)
    fr = est.find_fractals(merged)
    bi = est.build_bi(merged, fr)
    assert len(bi) >= 1
    assert bi[0].direction == "up"
    assert bi[0].start_idx == 1 and bi[0].end_idx == 5


# --------------------------------------------------------------------------- #
# 中枢
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_zhongshu_zg_zd_from_first_three_bi():
    # 手工三笔重叠: up[10→12], down[12→10.5], up[10.5→12.5]
    # ZG = min(12, 12, 12.5)=12, ZD = max(10, 10.5, 10.5)=10.5
    bi = [
        est.Bi("up", 0, 3, 10.0, 12.0),
        est.Bi("down", 3, 6, 12.0, 10.5),
        est.Bi("up", 6, 9, 10.5, 12.5),
    ]
    zs = est.build_zhongshu(bi)
    assert len(zs) == 1
    assert zs[0].zg == pytest.approx(12.0)
    assert zs[0].zd == pytest.approx(10.5)


@pytest.mark.unit
def test_zhongshu_no_overlap_returns_empty():
    bi2 = [
        est.Bi("up", 0, 3, 10.0, 11.0),
        est.Bi("down", 3, 6, 11.0, 10.5),
        est.Bi("up", 6, 9, 12.5, 13.0),  # 低点 12.5 > ZG=11 → 无重叠
    ]
    assert est.build_zhongshu(bi2) == []


# --------------------------------------------------------------------------- #
# 因果性: 信号确认前不得成交
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_chan_signal_not_actionable_before_fractal_confirmed():
    # 底分型第三根 K 线收盘前, 系统不得知道该分型存在
    bars = _bars([
        (10, 11, 9, 10),
        (9, 10, 8, 9),          # 底分型中间
        (9.5, 10.5, 9.2, 10),   # 第三根: 收盘才确认
    ])
    merged = est.merge_inclusion(bars[:2])
    fr_early = est.find_fractals(merged)
    assert all(f.kind != "bottom" or f.idx != 1 for f in fr_early)
    merged_full = est.merge_inclusion(bars)
    fr_late = est.find_fractals(merged_full)
    assert any(f.kind == "bottom" and f.idx == 1 for f in fr_late)


@pytest.mark.unit
def test_chan_engine_confirms_fractal_only_after_right_neighbor_final():
    # 流式引擎: 右邻未定型 (被包含吸收中) 时不得确认分型
    eng = est.ChanEngine()
    eng.feed(10, 11, 9, 10)
    eng.feed(9, 10, 8, 9)       # 底分型中间
    eng.feed(9.5, 10.5, 9.2, 10)  # 右邻追加 → 仍只确认到 len-3
    assert not eng.bis
    eng.feed(10, 11, 9.8, 10.8)  # 再一根 → bar1 处分型可确认 (但仅一根无法成笔)
    assert eng._anchor is not None


@pytest.mark.unit
def test_chan_backtest_executes_next_open_not_signal_bar():
    df = est.make_synthetic_third_buy_series()
    trades = est.run_chan_on_series(df, cost=0.0)
    assert trades, "合成三买序列应产生至少一笔交易"
    t0 = trades[0]
    sig_i = df.index[df["trade_date"] == t0["signal_date"]][0]
    exec_i = df.index[df["trade_date"] == t0["entry_date"]][0]
    assert exec_i == sig_i + 1
    assert t0["entry_price"] == pytest.approx(float(df["open"].iloc[exec_i]))


# --------------------------------------------------------------------------- #
# 隔日模式
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_geri_breakout_buys_next_open_and_exits_next_day_close():
    df = est.make_synthetic_breakout_series()
    trades = est.run_geri_on_series(df, cost=0.0)
    assert trades, "合成突破序列应产生至少一笔交易"
    t0 = trades[0]
    assert t0["buy_type"] == "breakout"
    sig_i = df.index[df["trade_date"] == t0["signal_date"]][0]
    ent_i = df.index[df["trade_date"] == t0["entry_date"]][0]
    assert ent_i == sig_i + 1
    assert t0["entry_price"] == pytest.approx(float(df["open"].iloc[ent_i]))
    ext_i = df.index[df["trade_date"] == t0["exit_date"]][0]
    assert ext_i >= ent_i + 1


@pytest.mark.unit
def test_geri_no_chase_when_open_gaps_too_high():
    df = est.make_synthetic_breakout_series(gap_open=1.05)
    trades = est.run_geri_on_series(df, cost=0.0)
    assert all(t["buy_type"] != "breakout" for t in trades)


@pytest.mark.unit
def test_geri_pullback_requires_shrink_volume_to_ma():
    df = est.make_synthetic_pullback_series()
    trades = est.run_geri_on_series(df, cost=0.0)
    assert any(t["buy_type"] == "pullback" for t in trades)


@pytest.mark.unit
def test_geri_breakout_closed_in_bear_env():
    df = est.make_synthetic_breakout_series()
    env = est.make_bear_env_df()  # 覆盖信号日且 close<MA60 的熊市环境
    trades = est.run_geri_on_series(df, cost=0.0, env_df=env)
    assert all(t["buy_type"] != "breakout" for t in trades)
