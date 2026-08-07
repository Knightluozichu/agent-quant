"""V32 尾部风控 (scripts/risk_overrides.py) 一致性快照测试.

背景: 修复风险项 A4 —— 回测-实盘一致性无自动化测试.
V3 改进版尾部风控纯函数层已部署实盘 (live_signal.py), 本测试锁定 assess()
的行为, 防止未来"顺手修正"导致口径漂移.

快照基准: data/v9_results/v3_tail_risk_ab.json 的 risk_events_full
(16 条 H1/H2 降仓事件, 本文件锁定其中 2026-02-02 事件).
数据源: data/cross_asset/*.parquet (与回测 load_data() 完全一致, 测试依赖).

回测怪癖锁定 (禁止单端"修正"):
  - vol20 除数错位: 行为由参数快照 + 事件快照间接冻结
  - 熔断日=调仓日 → 冷却立即解除: 见 test_cooldown_released_when_flush_day_is_rebalance_day
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

# 确保能 import scripts 模块 (与 test_no_lookahead.py 一致)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from risk_overrides import (  # noqa: E402
    ABS_WEAK,
    ACTION_EMERGENCY,
    ACTION_HOLD,
    ACTION_TRADE,
    DD_FLUSH,
    DD_HALF,
    DD_WARN,
    DECAY_THRESH,
    DEFENSE_SEQ,
    EXPO_REDUCE,
    H1_DD,
    H2_DAY,
    VOL_HV_THR,
    assess,
)
from run_qixing_v3 import DEFENSE, ETF_POOL, load_data  # noqa: E402

# 回测存档 risk_events_full 事件日: 2026-02-02 (161226 白银单日 -10% 触发 H2)
TD = date(2026, 2, 2)


@pytest.fixture(scope="module")
def env():
    """真实数据环境 (与回测同源): data 切片至 TD + 公共日历 + 当日索引.

    数据只保留到 TD (无未来函数口径, 与回测当日可用信息一致);
    idx_map 与回测 etf_data_at_date 一致: 各 code 在 TD 当日 (切片末行).
    """
    data = {
        c: df[df["trade_date"] <= TD].reset_index(drop=True)
        for c, df in load_data().items()
    }
    codes = [*ETF_POOL, DEFENSE]
    common_dates = sorted(
        set.intersection(*[set(data[c]["trade_date"]) for c in codes if c in data])
    )
    idx_map = {c: len(df) - 1 for c, df in data.items()}
    return data, common_dates, idx_map


def _holding_state(entry_price: float) -> dict:
    """构造持仓 161226 的标准 state (组合回撤 -1.9%, 不触发熔断层)."""
    return {
        "cash": 100_000.0,
        "shares": 10_000,
        "entry_price": entry_price,
        "peak_equity": 150_000.0,
        "risk_exposure": 1.0,
        "cooldown_until": None,
    }


# --------------------------------------------------------------------------- #
# 1. 参数快照
# --------------------------------------------------------------------------- #
def test_param_snapshot_matches_backtest_archive():
    """参数快照: 关键参数必须等于回测存档值 (禁止单端"修正").

    覆盖任务要求的核心参数 (VOL_HV_THR / EXPO_REDUCE / H1_DD / DD_FLUSH /
    DEFENSE_SEQ 顺序), 并顺带锁定其余风控常量与动作枚举 (口径漂移防线).
    这些参数与 exp_v32_tail_risk.py 逐字一致, 任一改动必须走全量验证.
    """
    # 任务要求的关键参数
    assert VOL_HV_THR == 0.45
    assert EXPO_REDUCE == 0.7
    assert H1_DD == -0.15
    assert DD_FLUSH == -0.30
    assert DEFENSE_SEQ == ("511260", "511220", "511880")  # 十年国债→城投债→货币

    # 顺带锁定其余常量 (防"顺手修正"漂移)
    assert H2_DAY == -0.05
    assert DECAY_THRESH == -0.02
    assert ABS_WEAK == 0.08
    assert (DD_WARN, DD_HALF) == (-0.12, -0.25)
    assert (ACTION_HOLD, ACTION_TRADE, ACTION_EMERGENCY) == (
        "hold", "trade", "emergency_defense",
    )


# --------------------------------------------------------------------------- #
# 2. 固定输入 → 固定输出
# --------------------------------------------------------------------------- #
def test_fixed_input_yields_fixed_output(env):
    """固定输入 → 固定输出: 同输入两次 assess 结果逐字段一致 (确定性).

    锁定: assess 是纯函数 (无随机/无状态副作用), 同一 (target/holding/state/
    data/td) 必须产出逐字段相同的 RiskDecision, 否则实盘与回测将无法对账.
    """
    data, common_dates, idx_map = env
    state = _holding_state(entry_price=4.722 / 1.538)  # 2026-02-02 自买入盈利 53.8%

    r1 = assess(target=None, holding="161226", state=state, data=data, td=TD,
                idx_map=idx_map, is_rebalance=False, common_dates=common_dates)
    r2 = assess(target=None, holding="161226", state=state, data=data, td=TD,
                idx_map=idx_map, is_rebalance=False, common_dates=common_dates)

    # 确定性: 两次结果逐字段一致
    assert r1 == r2

    # 关键字段类型
    assert isinstance(r1.exposure, float)
    assert isinstance(r1.action, str)
    assert isinstance(r1.events, list)
    assert r1.final_target is None or isinstance(r1.final_target, str)
    assert r1.cooldown_until is None or isinstance(r1.cooldown_until, date)

    # 事件内容确定: 同日同状态 → 同一事件序列
    assert [e["type"] for e in r1.events] == ["改进-H1/H2降仓"]


# --------------------------------------------------------------------------- #
# 3. 回测事件快照
# --------------------------------------------------------------------------- #
def test_snapshot_20260202_h1h2_event(env):
    """回测事件快照: 2026-02-02 事件与 v3_tail_risk_ab.json 逐字一致.

    存档条目: {'date': '2026-02-02', 'type': '改进-H1/H2降仓', 'from': '161226',
               'reason': 'entry_dd=53.8% day=-10.0%'}
    用真实数据 (161226 白银 2026-02-02 收 4.722, 前收 5.247) 构造等价 state
    (自买入盈利 53.8%) → assess 必须触发同类型事件:
      - exposure <= 0.7 (适度降仓, 不切防御)
      - action = hold (非调仓日仅记录, 不立即交易)
    """
    data, common_dates, idx_map = env
    cur = float(data["161226"].iloc[idx_map["161226"]]["close"])  # 4.722
    state = _holding_state(entry_price=cur / 1.538)  # 反推 entry_dd≈53.8%

    r = assess(target=None, holding="161226", state=state, data=data, td=TD,
               idx_map=idx_map, is_rebalance=False, common_dates=common_dates)

    h1h2 = [e for e in r.events if e["type"] == "改进-H1/H2降仓"]
    assert h1h2, f"未触发 H1/H2 降仓: events={r.events}"
    assert h1h2[0]["date"] == "2026-02-02"
    # 与回测存档 reason 逐字一致 (H2 当日 -10% 触发, H1 自买入盈利未触发)
    assert h1h2[0]["reason"] == "entry_dd=53.8% day=-10.0%"
    assert r.exposure <= EXPO_REDUCE  # 0.7 适度降仓
    assert r.action == ACTION_HOLD    # 非调仓日仅记录


# --------------------------------------------------------------------------- #
# 4. 边界
# --------------------------------------------------------------------------- #
def test_legacy_state_without_risk_fields(env):
    """旧 state 兼容: 缺 risk_* 字段 (pre-V32 state.json) 不抛错.

    实盘 load_state() 会 setdefault 补齐字段, 但旧存档/回测适配可能直接传裸
    dict; assess 必须对缺失字段按默认值处理 (risk_exposure=1.0 等),
    不能因 KeyError 中断, 否则升级期实盘会崩.
    """
    data, common_dates, idx_map = env
    # 无 entry_price / peak_equity / risk_exposure / cooldown_until
    state = {"cash": 100_000.0, "shares": 0}

    r = assess(target=None, holding=None, state=state, data=data, td=TD,
               idx_map=idx_map, is_rebalance=False, common_dates=common_dates)

    assert r.exposure == 1.0            # 默认满仓暴露
    assert r.action == ACTION_HOLD
    assert r.final_target is None
    assert r.cooldown_until is None
    assert r.events == []               # 无风险事件


def test_td_outside_common_dates_raises_value_error(env):
    """td 不在 common_dates → ValueError (调用约定: td 必须是公共日历成员).

    该行为是显式约定而非缺陷: 调用方 (live_signal/回测) 保证 td 来自
    get_trading_dates/公共日历交集。锁定此边界, 防止静默错误导致
    td_pos 越界后产生无未来函数违规.
    """
    data, common_dates, idx_map = env
    state = {"cash": 100_000.0, "shares": 0}

    with pytest.raises(ValueError):
        assess(target=None, holding=None, state=state, data=data,
               td=date(2099, 1, 1), idx_map=idx_map,
               is_rebalance=False, common_dates=common_dates)


# --------------------------------------------------------------------------- #
# 5. 熔断
# --------------------------------------------------------------------------- #
def test_emergency_defense_on_flush(env):
    """组合熔断: peak_equity 远高于 equity (dd<-30%) → emergency_defense.

    断言: action=emergency_defense (非调仓日也强制切防御), cooldown_until=td
    (熔断后冷却到下一个调仓日), final_target 按 DEFENSE_SEQ 优先级选防御,
    exposure 重置 1.0 (熔断清仓后按新目标全额买入).
    """
    data, common_dates, idx_map = env
    state = {
        "cash": 50_000.0,
        "shares": 0,
        "entry_price": 0.0,
        "peak_equity": 100_000.0,  # dd = -50% < DD_FLUSH (-0.30)
        "risk_exposure": 1.0,
        "cooldown_until": None,
    }

    r = assess(target="518880", holding=None, state=state, data=data, td=TD,
               idx_map=idx_map, is_rebalance=False, common_dates=common_dates)

    assert r.action == ACTION_EMERGENCY
    assert r.cooldown_until == TD
    assert r.final_target in DEFENSE_SEQ  # 511260(缺数据跳过)→511220/511880
    assert r.exposure == 1.0
    assert any(e["type"] == "熔断-30%清仓" for e in r.events)


# --------------------------------------------------------------------------- #
# 6. 回测怪癖锁定: 熔断日=调仓日 → 冷却立即解除
# --------------------------------------------------------------------------- #
def test_cooldown_released_when_flush_day_is_rebalance_day(env):
    """回测怪癖锁定: 熔断日=调仓日 → 冷却立即解除 (禁止"修正").

    回测语义: 冷却在下一个调仓日解除; 若熔断日本身就是调仓日, 回测当日
    即解除冷却并继续正常调仓 (cooldown_until=None)。实盘必须与回测保持
    同一节奏, 若单端"修正"为跨日解除, 调仓网格将与回测逐渐错位.
    """
    data, common_dates, idx_map = env
    state = {
        "cash": 50_000.0,
        "shares": 0,
        "entry_price": 0.0,
        "peak_equity": 100_000.0,
        "risk_exposure": 1.0,
        "cooldown_until": TD,  # 熔断日 = 调仓日
    }

    r = assess(target=None, holding=None, state=state, data=data, td=TD,
               idx_map=idx_map, is_rebalance=True, common_dates=common_dates)

    assert r.cooldown_until is None  # 冷却立即解除 (回测怪癖, 必须保留)
    assert r.action == ACTION_TRADE   # 走正常调仓路径
