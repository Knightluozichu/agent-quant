"""V3 改进版尾部风控纯函数层 (回测/实盘共用).

从 exp_v32_tail_risk.py 逐行提炼改进版逻辑 (已通过全部验证:
IS/OOS + 滚动4段3-4 + 参数扰动±20% + 成本2x/3x; 全周期收益+50%/回撤减半).

改进版机制:
  1. 高波动(vol20>0.45) + 动量衰减三重确认(Δs<-0.02 & close<MA10 & s<0.08)
     → 适度降仓 0.7 (不切防御, 保留反弹)
  2. H1/H2 (自买入回撤<-15% / 当日<-5%) → 降仓 0.7 (不切防御)
  3. 组合回撤 -12%~-25% → 降仓 0.8; -30% → 熔断清仓切防御 + 冷却至下个调仓日
  4. 防御优先级: 十年国债(511260)→城投债(511220)→货币(511880) (数据缺失自动跳过)
  5. 目标不可交易 (停牌/涨跌停/无价) → 切防御

一致性铁律:
  - 本模块是唯一风控实现, 回测(run_v3_risk)与实盘(live_signal)共用同一数学逻辑
  - 回测怪癖 (vol20除数错位/熔断日=调仓日冷却立即解除) 原样保留, 禁止单端"修正"
  - 参数与 exp_v32_tail_risk.py 逐字一致, 禁止单端修改
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np

# === 风控参数 (与 exp_v32_tail_risk.py 一致) ===
VOL_HV_THR = 0.45        # 高波动阈值
EXPO_REDUCE = 0.7        # 适度降仓比例
DECAY_THRESH = -0.02     # 动量5日变化阈值 (三重确认1)
ABS_WEAK = 0.08          # 绝对动量弱化阈值 (三重确认3)
H1_DD = -0.15            # 自买入回撤硬触发
H2_DAY = -0.05           # 当日跌幅硬触发
DD_WARN, DD_HALF, DD_FLUSH = -0.12, -0.25, -0.30  # 组合熔断分级
DEFENSE_SEQ = ("511260", "511220", "511880")       # 十年国债→城投债→货币 (缺数据跳过)

# 动作枚举
ACTION_HOLD = "hold"                # 无交易 (仅状态更新)
ACTION_TRADE = "trade"              # 调仓日正常交易 (exposure 参与买入折算)
ACTION_EMERGENCY = "emergency_defense"  # 组合熔断, 非调仓日也强制切防御


@dataclass(frozen=True)
class RiskDecision:
    """风控评估结果 (纯数据, 可 JSON 序列化)."""
    final_target: str | None
    exposure: float
    action: str
    reason: str
    events: list[dict[str, Any]] = field(default_factory=list)
    cooldown_until: date | None = None


def assess(
    *,
    target: str | None,
    holding: str | None,
    state: dict[str, Any],
    data: dict[str, Any],
    td: date,
    idx_map: dict[str, int],
    is_rebalance: bool,
    common_dates: list[Any],
    spot_map: dict[str, dict[str, Any]] | None = None,
) -> RiskDecision:
    """单日风控评估 (14:50 口径, 无未来函数).

    Args:
        target: select_target 原始目标 (含实时急跌过滤后)
        holding: 当前持仓代码 (None=空仓)
        state: 实盘 state 或回测适配 dict (需含 cash/shares/entry_price/
               peak_equity/risk_exposure/cooldown_until)
        data: {code: DataFrame} 历史数据 (含 trade_date/close 列)
        td: 信号日 (公共日历成员)
        idx_map: {code: 当日数据索引} (live_signal.build_etf_data_at_date)
        is_rebalance: 是否调仓日 (绝对网格)
        common_dates: 公共交易日历 (list[date])
        spot_map: 腾讯实时 {code: {price, prev_close}} (实盘传, 回测传 None)
    """
    spot_map = spot_map or {}
    td_pos = common_dates.index(td)
    exposure = float(state.get("risk_exposure", 1.0))
    cooldown_raw = state.get("cooldown_until")
    cooldown: date | None = None
    if isinstance(cooldown_raw, str) and cooldown_raw:
        cooldown = date.fromisoformat(cooldown_raw)
    elif isinstance(cooldown_raw, date):
        cooldown = cooldown_raw
    entry_price = float(state.get("entry_price", 0.0) or 0.0)
    peak_equity = float(state.get("peak_equity", 0.0))
    events: list[dict[str, Any]] = []

    # ---- 内部工具 (与回测逐行等价) ----
    def _series(code: str) -> np.ndarray:
        df = data[code]
        return np.asarray(df["close"].values[: idx_map[code] + 1], dtype=float)

    def _price(code: str) -> float:
        df = data[code]
        return float(df.iloc[idx_map[code]]["close"])

    def _close_at(code: str, at_pos: int) -> np.ndarray:
        d = common_dates[at_pos]
        sub = data[code][data[code]["trade_date"] <= d]
        return np.asarray(sub["close"].values, dtype=float)

    def _mom(code: str, period: int, at_pos: int | None = None) -> float:
        close = _series(code) if at_pos is None else _close_at(code, at_pos)
        if len(close) <= period or close[-period - 1] <= 0:
            return 0.0
        return float(close[-1] / close[-period - 1] - 1.0)

    def _mom_score(code: str, at_pos: int | None = None) -> float:
        return 0.5 * _mom(code, 10, at_pos) + 0.5 * _mom(code, 20, at_pos)

    def _vol20(code: str) -> float:
        """20日年化波动 (保留回测除数错位怪癖, 禁止"修正")."""
        close = _series(code)
        if len(close) < 21:
            return 0.35
        dr = np.diff(close[-21:]) / close[-21:-1]
        return float(np.std(dr) * np.sqrt(252))

    def _pick_defense() -> str:
        """防御优先级: 十年国债→城投债→货币 (债券需10日动量>0, 缺数据自动跳过)."""
        for code in DEFENSE_SEQ:
            if code not in data or code not in idx_map:
                continue
            if code == "511880":
                return code
            if _mom(code, 10) > 0:
                return code
        return "511880"

    def _tradable(code: str) -> bool:
        row = data[code]
        price = float(row.iloc[idx_map[code]]["close"])
        if price <= 0:
            return False
        if idx_map[code] > 0:
            prev = float(row.iloc[idx_map[code] - 1]["close"])
            if prev > 0 and abs(price / prev - 1) >= 0.099:
                return False
        return True

    # ---- 组合净值/回撤 (14:50 口径) ----
    equity = float(state.get("cash", 0.0))
    if holding and holding in data:
        px = float(spot_map[holding]["price"]) if holding in spot_map else _price(holding)
        equity += float(state.get("shares", 0.0)) * px
    if equity > peak_equity:
        peak_equity = equity
    dd = equity / peak_equity - 1.0 if peak_equity > 0 else 0.0

    # === 层3 日频硬触发 H1/H2 → 降仓 0.7 (不切防御) ===
    if holding and holding != "511880" and cooldown is None:
        cur = float(spot_map[holding]["price"]) if holding in spot_map else _price(holding)
        entry_dd = (cur / entry_price - 1.0) if entry_price > 0 else 0.0
        if holding in spot_map:
            prev = float(spot_map[holding]["prev_close"])
        else:
            prev = (_price_at_pos(holding, td_pos - 1, data, common_dates)
                    if td_pos > 0 else cur)
        day_ret = (cur / prev - 1.0) if prev > 0 else 0.0
        if entry_dd < H1_DD or day_ret < H2_DAY:
            exposure = min(exposure, EXPO_REDUCE)
            events.append({"date": str(td), "type": "改进-H1/H2降仓",
                           "reason": f"entry_dd={entry_dd:.1%} day={day_ret:.1%}"})

    # === 层4 组合熔断 ===
    if cooldown is None:
        if dd < DD_FLUSH:
            target_d = _pick_defense()
            events.append({"date": str(td), "type": "熔断-30%清仓",
                           "reason": f"dd={dd:.1%}, 切防御 {target_d}"})
            return RiskDecision(
                final_target=target_d, exposure=1.0, action=ACTION_EMERGENCY,
                reason=f"组合回撤{dd:.1%}触发熔断, 切防御{target_d}",
                events=events, cooldown_until=td)
        if dd < DD_HALF:
            events.append({"date": str(td), "type": "熔断-25%告警",
                           "reason": f"dd={dd:.1%}"})
        elif dd < DD_WARN:
            exposure = min(exposure, 0.8)
            events.append({"date": str(td), "type": "熔断-12%降仓",
                           "reason": f"dd={dd:.1%}, exposure=0.8"})

    # === 冷却解除: 熔断后下一调仓日解除 (回测怪癖: 熔断日=调仓日则立即解除) ===
    if cooldown is not None and is_rebalance:
        cooldown = None

    # === 调仓日: V3 决策叠加风控 ===
    if is_rebalance and cooldown is None and target and target in data and target in idx_map:
        close = _series(target)
        ma10 = float(np.mean(close[-10:])) if len(close) >= 10 else 0.0
        mom5_prev = _mom_score(target, td_pos - 5) if td_pos >= 5 else 0.0
        delta_s = _mom_score(target) - mom5_prev
        vol_t = _vol20(target)
        s_score = _mom_score(target)
        decay_triple = (delta_s < DECAY_THRESH and close[-1] < ma10
                        and s_score < ABS_WEAK)
        # 改进核心: 仅"高波动 + 动量衰减三重确认"触发适度降仓
        if vol_t > VOL_HV_THR and decay_triple:
            exposure = min(exposure, EXPO_REDUCE)
            events.append({"date": str(td), "type": "改进-高波动衰减降仓",
                           "reason": f"vol={vol_t:.2f} delta_s={delta_s:.4f} s={s_score:.3f}"})
        # 目标不可交易 → 切防御
        if not _tradable(target):
            events.append({"date": str(td), "type": "目标不可交易",
                           "reason": f"{target} 停牌/涨跌停/无价, 切防御"})
            target = _pick_defense()

    return RiskDecision(
        final_target=target, exposure=exposure,
        action=(ACTION_TRADE if is_rebalance else ACTION_HOLD),
        reason="", events=events, cooldown_until=cooldown)


def _price_at_pos(code: str, at_pos: int, data: dict[str, Any],
                  common_dates: list[Any]) -> float:
    """任意公共日历位置的价格 (用于 H2 昨收)."""
    if at_pos < 0:
        return 0.0
    d = common_dates[at_pos]
    row = data[code][data[code]["trade_date"] == d]
    return float(row.iloc[0]["close"]) if not row.empty else 0.0
