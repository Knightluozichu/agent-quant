"""策略定义: Strategy dataclass + V3封装(内置基准) + 稻草人(过拟合反例) + 注册表.

新增策略只需: 实现 select 函数 + 填 hypothesis(经济逻辑) + 定 param_grid, 然后注册到 REGISTRY.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import run_qixing_v3 as rq


@dataclass
class Strategy:
    name: str
    hypothesis: str  # 经济逻辑, 必填(空则否决)
    params: dict  # 默认参数
    param_grid: dict  # {参数: [候选值]} 稳健性扫描范围
    select: Callable  # select(data, idx_map, holding, params) -> 目标代码
    rebalance_days: int = 5


# --------------------------------------------------------------------------- #
# V3 (内置基准策略, 封装 run_qixing_v3.select_target)
# --------------------------------------------------------------------------- #
def v3_select(data: dict, idx_map: dict, holding, params: dict):
    # 通过设置模块全局参数实现参数化 (单线程实验室, 安全)
    rq.MOM_PERIODS = tuple(params.get("mom_periods", (10, 20)))
    rq.MOM_WEIGHTS = tuple(params.get("mom_weights", (0.5, 0.5)))
    target, _c, _b, _a = rq.select_target(data, idx_map, holding)
    return target


V3 = Strategy(
    name="v3",
    hypothesis=(
        "跨资产动量轮动: 在商品/海外/A股/债券中持有动量最强者, A股走弱时回避创业板, "
        "全部转弱切货币基金防御。经济逻辑: 趋势跟踪 + 跨资产风险分散。"
    ),
    params={"mom_periods": (10, 20), "mom_weights": (0.5, 0.5), "rebalance_days": 5},
    param_grid={
        "mom_periods": [(10, 20), (20, 60), (10, 30), (20, 40), (5, 20)],
        "mom_weights": [(0.5, 0.5), (0.4, 0.6), (0.6, 0.4)],
        "rebalance_days": [3, 5, 7, 10],
    },
    select=v3_select,
    rebalance_days=5,
)


# --------------------------------------------------------------------------- #
# 稻草人: 反动量(故意错误逻辑), 用于验证实验室能否识别无效/过拟合策略
# --------------------------------------------------------------------------- #
def strawman_select(data: dict, idx_map: dict, holding, params: dict):
    """选动量最差的ETF (反动量, 故意错误)."""
    rq.MOM_PERIODS = tuple(params.get("mom_periods", (10, 20)))
    rq.MOM_WEIGHTS = tuple(params.get("mom_weights", (0.5, 0.5)))
    scores = []
    for code in rq.ETF_POOL:
        if code not in idx_map:
            continue
        df = data[code]
        close = df["close"].values[: idx_map[code] + 1].astype(float)
        if len(close) < 121:
            continue
        scores.append((code, rq.calc_momentum_score(close)))
    if not scores:
        return rq.DEFENSE
    scores.sort(key=lambda x: x[1])  # 升序: 最弱在前
    return scores[0][0]  # 选最弱 (错误逻辑)


STRAWMAN = Strategy(
    name="strawman",
    hypothesis="(故意错误)买动量最弱者博反弹 —— 用于验证实验室能否识别无效策略。",
    params={"mom_periods": (10, 20), "mom_weights": (0.5, 0.5), "rebalance_days": 5},
    param_grid={
        "mom_periods": [(10, 20), (5, 10), (20, 60)],
        "rebalance_days": [5, 10],
    },
    select=strawman_select,
    rebalance_days=5,
)


REGISTRY: dict[str, Strategy] = {
    "v3": V3,
    "strawman": STRAWMAN,
}
