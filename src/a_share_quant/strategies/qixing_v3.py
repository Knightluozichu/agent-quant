"""七星V3 策略桥接模块.

P3-E1 架构统一: 本模块是 scripts/run_qixing_v3.py (规范实现) 与
a_share_quant 包结构之间的过渡桥梁, 允许渐进迁移而不破坏现有代码.

职责:
    1. 通过 load_params() 从 YAML 加载并校验策略参数 (单一事实来源)
    2. 按文件路径动态加载 scripts/run_qixing_v3.py, 复用其规范函数实现
    3. 以与原脚本一致的常量名 (大写) 暴露参数, 便于现有代码平滑切换

迁移路径:
    - 现状: 函数实现仍在脚本中, 使用脚本自身的模块级常量
    - 本模块: 暴露 YAML 校验后的 params + 脚本函数的引用
    - 未来: 将函数改造为接受 StrategyParams 参数, 彻底去除硬编码

注意:
    重新导出的函数 (select_target / calc_momentum_score 等) 当前仍读取
    scripts/run_qixing_v3.py 的模块级常量. 本模块导出的常量 (FEE/SLIPPAGE
    等) 来自 YAML 校验结果. 二者应保持一致; 若不一致以 YAML 为准,
    后续任务将使函数完全参数化.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from a_share_quant.strategies.params import StrategyParams, load_params

# ---------------------------------------------------------------------------
# 1. 加载并校验策略参数 (来自 YAML)
# ---------------------------------------------------------------------------
params: StrategyParams = load_params()

# ---------------------------------------------------------------------------
# 2. 动态加载 scripts/run_qixing_v3.py (规范实现)
# ---------------------------------------------------------------------------
# src/a_share_quant/strategies/qixing_v3.py -> parents[3] = 项目根目录
_SCRIPT_PATH: Path = Path(__file__).resolve().parents[3] / "scripts" / "run_qixing_v3.py"


def _load_script_module() -> Any:
    """按文件路径加载 scripts/run_qixing_v3.py 为模块.

    使用 importlib 而非普通 import, 因为 scripts/ 不是正式 Python 包,
    这样可在不修改 sys.path、不破坏现有代码的前提下复用其实现.
    """
    if not _SCRIPT_PATH.exists():
        msg = f"Cannot find canonical script: {_SCRIPT_PATH}"
        raise FileNotFoundError(msg)

    spec = importlib.util.spec_from_file_location("run_qixing_v3", _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        msg = f"Cannot load module spec from {_SCRIPT_PATH}"
        raise ImportError(msg)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_script: Any = _load_script_module()

# ---------------------------------------------------------------------------
# 3. 重新导出脚本的关键函数 (复用规范实现)
# ---------------------------------------------------------------------------
calc_momentum_score = _script.calc_momentum_score
check_short_momentum = _script.check_short_momentum
check_volume_spike = _script.check_volume_spike
check_single_day_drop = _script.check_single_day_drop
check_a_share_weak = _script.check_a_share_weak
select_target = _script.select_target
run_qixing_v3 = _script.run_qixing_v3
run_qixing_v3_no_lookahead = _script.run_qixing_v3_no_lookahead
run_cost_stress_test = _script.run_cost_stress_test
load_data = _script.load_data

# ---------------------------------------------------------------------------
# 4. 以原脚本常量名暴露 YAML 参数 (兼容现有代码)
# ---------------------------------------------------------------------------
ETF_POOL: dict[str, str] = params.universe.etf_pool
DEFENSE: str = params.universe.defense
A_SHARE_ETF: str = params.universe.a_share_etf
CATEGORIES: dict[str, list[str]] = params.universe.categories

FEE: float = params.trading.fee
SLIPPAGE: float = params.trading.slippage
REBALANCE_DAYS: int = params.trading.rebalance_days
SWITCH_THRESHOLD: float = params.trading.switch_threshold

MOM_WEIGHTS: tuple[float, ...] = params.momentum.mom_weights
MOM_PERIODS: tuple[int, ...] = params.momentum.mom_periods
SHORT_MOM_DAYS: int = params.momentum.short_mom_days
LONG_MOM_PERIOD: int = params.momentum.long_mom_period

VOL_SPIKE_RATIO: float = params.filters.vol_spike_ratio
DROP_THRESHOLD: float = params.filters.drop_threshold
DROP_LOOKBACK: int = params.filters.drop_lookback
PROFIT_PROTECTION_DD: float = params.filters.profit_protection_dd
A_SHARE_MA: int = params.filters.a_share_ma

USE_SHORT_MOM_FILTER: bool = params.flags.use_short_mom_filter
USE_VOL_SPIKE_FILTER: bool = params.flags.use_vol_spike_filter
USE_DROP_FILTER: bool = params.flags.use_drop_filter
USE_LONG_MOM_FILTER: bool = params.flags.use_long_mom_filter
USE_PROFIT_PROTECTION: bool = params.flags.use_profit_protection
USE_A_SHARE_FILTER: bool = params.flags.use_a_share_filter
USE_BEARISH_DAY_FILTER: bool = params.flags.use_bearish_day_filter
USE_CATEGORY_SWITCH: bool = params.flags.use_category_switch


__all__ = [
    "A_SHARE_ETF",
    "A_SHARE_MA",
    "CATEGORIES",
    "DEFENSE",
    "DROP_LOOKBACK",
    "DROP_THRESHOLD",
    "ETF_POOL",
    "FEE",
    "LONG_MOM_PERIOD",
    "MOM_PERIODS",
    "MOM_WEIGHTS",
    "PROFIT_PROTECTION_DD",
    "REBALANCE_DAYS",
    "SHORT_MOM_DAYS",
    "SLIPPAGE",
    "SWITCH_THRESHOLD",
    "USE_A_SHARE_FILTER",
    "USE_BEARISH_DAY_FILTER",
    "USE_CATEGORY_SWITCH",
    "USE_DROP_FILTER",
    "USE_LONG_MOM_FILTER",
    "USE_PROFIT_PROTECTION",
    "USE_SHORT_MOM_FILTER",
    "USE_VOL_SPIKE_FILTER",
    "VOL_SPIKE_RATIO",
    "StrategyParams",
    "calc_momentum_score",
    "check_a_share_weak",
    "check_short_momentum",
    "check_single_day_drop",
    "check_volume_spike",
    "load_data",
    "load_params",
    "params",
    "run_cost_stress_test",
    "run_qixing_v3",
    "run_qixing_v3_no_lookahead",
    "select_target",
]
