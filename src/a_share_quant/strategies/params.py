"""七星V3 策略参数模型与加载.

P3-E1 架构统一: 将 scripts/run_qixing_v3.py 的硬编码常量提取为受 pydantic
校验的类型安全参数模型。本模块是策略参数的单一事实来源入口。

提供:
    - StrategyParams: 校验全部策略参数的 pydantic 模型 (含嵌套子模型)
    - load_params(): 从 YAML 读取并返回已校验的参数实例

所有正浮点、权重求和、周期合法性等约束均由模型校验器强制执行,
无约束 dict 不进入核心接口 (符合 AGENTS.md 工程约束)。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# 默认配置路径: <project_root>/config/strategy_params.yaml
# src/a_share_quant/strategies/params.py -> parents[3] = 项目根目录
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH: Path = Path(__file__).resolve().parents[3] / "config" / "strategy_params.yaml"

# 动量权重求和容差 (浮点比较)
_WEIGHT_SUM_TOL: float = 1e-6


# =============================================================================
# 嵌套子模型
# =============================================================================
class UniverseParams(BaseModel):
    """标的池配置."""

    model_config = ConfigDict(extra="forbid")

    etf_pool: dict[str, str] = Field(..., description="主交易ETF池 code -> 名称")
    defense: str = Field(..., min_length=6, max_length=6, description="防御标的代码")
    a_share_etf: str = Field(..., min_length=6, max_length=6, description="A股代表标的代码")
    categories: dict[str, list[str]] = Field(..., description="多类别ETF池 类别名 -> 代码列表")

    @field_validator("etf_pool")
    @classmethod
    def _etf_pool_non_empty(cls, v: dict[str, str]) -> dict[str, str]:
        if not v:
            raise ValueError("etf_pool must not be empty")
        return v

    @model_validator(mode="after")
    def _check_references(self) -> UniverseParams:
        # 防御标的与A股代表标的不必在主池中, 但若在则代码需一致
        for code in [self.defense, self.a_share_etf]:
            if not code.isdigit():
                raise ValueError(f"ETF code must be numeric, got {code!r}")
        # categories 中所有代码应在 etf_pool 中 (类别切换的候选来自主池)
        pool_codes = set(self.etf_pool.keys())
        for cat_name, codes in self.categories.items():
            for code in codes:
                if code not in pool_codes:
                    raise ValueError(
                        f"category '{cat_name}' contains code {code!r} not in etf_pool"
                    )
        return self


class TradingParams(BaseModel):
    """交易成本与调仓参数."""

    model_config = ConfigDict(extra="forbid")

    fee: float = Field(..., gt=0.0, description="单边手续费率 (必须为正)")
    slippage: float = Field(..., gt=0.0, description="单边滑点率 (必须为正)")
    rebalance_days: int = Field(..., ge=1, description="调仓频率 (交易日, >=1)")
    switch_threshold: float = Field(..., ge=0.0, description="换仓阈值参考 (>=0)")


class MomentumParams(BaseModel):
    """动量评分参数."""

    model_config = ConfigDict(extra="forbid")

    mom_weights: tuple[float, ...] = Field(..., description="动量权重 (须求和≈1.0)")
    mom_periods: tuple[int, ...] = Field(..., description="动量周期 (须与权重等长)")
    short_mom_days: int = Field(..., ge=1, description="短期动量过滤窗口 (>=1)")
    long_mom_period: int = Field(..., ge=1, description="长周期动量过滤窗口 (>=1)")

    @model_validator(mode="after")
    def _validate_weights_and_periods(self) -> MomentumParams:
        if len(self.mom_weights) == 0:
            raise ValueError("mom_weights must not be empty")
        if len(self.mom_periods) != len(self.mom_weights):
            raise ValueError(
                f"mom_periods length ({len(self.mom_periods)}) must equal "
                f"mom_weights length ({len(self.mom_weights)})"
            )
        if any(w < 0 for w in self.mom_weights):
            raise ValueError("mom_weights must be non-negative")
        if any(p < 1 for p in self.mom_periods):
            raise ValueError("mom_periods must be >= 1")
        total = sum(self.mom_weights)
        if abs(total - 1.0) > _WEIGHT_SUM_TOL:
            raise ValueError(f"mom_weights must sum to 1.0 (got {total})")
        return self


class FilterParams(BaseModel):
    """过滤器参数."""

    model_config = ConfigDict(extra="forbid")

    vol_spike_ratio: float = Field(..., gt=0.0, description="放量阈值 (必须为正)")
    drop_threshold: float = Field(..., lt=0.0, description="单日跌幅阈值 (必须为负)")
    drop_lookback: int = Field(..., ge=1, description="跌幅检查回看天数 (>=1)")
    profit_protection_dd: float = Field(..., gt=0.0, description="盈利保护回撤阈值 (必须为正)")
    a_share_ma: int = Field(..., ge=1, description="A股走弱判断MA周期 (>=1)")


class FlagParams(BaseModel):
    """功能开关."""

    model_config = ConfigDict(extra="forbid")

    use_short_mom_filter: bool
    use_vol_spike_filter: bool
    use_drop_filter: bool
    use_long_mom_filter: bool
    use_profit_protection: bool
    use_a_share_filter: bool
    use_bearish_day_filter: bool
    use_category_switch: bool


class ChangelogEntry(BaseModel):
    """变更记录条目."""

    model_config = ConfigDict(extra="forbid")

    version: str
    date: str
    note: str


# =============================================================================
# 顶层参数模型
# =============================================================================
class StrategyParams(BaseModel):
    """七星V3 策略参数 (顶层模型).

    校验全部策略参数并提供类型安全访问。所有正浮点、权重求和、周期合法性
    等约束由嵌套子模型的校验器强制执行。

    用法::

        from a_share_quant.strategies.params import load_params
        params = load_params()
        print(params.trading.fee)            # 0.0005
        print(params.flags.use_drop_filter)  # True
        print(params.momentum.mom_weights)   # (0.5, 0.5)
    """

    model_config = ConfigDict(extra="forbid")

    version: str = Field(..., min_length=1, description="参数版本号")
    name: str = Field(..., min_length=1, description="策略名称")
    description: str = Field("", description="策略描述")
    universe: UniverseParams
    trading: TradingParams
    momentum: MomentumParams
    filters: FilterParams
    flags: FlagParams
    changelog: list[ChangelogEntry] = Field(default_factory=list)


# =============================================================================
# 加载函数
# =============================================================================
def load_params(path: Path | str | None = None) -> StrategyParams:
    """从 YAML 文件加载并校验策略参数.

    Args:
        path: YAML 配置路径。默认为 ``<project_root>/config/strategy_params.yaml``。

    Returns:
        已校验的 StrategyParams 实例。

    Raises:
        FileNotFoundError: 配置文件不存在。
        pydantic.ValidationError: 参数非法 (负费率、权重不和为1等)。
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        msg = f"Strategy params config not found: {config_path}"
        raise FileNotFoundError(msg)

    with config_path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    return StrategyParams.model_validate(raw)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "ChangelogEntry",
    "FilterParams",
    "FlagParams",
    "MomentumParams",
    "StrategyParams",
    "TradingParams",
    "UniverseParams",
    "load_params",
]
