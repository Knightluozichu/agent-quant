"""Tests for 七星V3 strategy parameter model and YAML loading.

P3-E1 架构统一: 验证 StrategyParams 模型与 load_params() 加载逻辑。
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from a_share_quant.strategies.params import (
    DEFAULT_CONFIG_PATH,
    StrategyParams,
    load_params,
)

# 项目根目录下的 YAML 配置
PROJECT_ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = PROJECT_ROOT / "config" / "strategy_params.yaml"


# =============================================================================
# Fixtures / helpers
# =============================================================================
def _valid_params_dict() -> dict:
    """返回一份与 strategy_params.yaml 等价的合法参数 dict (深拷贝用于变异测试)."""
    with YAML_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def valid_params() -> dict:
    """合法参数 dict (每用例独立深拷贝)."""
    return _valid_params_dict()


# =============================================================================
# 加载测试
# =============================================================================
@pytest.mark.unit
class TestLoadParams:
    """验证从 YAML 加载策略参数."""

    def test_default_path_points_to_config(self) -> None:
        """DEFAULT_CONFIG_PATH 指向 config/strategy_params.yaml."""
        assert DEFAULT_CONFIG_PATH == YAML_PATH
        assert DEFAULT_CONFIG_PATH.exists()

    def test_load_params_succeeds(self) -> None:
        """load_params() 成功返回 StrategyParams 实例."""
        params = load_params()
        assert isinstance(params, StrategyParams)

    def test_load_params_explicit_path(self) -> None:
        """显式传入路径也能加载."""
        params = load_params(YAML_PATH)
        assert isinstance(params, StrategyParams)
        assert params.version == "3.0.0"

    def test_load_params_missing_file_raises(self, tmp_path: Path) -> None:
        """配置文件不存在时抛出 FileNotFoundError."""
        missing = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError):
            load_params(missing)


# =============================================================================
# 字段完整性测试
# =============================================================================
@pytest.mark.unit
class TestRequiredFields:
    """验证所有必需字段存在且类型正确."""

    def test_all_top_level_fields_present(self, valid_params: dict) -> None:
        """顶层字段齐全."""
        params = StrategyParams.model_validate(valid_params)
        for field in ["version", "name", "description", "universe", "trading",
                      "momentum", "filters", "flags", "changelog"]:
            assert hasattr(params, field), f"missing top-level field: {field}"

    def test_universe_fields(self, valid_params: dict) -> None:
        """标的池字段与脚本常量一致."""
        params = StrategyParams.model_validate(valid_params)
        assert params.universe.defense == "511880"
        assert params.universe.a_share_etf == "159915"
        assert "518880" in params.universe.etf_pool
        assert "159915" in params.universe.etf_pool
        assert set(params.universe.categories.keys()) == {"商品", "海外", "A股", "债券"}

    def test_trading_fields(self, valid_params: dict) -> None:
        """交易成本字段与脚本常量一致."""
        params = StrategyParams.model_validate(valid_params)
        assert params.trading.fee == pytest.approx(0.0005)
        assert params.trading.slippage == pytest.approx(0.001)
        assert params.trading.rebalance_days == 5

    def test_momentum_fields(self, valid_params: dict) -> None:
        """动量字段与脚本常量一致."""
        params = StrategyParams.model_validate(valid_params)
        assert params.momentum.mom_weights == (0.5, 0.5)
        assert params.momentum.mom_periods == (10, 20)
        assert params.momentum.short_mom_days == 10

    def test_filter_fields(self, valid_params: dict) -> None:
        """过滤器字段与脚本常量一致."""
        params = StrategyParams.model_validate(valid_params)
        assert params.filters.drop_threshold == pytest.approx(-0.03)
        assert params.filters.drop_lookback == 5
        assert params.filters.vol_spike_ratio == pytest.approx(2.5)
        assert params.filters.a_share_ma == 15
        assert params.filters.profit_protection_dd == pytest.approx(0.05)

    def test_all_flags_present(self, valid_params: dict) -> None:
        """全部 USE_* 开关存在且为 bool."""
        params = StrategyParams.model_validate(valid_params)
        expected_flags = {
            "use_short_mom_filter": False,
            "use_vol_spike_filter": False,
            "use_drop_filter": True,
            "use_long_mom_filter": False,
            "use_profit_protection": False,
            "use_a_share_filter": True,
            "use_bearish_day_filter": False,
            "use_category_switch": False,
        }
        for name, expected in expected_flags.items():
            assert getattr(params.flags, name) is expected, f"flag {name} mismatch"

    def test_changelog_present(self, valid_params: dict) -> None:
        """changelog 非空且首条版本为 3.0.0."""
        params = StrategyParams.model_validate(valid_params)
        assert len(params.changelog) >= 1
        assert params.changelog[0].version == "3.0.0"


# =============================================================================
# 校验器测试 — 非法参数被拒绝
# =============================================================================
@pytest.mark.unit
class TestValidators:
    """验证非法参数被拒绝."""

    def test_negative_fee_rejected(self, valid_params: dict) -> None:
        """负费率被拒绝."""
        bad = copy.deepcopy(valid_params)
        bad["trading"]["fee"] = -0.001
        with pytest.raises(ValidationError):
            StrategyParams.model_validate(bad)

    def test_zero_fee_rejected(self, valid_params: dict) -> None:
        """零费率被拒绝 (必须为正)."""
        bad = copy.deepcopy(valid_params)
        bad["trading"]["fee"] = 0.0
        with pytest.raises(ValidationError):
            StrategyParams.model_validate(bad)

    def test_negative_slippage_rejected(self, valid_params: dict) -> None:
        """负滑点被拒绝."""
        bad = copy.deepcopy(valid_params)
        bad["trading"]["slippage"] = -0.002
        with pytest.raises(ValidationError):
            StrategyParams.model_validate(bad)

    def test_weights_not_summing_to_one_rejected(self, valid_params: dict) -> None:
        """动量权重求和不为1被拒绝."""
        bad = copy.deepcopy(valid_params)
        bad["momentum"]["mom_weights"] = [0.5, 0.4]  # sum = 0.9
        with pytest.raises(ValidationError):
            StrategyParams.model_validate(bad)

    def test_weights_and_periods_length_mismatch_rejected(
        self, valid_params: dict
    ) -> None:
        """动量权重与周期长度不一致被拒绝."""
        bad = copy.deepcopy(valid_params)
        bad["momentum"]["mom_weights"] = [0.5, 0.3, 0.2]  # 3 个权重
        # mom_periods 仍为 [10, 20] (2 个)
        with pytest.raises(ValidationError):
            StrategyParams.model_validate(bad)

    def test_non_negative_drop_threshold_rejected(
        self, valid_params: dict
    ) -> None:
        """跌幅阈值为正被拒绝 (必须为负)."""
        bad = copy.deepcopy(valid_params)
        bad["filters"]["drop_threshold"] = 0.03
        with pytest.raises(ValidationError):
            StrategyParams.model_validate(bad)

    def test_zero_rebalance_days_rejected(self, valid_params: dict) -> None:
        """调仓天数为0被拒绝."""
        bad = copy.deepcopy(valid_params)
        bad["trading"]["rebalance_days"] = 0
        with pytest.raises(ValidationError):
            StrategyParams.model_validate(bad)

    def test_negative_vol_spike_ratio_rejected(self, valid_params: dict) -> None:
        """放量阈值为负被拒绝."""
        bad = copy.deepcopy(valid_params)
        bad["filters"]["vol_spike_ratio"] = -1.0
        with pytest.raises(ValidationError):
            StrategyParams.model_validate(bad)

    def test_category_code_not_in_pool_rejected(self, valid_params: dict) -> None:
        """类别中包含不在 etf_pool 的代码被拒绝."""
        bad = copy.deepcopy(valid_params)
        bad["universe"]["categories"]["商品"].append("999999")
        with pytest.raises(ValidationError):
            StrategyParams.model_validate(bad)

    def test_extra_field_rejected(self, valid_params: dict) -> None:
        """未知字段被拒绝 (extra=forbid)."""
        bad = copy.deepcopy(valid_params)
        bad["trading"]["unexpected_field"] = 1.0
        with pytest.raises(ValidationError):
            StrategyParams.model_validate(bad)


# =============================================================================
# 桥接模块测试 (qixing_v3 常量与 YAML 一致)
# =============================================================================
@pytest.mark.unit
class TestBridgeModule:
    """验证桥接模块导出的常量与 YAML 参数一致."""

    def test_bridge_constants_match_yaml(self) -> None:
        """qixing_v3 桥接模块的大写常量来自 YAML 校验结果."""
        from a_share_quant.strategies import qixing_v3 as bridge

        assert pytest.approx(0.0005) == bridge.FEE
        assert pytest.approx(0.001) == bridge.SLIPPAGE
        assert bridge.REBALANCE_DAYS == 5
        assert bridge.MOM_WEIGHTS == (0.5, 0.5)
        assert bridge.MOM_PERIODS == (10, 20)
        assert bridge.DEFENSE == "511880"
        assert bridge.USE_DROP_FILTER is True
        assert bridge.USE_A_SHARE_FILTER is True
        assert bridge.USE_CATEGORY_SWITCH is False

    def test_bridge_reexports_functions(self) -> None:
        """桥接模块重新导出脚本的关键函数."""
        from a_share_quant.strategies import qixing_v3 as bridge

        for name in [
            "select_target",
            "calc_momentum_score",
            "check_short_momentum",
            "check_volume_spike",
            "check_single_day_drop",
            "check_a_share_weak",
            "run_qixing_v3",
            "run_qixing_v3_no_lookahead",
        ]:
            assert callable(getattr(bridge, name)), f"bridge missing callable: {name}"
