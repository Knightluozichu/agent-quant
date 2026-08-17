"""Tests for parameter versioning and Champion/Challenger framework (P3-E2)."""

from __future__ import annotations

import json
from datetime import date

import pytest

from a_share_quant.evolution.param_versioning import (
    DEFAULT_MIN_TRADES,
    ParamRegistry,
    ParamVersion,
    PromotionError,
    PromotionMetrics,
    ShadowTracker,
    ShadowTrade,
    VersionStatus,
)

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def registry_path(tmp_path):
    """临时注册表路径，测试间互不干扰。"""
    return tmp_path / "param_registry.json"


@pytest.fixture
def registry(registry_path):
    """干净的 ParamRegistry。"""
    return ParamRegistry(registry_path=registry_path)


def _valid_metrics(
    *,
    n_trades: int = 50,
    oos_sharpe: float = 1.5,
    oos_sharpe_ci_lower: float = 1.2,
    max_drawdown: float = -0.10,
    cost_stress_passed: bool = True,
    manual_approved: bool = True,
) -> PromotionMetrics:
    """构造一份满足默认门槛的晋升指标。"""
    return PromotionMetrics(
        n_trades=n_trades,
        oos_sharpe=oos_sharpe,
        oos_sharpe_ci_lower=oos_sharpe_ci_lower,
        cost_stress_passed=cost_stress_passed,
        max_drawdown=max_drawdown,
        manual_approved=manual_approved,
    )


# ----------------------------------------------------------------------
# ParamVersion
# ----------------------------------------------------------------------


class TestParamVersion:
    def test_params_hash_stable_regardless_of_key_order(self):
        v1 = ParamVersion(
            version_id="v001",
            params={"a": 1, "b": 2},
            changelog="init",
        )
        v2 = ParamVersion(
            version_id="v002",
            params={"b": 2, "a": 1},  # 相同参数，顺序不同
            changelog="init",
        )
        assert v1.params_hash == v2.params_hash

    def test_default_status_is_challenger(self):
        v = ParamVersion(version_id="v001", params={"x": 1}, changelog="init")
        assert v.status == VersionStatus.CHALLENGER
        assert v.metrics is None


# ----------------------------------------------------------------------
# ParamRegistry: register
# ----------------------------------------------------------------------


class TestRegister:
    def test_register_new_challenger(self, registry):
        v = registry.register(
            params={"DROP_LOOKBACK": 5, "REBALANCE_DAYS": 5},
            changelog="七星V3 初始参数",
        )
        # 新注册版本应为 challenger
        assert v.status == VersionStatus.CHALLENGER
        assert v.version_id == "v001"
        assert v.params == {"DROP_LOOKBACK": 5, "REBALANCE_DAYS": 5}
        # 注册表应包含该版本
        assert v in registry.list_versions()
        assert registry.get_champion() is None  # 尚无 champion
        assert registry.get_challengers() == [v]

    def test_version_id_increments(self, registry):
        v1 = registry.register(params={"a": 1}, changelog="c1")
        v2 = registry.register(params={"a": 2}, changelog="c2")
        assert v1.version_id == "v001"
        assert v2.version_id == "v002"

    def test_parent_lineage_tracked(self, registry):
        v1 = registry.register(params={"a": 1}, changelog="c1")
        v2 = registry.register(params={"a": 2}, changelog="c2", parent_version_id=v1.version_id)
        assert v2.parent_version_id == "v001"


# ----------------------------------------------------------------------
# ParamRegistry: promote
# ----------------------------------------------------------------------


class TestPromote:
    def test_promote_challenger_to_champion(self, registry):
        v1 = registry.register(params={"DROP_LOOKBACK": 5}, changelog="init")
        promoted = registry.promote(v1.version_id, _valid_metrics())

        assert promoted.status == VersionStatus.CHAMPION
        assert promoted.promoted_at is not None
        assert promoted.metrics is not None
        assert promoted.metrics["oos_sharpe"] == 1.5
        assert registry.get_champion().version_id == "v001"
        assert registry.get_challengers() == []

    def test_reject_promotion_without_enough_samples(self, registry):
        v1 = registry.register(params={"DROP_LOOKBACK": 5}, changelog="init")
        # 样本不足 (< 30)
        bad_metrics = _valid_metrics(n_trades=10)

        with pytest.raises(PromotionError) as exc_info:
            registry.promote(v1.version_id, bad_metrics)

        assert "insufficient trades" in str(exc_info.value)
        # 失败不应改变版本状态
        assert v1.status == VersionStatus.CHALLENGER
        assert registry.get_champion() is None

    def test_reject_promotion_without_manual_approval(self, registry):
        v1 = registry.register(params={"DROP_LOOKBACK": 5}, changelog="init")
        bad_metrics = _valid_metrics(manual_approved=False)

        with pytest.raises(PromotionError) as exc_info:
            registry.promote(v1.version_id, bad_metrics)

        assert "manual approval" in str(exc_info.value)
        assert v1.status == VersionStatus.CHALLENGER

    def test_reject_promotion_when_sharpe_ci_not_better(self, registry):
        # 先建立 champion (Sharpe 1.5)
        v1 = registry.register(params={"a": 1}, changelog="c1")
        registry.promote(v1.version_id, _valid_metrics(oos_sharpe=1.5, oos_sharpe_ci_lower=1.2))

        v2 = registry.register(params={"a": 2}, changelog="c2")
        # CI 下界 1.0 未超过 champion Sharpe 1.5
        bad_metrics = _valid_metrics(oos_sharpe=1.1, oos_sharpe_ci_lower=1.0, max_drawdown=-0.08)

        with pytest.raises(PromotionError) as exc_info:
            registry.promote(v2.version_id, bad_metrics)

        assert "OOS Sharpe CI lower" in str(exc_info.value)
        assert v2.status == VersionStatus.CHALLENGER
        assert registry.get_champion().version_id == "v001"

    def test_reject_promotion_when_drawdown_worse(self, registry):
        v1 = registry.register(params={"a": 1}, changelog="c1")
        registry.promote(
            v1.version_id,
            _valid_metrics(oos_sharpe=1.5, oos_sharpe_ci_lower=1.2, max_drawdown=-0.10),
        )

        v2 = registry.register(params={"a": 2}, changelog="c2")
        # Sharpe CI 满足，但回撤更差 (-0.20 < -0.10)
        bad_metrics = _valid_metrics(
            oos_sharpe=2.0,
            oos_sharpe_ci_lower=1.6,
            max_drawdown=-0.20,
        )

        with pytest.raises(PromotionError) as exc_info:
            registry.promote(v2.version_id, bad_metrics)

        assert "drawdown" in str(exc_info.value)

    def test_promote_archives_old_champion(self, registry):
        v1 = registry.register(params={"a": 1}, changelog="c1")
        registry.promote(v1.version_id, _valid_metrics(oos_sharpe=1.5, max_drawdown=-0.10))

        v2 = registry.register(params={"a": 2}, changelog="c2")
        # v2 在所有门槛上优于 v1
        registry.promote(
            v2.version_id,
            _valid_metrics(oos_sharpe=1.8, oos_sharpe_ci_lower=1.6, max_drawdown=-0.08),
        )

        assert registry.get_champion().version_id == "v002"
        assert v1.status == VersionStatus.ARCHIVED
        assert v1.archived_at is not None
        assert v2.status == VersionStatus.CHAMPION

    def test_promote_unknown_version_raises(self, registry):
        with pytest.raises(KeyError):
            registry.promote("v999", _valid_metrics())

    def test_promote_accepts_dict_metrics(self, registry):
        v1 = registry.register(params={"a": 1}, changelog="c1")
        metrics_dict = {
            "n_trades": 40,
            "oos_sharpe": 1.5,
            "oos_sharpe_ci_lower": 1.2,
            "cost_stress_passed": True,
            "max_drawdown": -0.10,
            "manual_approved": True,
        }
        registry.promote(v1.version_id, metrics_dict)
        assert registry.get_champion().version_id == "v001"


# ----------------------------------------------------------------------
# ParamRegistry: rollback
# ----------------------------------------------------------------------


class TestRollback:
    def test_rollback_to_previous_version(self, registry):
        # v1 -> champion, v2 -> champion (v1 archived), rollback to v1
        v1 = registry.register(params={"DROP_LOOKBACK": 5}, changelog="init")
        registry.promote(v1.version_id, _valid_metrics(oos_sharpe=1.5, max_drawdown=-0.10))

        v2 = registry.register(params={"DROP_LOOKBACK": 3}, changelog="tune lookback")
        registry.promote(
            v2.version_id,
            _valid_metrics(oos_sharpe=1.8, oos_sharpe_ci_lower=1.6, max_drawdown=-0.08),
        )
        assert registry.get_champion().version_id == "v002"
        assert v1.status == VersionStatus.ARCHIVED

        # 回滚到 v1
        restored = registry.rollback(v1.version_id)
        assert restored.status == VersionStatus.CHAMPION
        assert restored.version_id == "v001"
        # v2 被归档
        assert v2.status == VersionStatus.ARCHIVED
        assert v2.archived_at is not None
        assert registry.get_champion().version_id == "v001"

    def test_rollback_unknown_version_raises(self, registry):
        with pytest.raises(KeyError):
            registry.rollback("v999")

    def test_rollback_to_current_champion_raises(self, registry):
        v1 = registry.register(params={"a": 1}, changelog="c1")
        registry.promote(v1.version_id, _valid_metrics())

        with pytest.raises(PromotionError):
            registry.rollback(v1.version_id)


# ----------------------------------------------------------------------
# ParamRegistry: persistence
# ----------------------------------------------------------------------


class TestPersistence:
    def test_persistence_to_json_file(self, registry, registry_path):
        v1 = registry.register(params={"DROP_LOOKBACK": 5}, changelog="init")
        registry.promote(v1.version_id, _valid_metrics())

        # 文件确实写入磁盘
        assert registry_path.exists()
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
        assert "versions" in raw
        assert "audit_log" in raw
        assert len(raw["versions"]) == 1
        assert raw["versions"][0]["status"] == "champion"

        # 用同一路径重新加载，状态应一致
        reloaded = ParamRegistry(registry_path=registry_path)
        champ = reloaded.get_champion()
        assert champ is not None
        assert champ.version_id == "v001"
        assert champ.status == VersionStatus.CHAMPION
        assert champ.params == {"DROP_LOOKBACK": 5}
        assert champ.metrics is not None
        assert len(reloaded.list_versions()) == 1
        # 审计日志也应恢复
        assert len(reloaded.get_audit_log()) >= 2  # register + promote

    def test_empty_registry_load_when_file_missing(self, registry_path):
        reg = ParamRegistry(registry_path=registry_path)
        assert reg.list_versions() == []
        assert reg.get_champion() is None

    def test_save_creates_parent_directory(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "param_registry.json"
        reg = ParamRegistry(registry_path=nested)
        reg.register(params={"a": 1}, changelog="init")
        assert nested.exists()


# ----------------------------------------------------------------------
# ShadowTracker
# ----------------------------------------------------------------------


class TestShadowTracker:
    def test_can_promote_false_below_min_samples(self):
        tracker = ShadowTracker(min_trades=30)
        assert tracker.n_trades == 0
        assert not tracker.can_promote()

        for i in range(29):
            tracker.record_trade(
                ShadowTrade(
                    trade_id=f"t{i}",
                    symbol="510300",
                    entry_date=date(2024, 1, 1),
                    exit_date=date(2024, 1, 5),
                    return_pct=0.01,
                )
            )
        assert tracker.n_trades == 29
        assert not tracker.can_promote()

    def test_can_promote_true_at_min_samples(self):
        tracker = ShadowTracker(min_trades=30)
        for i in range(30):
            tracker.record_trade(
                ShadowTrade(
                    trade_id=f"t{i}",
                    symbol="510300",
                    entry_date=date(2024, 1, 1),
                    exit_date=date(2024, 1, 5),
                    return_pct=0.01,
                )
            )
        assert tracker.can_promote()

    def test_record_trade_accepts_dict(self):
        tracker = ShadowTracker(min_trades=2)
        tracker.record_trade(
            {
                "trade_id": "t1",
                "symbol": "510300",
                "entry_date": "2024-01-01",
                "exit_date": "2024-01-05",
                "return_pct": 0.02,
                "pnl": 100.0,
            }
        )
        assert tracker.n_trades == 1

    def test_summary_empty(self):
        tracker = ShadowTracker()
        s = tracker.summary()
        assert s["n_trades"] == 0
        assert s["win_rate"] == 0.0
        assert s["sharpe"] == 0.0
        assert s["max_drawdown"] == 0.0

    def test_summary_computes_metrics(self):
        tracker = ShadowTracker(min_trades=3)
        # 2 wins, 1 loss
        tracker.record_trade(
            ShadowTrade(
                trade_id="t1",
                symbol="510300",
                entry_date=date(2024, 1, 1),
                exit_date=date(2024, 1, 5),
                return_pct=0.05,
            )
        )
        tracker.record_trade(
            ShadowTrade(
                trade_id="t2",
                symbol="510300",
                entry_date=date(2024, 1, 6),
                exit_date=date(2024, 1, 10),
                return_pct=0.03,
            )
        )
        tracker.record_trade(
            ShadowTrade(
                trade_id="t3",
                symbol="510300",
                entry_date=date(2024, 1, 11),
                exit_date=date(2024, 1, 15),
                return_pct=-0.04,
            )
        )

        s = tracker.summary()
        assert s["n_trades"] == 3
        assert s["win_rate"] == pytest.approx(2 / 3)
        assert s["avg_return"] == pytest.approx((0.05 + 0.03 - 0.04) / 3)
        # 累计: 1.05 * 1.03 * 0.96 - 1
        assert s["total_return"] == pytest.approx(1.05 * 1.03 * 0.96 - 1.0)
        # 最大回撤非正
        assert s["max_drawdown"] <= 0.0
        assert s["sharpe"] > 0.0  # 均值为正

    def test_to_promotion_metrics_bridges_to_registry(self, registry_path):
        # tracker 与 registry 使用相同的样本门槛
        tracker = ShadowTracker(min_trades=5)
        registry = ParamRegistry(registry_path=registry_path, min_trades=5)
        for i in range(5):
            tracker.record_trade(
                ShadowTrade(
                    trade_id=f"t{i}",
                    symbol="510300",
                    entry_date=date(2024, 1, 1),
                    exit_date=date(2024, 1, 5),
                    return_pct=0.02,
                )
            )
        assert tracker.can_promote()

        v1 = registry.register(params={"DROP_LOOKBACK": 5}, changelog="init")
        metrics = tracker.to_promotion_metrics(
            oos_sharpe=1.5,
            oos_sharpe_ci_lower=1.2,
            cost_stress_passed=True,
            manual_approved=True,
        )
        # 样本量来自 tracker
        assert metrics.n_trades == 5
        registry.promote(v1.version_id, metrics)
        assert registry.get_champion().version_id == "v001"

    def test_default_min_trades_is_30(self):
        tracker = ShadowTracker()
        assert tracker.min_trades == DEFAULT_MIN_TRADES
