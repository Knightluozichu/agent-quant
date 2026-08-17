"""Parameter versioning and Champion/Challenger framework.

P3-E2: 参数版本化 + 回滚

为七星V3 (QixingV3) 策略提供参数版本管理与受控进化:

1. ``ParamVersion``    —— 单个参数版本 (champion / challenger / archived)
2. ``ParamRegistry``   —— 版本注册表，持久化到 JSON，支持注册 / 晋升 / 回滚
3. ``ShadowTracker``   —— 影子交易跟踪，累积样本以供晋升评估

晋升门槛 (保守，需同时满足):
- 最少 ``min_trades`` (默认 30) 笔交易，保证样本充分性
- OOS Sharpe 置信区间下界 > 当前 champion 的 Sharpe 点估计
- 成本压力测试 (cost stress test) 通过
- 最大回撤不劣于 champion (drawdown 以负小数表示，越接近 0 越好)
- 人工批准标志 (manual_approved) 必须为 True

设计原则 (对齐 AGENTS.md 自进化治理):
- 永不自动部署到实盘 (live trading)
- 晋升需统计显著性，单笔盈亏不作为因子有效证据
- 保留完整审计轨迹 (audit_log)，能复盘、能回滚
- 注册表持久化到 ``data/evolution/param_registry.json``，可复现
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

# 注册表默认持久化路径 (相对仓库根目录)
DEFAULT_REGISTRY_PATH: Path = Path("data/evolution/param_registry.json")

# 影子交易晋升所需的最小样本量
DEFAULT_MIN_TRADES: int = 30


class VersionStatus(StrEnum):
    """参数版本生命周期状态。"""

    CHAMPION = "champion"  # 当前生效的最佳版本
    CHALLENGER = "challenger"  # 正在影子测试的候选版本
    ARCHIVED = "archived"  # 已归档 (被替代或回滚离开)


class ParamVersion(BaseModel):
    """单个参数版本。

    Attributes:
        version_id:         版本唯一标识 (如 ``v001``)
        created_at:         创建时间 (UTC)
        params:             策略参数字典
        changelog:          变更说明
        status:             当前生命周期状态
        metrics:            晋升时记录的评估指标 (未评估时为 None)
        promoted_at:        最近一次晋升为 champion 的时间
        archived_at:        归档时间
        parent_version_id:  父版本标识，用于追踪血缘 lineage
    """

    version_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    params: dict[str, Any]
    changelog: str
    status: VersionStatus = VersionStatus.CHALLENGER
    metrics: dict[str, Any] | None = None
    promoted_at: datetime | None = None
    archived_at: datetime | None = None
    parent_version_id: str | None = None

    @property
    def params_hash(self) -> str:
        """参数字典的稳定短哈希，用于复现性与回归校验。

        与 ``EvolutionManager.config_hash`` 语义一致: 相同参数 (任意顺序) 产生相同哈希。
        """
        payload = json.dumps(self.params, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class PromotionMetrics(BaseModel):
    """晋升评估指标。

    所有字段均为晋升门槛的输入。``max_drawdown`` 以负小数表示
    (如 ``-0.15`` 表示 15% 回撤)，与 ``data/qixing_results`` 结果文件约定一致。

    Attributes:
        n_trades:              样本交易笔数
        oos_sharpe:            OOS (样本外) Sharpe 点估计
        oos_sharpe_ci_lower:   OOS Sharpe 置信区间下界
        cost_stress_passed:    成本压力测试是否通过
        max_drawdown:          最大回撤 (负小数，<= 0)
        manual_approved:       人工批准标志
    """

    n_trades: int = Field(ge=0)
    oos_sharpe: float
    oos_sharpe_ci_lower: float
    cost_stress_passed: bool = False
    max_drawdown: float
    manual_approved: bool = False

    @field_validator("max_drawdown")
    @classmethod
    def _drawdown_non_positive(cls, v: float) -> float:
        """回撤约定为非正小数，避免符号歧义导致比较错误。"""
        if v > 0:
            msg = "max_drawdown must be <= 0 (use negative decimal, e.g. -0.15)"
            raise ValueError(msg)
        return v


class PromotionError(ValueError):
    """晋升条件未满足时抛出。

    Attributes:
        reasons: 未满足的门槛列表，便于审计与展示。
    """

    def __init__(self, reasons: list[str]):
        self.reasons = list(reasons)
        super().__init__("; ".join(self.reasons) if self.reasons else "promotion rejected")


class ShadowTrade(BaseModel):
    """单笔影子交易记录。

    Attributes:
        trade_id:    交易唯一标识
        symbol:      标的代码
        entry_date:  入场日
        exit_date:   出场日
        return_pct:  区间收益率 (小数，如 0.05 表示 5%)
        pnl:         绝对盈亏 (可选，默认 0.0)
    """

    trade_id: str
    symbol: str
    entry_date: date
    exit_date: date
    return_pct: float
    pnl: float = 0.0


class ParamRegistry:
    """参数版本注册表: Champion / Challenger 管理。

    所有变更即时持久化到 ``registry_path`` (JSON)，进程重启后可恢复。
    任意时刻最多一个 champion；晋升新 champion 时旧 champion 自动归档。

    Example:
        >>> registry = ParamRegistry()
        >>> v1 = registry.register({"DROP_LOOKBACK": 5}, changelog="init")
        >>> metrics = PromotionMetrics(
        ...     n_trades=50, oos_sharpe=1.5, oos_sharpe_ci_lower=1.2,
        ...     cost_stress_passed=True, max_drawdown=-0.10, manual_approved=True,
        ... )
        >>> registry.promote(v1.version_id, metrics)  # doctest: +SKIP
    """

    def __init__(
        self,
        registry_path: Path | None = None,
        min_trades: int = DEFAULT_MIN_TRADES,
    ) -> None:
        self.registry_path: Path = registry_path or DEFAULT_REGISTRY_PATH
        self.min_trades: int = min_trades
        self._versions: list[ParamVersion] = []
        self._audit_log: list[dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def register(
        self,
        params: dict[str, Any],
        changelog: str,
        parent_version_id: str | None = None,
    ) -> ParamVersion:
        """注册一个新的 challenger 版本。

        Args:
            params:             策略参数字典
            changelog:          变更说明
            parent_version_id:  父版本标识 (血缘追踪)

        Returns:
            新创建的 ``ParamVersion`` (状态为 challenger)。
        """
        version = ParamVersion(
            version_id=self._next_version_id(),
            params=dict(params),
            changelog=changelog,
            status=VersionStatus.CHALLENGER,
            parent_version_id=parent_version_id,
        )
        self._versions.append(version)
        self._audit(
            "register",
            version.version_id,
            {"changelog": changelog, "parent": parent_version_id},
        )
        self._save()
        return version

    def promote(
        self,
        version_id: str,
        metrics: PromotionMetrics | dict[str, Any],
    ) -> ParamVersion:
        """将 challenger 晋升为 champion。

        需通过全部晋升门槛 (见模块 docstring)。失败时抛出 ``PromotionError``，
        不修改任何版本状态，但会把拒绝原因写入审计日志。

        Args:
            version_id:  待晋升版本标识
            metrics:     晋升评估指标 (``PromotionMetrics`` 或等价 dict)

        Returns:
            晋升后的 ``ParamVersion`` (状态为 champion)。

        Raises:
            KeyError:        版本不存在
            PromotionError:  晋升门槛未满足
        """
        target = self._find(version_id)
        if target is None:
            msg = f"version not found: {version_id}"
            raise KeyError(msg)
        if target.status == VersionStatus.CHAMPION:
            msg = f"version {version_id} is already champion"
            raise PromotionError([msg])

        metrics_obj = (
            metrics
            if isinstance(metrics, PromotionMetrics)
            else PromotionMetrics.model_validate(metrics)
        )

        reasons = self._validate_promotion(metrics_obj)
        if reasons:
            self._audit("promote_rejected", version_id, {"reasons": reasons})
            self._save()
            raise PromotionError(reasons)

        current = self.get_champion()
        if current is not None and current.version_id != target.version_id:
            current.status = VersionStatus.ARCHIVED
            current.archived_at = self._now()

        target.status = VersionStatus.CHAMPION
        target.promoted_at = self._now()
        target.metrics = metrics_obj.model_dump(mode="json")

        self._audit(
            "promote",
            target.version_id,
            {"archived": current.version_id if current else None},
        )
        self._save()
        return target

    def rollback(self, version_id: str) -> ParamVersion:
        """回滚到指定历史版本。

        将当前 champion 归档，并把目标版本重新标记为 champion。
        目标版本通常是被替代的旧 champion。

        Args:
            version_id:  要恢复的版本标识

        Returns:
            恢复后的 ``ParamVersion`` (状态为 champion)。

        Raises:
            KeyError:        版本不存在
            PromotionError:  目标已是当前 champion
        """
        target = self._find(version_id)
        if target is None:
            msg = f"version not found: {version_id}"
            raise KeyError(msg)
        if target.status == VersionStatus.CHAMPION:
            msg = f"cannot rollback to current champion {version_id}"
            raise PromotionError([msg])

        current = self.get_champion()
        if current is not None:
            current.status = VersionStatus.ARCHIVED
            current.archived_at = self._now()

        target.status = VersionStatus.CHAMPION
        target.promoted_at = self._now()

        self._audit(
            "rollback",
            target.version_id,
            {"archived": current.version_id if current else None},
        )
        self._save()
        return target

    def get_champion(self) -> ParamVersion | None:
        """返回当前 champion 版本 (无则返回 None)。

        champion 的参数可通过 ``.params`` 访问。
        """
        for v in self._versions:
            if v.status == VersionStatus.CHAMPION:
                return v
        return None

    def get_challengers(self) -> list[ParamVersion]:
        """返回所有活跃 challenger 版本。"""
        return [v for v in self._versions if v.status == VersionStatus.CHALLENGER]

    def list_versions(self) -> list[ParamVersion]:
        """返回完整版本历史 (按注册顺序)。"""
        return list(self._versions)

    def get_audit_log(self) -> list[dict[str, Any]]:
        """返回审计日志副本。"""
        return [dict(entry) for entry in self._audit_log]

    # ------------------------------------------------------------------
    # 晋升门槛校验
    # ------------------------------------------------------------------
    def _validate_promotion(self, metrics: PromotionMetrics) -> list[str]:
        """校验晋升门槛，返回未满足项的列表 (空列表表示通过)。"""
        reasons: list[str] = []

        if metrics.n_trades < self.min_trades:
            reasons.append(f"insufficient trades: {metrics.n_trades} < min {self.min_trades}")
        if not metrics.manual_approved:
            reasons.append("manual approval required (manual_approved=False)")
        if not metrics.cost_stress_passed:
            reasons.append("cost stress test not passed")

        champion = self.get_champion()
        if champion is not None and champion.metrics is not None:
            champ_sharpe = float(champion.metrics.get("oos_sharpe", 0.0))
            if not (metrics.oos_sharpe_ci_lower > champ_sharpe):
                reasons.append(
                    f"OOS Sharpe CI lower {metrics.oos_sharpe_ci_lower:.4f} "
                    f"not > champion Sharpe {champ_sharpe:.4f}"
                )
            champ_dd = float(champion.metrics.get("max_drawdown", -1.0))
            if metrics.max_drawdown < champ_dd:
                reasons.append(
                    f"drawdown {metrics.max_drawdown:.4f} worse than champion {champ_dd:.4f}"
                )
        return reasons

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _find(self, version_id: str) -> ParamVersion | None:
        for v in self._versions:
            if v.version_id == version_id:
                return v
        return None

    def _next_version_id(self) -> str:
        max_n = 0
        for v in self._versions:
            suffix = v.version_id[1:] if v.version_id.startswith("v") else ""
            if suffix.isdigit():
                max_n = max(max_n, int(suffix))
        return f"v{max_n + 1:03d}"

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    def _audit(self, action: str, version_id: str, detail: dict[str, Any]) -> None:
        self._audit_log.append(
            {
                "timestamp": self._now().isoformat(),
                "action": action,
                "version_id": version_id,
                "detail": detail,
            }
        )

    def _load(self) -> None:
        if not self.registry_path.exists():
            return
        data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        self._versions = [ParamVersion.model_validate(v) for v in data.get("versions", [])]
        self._audit_log = [dict(e) for e in data.get("audit_log", [])]

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "versions": [v.model_dump(mode="json") for v in self._versions],
            "audit_log": self._audit_log,
        }
        self.registry_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class ShadowTracker:
    """影子交易跟踪器: 累积候选参数的样本外交易结果。

    影子交易不触发真实下单，仅记录候选版本在历史/实时行情中的模拟交易，
    累积到最小样本量 (默认 30 笔) 后方可提交晋升评估。

    Example:
        >>> tracker = ShadowTracker(min_trades=30)
        >>> tracker.can_promote()
        False
        >>> for i in range(30):  # doctest: +SKIP
        ...     tracker.record_trade(ShadowTrade(
        ...         trade_id=f"t{i}", symbol="510300",
        ...         entry_date=date(2024, 1, 1), exit_date=date(2024, 1, 5),
        ...         return_pct=0.01,
        ...     ))
        >>> tracker.can_promote()  # doctest: +SKIP
        True
    """

    def __init__(self, min_trades: int = DEFAULT_MIN_TRADES) -> None:
        self.min_trades: int = min_trades
        self._trades: list[ShadowTrade] = []

    def record_trade(self, trade_result: ShadowTrade | dict[str, Any]) -> None:
        """记录一笔影子交易。

        Args:
            trade_result: ``ShadowTrade`` 或等价 dict。
        """
        trade = (
            trade_result
            if isinstance(trade_result, ShadowTrade)
            else ShadowTrade.model_validate(trade_result)
        )
        self._trades.append(trade)

    @property
    def n_trades(self) -> int:
        """已记录的影子交易笔数。"""
        return len(self._trades)

    def can_promote(self) -> bool:
        """是否已达到最小样本量，可提交晋升评估。"""
        return self.n_trades >= self.min_trades

    def summary(self) -> dict[str, Any]:
        """返回影子表现汇总。

        Returns:
            包含 ``n_trades`` / ``win_rate`` / ``avg_return`` /
            ``total_return`` / ``sharpe`` / ``max_drawdown`` 的字典。
            无交易时各指标为 0.0。

        Note:
            ``sharpe`` 按每笔收益的均值/标准差计算并按 ``sqrt(252)`` 年化，
            为粗略估计; ``max_drawdown`` 基于累计收益曲线，以负小数表示。
        """
        n = len(self._trades)
        if n == 0:
            return {
                "n_trades": 0,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "total_return": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
            }

        rets = [t.return_pct for t in self._trades]
        wins = sum(1 for r in rets if r > 0)
        win_rate = wins / n
        mean_return = sum(rets) / n

        # 累计收益曲线与最大回撤
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in rets:
            equity *= 1.0 + r
            peak = max(peak, equity)
            dd = equity / peak - 1.0 if peak > 0 else 0.0
            max_dd = min(max_dd, dd)
        total_return = equity - 1.0

        variance = sum((r - mean_return) ** 2 for r in rets) / n
        std = math.sqrt(variance)
        sharpe = mean_return / std * math.sqrt(252) if std > 0 else 0.0

        return {
            "n_trades": n,
            "win_rate": win_rate,
            "avg_return": mean_return,
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
        }

    def to_promotion_metrics(
        self,
        *,
        oos_sharpe: float,
        oos_sharpe_ci_lower: float,
        cost_stress_passed: bool = False,
        manual_approved: bool = False,
    ) -> PromotionMetrics:
        """根据影子汇总构建 ``PromotionMetrics``。

        样本数与回撤取自 ``summary()``，Sharpe 相关字段与审批标志由调用方提供。
        便于将 ShadowTracker 与 ``ParamRegistry.promote`` 串联。
        """
        s = self.summary()
        return PromotionMetrics(
            n_trades=int(s["n_trades"]),
            oos_sharpe=oos_sharpe,
            oos_sharpe_ci_lower=oos_sharpe_ci_lower,
            cost_stress_passed=cost_stress_passed,
            max_drawdown=float(s["max_drawdown"]),
            manual_approved=manual_approved,
        )


__all__ = [
    "DEFAULT_MIN_TRADES",
    "DEFAULT_REGISTRY_PATH",
    "ParamRegistry",
    "ParamVersion",
    "PromotionError",
    "PromotionMetrics",
    "ShadowTracker",
    "ShadowTrade",
    "VersionStatus",
]
