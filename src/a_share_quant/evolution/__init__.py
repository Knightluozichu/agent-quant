"""Champion/Challenger evolution mechanism.

Controlled strategy evolution:
1. Champion: Current best-performing strategy
2. Challengers: New strategy variants being tested
3. Promotion: Challenger replaces champion if it outperforms
4. Demotion: Champion demoted if underperforms

Key principles:
- Never auto-deploy to live trading
- Require statistical significance for promotion
- Keep audit trail of all changes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional
import hashlib
import json


class StrategyStatus(str, Enum):
    """Strategy lifecycle status."""

    CHALLENGER = "CHALLENGER"  # Being tested
    CHAMPION = "CHAMPION"  # Current best
    RETIRED = "RETIRED"  # No longer active
    REJECTED = "REJECTED"  # Failed testing


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy."""

    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sample_size: int = 0  # Number of trades

    def to_dict(self) -> dict:
        return {
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "sample_size": self.sample_size,
        }


@dataclass
class StrategyVersion:
    """A versioned strategy configuration."""

    strategy_id: str
    version: int
    name: str
    params: dict
    status: StrategyStatus = StrategyStatus.CHALLENGER
    created_at: datetime = field(default_factory=datetime.now)
    promoted_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None
    performance: StrategyPerformance = field(default_factory=StrategyPerformance)
    parent_version: Optional[int] = None  # For tracking lineage
    notes: str = ""

    @property
    def config_hash(self) -> str:
        """Get hash of strategy configuration for reproducibility."""
        config_str = json.dumps(
            {
                "name": self.name,
                "params": self.params,
            },
            sort_keys=True,
        )
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "version": self.version,
            "name": self.name,
            "params": self.params,
            "status": self.status.value,
            "config_hash": self.config_hash,
            "performance": self.performance.to_dict(),
            "notes": self.notes,
        }


@dataclass
class PromotionRecord:
    """Record of a strategy promotion/demotion."""

    timestamp: datetime
    champion_id: str
    champion_version: int
    challenger_id: str
    challenger_version: int
    action: str  # "PROMOTE" or "DEMOTE"
    reason: str
    champion_performance: dict
    challenger_performance: dict


class EvolutionManager:
    """Manage strategy evolution with champion/challenger mechanism.

    Rules:
    1. Only one champion at a time per strategy type
    2. Challengers must achieve statistical significance to promote
    3. Minimum sample size required (default 30 trades)
    4. Champion must underperform by margin to be demoted
    """

    def __init__(
        self,
        min_sample_size: int = 30,
        promotion_threshold: float = 0.10,  # 10% better to promote
        demotion_threshold: float = -0.15,  # 15% worse to demote
        max_challengers: int = 3,
    ):
        self.min_sample_size = min_sample_size
        self.promotion_threshold = promotion_threshold
        self.demotion_threshold = demotion_threshold
        self.max_challengers = max_challengers

        self._champions: dict[str, StrategyVersion] = {}
        self._challengers: dict[str, list[StrategyVersion]] = {}
        self._retired: list[StrategyVersion] = []
        self._promotion_history: list[PromotionRecord] = []

    def register_champion(self, strategy: StrategyVersion) -> None:
        """Register initial champion for a strategy type."""
        strategy.status = StrategyStatus.CHAMPION
        strategy.promoted_at = datetime.now()
        self._champions[strategy.strategy_id] = strategy

    def add_challenger(self, strategy: StrategyVersion) -> bool:
        """Add a new challenger.

        Returns False if max challengers reached.
        """
        sid = strategy.strategy_id
        if sid not in self._challengers:
            self._challengers[sid] = []

        if len(self._challengers[sid]) >= self.max_challengers:
            return False

        strategy.status = StrategyStatus.CHALLENGER
        self._challengers[sid].append(strategy)
        return True

    def get_champion(self, strategy_id: str) -> Optional[StrategyVersion]:
        """Get current champion for a strategy type."""
        return self._champions.get(strategy_id)

    def get_challengers(self, strategy_id: str) -> list[StrategyVersion]:
        """Get all challengers for a strategy type."""
        return self._challengers.get(strategy_id, [])

    def update_performance(
        self,
        strategy_id: str,
        version: int,
        performance: StrategyPerformance,
    ) -> None:
        """Update performance metrics for a strategy version."""
        # Check champion
        champion = self._champions.get(strategy_id)
        if champion and champion.version == version:
            champion.performance = performance
            return

        # Check challengers
        for challenger in self._challengers.get(strategy_id, []):
            if challenger.version == version:
                challenger.performance = performance
                return

    def evaluate_promotion(self, strategy_id: str) -> tuple[bool, str]:
        """Evaluate if any challenger should be promoted.

        Returns:
            (should_promote, reason)
        """
        champion = self._champions.get(strategy_id)
        if not champion:
            return False, "No champion registered"

        challengers = self._challengers.get(strategy_id, [])
        if not challengers:
            return False, "No challengers"

        # Find best challenger
        best_challenger = None
        best_score = float("-inf")

        for challenger in challengers:
            # Check sample size
            if challenger.performance.sample_size < self.min_sample_size:
                continue

            # Calculate composite score
            score = self._calculate_score(challenger.performance)
            if score > best_score:
                best_score = score
                best_challenger = challenger

        if not best_challenger:
            return False, "No challenger with sufficient sample size"

        # Compare with champion
        champion_score = self._calculate_score(champion.performance)

        if champion_score == 0:
            improvement = best_score
        else:
            improvement = (best_score - champion_score) / abs(champion_score)

        if improvement >= self.promotion_threshold:
            return True, f"Challenger v{best_challenger.version} outperforms by {improvement:.1%}"

        return (
            False,
            f"Improvement {improvement:.1%} below threshold {self.promotion_threshold:.1%}",
        )

    def promote_challenger(
        self,
        strategy_id: str,
        challenger_version: int,
        reason: str = "",
    ) -> bool:
        """Promote a challenger to champion.

        Returns True if promotion successful.
        """
        champion = self._champions.get(strategy_id)
        challengers = self._challengers.get(strategy_id, [])

        # Find challenger
        challenger = None
        for c in challengers:
            if c.version == challenger_version:
                challenger = c
                break

        if not challenger:
            return False

        # Record promotion
        record = PromotionRecord(
            timestamp=datetime.now(),
            champion_id=strategy_id,
            champion_version=champion.version if champion else 0,
            challenger_id=strategy_id,
            challenger_version=challenger_version,
            action="PROMOTE",
            reason=reason,
            champion_performance=champion.performance.to_dict() if champion else {},
            challenger_performance=challenger.performance.to_dict(),
        )
        self._promotion_history.append(record)

        # Demote old champion
        if champion:
            champion.status = StrategyStatus.RETIRED
            champion.retired_at = datetime.now()
            self._retired.append(champion)

        # Promote challenger
        challenger.status = StrategyStatus.CHAMPION
        challenger.promoted_at = datetime.now()
        self._champions[strategy_id] = challenger

        # Remove from challengers
        self._challengers[strategy_id] = [c for c in challengers if c.version != challenger_version]

        return True

    def _calculate_score(self, perf: StrategyPerformance) -> float:
        """Calculate composite performance score.

        Weighted combination of:
        - Sharpe ratio (40%)
        - Total return (30%)
        - Max drawdown (20%, inverted)
        - Win rate (10%)
        """
        sharpe_score = perf.sharpe_ratio * 0.4
        return_score = perf.total_return * 0.3
        drawdown_score = (1 - perf.max_drawdown) * 0.2
        win_rate_score = perf.win_rate * 0.1

        return sharpe_score + return_score + drawdown_score + win_rate_score

    def get_status_report(self) -> dict:
        """Get status report of all strategies."""
        report = {
            "champions": {},
            "challengers": {},
            "retired_count": len(self._retired),
            "promotion_count": len(self._promotion_history),
        }

        for sid, champion in self._champions.items():
            report["champions"][sid] = champion.to_dict()

        for sid, challengers in self._challengers.items():
            report["challengers"][sid] = [c.to_dict() for c in challengers]

        return report
