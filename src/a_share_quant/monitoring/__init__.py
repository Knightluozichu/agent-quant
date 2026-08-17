"""Drift detection for strategy monitoring.

Detects:
1. Factor drift: Changes in factor distributions
2. Performance drift: Degradation in strategy performance
3. Regime drift: Changes in market regime distribution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
from scipy import stats

# =============================================================================
# Drift Detection Results
# =============================================================================


@dataclass
class DriftAlert:
    """A drift detection alert."""

    alert_id: str
    drift_type: str  # FACTOR, PERFORMANCE, REGIME
    severity: str  # INFO, WARNING, CRITICAL
    metric_name: str
    baseline_value: float
    current_value: float
    drift_score: float  # 0-1, higher = more drift
    detected_at: datetime = field(default_factory=datetime.now)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "drift_type": self.drift_type,
            "severity": self.severity,
            "metric_name": self.metric_name,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "drift_score": self.drift_score,
            "detected_at": self.detected_at.isoformat(),
            "description": self.description,
        }


# =============================================================================
# Statistical Drift Detection
# =============================================================================


class StatisticalDriftDetector:
    """Detect drift using statistical tests."""

    def __init__(
        self,
        window_size: int = 60,  # Baseline window
        sensitivity: float = 0.05,  # P-value threshold
    ):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self._alert_counter = 0

    def detect_distribution_drift(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        metric_name: str,
    ) -> DriftAlert | None:
        """Detect drift using KS test."""
        if len(baseline) < 10 or len(current) < 10:
            return None

        # Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(baseline, current)

        if p_value < self.sensitivity:
            self._alert_counter += 1
            severity = "CRITICAL" if p_value < 0.01 else "WARNING"

            return DriftAlert(
                alert_id=f"DRIFT_{self._alert_counter:06d}",
                drift_type="FACTOR",
                severity=severity,
                metric_name=metric_name,
                baseline_value=float(np.mean(baseline)),
                current_value=float(np.mean(current)),
                drift_score=ks_stat,
                description=f"KS test: stat={ks_stat:.4f}, p={p_value:.4f}",
            )

        return None

    def detect_mean_drift(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        metric_name: str,
    ) -> DriftAlert | None:
        """Detect drift in mean using t-test."""
        if len(baseline) < 10 or len(current) < 10:
            return None

        _t_stat, p_value = stats.ttest_ind(baseline, current)

        if p_value < self.sensitivity:
            self._alert_counter += 1
            baseline_mean = float(np.mean(baseline))
            current_mean = float(np.mean(current))
            drift_pct = (
                abs(current_mean - baseline_mean) / abs(baseline_mean) if baseline_mean != 0 else 0
            )

            severity = "CRITICAL" if drift_pct > 0.5 else "WARNING"

            return DriftAlert(
                alert_id=f"DRIFT_{self._alert_counter:06d}",
                drift_type="FACTOR",
                severity=severity,
                metric_name=metric_name,
                baseline_value=baseline_mean,
                current_value=current_mean,
                drift_score=drift_pct,
                description=f"Mean shift: {baseline_mean:.4f} → {current_mean:.4f}",
            )

        return None


# =============================================================================
# Performance Drift Monitor
# =============================================================================


@dataclass
class PerformanceWindow:
    """Performance metrics for a time window."""

    start_date: date
    end_date: date
    total_return: float
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float
    trade_count: int


class PerformanceDriftMonitor:
    """Monitor strategy performance for degradation."""

    def __init__(
        self,
        baseline_window: int = 60,  # Days
        alert_threshold: float = 0.30,  # 30% degradation
    ):
        self.baseline_window = baseline_window
        self.alert_threshold = alert_threshold
        self._baseline: PerformanceWindow | None = None
        self._history: list[PerformanceWindow] = []
        self._detector = StatisticalDriftDetector()

    def set_baseline(self, window: PerformanceWindow) -> None:
        """Set baseline performance."""
        self._baseline = window

    def add_observation(self, window: PerformanceWindow) -> None:
        """Add a new performance observation."""
        self._history.append(window)

    def check_drift(self) -> list[DriftAlert]:
        """Check for performance drift."""
        if not self._baseline or len(self._history) < 5:
            return []

        alerts = []

        # Get recent performance
        recent = self._history[-5:]
        recent_returns = [w.total_return for w in recent]
        recent_sharpes = [w.sharpe_ratio for w in recent]

        # Check return degradation
        avg_recent_return = np.mean(recent_returns)
        if self._baseline.total_return != 0:
            return_drift = (self._baseline.total_return - avg_recent_return) / abs(
                self._baseline.total_return
            )
            if return_drift > self.alert_threshold:
                self._detector._alert_counter += 1
                alerts.append(
                    DriftAlert(
                        alert_id=f"PERF_{self._detector._alert_counter:06d}",
                        drift_type="PERFORMANCE",
                        severity="CRITICAL" if return_drift > 0.5 else "WARNING",
                        metric_name="total_return",
                        baseline_value=self._baseline.total_return,
                        current_value=avg_recent_return,
                        drift_score=return_drift,
                        description=f"Return degraded by {return_drift:.1%}",
                    )
                )

        # Check Sharpe degradation
        avg_recent_sharpe = np.mean(recent_sharpes)
        if self._baseline.sharpe_ratio != 0:
            sharpe_drift = (self._baseline.sharpe_ratio - avg_recent_sharpe) / abs(
                self._baseline.sharpe_ratio
            )
            if sharpe_drift > self.alert_threshold:
                self._detector._alert_counter += 1
                alerts.append(
                    DriftAlert(
                        alert_id=f"PERF_{self._detector._alert_counter:06d}",
                        drift_type="PERFORMANCE",
                        severity="WARNING",
                        metric_name="sharpe_ratio",
                        baseline_value=self._baseline.sharpe_ratio,
                        current_value=avg_recent_sharpe,
                        drift_score=sharpe_drift,
                        description=f"Sharpe degraded by {sharpe_drift:.1%}",
                    )
                )

        return alerts


# =============================================================================
# Regime Drift Monitor
# =============================================================================


class RegimeDriftMonitor:
    """Monitor market regime distribution changes."""

    def __init__(self, baseline_distribution: dict[str, float] | None = None):
        self._baseline = baseline_distribution or {}
        self._current_counts: dict[str, int] = {}
        self._total_observations = 0

    def set_baseline(self, distribution: dict[str, float]) -> None:
        """Set baseline regime distribution."""
        self._baseline = distribution

    def observe(self, regime_state: str) -> None:
        """Record a regime observation."""
        self._current_counts[regime_state] = self._current_counts.get(regime_state, 0) + 1
        self._total_observations += 1

    def get_current_distribution(self) -> dict[str, float]:
        """Get current regime distribution."""
        if self._total_observations == 0:
            return {}
        return {
            state: count / self._total_observations for state, count in self._current_counts.items()
        }

    def check_drift(self) -> DriftAlert | None:
        """Check for regime distribution drift."""
        if not self._baseline or self._total_observations < 20:
            return None

        current = self.get_current_distribution()

        # Calculate KL divergence
        kl_div = 0.0
        for state in set(list(self._baseline.keys()) + list(current.keys())):
            p = self._baseline.get(state, 0.001)
            q = current.get(state, 0.001)
            if p > 0 and q > 0:
                kl_div += p * np.log(p / q)

        if kl_div > 0.5:  # Threshold for significant drift
            return DriftAlert(
                alert_id="REGIME_DRIFT",
                drift_type="REGIME",
                severity="WARNING" if kl_div < 1.0 else "CRITICAL",
                metric_name="regime_distribution",
                baseline_value=0.0,
                current_value=kl_div,
                drift_score=min(1.0, kl_div / 2),
                description=f"KL divergence: {kl_div:.4f}",
            )

        return None
