"""V4 stale-leader momentum-failure breaker research.

The canonical V4 selector remains frozen.  The breaker acts only after an
observable absolute shock and only when V4 would otherwise keep the current
holding.  It rotates to a qualified positive-momentum alternative, or exits to
cash until the next fixed grid when no such alternative exists.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts import exp_v3g_relative_rotation as rr
    from scripts import exp_v4_risk_budget as rb
    from scripts import qixing_v4 as v4
    from scripts import risk_overrides as ro
    from scripts import run_qixing_v3 as rq
    from scripts.exp_v3g_full_pool_fast_slow import run_full_pool_strategy
    from scripts.exp_v4_stop import run_v4_with_stop
except ModuleNotFoundError:
    import exp_v3g_relative_rotation as rr
    import exp_v4_risk_budget as rb
    import qixing_v4 as v4
    import risk_overrides as ro
    import run_qixing_v3 as rq
    from exp_v3g_full_pool_fast_slow import run_full_pool_strategy
    from exp_v4_stop import run_v4_with_stop


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT = PROJECT_ROOT / "data" / "v9_results" / "v4_momentum_failure.json"
INITIAL_CAPITAL = 100_000.0
SELECTED = "stale_leader_best_or_cash"


def threshold_neighborhood() -> dict[str, tuple[float, float]]:
    """One-axis +/-20% perturbations; center remains the selected rule."""
    return {
        "center": (-0.05, -0.10),
        "day_stricter_80pct": (-0.04, -0.10),
        "day_looser_120pct": (-0.06, -0.10),
        "three_day_stricter_80pct": (-0.05, -0.08),
        "three_day_looser_120pct": (-0.05, -0.12),
    }


def replay_documented_failure() -> dict[str, Any]:
    """Replay the user's causal path after the first observed silver shock."""
    silver: np.ndarray[Any, np.dtype[np.float64]] = np.asarray(
        [-0.05, -0.08, -0.03], dtype=np.float64
    )
    gold: np.ndarray[Any, np.dtype[np.float64]] = np.asarray([0.01, 0.02, 0.03], dtype=np.float64)
    v4_return = float(np.prod(1.0 + silver) - 1.0)
    cash_return = float(1.0 + silver[0] - 1.0)
    replacement_return = float((1.0 + silver[0]) * (1.0 + gold[1]) * (1.0 + gold[2]) - 1.0)
    return {
        "silver_returns": silver.tolist(),
        "gold_returns": gold.tolist(),
        "v4_stale_holding_return": v4_return,
        "breaker_cash_return": cash_return,
        "breaker_qualified_gold_return": replacement_return,
        "decision_timing": [
            "day1 silver shock is borne before it becomes observable",
            "day1 14:50 shock breaks stale rank and selects gold or cash",
            "day2/day3 no longer hold silver",
        ],
        "first_shock_is_unavoidable": True,
    }


def audited(result: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(result["metrics"])
    metrics.update(rb._risk_metrics(result["equity_curve"]))
    annual = result.get("annual", rb._annual_rows(result["equity_curve"]))
    metrics["negative_years"] = int(sum(row["return"] < 0.0 for row in annual))
    return {
        **result,
        "metrics": metrics,
        "annual": annual,
        "rolling_252": rb._rolling_252(result["equity_curve"]),
    }


def compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "params": result.get("params", asdict(v4.V4_PARAMS)),
        "metrics": result["metrics"],
        "annual": result["annual"],
        "rolling_252": result["rolling_252"],
        "stop_events": result.get("stop_events", []),
    }


def run_candidate(
    data: dict[str, pd.DataFrame],
    mode: str,
    *,
    cost_multiplier: float,
    day_threshold: float = -0.05,
    three_day_threshold: float = -0.10,
) -> dict[str, Any]:
    return audited(
        run_v4_with_stop(
            data,
            stop_mode="disabled",
            cost_multiplier=cost_multiplier,
            momentum_failure_mode=mode,
            momentum_failure_day_threshold=day_threshold,
            momentum_failure_three_day_threshold=three_day_threshold,
        )
    )


def recent_cluster(baseline: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    dates = pd.to_datetime(baseline["equity_curve"]["trade_date"])
    mask = (dates >= "2026-08-10") & (dates <= "2026-08-11")
    base_returns = rb._curve_returns(baseline["equity_curve"])[mask]
    selected_returns = rb._curve_returns(selected["equity_curve"])[mask]
    base_cluster = float(np.prod(1.0 + base_returns) - 1.0)
    selected_cluster = float(np.prod(1.0 + selected_returns) - 1.0)
    reduction = 1.0 - abs(selected_cluster) / abs(base_cluster) if base_cluster < 0.0 else 0.0
    return {
        "start": "2026-08-10",
        "end": "2026-08-11",
        "v4_daily_returns": base_returns.tolist(),
        "breaker_daily_returns": selected_returns.tolist(),
        "v4_compound_return": base_cluster,
        "breaker_compound_return": selected_cluster,
        "loss_reduction": reduction,
        "same_history_warning": "retrospective close-price proxy, not live OOS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V4 stale-leader failure research")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    args = parser.parse_args()
    if args.bootstrap_samples < 500:
        raise ValueError("use at least 500 bootstrap samples")
    if ro.EXPO_REDUCE != 1.0 or ro.H3_EXPO_REDUCE != 1.0:
        raise RuntimeError("research expects disabled legacy downsizing")

    data = rq.load_data()
    baseline = audited(run_full_pool_strategy(data, v4.V4_PARAMS))
    disabled_control = audited(run_v4_with_stop(data, stop_mode="disabled"))
    parity_max_abs = float(
        np.max(
            np.abs(
                np.asarray(baseline["equity_curve"]["equity"], dtype=float)
                - np.asarray(disabled_control["equity_curve"]["equity"], dtype=float)
            )
        )
    )
    if parity_max_abs > 1e-6:
        raise RuntimeError("disabled breaker does not reproduce canonical V4")
    candidates = {
        "stale_leader_cash": run_candidate(data, "stale_leader_cash", cost_multiplier=1.0),
        SELECTED: run_candidate(data, SELECTED, cost_multiplier=1.0),
    }
    broad_cash = audited(run_v4_with_stop(data, stop_mode="fixed_shock_cash", cost_multiplier=1.0))
    selected = candidates[SELECTED]

    cost_pressure: dict[str, Any] = {}
    for cost in (0.0, 1.0, 2.0, 3.0):
        cost_baseline = audited(run_full_pool_strategy(data, v4.V4_PARAMS, cost_multiplier=cost))
        cost_pressure[f"{cost:.0f}x"] = {
            "baseline_v4": compact(cost_baseline),
            SELECTED: compact(run_candidate(data, SELECTED, cost_multiplier=cost)),
        }

    neighbors: list[dict[str, Any]] = []
    for name, (day_threshold, three_day_threshold) in threshold_neighborhood().items():
        result = run_candidate(
            data,
            SELECTED,
            cost_multiplier=1.0,
            day_threshold=day_threshold,
            three_day_threshold=three_day_threshold,
        )
        metrics = result["metrics"]
        neighbor_recent = recent_cluster(baseline, result)
        neighbors.append(
            {
                "name": name,
                "day_threshold": day_threshold,
                "three_day_threshold": three_day_threshold,
                "final_value": metrics["final_value"],
                "cagr": metrics["cagr"],
                "max_drawdown": metrics["max_drawdown"],
                "sharpe": metrics["sharpe"],
                "stop_triggers": metrics["stop_triggers"],
                "cagr_retention": metrics["cagr"] / baseline["metrics"]["cagr"],
                "recent_cluster_loss_reduction": neighbor_recent["loss_reduction"],
                "specified_cluster_passed": bool(
                    metrics["cagr"] / baseline["metrics"]["cagr"] >= 0.90
                    and neighbor_recent["loss_reduction"] >= 0.40
                ),
            }
        )

    recent = recent_cluster(baseline, selected)
    documented = replay_documented_failure()
    baseline_metrics = baseline["metrics"]
    selected_metrics = selected["metrics"]
    cagr_retention = selected_metrics["cagr"] / baseline_metrics["cagr"]
    mdd_improvement = 1.0 - abs(selected_metrics["max_drawdown"]) / abs(
        baseline_metrics["max_drawdown"]
    )
    acceptance_checks = {
        "cagr_retention_at_least_90pct": cagr_retention >= 0.90,
        "global_mdd_reduction_at_least_10pct": mdd_improvement >= 0.10,
        "documented_cluster_loss_at_most_6pct": (
            documented["breaker_qualified_gold_return"] >= -0.06
        ),
        "recent_cluster_loss_reduction_at_least_40pct": (recent["loss_reduction"] >= 0.40),
        "no_additional_negative_year": (
            selected_metrics["negative_years"] <= baseline_metrics["negative_years"]
        ),
        "positive_at_2x_cost": (
            cost_pressure["2x"][SELECTED]["metrics"]["final_value"] > INITIAL_CAPITAL
        ),
    }
    bootstrap = {
        str(block): rb._bootstrap_comparison(
            baseline,
            selected,
            bootstrap_samples=args.bootstrap_samples,
            block_size=block,
            seed=20261100 + block,
        )
        for block in (5, 20, 60)
    }
    segments = {
        label: {
            "baseline_v4": rr.segment_metrics(baseline["equity_curve"], start, end),
            SELECTED: rr.segment_metrics(selected["equity_curve"], start, end),
        }
        for label, start, end in (
            ("retrospective_2020_2023", "2020-06-19", "2023-12-29"),
            ("retrospective_2024_current", "2024-01-01", "2026-08-11"),
        )
    }
    classification = (
        "targeted_cluster_breaker_only_requires_frozen_oos"
        if not all(acceptance_checks.values())
        else "global_candidate_requires_frozen_oos"
    )
    payload: dict[str, Any] = {
        "meta": {
            "strategy": "frozen V4 plus stale-leader momentum-failure breaker",
            "data_start": str(baseline["equity_curve"]["trade_date"].iloc[0].date()),
            "data_end": str(baseline["equity_curve"]["trade_date"].iloc[-1].date()),
            "observations": len(baseline["equity_curve"]),
            "design_date": "2026-08-12",
            "live_post_design_oos_observations": 0,
            "selected_before_formal_run": SELECTED,
            "main_candidate_count": len(candidates),
            "neighborhood_points": len(neighbors),
            "adaptive_context_warning": (
                "the rule was proposed after observing V4 and prior stop research; "
                "historical results are model-development evidence only"
            ),
            "no_lookahead": (
                "T shock and V4 state decide the T 14:50 close-proxy trade; "
                "forward event returns are reporting only"
            ),
            "production_status": "research_only_no_production_change",
            "disabled_breaker_parity_max_abs_equity_difference": parity_max_abs,
        },
        "theory": {
            "formula": ("STOP_t = I[V4 would retain held asset] * I[r_1d<=-5% or r_3d<=-10%]"),
            "replacement": (
                "strongest eligible alternative with positive slow momentum and trend, "
                "and both 3d/5d return above the failed holding; otherwise cash"
            ),
            "reentry": "cash exits wait until the next fixed V4 rebalance grid",
            "threshold_source": (
                "loss-budget thresholds inherited from the documented accident, "
                "not selected by return maximization"
            ),
            "identifiability_limit": (
                "the first unobserved shock cannot be avoided by a causal daily rule"
            ),
        },
        "baseline_v4": compact(baseline),
        "broad_fixed_cash_control": compact(broad_cash),
        "candidates": {name: compact(result) for name, result in candidates.items()},
        "documented_failure_replay": documented,
        "recent_cluster_replay": recent,
        "acceptance": {
            "cagr_retention": cagr_retention,
            "global_mdd_improvement": mdd_improvement,
            "checks": acceptance_checks,
            "global_joint_passed": all(acceptance_checks.values()),
            "specified_cluster_passed": bool(
                acceptance_checks["cagr_retention_at_least_90pct"]
                and acceptance_checks["documented_cluster_loss_at_most_6pct"]
                and acceptance_checks["recent_cluster_loss_reduction_at_least_40pct"]
            ),
            "threshold_neighborhood_specified_cluster_passes": sum(
                row["specified_cluster_passed"] for row in neighbors
            ),
            "threshold_neighborhood_specified_cluster_pass_rate": (
                sum(row["specified_cluster_passed"] for row in neighbors) / len(neighbors)
            ),
        },
        "threshold_neighborhood": neighbors,
        "cost_pressure": cost_pressure,
        "retrospective_segments": segments,
        "paired_block_bootstrap": bootstrap,
        "assessment": {
            "classification": classification,
            "interpretation": (
                "The breaker can stop a second clustered loss while retaining most "
                "historical CAGR, but it is not a general maximum-drawdown solution."
            ),
            "production_status": "research_only_no_production_change",
        },
    }

    print("\nV4 stale-leader momentum-failure breaker")
    for name, result in [
        ("baseline_v4", baseline),
        ("broad_fixed_cash", broad_cash),
        *candidates.items(),
    ]:
        metrics = result["metrics"]
        print(
            f"{name:<28} final={metrics['final_value']:>11,.0f} "
            f"CAGR={metrics['cagr']:>7.2%} Sharpe={metrics['sharpe']:>6.3f} "
            f"MDD={metrics['max_drawdown']:>7.2%} "
            f"stops={metrics.get('stop_triggers', 0):>2}"
        )
    print(
        f"\nselected retention={cagr_retention:.1%} "
        f"MDD improvement={mdd_improvement:.1%} "
        f"recent cluster={recent['v4_compound_return']:.2%} -> "
        f"{recent['breaker_compound_return']:.2%} "
        f"assessment={classification}"
    )
    if args.save:
        OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
        print(f"saved: {OUTPUT}")


if __name__ == "__main__":
    main()
