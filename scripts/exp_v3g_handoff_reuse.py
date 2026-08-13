"""Reuse boundary test for V3-G pair handoffs inside the production universe.

Three layers are compared without changing production code:

1. canonical server V3-G with no non-grid handoff;
2. global-confirmed handoff, where the peer must also be the full-pool leader;
3. internal-pair handoff, where only the declared pair's 10/20-day scores matter.

All signals use T and earlier.  Execution remains the server's T-day 14:50
close approximation and all downsize layers remain disabled.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

try:
    from scripts import risk_overrides as ro
    from scripts import run_qixing_v3 as rq
    from scripts.exp_v3g_pair_multifactor import (
        PairFactorParams,
        compact,
        run_pair_strategy,
    )
    from scripts.exp_v3g_relative_rotation import (
        RotationParams,
        run_strategy,
        segment_metrics,
    )
except ModuleNotFoundError:
    import risk_overrides as ro
    import run_qixing_v3 as rq
    from exp_v3g_pair_multifactor import (
        PairFactorParams,
        compact,
        run_pair_strategy,
    )
    from exp_v3g_relative_rotation import (
        RotationParams,
        run_strategy,
        segment_metrics,
    )

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT = PROJECT_ROOT / "data" / "v9_results" / "v3g_handoff_reuse.json"

PAIR_GROUPS: dict[str, tuple[str, str]] = {
    "precious": ("518880", "161226"),
    "commodity": ("159985", "501018"),
    "equity": ("513100", "159915"),
    "defense": ("511220", "511880"),
}
GROUP_NAMES = {
    "precious": "黄金-白银",
    "commodity": "豆粕-原油",
    "equity": "纳指-创业板",
    "defense": "城投债-货币",
}


def global_params(
    groups: tuple[tuple[str, str], ...],
) -> RotationParams:
    """Require the proposed peer to be the full production-pool leader."""
    return RotationParams(
        relative_gap=0.005,
        holding_drawdown=0.0,
        persistence_hits=1,
        fast_drawdown=None,
        minimum_hold_days=3,
        scope_groups=groups,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="V3-G handoff reuse test")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    if ro.EXPO_REDUCE != 1.0 or ro.H3_EXPO_REDUCE != 1.0:
        raise RuntimeError("Current V3-G requires all downsize layers disabled")

    data = rq.load_data()
    slow = PairFactorParams.slow_only()
    all_groups = tuple(PAIR_GROUPS.values())

    baseline = run_pair_strategy(data, PairFactorParams.disabled())
    global_results: dict[str, dict[str, Any]] = {}
    internal_results: dict[str, dict[str, Any]] = {}
    for name, pair in PAIR_GROUPS.items():
        global_results[name] = run_strategy(data, global_params((pair,)))
        internal_results[name] = run_pair_strategy(data, slow, pair_groups=(pair,))

    combinations = {
        "precious_commodity": (PAIR_GROUPS["precious"], PAIR_GROUPS["commodity"]),
        "precious_equity": (PAIR_GROUPS["precious"], PAIR_GROUPS["equity"]),
        "precious_defense": (PAIR_GROUPS["precious"], PAIR_GROUPS["defense"]),
        "all": all_groups,
    }
    global_combinations = {
        name: run_strategy(data, global_params(groups)) for name, groups in combinations.items()
    }
    internal_combinations = {
        name: run_pair_strategy(data, slow, pair_groups=groups)
        for name, groups in combinations.items()
    }
    precious_hybrid_params = PairFactorParams.slow_only(confirmation_hits=2)
    precious_hybrid = run_pair_strategy(
        data,
        precious_hybrid_params,
        pair_groups=(PAIR_GROUPS["precious"],),
        global_confirmation_immediate=True,
    )

    focal_runners = {
        "baseline": ("internal", ()),
        "precious_global": ("global", (PAIR_GROUPS["precious"],)),
        "precious_internal": ("internal", (PAIR_GROUPS["precious"],)),
        "precious_hybrid": ("hybrid", (PAIR_GROUPS["precious"],)),
        "commodity_global": ("global", (PAIR_GROUPS["commodity"],)),
        "precious_commodity_global": (
            "global",
            combinations["precious_commodity"],
        ),
        "all_global": ("global", all_groups),
        "all_internal": ("internal", all_groups),
    }
    costs: dict[str, Any] = {}
    for multiplier in (1.0, 2.0, 3.0):
        rows: dict[str, Any] = {}
        for name, (mode, groups) in focal_runners.items():
            if name == "baseline":
                result = run_pair_strategy(
                    data, PairFactorParams.disabled(), cost_multiplier=multiplier
                )
            elif mode == "global":
                result = run_strategy(data, global_params(groups), cost_multiplier=multiplier)
            elif mode == "hybrid":
                result = run_pair_strategy(
                    data,
                    precious_hybrid_params,
                    cost_multiplier=multiplier,
                    pair_groups=groups,
                    global_confirmation_immediate=True,
                )
            else:
                result = run_pair_strategy(
                    data, slow, cost_multiplier=multiplier, pair_groups=groups
                )
            rows[name] = compact(result)
        costs[f"{multiplier:.0f}x"] = rows

    global_sensitivity: dict[str, Any] = {}
    sensitivity_scopes = {
        "precious": (PAIR_GROUPS["precious"],),
        "commodity": (PAIR_GROUPS["commodity"],),
        "precious_commodity": combinations["precious_commodity"],
    }
    for scope_name, groups in sensitivity_scopes.items():
        rows: dict[str, Any] = {}
        base_params = global_params(groups)
        for gap in (0.0, 0.0025, 0.005, 0.0075, 0.010):
            for hits in (1, 2):
                params = replace(
                    base_params,
                    relative_gap=gap,
                    persistence_hits=hits,
                )
                key = f"gap{gap:.2%}_hits{hits}"
                rows[key] = compact(run_strategy(data, params))
        global_sensitivity[scope_name] = rows

    all_named = {
        "baseline": baseline,
        "precious_hybrid": precious_hybrid,
        **{f"{name}_global": result for name, result in global_results.items()},
        **{f"{name}_internal": result for name, result in internal_results.items()},
        **{f"{name}_global": result for name, result in global_combinations.items()},
        **{f"{name}_internal": result for name, result in internal_combinations.items()},
    }

    segments: dict[str, Any] = {}
    for label, start, end in (
        ("IS", "2020-06-19", "2023-12-29"),
        ("OOS", "2024-01-01", "2026-08-10"),
    ):
        segments[label] = {
            name: segment_metrics(result["equity_curve"], start, end)
            for name, result in all_named.items()
        }

    print("\nvariant                    final    CAGR Sharpe     MDD legs early rel5d")
    for name, result in all_named.items():
        metrics = result["metrics"]
        print(
            f"{name:<26} {metrics['final_value']:>9,.0f} "
            f"{metrics['cagr']:>6.1%} {metrics['sharpe']:>6.2f} "
            f"{metrics['max_drawdown']:>7.1%} {metrics['trade_legs']:>4} "
            f"{metrics['early_rotations']:>5} "
            f"{metrics['ex_post_relative_5d_avg']:>6.1%}"
        )

    print("\ncost pressure")
    for label, rows in costs.items():
        print(
            label,
            " ".join(f"{name}={row['metrics']['final_value']:,.0f}" for name, row in rows.items()),
        )

    payload = {
        "meta": {
            "strategy": "canonical server V3-G; all downsize layers disabled",
            "execution": "T-day 14:50 close approximation",
            "initial_capital": 100_000.0,
            "slow_rule": "peer 0.5*r10+0.5*r20 leads by >=0.5pp; 3-day lock",
            "signal_inputs": "T and earlier only",
            "groups": {
                name: {"label": GROUP_NAMES[name], "codes": list(pair)}
                for name, pair in PAIR_GROUPS.items()
            },
        },
        "results": {name: compact(result) for name, result in all_named.items()},
        "global_sensitivity": global_sensitivity,
        "cost_pressure": costs,
        "segments": segments,
    }
    if args.save:
        OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
        print(f"\nsaved: {OUTPUT}")


if __name__ == "__main__":
    main()
