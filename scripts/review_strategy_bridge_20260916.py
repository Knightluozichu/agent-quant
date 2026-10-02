"""Frozen 36-day research bridge, no personal state, network, deployment or optimization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
from scripts import live_signal as live
from scripts import run_qixing_v3 as rq
from scripts.review_oil_proxy_20260916 import prepare_bars
from scripts.review_rotation_20260916 import replay_next_open
from scripts.review_since_entry_20260916 import END, INITIAL, START, summarize_trades

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "artifacts/since_entry_20260916"
DEFAULT_OUT = ROOT / "artifacts/strategy_implementation_20260916/research/bridge_36d"
SOURCES = [
    "scripts/review_strategy_bridge_20260916.py",
    "scripts/review_rotation_20260916.py",
    "scripts/review_since_entry_20260916.py",
    "scripts/review_oil_proxy_20260916.py",
    "scripts/live_signal.py",
    "scripts/qixing_v4.py",
    "scripts/run_qixing_v3.py",
    "scripts/risk_overrides.py",
    "scripts/notify.py",
    "pyproject.toml",
    "uv.lock",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def buggy_record(state: dict[str, Any], overlay: dict[str, Any], _snapshot: dict[str, Any]) -> None:
    """Explicit invalid diagnostic: reproduce the pre-fix caller, not a strategy."""
    state["v4_state"]["candidate_history"] = overlay["history"]


def run_bridge(output: Path) -> dict[str, Any]:
    # Refuse overwrite; earlier reports are evidence, not an output scratchpad.
    output.mkdir(parents=True, exist_ok=False)
    reference = json.loads((FROZEN / "review.json").read_text())
    original = dict(rq.ETF_POOL)
    paths = {c: FROZEN / "inputs" / f"{c}.parquet" for c in [*original, rq.DEFENSE]}
    for code, path in paths.items():
        if sha256(path) != reference["input_sha256"][code]:
            raise ValueError(f"Frozen input hash mismatch: {code}")
    data = {c: prepare_bars(pd.read_parquet(p), c, END) for c, p in paths.items()}
    dates = sorted(set.intersection(*(set(frame.trade_date) for frame in data.values())))
    grid = set(dates[130 :: rq.REBALANCE_DAYS])
    assert START in dates and END in dates and str(dates[130]) == reference["grid_anchor"]
    config = {
        "start": START,
        "end": END,
        "initial_capital": INITIAL,
        "v4_params": asdict(live.v4.V4_PARAMS),
        "momentum_periods": rq.MOM_PERIODS,
        "momentum_weights": rq.MOM_WEIGHTS,
        "rebalance_days": rq.REBALANCE_DAYS,
        "grid_anchor": dates[130],
        "one_way_cost": rq.FEE + rq.SLIPPAGE,
        "cash_fraction": 0.99,
        "lot_size": 100,
        "force_entry_on_start": False,
        "early_lock_starts": "fill day, not production pending/signal day",
        "execution": "T close signal; next common observed open; final close mark",
        "initial_state": "fresh cash; no personal trades or inherited risk/confirmation state",
        "research_only": True,
        "independent_oos": False,
        "historical_deployment_replay": False,
        "august_release_manifest": "missing",
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    assert asdict(live.v4.V4_PARAMS) == reference["v4_params"]
    assert list(rq.MOM_PERIODS) == reference["momentum_periods"]
    assert list(rq.MOM_WEIGHTS) == reference["momentum_weights"]
    assert reference["single_side_cost"] == rq.FEE + rq.SLIPPAGE
    write_json(output / "config.json", config)
    write_json(output / "calendar.json", {"dates": dates, "rebalance_grid": sorted(grid)})
    bundle_paths = [ROOT / p for p in SOURCES] + list(paths.values()) + [FROZEN / "review.json"]
    before = {str(p.relative_to(ROOT)): sha256(p) for p in bundle_paths}
    for path in bundle_paths:
        dest = output / "bundle" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        assert sha256(dest) == before[str(path.relative_to(ROOT))]
    scenarios = {
        "v3g_original": ("V3-G", original, False),
        "v4_original": ("V4", original, False),
        "v4_exclude_oil": ("V4", {c: n for c, n in original.items() if c != "501018"}, False),
        "BUGGY_v4_original": ("V4", original, True),
        "BUGGY_v4_exclude_oil": ("V4", {c: n for c, n in original.items() if c != "501018"}, True),
    }
    results = {}
    prices = {c: float(frame.close.iloc[-1]) for c, frame in data.items()}
    for name, (mode, pool, buggy) in scenarios.items():
        pool_version = (
            "research-" + hashlib.sha256(",".join(sorted(pool)).encode()).hexdigest()[:12]
        )
        record_patch = (
            patch.object(live, "_record_v4_state", buggy_record) if buggy else nullcontext()
        )
        with (
            patch.object(live, "ETF_POOL", pool),
            patch.object(live, "POOL_VERSION", pool_version),
            record_patch,
        ):
            replay = replay_next_open(
                data, start=START, end=END, grid=grid, calendar=dates, mode=mode
            )
        accounting = summarize_trades(replay["trades"], prices, cost=rq.FEE + rq.SLIPPAGE)
        assert math.isclose(
            accounting["ending_equity"], replay["metrics"]["final_equity"], abs_tol=1e-7
        )
        assert replay["curve"].iloc[0].equity == INITIAL
        assert all(str(START) <= t["signal_date"] < t["date"] for t in replay["trades"])
        if "exclude_oil" in name:
            assert all(t["code"] != "501018" for t in replay["trades"])
        dest = output / name
        dest.mkdir()
        replay["curve"].to_csv(dest / "curve.csv", index=False)
        pd.DataFrame(accounting["trades"]).to_csv(dest / "trades.csv", index=False)
        for key in ("decisions", "initial_state", "final_state", "unfilled_next_open"):
            write_json(dest / f"{key}.json", replay[key])
        write_json(dest / "accounting.json", accounting)
        results[name] = {
            **replay["metrics"],
            "mode": mode,
            "pool": pool,
            "pool_version": pool_version,
            "valid_strategy": not buggy,
            "trade_legs": len(replay["trades"]),
            "estimated_cost_paid": accounting["estimated_cost_paid"],
            "early_signals": sum(d["v4_triggered"] and mode == "V4" for d in replay["decisions"]),
        }
    write_json(output / "summary.json", results)
    after = {str(p.relative_to(ROOT)): sha256(p) for p in bundle_paths}
    if before != after:
        raise RuntimeError("Source/input changed while replaying; do not use this run")
    write_json(
        output / "manifest.json",
        {
            "source_and_input_sha256": before,
            "config_sha256": sha256(output / "config.json"),
            "output_sha256": {
                str(p.relative_to(output)): sha256(p)
                for p in sorted(output.rglob("*"))
                if p.is_file()
            },
        },
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(run_bridge(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
