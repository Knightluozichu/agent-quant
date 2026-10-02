"""Version-selection comparison under one causal execution/risk contract; research only."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import platform
import shutil
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
from scripts import live_signal as live
from scripts import run_qixing_v3 as rq
from scripts.review_oil_proxy_20260916 import prepare_bars
from scripts.review_rotation_20260916 import replay_next_open
from scripts.review_since_entry_20260916 import INITIAL, START, summarize_trades
from scripts.review_strategy_bridge_20260916 import SOURCES, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "artifacts/version_comparison_20260916"
HISTORICAL = {
    "v3": {
        "commit": "18618428fd39deabdadc4ecdddf63d3d02b9bf88",
        "file": "v3_pre_gate.py",
        "sha256": "1264e825631d43c98d6b1fe154bbe18685f0853d9f08e9d129da4cb61ac509c5",
        "description": "Pre-gate V3 selector: drop filter 5 days, MA15; no gate/buffer exemption",
    },
    "v3_entry": {
        "commit": "8bc38faf792b2cdf943f750c73f09f68848f754c",
        "file": "v3_entry_release.py",
        "sha256": "c4ca6b0c6268c71e37b56cc36102be1de117c51dab57fca19f08884360f5622d",
        "description": "July 28 initial V3 selector sensitivity; 3-day drop filter and MA20",
    },
}
SELECTOR_FUNCTIONS = {
    "calc_momentum_score",
    "check_short_momentum",
    "check_volume_spike",
    "check_single_day_drop",
    "check_a_share_weak",
    "select_target",
}


def historical_selector(path: Path, expected_hash: str, pool: dict[str, str]) -> Any:
    """Load only literal constants and selector functions from hash-pinned repository code.

    Do not import the historical module: its top-level code creates output directories.
    This is a trusted-source extractor, not a sandbox for arbitrary uploaded Python.
    """
    if sha256(path) != expected_hash:
        raise ValueError("Historical source hash mismatch")
    tree = ast.parse(path.read_text())
    namespace: dict[str, Any] = {"np": np, "__name__": "historical_selector"}
    functions: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id.isupper():
                try:
                    namespace[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
        elif isinstance(node, ast.FunctionDef) and node.name in SELECTOR_FUNCTIONS:
            functions.append(node)
    namespace["ETF_POOL"] = dict(pool)
    # Execute only hash-pinned, trusted repository selector definitions.
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return namespace["select_target"]


def run_comparison(output: Path, frozen: Path = FROZEN) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    reference = json.loads((frozen / "inputs.json").read_text())
    end = date.fromisoformat(reference["end"])
    original = dict(rq.ETF_POOL)
    paths = {c: frozen / "inputs" / f"{c}.parquet" for c in [*original, rq.DEFENSE]}
    for code, path in paths.items():
        if sha256(path) != reference["input_sha256"][code]:
            raise ValueError(f"Frozen input hash mismatch: {code}")
    data = {c: prepare_bars(pd.read_parquet(p), c, end) for c, p in paths.items()}
    dates = sorted(set.intersection(*(set(frame.trade_date) for frame in data.values())))
    assert START in dates and end in dates
    assert str(dates[130]) == reference["grid_anchor"]
    grid = set(dates[130 :: rq.REBALANCE_DAYS])
    sources = [ROOT / p for p in [*SOURCES, "scripts/review_versions_20260916.py"]]
    sources += [*paths.values(), frozen / "inputs.json"]
    sources += [frozen / "historical_sources" / spec["file"] for spec in HISTORICAL.values()]
    before = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    for path in sources:
        dest = output / "bundle" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    config = {
        "start": str(START),
        "end": str(end),
        "initial_capital": INITIAL,
        "start_basis": "Earliest ledger date, not verified first-ever investment",
        "historical_selector_sources": HISTORICAL,
        "v4_params": asdict(live.v4.V4_PARAMS),
        "execution": "T close signal; next observed open; final close mark; no terminal sale",
        "risk_contract": "Shared current risk: reductions off, -30% emergency defense retained",
        "interpretation": "Archived selectors, common execution/risk; NOT historic live returns",
        "v3g": "Current final configuration, reductions disabled per 2026-08-11 decision",
        "single_side_cost": rq.FEE + rq.SLIPPAGE,
        "initial_state": "100000 cash, no personal fills or inherited risk/confirmation state",
        "force_entry_on_start": False,
        "cash_fraction": 0.99,
        "lot_size": 100,
        "grid_anchor": str(dates[130]),
        "rebalance_days": rq.REBALANCE_DAYS,
        "early_lock_starts": "fill day; differs from production pending/signal day",
        "research_only": True,
        "independent_oos": False,
        "parameter_search": False,
        "product_eligibility_inferred": False,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    write_json(output / "config.json", config)
    write_json(output / "calendar.json", {"dates": dates, "rebalance_grid": sorted(grid)})
    results = {}
    prices = {c: float(f.close.iloc[-1]) for c, f in data.items()}
    for version in ("v3", "v3g", "v4", "v3_entry"):
        for pool_name in ("original", "exclude_oil"):
            pool = {c: n for c, n in original.items() if pool_name == "original" or c != "501018"}
            pool_version = (
                "research-" + hashlib.sha256(",".join(sorted(pool)).encode()).hexdigest()[:12]
            )
            selector_patch: AbstractContextManager[Any] = nullcontext()
            if version in HISTORICAL:
                spec = HISTORICAL[version]
                selector = historical_selector(
                    frozen / "historical_sources" / spec["file"], spec["sha256"], pool
                )
                selector_patch = patch.object(live, "select_target", selector)
            with (
                patch.object(live, "ETF_POOL", pool),
                patch.object(live, "POOL_VERSION", pool_version),
                selector_patch,
            ):
                replay = replay_next_open(
                    data,
                    start=START,
                    end=end,
                    grid=grid,
                    calendar=dates,
                    mode="V4" if version == "v4" else "V3-G",
                )
            accounting = summarize_trades(replay["trades"], prices, cost=rq.FEE + rq.SLIPPAGE)
            assert math.isclose(
                accounting["ending_equity"], replay["metrics"]["final_equity"], abs_tol=1e-7
            )
            assert replay["curve"].iloc[0].equity == INITIAL
            assert all(str(START) <= t["signal_date"] < t["date"] for t in replay["trades"])
            assert pool_name == "original" or all(t["code"] != "501018" for t in replay["trades"])
            name = f"{version}_{pool_name}"
            dest = output / name
            dest.mkdir()
            replay["curve"].to_csv(dest / "curve.csv", index=False)
            pd.DataFrame(accounting["trades"]).to_csv(dest / "trades.csv", index=False)
            for key in ("decisions", "initial_state", "final_state", "unfilled_next_open"):
                write_json(dest / f"{key}.json", replay[key])
            write_json(dest / "accounting.json", accounting)
            results[name] = {
                **replay["metrics"],
                "version": version,
                "pool": pool,
                "pool_version": pool_version,
                "pnl": accounting["total_pnl"],
                "trade_legs": len(replay["trades"]),
                "estimated_cost_paid": accounting["estimated_cost_paid"],
                "early_signals": sum(
                    d["v4_triggered"] and version == "v4" for d in replay["decisions"]
                ),
                "role": "initial-release sensitivity"
                if version == "v3_entry"
                else "main comparison",
            }
    write_json(output / "summary.json", results)
    if before != {str(p.relative_to(ROOT)): sha256(p) for p in sources}:
        raise RuntimeError("Source/input changed during replay")
    write_json(
        output / "manifest.json",
        {
            "source_and_input_sha256": before,
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
    parser.add_argument("--output", type=Path, default=FROZEN / "results")
    parser.add_argument("--frozen", type=Path, default=FROZEN)
    args = parser.parse_args()
    print(json.dumps(run_comparison(args.output, args.frozen), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
