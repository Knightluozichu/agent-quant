"""Freeze the latest V3-G full-pool fast/slow research under stable names.

The archive is built from the completed full replay result.  It preserves the
three variants requested for comparison, validates their exact parameters,
and records source/code hashes so a reused name cannot silently drift.

Usage: uv run python scripts/archive_v3g_latest_research.py
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).parent.parent
SOURCE = PROJECT_ROOT / "data" / "v9_results" / "v3g_full_pool_fast_slow.json"
OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "qixing_results"
    / "v3g_latest_research_20260811.json"
)

EXPECTED_PARAMS: dict[str, dict[str, Any]] = {
    "baseline": {
        "enabled": False,
        "mode": "slow",
        "slow_gap": 0.005,
        "fast_5d_gap": 0.015,
        "fast_3d_gap": 0.0075,
        "minimum_target_momentum": 0.0,
        "minimum_hold_days": 3,
        "confirmation_hits": 1,
        "confirmation_window": 2,
    },
    "slow_gap1_confirm2": {
        "enabled": True,
        "mode": "slow",
        "slow_gap": 0.01,
        "fast_5d_gap": 0.015,
        "fast_3d_gap": 0.0075,
        "minimum_target_momentum": 0.0,
        "minimum_hold_days": 3,
        "confirmation_hits": 2,
        "confirmation_window": 2,
    },
    "consensus_strict": {
        "enabled": True,
        "mode": "consensus",
        "slow_gap": 0.0075,
        "fast_5d_gap": 0.0225,
        "fast_3d_gap": 0.01125,
        "minimum_target_momentum": 0.0,
        "minimum_hold_days": 3,
        "confirmation_hits": 2,
        "confirmation_window": 2,
    },
}

VARIANT_SPECS = {
    "V3-G_BASELINE": {
        "source_key": "baseline",
        "display_name": "原始 V3-G",
        "lineage": "canonical server V3-G",
        "classification": "production_reference",
        "definition": "服务器 V3-G 原路径；5 日网格，关闭全部降仓层。",
        "verdict": "当前生产参考；本次归档不修改服务器策略。",
    },
    "V3-G_FULL_POOL_SLOW_1PP_CONFIRM2": {
        "source_key": "slow_gap1_confirm2",
        "display_name": "全池慢动量领先 1pp、2 日确认",
        "lineage": "V3-G + full-pool slow momentum handoff",
        "classification": "research_only_not_approved",
        "definition": (
            "非网格日仅当全池慢动量 Top1 相对持仓领先至少 1pp，且连续 2 日确认时换仓；"
            "慢动量=0.5*r10+0.5*r20，最短持有 3 日。"
        ),
        "verdict": (
            "1x 终值提高但夏普下降、回撤扩大；2x 相对基线无实质优势，3x 成本压力下失效。"
        ),
    },
    "V4": {
        "source_key": "consensus_strict",
        "display_name": "V4（V3-G + 全池严格快慢共振）",
        "lineage": "V3-G + full-pool strict fast/slow consensus",
        "classification": "research_only_not_approved",
        "definition": (
            "非网格日要求同一资产同时为全池快、慢 Top1；慢领先 0.75pp，"
            "5 日快动量领先 2.25pp，3 日快动量领先 1.125pp，连续 2 日确认，"
            "最短持有 3 日。"
        ),
        "verdict": (
            "1x 表现最好，但样本内几乎无增量且回撤扩大；优势集中于 2024-2026，"
            "3x 成本压力下出现非线性风险反馈并失效。"
        ),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source(source: dict[str, Any]) -> None:
    meta = source.get("meta", {})
    if meta.get("strategy") != "canonical server V3-G; all downsize layers disabled":
        raise ValueError("source is not the canonical server V3-G replay")
    if meta.get("execution") != "T-day 14:50 close approximation":
        raise ValueError("source execution drift: expected T-day 14:50")
    if meta.get("initial_capital") != 100_000.0:
        raise ValueError("source capital drift: expected 100000")
    if meta.get("signal_inputs") != "T and earlier only":
        raise ValueError("source signal timing is not lookahead-safe")

    named_variants = source.get("named_variants", {})
    for source_key, expected in EXPECTED_PARAMS.items():
        actual = named_variants.get(source_key, {}).get("params")
        if actual != expected:
            raise ValueError(
                f"parameter drift for {source_key}: expected {expected!r}, got {actual!r}"
            )


def build_archive(
    source: dict[str, Any],
    *,
    source_sha256: str,
    artifact_hashes: dict[str, str],
    archived_at: str,
) -> dict[str, Any]:
    """Build a deterministic archive payload after validating frozen parameters."""
    _validate_source(source)
    variants: dict[str, Any] = {}

    for archive_name, spec in VARIANT_SPECS.items():
        source_key = spec["source_key"]
        result = source["named_variants"][source_key]
        variants[archive_name] = {
            "display_name": spec["display_name"],
            "lineage": spec["lineage"],
            "source_key": source_key,
            "classification": spec["classification"],
            "definition": spec["definition"],
            "params": copy.deepcopy(result["params"]),
            "metrics": copy.deepcopy(result["metrics"]),
            "yearly": copy.deepcopy(result["yearly"]),
            "segments": {
                segment: copy.deepcopy(source["segments"][segment][source_key])
                for segment in ("IS", "OOS")
            },
            "cost_pressure": {
                multiplier: copy.deepcopy(
                    source["cost_pressure"][multiplier][source_key]["metrics"]
                )
                for multiplier in ("1x", "2x", "3x")
            },
            "verdict": spec["verdict"],
        }

    baseline = variants["V3-G_BASELINE"]
    slow = variants["V3-G_FULL_POOL_SLOW_1PP_CONFIRM2"]
    strict = variants["V4"]
    return {
        "archive_id": "V3G-LATEST-20260811-FULL-POOL-FAST-SLOW",
        "archived_at": archived_at,
        "research_as_of": baseline["metrics"]["end"],
        "scope": {
            "initial_capital": source["meta"]["initial_capital"],
            "execution": source["meta"]["execution"],
            "signal_inputs": source["meta"]["signal_inputs"],
            "server_identity": (
                "scripts/live_signal.py -> scripts/run_qixing_v3.py::select_target -> "
                "scripts/risk_overrides.py::assess"
            ),
            "downsize_layers": "disabled",
        },
        "source": {
            "path": str(SOURCE.relative_to(PROJECT_ROOT)),
            "sha256": source_sha256,
            "experiment_script": "scripts/exp_v3g_full_pool_fast_slow.py",
            "reproduce": "uv run python scripts/exp_v3g_full_pool_fast_slow.py",
            "artifact_hashes": artifact_hashes,
        },
        "production_status": {
            "active_variant": "V3-G_BASELINE",
            "changed_by_this_archive": False,
            "research_variants_deployed": False,
        },
        "variants": variants,
        "research_conclusions": [
            (
                "严格快慢共振 1x 终值为 "
                f"{strict['metrics']['final_value']:.2f}，高于基线 "
                f"{baseline['metrics']['final_value']:.2f}；但 IS 终值仅 "
                f"{strict['segments']['IS']['final_value']:.2f} 对 "
                f"{baseline['segments']['IS']['final_value']:.2f}，优势主要来自 OOS 近期阶段。"
            ),
            (
                "慢动量领先 1pp、2 日确认的 1x 终值为 "
                f"{slow['metrics']['final_value']:.2f}，但夏普 "
                f"{slow['metrics']['sharpe']:.4f} 低于基线 "
                f"{baseline['metrics']['sharpe']:.4f}，最大回撤也更深。"
            ),
            (
                "两套全池覆盖层在 3x 成本压力下均发生策略路径/风控反馈失稳；"
                "该压力结果不是线性手续费扣减，不能据 1x 终值批准上线。"
            ),
            "归档动作不改策略代码、不触发实盘订单；当前生产参考仍是原始 V3-G。",
        ],
    }


def main() -> None:
    source = json.loads(SOURCE.read_text())
    artifact_paths = (
        SOURCE,
        PROJECT_ROOT / "scripts" / "exp_v3g_full_pool_fast_slow.py",
        PROJECT_ROOT / "scripts" / "live_signal.py",
        PROJECT_ROOT / "scripts" / "run_qixing_v3.py",
        PROJECT_ROOT / "scripts" / "risk_overrides.py",
    )
    artifact_hashes = {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in artifact_paths
    }
    archive = build_archive(
        source,
        source_sha256=sha256_file(SOURCE),
        artifact_hashes=artifact_hashes,
        archived_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(archive, indent=2, ensure_ascii=False) + "\n")
    print(f"Archived {len(archive['variants'])} variants to {OUTPUT}")


if __name__ == "__main__":
    main()
