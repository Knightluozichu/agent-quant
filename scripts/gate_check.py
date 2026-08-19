"""上线闸门 (gate check): 读过拟合审计 JSON, 输出 allow/block 决定.

把 "审计不过则不得上实盘" 从文字纪律变成可执行检查 (I-V4A-02)。
部署 checklist / 复审流程中调用:

  python scripts/gate_check.py data/v9_results/v4_overfit_audit.json
  python scripts/gate_check.py audit.json --min-oos-days 252 --json

退出码:
  0 = allow    (全部阻断规则通过, 可上/维持实盘)
  1 = block    (任一阻断规则失败, 只允许影子模式 shadow)
  2 = 输入无效 (文件缺失/JSON 损坏/关键字段缺失)

阻断规则 (任一失败即 block):
  R1 assessment.statistical_increment_pass 必须为 true
  R2 assessment.classification 不得为 high_overfit_risk
  R3 White 现实检验 p 值 (block=20) < 0.05
  R4 移动块自助法 (block=20) 年化相对收益 95% CI 下界 > 0
  R5 meta.live_oos_observations >= --min-oos-days (默认 252, 对应 I-V4-05 复审纪律)

告警 (不阻断, 但必须展示):
  W1 meta.trial_count_is_lower_bound 为 true → 真实试验次数大于审计口径, p 值系统性偏乐观
  W2 post_2024_share_of_total_relative_log_return > 0.8 → 超额集中于单一 regime, 不外推

设计约束: 仅标准库, 服务器生产 venv (精简集) 可直接运行。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

WHITE_BLOCK = "20"  # 与审计主口径一致 (block=20)
P_THRESHOLD = 0.05
CONCENTRATION_WARN = 0.8


@dataclass
class RuleResult:
    rule: str
    passed: bool
    blocking: bool
    detail: str


@dataclass
class GateResult:
    verdict: str  # "allow" | "block"
    rules: list[RuleResult] = field(default_factory=list)

    @property
    def failed_blocking(self) -> list[RuleResult]:
        return [r for r in self.rules if r.blocking and not r.passed]

    @property
    def warnings(self) -> list[RuleResult]:
        return [r for r in self.rules if not r.blocking and not r.passed]


class AuditInputError(Exception):
    """审计 JSON 缺失关键字段, 无法裁决 (fail-closed 的输入侧)."""


def _require(obj: dict[str, Any], path: str) -> Any:
    """按 'a/b/c' 路径取嵌套字段, 缺失即 AuditInputError."""
    cur: Any = obj
    for part in path.split("/"):
        if not isinstance(cur, dict) or part not in cur:
            raise AuditInputError(f"缺少字段: {path}")
        cur = cur[part]
    return cur


def evaluate(audit: dict[str, Any], min_oos_days: int = 252) -> GateResult:
    """对单份审计 JSON 做闸门裁决. 纯函数, 可直接单测."""
    rules: list[RuleResult] = []

    # R1: 统计增量检验
    stat_pass = _require(audit, "assessment/statistical_increment_pass")
    rules.append(
        RuleResult(
            "R1 statistical_increment_pass",
            bool(stat_pass),
            True,
            f"statistical_increment_pass={stat_pass}",
        )
    )

    # R2: 综合分级
    classification = _require(audit, "assessment/classification")
    rules.append(
        RuleResult(
            "R2 classification != high_overfit_risk",
            classification != "high_overfit_risk",
            True,
            f"classification={classification!r}",
        )
    )

    # R3: White 现实检验
    white_p = float(_require(audit, f"incremental/white_reality_check/{WHITE_BLOCK}/p_value"))
    rules.append(
        RuleResult(
            "R3 white_reality_check p < 0.05",
            white_p < P_THRESHOLD,
            True,
            f"p={white_p:.4f} (block={WHITE_BLOCK}, 阈值 {P_THRESHOLD})",
        )
    )

    # R4: 自助法 CI 下界
    ci95 = _require(audit, f"incremental/moving_block_bootstrap/{WHITE_BLOCK}/ci95")
    ci_lo = float(ci95[0])
    rules.append(
        RuleResult(
            "R4 bootstrap CI95 下界 > 0",
            ci_lo > 0.0,
            True,
            f"ci95=[{ci_lo:.4f}, {float(ci95[1]):.4f}] (block={WHITE_BLOCK})",
        )
    )

    # R5: 真实样本外天数
    oos = int(_require(audit, "meta/live_oos_observations"))
    rules.append(
        RuleResult(
            f"R5 live_oos_observations >= {min_oos_days}",
            oos >= min_oos_days,
            True,
            f"live_oos_observations={oos}",
        )
    )

    # W1: 试验计数仅为下界 → 通过了的 p 值仍偏乐观
    lower_bound = audit.get("meta", {}).get("trial_count_is_lower_bound")
    if lower_bound is not None:
        rules.append(
            RuleResult(
                "W1 试验计数完整性",
                not bool(lower_bound),
                False,
                f"trial_count_is_lower_bound={lower_bound} (真实试验更多, p 值偏乐观)",
            )
        )

    # W2: 超额集中于单一 regime
    share = (
        audit.get("incremental", {})
        .get("annual", {})
        .get("post_2024_share_of_total_relative_log_return")
    )
    if share is not None:
        rules.append(
            RuleResult(
                "W2 超额 regime 集中度",
                float(share) <= CONCENTRATION_WARN,
                False,
                f"post_2024_share={float(share):.3f} (告警线 {CONCENTRATION_WARN})",
            )
        )

    verdict = "allow" if not [r for r in rules if r.blocking and not r.passed] else "block"
    return GateResult(verdict=verdict, rules=rules)


def _render_text(path: Path, result: GateResult) -> str:
    lines = [f"闸门裁决: {result.verdict.upper()}  (审计文件: {path})", ""]
    for r in result.rules:
        if r.blocking:
            mark = "✅" if r.passed else "❌"
            lines.append(f"  {mark} [{r.rule}] {r.detail}")
    if result.warnings:
        lines.append("")
        lines.append("  告警 (不阻断):")
        for r in result.warnings:
            lines.append(f"  ⚠️  [{r.rule}] {r.detail}")
    lines.append("")
    if result.verdict == "block":
        lines.append("结论: 审计未通过 → 只允许影子模式 (shadow), 不得上/维持实盘仓位。")
    else:
        lines.append("结论: 审计通过。注意告警项仍需在决策记录中显式回应。")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="上线闸门: 审计不过则只允许影子模式")
    parser.add_argument("audit_json", type=Path, help="过拟合审计结果 JSON 路径")
    parser.add_argument(
        "--min-oos-days", type=int, default=252, help="R5 最低真实样本外交易日 (默认 252)"
    )
    parser.add_argument("--json", action="store_true", help="机器可读 JSON 输出")
    args = parser.parse_args(argv)

    if not args.audit_json.is_file():
        print(f"❌ 审计文件不存在: {args.audit_json}", file=sys.stderr)
        return 2
    try:
        audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
        result = evaluate(audit, min_oos_days=args.min_oos_days)
    except (json.JSONDecodeError, AuditInputError, TypeError, ValueError, IndexError) as exc:
        print(f"❌ 审计输入无效 (fail-closed, 视同 block): {exc}", file=sys.stderr)
        return 2

    if args.json:
        payload = {
            "audit_file": str(args.audit_json),
            "verdict": result.verdict,
            "rules": [
                {
                    "rule": r.rule,
                    "passed": r.passed,
                    "blocking": r.blocking,
                    "detail": r.detail,
                }
                for r in result.rules
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(_render_text(args.audit_json, result))
    return 0 if result.verdict == "allow" else 1


if __name__ == "__main__":
    raise SystemExit(main())
