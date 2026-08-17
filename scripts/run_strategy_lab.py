"""策略实验室 CLI — 轻量级策略防过拟合检验工具.

用法:
  uv run python scripts/run_strategy_lab.py --list      # 列出可用策略
  uv run python scripts/run_strategy_lab.py v3          # 检验 V3 (内置基准, 应"通过")
  uv run python scripts/run_strategy_lab.py strawman    # 检验稻草人(反动量, 应"否决")

新增策略: 在 scripts/strategy_lab/strategies.py 实现 select 函数 + 填 hypothesis
          + 定 param_grid, 注册到 REGISTRY 即可。

5项检验: 全周期回测 / 样本外验证 / 参数稳健性 / 滚动一致性 / 基准对比。
判定: 通过(可考虑采纳) / 警惕(部分达标) / 否决(过拟合或无效)。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from strategy_lab.strategies import REGISTRY
from strategy_lab.validate import run_validation


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("--list", "-l"):
        print("  可用策略:")
        for name, s in REGISTRY.items():
            print(f"    {name:12s} — {s.hypothesis[:48]}...")
        print("\n  用法: uv run python scripts/run_strategy_lab.py <策略名>")
        return

    name = args[0]
    if name not in REGISTRY:
        print(f"  ❌ 未知策略: {name}")
        print(f"     可用: {list(REGISTRY.keys())}")
        sys.exit(1)

    run_validation(REGISTRY[name])


if __name__ == "__main__":
    main()
