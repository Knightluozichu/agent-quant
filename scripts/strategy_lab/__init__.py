"""策略实验室 (Strategy Lab) — 轻量级策略防过拟合检验工具.

任何 ETF 轮动策略想法丢进去, 自动跑样本外验证、参数稳健性扫描、
滚动一致性、基准对比, 输出一份体检报告并给出"通过/警惕/否决"结论.

用法: uv run python scripts/run_strategy_lab.py <策略名>
"""
