# DECISIONS.md - 决策记录

> 记录项目中的重要技术和研究决策

## 格式

```
### D-XXX: 决策标题
- **日期**: YYYY-MM-DD
- **状态**: 已决定 | 已废弃 | 讨论中
- **背景**: 为什么需要这个决策
- **决策**: 最终选择
- **理由**: 为什么选择这个方案
- **影响**: 这个决策的影响范围
```

---

### D-001: 使用 Python 3.12 作为主要语言
- **日期**: 2026-07-20
- **状态**: 已决定
- **背景**: 需要选择项目主要编程语言
- **决策**: Python 3.12
- **理由**: Apple Silicon 兼容性、稳定性、第三方数据 SDK 支持、量化生态丰富
- **影响**: 全部代码

### D-002: 使用 uv 管理依赖
- **日期**: 2026-07-20
- **状态**: 已决定
- **背景**: 需要可靠的依赖管理工具
- **决策**: uv + pyproject.toml + uv.lock
- **理由**: 速度快、锁定依赖、现代化、不维护 requirements.txt
- **影响**: 项目构建和 CI

### D-003: 第一版自建事件驱动回测器
- **日期**: 2026-07-20
- **状态**: 已决定
- **背景**: 是否使用通用回测框架
- **决策**: 自建事件驱动回测器
- **理由**: A股 T+1、涨跌停、停牌、路径歧义等特殊规则需要精确控制；需完整保存信号、计划、订单、成交和归因链路
- **影响**: backtest 模块

### D-004: 数据源通过 Provider 接口隔离
- **日期**: 2026-07-20
- **状态**: 已决定
- **背景**: 需要支持多个数据源
- **决策**: DataProvider 协议 + MockProvider 默认
- **理由**: 无密钥也能开发测试；策略代码不直接调用第三方 SDK
- **影响**: data 模块

### D-005: 9 状态市场模型
- **日期**: 2026-07-20
- **状态**: 已决定
- **背景**: 如何划分市场状态
- **决策**: 方向(上涨/横盘/下跌) × 震荡强度(低/中/高) = 9 状态
- **理由**: 覆盖第一阶段必须行情，可解释，规则模型可实现
- **影响**: regimes、strategies 模块

### D-006: 第一阶段只做多，不融资融券
- **日期**: 2026-07-20
- **状态**: 已决定
- **背景**: 确定第一阶段交易范围
- **决策**: A股与场内 ETF，只做多，不融资、不融券、不使用期货期权
- **理由**: 降低复杂度和风险，适合家庭量化起步
- **影响**: 全部策略和执行模块

### D-007: 参数版本化采用独立 ParamRegistry + Pydantic 模型
- **日期**: 2026-07-31
- **状态**: 已决定
- **背景**: P3-E2 参数版本化+回滚，需要为七星V3策略提供可复盘、可回滚的参数治理
- **决策**: 新建 `evolution/param_versioning.py`，使用 Pydantic v2 模型 (`ParamVersion`/`PromotionMetrics`/`ShadowTrade`) 与 `ParamRegistry`/`ShadowTracker` 类；注册表持久化到 `data/evolution/param_registry.json`，与既有 `EvolutionManager` (dataclass, 策略级) 并存
- **理由**: Pydantic 模型提供严格 Schema 与 JSON 序列化；参数级版本化与策略级版本化关注点不同，独立模块职责清晰；晋升门槛保守 (≥30 笔交易、OOS Sharpe CI 下界 > champion、成本压力测试、回撤不劣化、人工批准)，符合"绝不把一次盈利当作因子有效证据"原则
- **影响**: evolution 模块；P10-007/P10-009/P10-012

### D-008: 七星V3 策略参数提取为 YAML + Pydantic 校验 (架构统一)
- **日期**: 2026-07-31
- **状态**: 已决定
- **背景**: scripts/run_qixing_v3.py 以模块级常量硬编码全部策略参数，无法被包内代码复用、校验或版本化，阻碍 P3-E2 参数治理与后续迁移
- **决策**: P3-E1 架构统一。新建 `config/strategy_params.yaml` 作为参数单一事实来源；新建 `src/a_share_quant/strategies/params.py` 用 Pydantic v2 `StrategyParams` (含嵌套子模型与 field/model 校验器) 加载并校验；新建 `src/a_share_quant/strategies/qixing_v3.py` 桥接模块，通过 importlib 按路径加载脚本并重新导出函数与常量，支持渐进迁移
- **理由**: 配置与实现分离，无约束 dict 不进入核心接口 (符合 AGENTS.md)；Pydantic 校验在加载期拦截非法参数 (负费率、权重不和为1等)；桥接模式不破坏现有脚本，可平滑迁移；不修改 scripts/run_qixing_v3.py 保证回测可复现性
- **影响**: strategies 模块；为 P3-E2 参数版本化提供输入；后续任务可将脚本函数改造为接受 StrategyParams 参数

### D-009: 风控告警模块为纯建议, 绝不自动交易 (P3-E4)
- **日期**: 2026-07-31
- **状态**: 已决定
- **背景**: FIX_PLAN E4 要求实现风控告警, 但 Phase 14 自动交易当前未授权 (ALLOW_LIVE_TRADING=false)。需要在不自动执行交易的前提下提供风控监控能力
- **决策**: 实现 `src/a_share_quant/risk/alerts.py` (RiskAlertLevel/RiskAlert/RiskMonitor) 和 `risk/checker.py` (RiskChecker)。四项检查: 日内回撤>5% (WARNING+Bark)、周回撤>3% (WARNING+减仓建议)、月回撤>8% (CRITICAL+全切现金建议)、单标的集中度<80% (INFO)。所有建议字符串包含 "此为建议, 需人工确认后手动执行"。Bark 推送委托 scripts/notify.py, notify 不可用时静默降级
- **理由**: 符合 FIX_PLAN "删除任何未单独授权的自动实盘交易设计" 硬约束；告警与建议分离于执行层, 可被 cron/web 调用但不触达券商接口；pending_action 字段为机器可读标签, 供未来独立授权后可选接入
- **影响**: risk 模块；可被 trade_server /api/status 或 cron 调用；为未来独立授权的自动交易提供前置风控判断输入

### D-010: 盈亏归因引擎采用独立 engine.py + Pydantic 模型 (P3-E3)
- **日期**: 2026-07-31
- **状态**: 已决定
- **背景**: 七星V3 策略需要 P&L 归因能力 (Phase 9), 原有 `attribution/__init__.py` 中已有 ResearchAttributionEngine (交易级通用归因), 但缺乏针对七星V3 动量因子分解、择时贡献、成本拖累和 MFE/MAE 的专用引擎
- **决策**: 创建 `src/a_share_quant/attribution/engine.py`, 包含新的 `AttributionEngine` (analyze 方法)、`AttributionReport`/`TradeAttribution` Pydantic 模型和 `generate_html_report` Jinja2 渲染。将原有 `__init__.py` 中的 `AttributionEngine` 重命名为 `ResearchAttributionEngine`、`TradeAttribution` 重命名为 `ResearchTradeAttribution`, 避免命名冲突。`__init__.py` 重新导出新引擎类作为包级 API
- **理由**: 七星V3 归因需求 (10d/20d 动量分解、rebalance vs buy-and-hold 择时、fee+slippage 成本、MFE/MAE) 与原有通用交易归因 (market/sector/factor/timing/exit 多维分解) 关注点不同, 独立引擎更清晰；Pydantic 模型提供结构化输出和 JSON 序列化, 便于报告生成和 API 返回；重命名而非覆盖保留原有测试不破坏
- **影响**: attribution 模块；tests/unit/test_attribution.py (13 项测试) + test_research_attribution.py 更新导入；覆盖 Phase 9 的 P9-003/004/006/007/008/011

## 2026-08-10: V3-G 门控版暴跌过滤上线

**背景**: DROP_FILTER（近5日单日跌>3%排除候选）不分真假跌，震荡上涨行情中49%被甩是误杀（后5日反弹），但被甩品种平均继续跌-1.81%，过滤整体净效果为正。

**方案**: 门控放行（ret60<0.01 放行平淡品种假摔）+ 缓冲豁免（放行持仓必须保持第一）+ H3止损（放行后缓跌>2%降仓0.3），三者正交互补。

**关键数据**: 
- 全周期 +10.0%（3,141,899），夏普 2.42，回撤 -20.0%（比基线收窄1.1pp）
- 分歧14次（样本量足够），参数扰动6/6全过，成本2x/3x全过
- 置换测试92%分位（11/12随机比真实差，1个异常值拉低分位）
- 调参过程验证了"参数全网格必须覆盖全部7个可调参数"的经验教训
- 跨池验证：换池后+9.3%→0%或-13.2%，证明机制是资产特定+环境特定的

**结论**: 证据强度"中等偏强"，用户决策接受上线。
