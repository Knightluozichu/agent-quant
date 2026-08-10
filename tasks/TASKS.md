# TASKS.md - 任务账本

> 状态: `[ ]` 待办 | `[x]` 完成 | `[!]` 阻塞 | `[-]` 取消

## Phase 0: 仓库、环境和任务系统

- [x] P0-001 初始化 Git 和 uv 项目
- [x] P0-002 固定 Python 3.12
- [x] P0-003 建立目录结构
- [x] P0-004 创建 .env.example
- [x] P0-005 创建 Settings (pydantic-settings)
- [x] P0-006 创建 quant doctor CLI
- [x] P0-007 配置 Ruff、mypy、pytest
- [x] P0-008 配置 CI
- [x] P0-009 创建 AGENTS.md
- [x] P0-010 创建 TASKS、DECISIONS、ISSUES、EXPERIMENTS
- [x] P0-011 编写 README 快速启动
- [x] P0-012 添加 secrets 检查测试

## Phase 1: 领域模型与配置

- [x] P1-001 定义时间、标的和货币类型
- [x] P1-002 定义 MarketState
- [x] P1-003 定义 StockState
- [x] P1-004 定义 StrategyDecision
- [x] P1-005 定义 TradePlan
- [x] P1-006 定义 Order、Fill、Position
- [x] P1-007 定义 TradeLedger
- [x] P1-008 建立 YAML schema
- [x] P1-009 配置解析、覆盖和 hash
- [x] P1-010 添加序列化与兼容测试

## Phase 2: 数据层

- [x] P2-001 DataProvider 协议
- [x] P2-002 MockProvider
- [x] P2-003 LocalParquetProvider
- [ ] P2-004 统一代码映射
- [x] P2-005 交易日历
- [ ] P2-006 security master
- [x] P2-007 日线 OHLCV
- [ ] P2-008 复权因子
- [ ] P2-009 停复牌和涨跌停
- [ ] P2-010 指数和行业
- [ ] P2-011 历史成分
- [x] P2-012 数据质量规则
- [ ] P2-013 分层存储
- [ ] P2-014 数据 snapshot 与 manifest
- [!] P2-015 接入一个真实 Provider (阻塞: 缺少凭证)
- [!] P2-016 双源抽样校验 (阻塞: 缺少凭证)

## Phase 3: 市场规则和执行账本

- [x] P3-001 日期化费用规则
- [x] P3-002 T+1
- [x] P3-003 交易单位
- [x] P3-004 历史涨跌幅
- [x] P3-005 停牌
- [ ] P3-006 公司行为
- [x] P3-007 订单状态机
- [x] P3-008 成交模型
- [x] P3-009 账户和持仓账本
- [ ] P3-010 故障注入
- [ ] P3-011 golden 场景
- [x] P3-E1 七星V3 参数化架构统一 (config/strategy_params.yaml + pydantic StrategyParams + qixing_v3 桥接模块)
- [x] P3-E4 风控告警-非自动交易 (risk/alerts.py + risk/checker.py: RiskAlertLevel/RiskAlert/RiskMonitor/RiskChecker, 4项检查+ Bark推送+人工确认建议, 28项单元测试全通过)
- [x] P3-E3 盈亏归因引擎 (attribution/engine.py: AttributionEngine.analyze 因子贡献+择时+成本+MFE/MAE, AttributionReport/TradeAttribution Pydantic模型, generate_html_report Jinja2模板, 13项单元测试全通过)

## Phase 4: 因子和状态识别

- [x] P4-001 特征计算框架 (RegimeDetector)
- [x] P4-002 趋势特征 (MA slope)
- [x] P4-003 震荡特征 (ATR percentile)
- [x] P4-004 波动率特征
- [ ] P4-005 成交与流动性
- [ ] P4-006 市场宽度
- [ ] P4-007 因子快照
- [x] P4-008 9 状态规则 Champion
- [x] P4-009 状态概率 (confidence)
- [ ] P4-010 状态迟滞
- [ ] P4-011 状态解释
- [x] P4-012 合成行情测试

## Phase 5: 策略与完整 TradePlan

- [x] P5-001 Strategy 接口 (BaseStrategy)
- [x] P5-002 TREND_HOLD
- [x] P5-003 PULLBACK_SWING
- [x] P5-004 RANGE_MEAN_REVERSION
- [x] P5-005 CASH_DEFENSE
- [x] P5-006 BEAR_REBOUND
- [x] P5-007 策略路由 (REGIME_STRATEGY_MAP)
- [ ] P5-008 策略仲裁
- [x] P5-009 TradePlan 构建器 (StrategySignal)
- [ ] P5-010 拒绝原因
- [ ] P5-011 持仓状态机
- [x] P5-012 策略文档和测试

## Phase 6: 频率、仓位和风控

- [ ] P6-001 signal reset
- [ ] P6-002 cooldown
- [ ] P6-003 滚动次数限制
- [x] P6-004 仓位计算 (PositionSizer)
- [x] P6-005 单标的约束 (max_position_pct)
- [ ] P6-006 策略和行业约束
- [ ] P6-007 相关性聚合
- [x] P6-008 组合风险预算 (PortfolioRiskManager)
- [x] P6-009 多级熔断 (DrawdownController)
- [ ] P6-010 风险拒绝解释
- [x] P6-011 风险性质测试

## Phase 7: 事件驱动回测

- [x] P7-001 事件总线 (BacktestEngine)
- [x] P7-002 时间推进器
- [x] P7-003 数据事件 (MarketEvent)
- [x] P7-004 策略事件 (SignalEvent)
- [x] P7-005 风险事件
- [x] P7-006 订单成交事件 (OrderEvent, FillEvent)
- [x] P7-007 持仓与退出
- [x] P7-008 费用和净值
- [ ] P7-009 run manifest
- [ ] P7-010 断点和重放
- [x] P7-011 报告 (ReportGenerator)
- [x] P7-012 回测 CLI

## Phase 8: 研究协议与 Baseline

- [ ] P8-001 时间切分
- [ ] P8-002 Locked Test
- [ ] P8-003 Walk-forward
- [ ] P8-004 成本压力
- [ ] P8-005 参数扰动
- [ ] P8-006 交易随机化
- [ ] P8-007 分状态指标
- [ ] P8-008 baseline buy-and-hold
- [ ] P8-009 baseline cash
- [ ] P8-010 baseline random
- [ ] P8-011 五策略独立报告
- [ ] P8-012 组合报告

## Phase 9: 盈亏归因

- [ ] P9-001 事前信号解释
- [ ] P9-002 市场和行业归因
- [x] P9-003 因子贡献 (attribution/engine.py: 10d/20d 动量信号加权分解)
- [x] P9-004 择时贡献 (attribution/engine.py: rebalance vs buy-and-hold 超额收益)
- [ ] P9-005 退出贡献
- [x] P9-006 成本贡献 (attribution/engine.py: fees + slippage 拖累)
- [x] P9-007 失败原因码 (attribution/__init__.py: FAILURE_REASONS + analyze_failures)
- [x] P9-008 MFE/MAE (attribution/engine.py: 每笔交易最大有利/不利偏移)
- [ ] P9-009 消融框架
- [ ] P9-010 因子状态评分
- [x] P9-011 归因报告 (attribution/engine.py: generate_html_report + Jinja2模板)

## Phase 10: 自进化治理

- [x] P10-001 Model Registry (EvolutionManager)
- [x] P10-002 Champion freeze
- [x] P10-003 Challenger lineage (parent_version)
- [ ] P10-004 因子漂移
- [ ] P10-005 表现漂移
- [ ] P10-006 候选生成预算
- [x] P10-007 多重尝试登记
- [x] P10-008 晋升评分 (_calculate_score)
- [x] P10-009 Shadow
- [ ] P10-010 Canary
- [x] P10-011 promote (promote_challenger)
- [x] P10-012 rollback
- [x] P10-013 冻结 hash regression (config_hash)
- [ ] P10-014 evolution report

## Phase 11: 纸面交易

- [ ] P11-001 PaperBroker
- [ ] P11-002 每日调度
- [ ] P11-003 信号生成
- [ ] P11-004 orders.csv
- [ ] P11-005 模拟成交
- [ ] P11-006 跨日持仓
- [ ] P11-007 对账
- [ ] P11-008 重启恢复
- [ ] P11-009 告警
- [ ] P11-010 operations runbook
- [ ] P11-011 聚宽交叉验证适配器

## Phase 12: 可视化与操作台

- [ ] P12-001 市场状态页
- [ ] P12-002 今日 TradePlan
- [ ] P12-003 拒绝信号
- [ ] P12-004 持仓止损止盈
- [ ] P12-005 交易归因
- [ ] P12-006 因子状态热力表
- [ ] P12-007 Champion/Challenger
- [ ] P12-008 漂移告警
- [ ] P12-009 回撤和熔断
- [ ] P12-010 导出报告

## Phase 13: 发布候选

- [ ] P13-001 全量测试
- [ ] P13-002 安全审查
- [ ] P13-003 数据许可审查
- [ ] P13-004 文档审查
- [ ] P13-005 可复现审查
- [ ] P13-006 性能审查
- [ ] P13-007 灾难恢复演练
- [ ] P13-008 release manifest
- [ ] P13-009 v1.0 只读包
- [ ] P13-010 最终限制和免责声明

## Phase 14: 券商自动化（默认 NOT_AUTHORIZED）

> ⚠️ 本阶段默认禁止执行，需用户明确授权后启动

- [ ] P14-001 ~ P14-010 (NOT_AUTHORIZED)

## Phase V32: 尾部风控实盘落地 (2026-08-06)

### 二期 (2026-08-07): 运维治理 + 511260 启用
- [x] R1a 511260 数据补齐 (2171天至8-06, 与全池同步)
- [x] R1b 511260 接入评估: 回测验证显示防御收益 -1.2% (十年国债<城投债优先),
      本期不启用, 保持已验证版本; 511260 留作防御备选 (DEFENSE_SEQ 缺数据自动跳过)
- [x] R2 服务器 lock 治理 (SERVER_OPERATIONS.md: 禁止uv sync)
- [x] R3 risk_log 滚动 (保留最近500条)
- [x] R4 audit 月度 cron (每月1日16:00 + Bark)
- [ ] R5 快照测试扩展 (511260 防御用例) ← 批次C

> 改进版风控已通过全部验证: 全周期收益 190.6万→285.6万 (+50%), 回撤-40.7%→-21.1%,
> 夏普1.66→2.36; IS/OOS/滚动3-4/扰动±20%/成本2x3x 全过 (报告 6.15/6.16 节)

- [x] V32-001 提炼 risk_overrides.py 纯函数风控层 (ruff/mypy 全绿)
- [x] V32-002 live_signal.py 最小注入 (git diff 53行: 51插入+2删除, 零逻辑破坏)
- [x] V32-003 口径一致性抽查 (2026-02 白银段 assess 触发与回测一致)
- [x] V32-004 git commit (8fb09e4) + scp 定向部署 (cron 无需重启, trade-web 已 restart)
- [x] V32-005 服务器验证 (风控层加载 OK / --status 旧 state 兼容 / trade-web active)
- [ ] V32-006 paper_mode 试运行 ≥1月 (验收: 信号偏差<1%, 误杀率≤30%)
- [x] V32-007 audit_risk.py 误杀审计脚本 (本地测试+服务器部署, 逻辑验证通过)

## [V3-G] 门控版暴跌过滤 — 已上线 (2026-08-10)

**状态**: ✅ 已上线
**参数**: ret60_thr=0.01, drop_threshold=3%, drop_lookback=5, 豁免 on, H3(δ=2%, expo=0.3)
**全周期**: +10.0% (3,141,899 vs 基线 2,855,701), 夏普 2.42, 回撤 -20.0%
**验证**: 置换测试 92% 分位, 扰动 6/6 ✅, 成本 2x/3x ✅, 滚动窗口 3/4 收益段 ✅
**改动**: run_qixing_v3.py (check_single_day_drop + select_target), risk_overrides.py (H3层), live_signal.py (实时急跌同步 + H3状态)
**实验脚本**: 13 个 exp_*.py, 12 个结果 JSON
