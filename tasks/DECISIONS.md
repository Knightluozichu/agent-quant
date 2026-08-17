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

**执行口径补充（2026-08-11）**: 用户最终确认关闭全部降仓层。服务器正式 V3-G
保持门控放行与缓冲豁免，但 `EXPO_REDUCE=1.0`、`H3_EXPO_REDUCE=1.0`；
上文 H3 降仓 0.3 仅保留为历史研究参数，不属于当前 V3-G。

## 2026-08-11: V4 获批用于下一调仓日

- **决策**: 展示名固定为 `V4`，内部唯一标识为
  `QIXING_V4_FULL_POOL_CONSENSUS_20260811`。V4 = 当前服务器 V3-G + 全池严格
  快慢共振覆盖层，使用 2/2 交易日确认和提前轮动后 3 交易日锁。
- **授权**: 用户明确要求立即在服务器实施，并从下一个调仓日启用。
- **执行边界**: 仍为 14:50 生成官方信号、网页人工填写真实成交并确认；不连接券商、
  不自动提交真实订单，`paper_mode=false` 保持不变。
- **冷启动**: 2026-08-11 未保存完整 14:50 V4 因子快照，不事后补记确认命中；
  V4 确认历史从部署后的首个自然 14:50 运行开始。
- **回滚**: 用独立 `strategy_mode.json` 在 `V4` / `V3-G` 间切换；若 V4 订单已
  真实成交，只回滚软件口径，不自动反向交易或恢复旧持仓快照。

## 2026-08-12: 本地盘中预演以服务器实盘状态为唯一账户输入

- **决策**: 使用 `scripts/sync_server_preview.py` 只读拉取服务器 `state.json`、
  `strategy_mode.json` 和 V4 发布清单；通过策略 ID、配置哈希和四个生产核心文件哈希
  校验后，备份并原子替换本地状态，再调用 `live_signal.py --dry-run`。
- **安全边界**: 不同步服务器 `.env`/`config.json`，不写服务器，不连接券商；预演后
  校验本地状态哈希未变化。14:50 之前的盘中结果不属于官方交易信号。
- **理由**: V4 提前轮动、2/2 确认和三交易日锁依赖真实持仓、现金与状态历史；仅同步
  策略模式会导致空仓口径与服务器实盘口径分裂。

## 2026-08-17: 部署体系与实盘外围修复的关键决策

- **部署只走 systemd**: trade-web.service (quant 用户/127.0.0.1/venv/journalctl)
  是唯一服务管理入口; 部署脚本不得 nohup/kill 绕过, 不得传 --host。
  新流程: scp → deploy/staging/ → 脚本备份(时间戳)→校验→安装→is-active+401 探测
  →失败 EXIT trap 自动回滚。理由: 3a20ac2 的 kill+nohup+0.0.0.0 与既有体系冲突,
  会产生 systemd 复活抢端口的"假性部署成功"。
- **服务器禁止 uv run**: 生产一律 /opt/quant/.venv/bin/python (SERVER_OPERATIONS.md
  既有红线, 本次写入部署脚本硬性检查)。
- **幂等锁顺序**: 先 idempotency.lock 后 state lock (quant_state.lock), 全仓库
  唯一同时持两把锁的路径是 /api/confirm 与 /api/trade, 顺序固定防死锁。
- **notify 退避语义定版**: timeout 固定 10s × 3 次 + sleep 2s→4s (最坏 36s);
  4xx 与 Bark 业务失败不重试; 重试耗尽落盘 notify_failures.log 供巡检。
  理由: 14:50 调仓窗口仅 10 分钟, 96s 级阻塞不可接受; 配置错误重试无意义。
- **A 股零股规则口径**: 买入必须 100 股整数倍, 卖出允许零股一次性清仓;
  confirm_order 与 record_manual_trade 统一为该口径 (卖出腿不再做 %100 校验)。

## 2026-08-17 (晚): lint 政策与运维细节补充

- **研究脚本 lint 政策**: exp_*/run_*/archive_*/strategy_lab 等一次性研究代码经
  pyproject per-file-ignores 放宽纯风格规则 (N806/E501/ARG001/B007/SIM/RET504 等),
  正确性规则 (F841/F401/B023/F601) 全仓强制。理由: 实验结果已归档, 风格返工
  无研究价值; 正确性规则可能掩盖真实 bug 故不放宽。生产脚本不在豁免 glob 内。
- **config.lock 降级策略**: 锁文件打不开时降级无锁 + 告警 (不 500), 与幂等 guard
  的"必须持锁"不同是有意的: 幂等保护资金路径 (宁停不错), config 保护会话
  token (丢了重登录即可)。
- **deepcopy 防污染保留**: 实测 0.2ms/次 (8 标的 2.3 万行), 无性能问题, 不改设计。
