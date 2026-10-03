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
- [x] P3-E5 V4 全池严格快慢共振生产化（研究/实盘共源内核、14:50官方快照、
      2日确认、3交易日锁、人工成交CAS、V3-G配置回滚）
- [x] P3-E6 V4 新仓失败保护研究（前三交易日、单日-5%/入场-10%、选择性替代、
      IS/OOS/成本/敏感性/合成压力；结论仅影子候选，未修改生产V4）
- [x] P3-E7 V4 两阶段入场回放（50%/60%首仓、持有满2日且目标/资格/慢动量/
      短趋势复核后补满；全周期/IS-OOS/成本/合成压力；全量应用不晋级）
- [x] P3-E8 V4 六年半过拟合数学审计（17 个历史候选多重试验校正、27 点参数
      邻域、CSCV/PBO、HAC/块自助法、年度与滚动集中度；判定高风险但非证伪）
- [x] P3-E9 V3-G 六年半过拟合数学审计（严格跌幅版V3对照、111次历史评估追溯、
      门控/动量分层CSCV-PBO、局部邻域与块自助法；风险高于V4增量层）
- [x] P3-E10 严格V3.5研究回放与审核（严格V3底座仅叠加冻结V4快慢共振，明确
      排除全部V3-G门控/豁免；四策略、冲击场景、成本、分段、邻域与多重检验）
- [x] P3-E11 V4 理论先行风险预算研究（冻结V4选品路径，比较波动率目标、CPPI、
      冲击刹车及组合；无未来函数回放、成本/分段/邻域/块自助法审核，仅列影子候选）
- [x] P3-E12 V4 原领涨动量失效断路器（仅在冲击后且V4不会自行退出时切换合格
      替代品或退现金；指定连续暴跌、全局MDD、成本、邻域和块自助法分层审核）
- [x] P3-E13 V4 多因子状态防护研究（绝对趋势、广度、相关集中、下行波动加速、
      领涨稳定性与冲击跨仓传递；三阶段否证、9点邻域、成本和块自助法审核）
- [x] P3-E6 服务器实盘状态只读同步与本地盘中预演（发布哈希校验、状态备份、
      原子替换、真实持仓/现金 dry-run、禁止同步秘密和写服务器）

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

## 2026-08-17: 3a20ac2 专家团审核与 P0-P2 全量修复

- [x] FIX-001 部署脚本返工: PROJECT_DIR 加 `/..`; 全程 systemctl(删 kill/nohup);
      删 `--host 0.0.0.0`; 禁 `uv run` 改 `.venv/bin/python`; staging+带时间戳备份
      在替换之前; EXIT trap 自动回滚; 服务器护栏(--server); .DEPLOYED_MANIFEST 写入;
      时间窗口修八进制/15:00 边界/TZ 死逻辑; token 走 mktemp 0600 curl -K 不上命令行
- [x] FIX-002 DEPLOY_CHECKLIST_20260817.md 按新流程重写 (sha256 双向校验替代行数,
      回滚路径通配, 删除生产触发确认教唆, scp 用户对齐 SERVER_OPERATIONS.md)
- [x] FIX-003 health_check.sh 适配 /api/health 鉴权 (config.json 取未过期 dict token,
      取不到则跳过不误报)
- [x] FIX-004 live_signal: 删卖出腿 %100 校验(零股可一次性清仓, 消除死锁);
      record_manual_trade 改 _today_sh; 停牌回退按 td 截断(防未来函数);
      inject_realtime 空集守卫; _SH_TZ 改 ZoneInfo+回退; 审计时间戳显式时区
- [x] FIX-005 trade_server: 幂等持久化原子写(tmp+fsync+replace)+idempotency.lock
      包住检查→执行→记录(锁顺序: idempotency→state); 损坏文件备份 .corrupt 重建;
      /api/refresh 真注入缓存; 关 /docs//redoc//openapi.json; 两处 date.today()
      改 _today_sh; _file_count 函数属性 hack 改模块级全局; token 校验 compare_digest
- [x] FIX-006 notify: 退避语义定版 timeout 固定 10s + sleep 2s→4s (最坏 36s);
      4xx/业务失败不重试; 5xx/网络重试; 三次耗尽写 data/live/notify_failures.log;
      import time 上移
- [x] FIX-007 测试补齐: test_live_signal +5 (零股卖出/停牌无未来函数/空集守卫/时区);
      test_trade_server 新建 12 用例 (幂等并发 4 线程/损坏文件/health 鉴权/docs 404);
      test_notify +5 (退避序列/4xx 不重试/5xx 重试/失败落盘无 key 泄漏)
- [x] FIX-008 验收: ruff check 变更文件全绿; pytest 全量 exit=0 (约 300+ 通过);
      mypy 对 live_signal/trade_server/notify 零新增错误 (消掉 _file_count 旧错误);
      bash -n 部署脚本语法通过; CI 门禁 (notify.py mypy) 通过
- [x] FIX-009 服务器首演完成 (2026-08-17 17:20, 备份 deploy/backup/20260817_172044):
      首演发现并修复 curl 状态码兜底拼接 bug (1daffdc), 二次执行全部通过;
      401 鉴权/is-active/journalctl 无异常/哈希一致/manifest 已写;
      health_check.sh 检查项 2 实测: 无未过期 token 时跳过不误报 ✓;
      数据新鲜度 ❌ 为预存在 (run_data_sync cron 21:30 未运行); 属主已归一化 quant

## 2026-08-17 (晚): I-FIX-01~05 遗留项清零

- [x] FIX-010 I-FIX-01 config 竞态: config.lock (fcntl.flock) 串行化
      require_token/login/logout/set_password 读-改-写; 锁顺序
      config.lock → idempotency.lock → state lock 全路径核对无反向获取
- [x] FIX-011 I-FIX-04 请求体限制: 纯 ASGI middleware, >64KB → 413,
      非法 Content-Length → 400; 测试 4 个新增 (并发 login 8 线程不丢 token/413/400)
- [x] FIX-012 I-FIX-02 全仓 ruff check 归零: src/ 139 个全修
      (DTZ 时区语义 15 处统一 Asia/Shanghai, pydantic 运行时注解文件保留运行时导入);
      研究脚本 per-file-ignores 放宽纯风格规则, 正确性规则 36 个逐个修复;
      tests/ 19 个修复; 全仓 ruff check = 0 errors, format --check 通过
- [x] FIX-013 I-FIX-03 确认符合设计意图, 关闭; I-FIX-05 实测 deepcopy 0.2ms, 关闭

## 2026-08-19: V4 实盘深度审查 (best-of-5 独立审查 + Verifier 合并)

- [x] REVIEW-001 5 路独立审查 qixing_v4/run_qixing_v3/live_signal/risk_overrides +
      审计 JSON + 账本, 关键论断逐条对照源文件复核 (White p=0.755 / CI 跨 0 /
      post-2024 集中度 99.1% / OOS=0 / pending 阻塞熔断 / H3 不持久化, 全部属实)
- [x] REVIEW-002 审查结论落账: ISSUES.md 新增 I-V4A-01~14
      (P0: pending 阻塞熔断、252 日冻结缺硬门禁; P1: 日历空集重置 bug、H3 持久化、
      双重扣费、网格锚定、快照三缺状态; P2: skip 锁失效/config_hash 覆盖缺口/
      拆分与分红复权/涨跌停 20cm/分品种滑点/3x 悬崖根因/数据工程)
- [x] REVIEW-003 上线闸门 scripts/gate_check.py (I-V4A-02): 纯标准库, 5 条阻断规则
      (statistical_increment_pass / classification / White p<0.05 / bootstrap CI 下界>0 /
      live OOS>=252) + 2 条告警 (试验数下界 / regime 集中度); exit 0=allow 1=block
      2=输入无效(fail-closed); 对归档 v4_overfit_audit.json 实测裁决 BLOCK
- [x] REVIEW-004 测试 tests/unit/test_gate_check.py 11 例全过 (含真实归档 JSON
      必须 block 的回归); ruff check/format 两文件全绿
- [x] REVIEW-005a 修复 I-V4A-01 (P0): pending 只抑制新交易信号不阻塞风控/熔断;
      熔断被阻塞时 timeSensitive 告警+非零退出; 抑制日不写 last_run_date 可重跑;
      新增测试 3 例, 全量回归绿, mypy 零新增 (基线 69)
- [ ] REVIEW-005b 待办: gate_check 接入部署 checklist 硬性阻断; pending 超期自动 expired

## 2026-08-20: dj.luozichu.ink 公网暴露

- [x] WEB-001 I-FIX-04 全局超时中间件 (15s→504, 纯 ASGI, 测试 +3, mypy 零新增),
      b9672e9 部署完成 (sha256 双向校验, trade-web 重启, 本地 401 正常)
- [x] WEB-002 nginx vhost dj.luozichu.ink → 127.0.0.1:8090
      (/etc/nginx/sites-available/qixing, 独立站点不动既有 vhost)
- [x] WEB-003 certbot --nginx 签发 Let's Encrypt 证书 (有效期至 2026-11-18,
      certbot.timer 自动续期 active), 80→443 跳转, 公网验证:
      https 根路径 200 / api 无 token 401 / http 301

## 2026-08-20 (午): 记账网页卡片式重构

- [x] WEB-004 scripts/web/index.html 重构: 标签页改 12 栅格卡片单页 (桌面双列/
      移动单列), 策略全景改 IntersectionObserver 滚动懒加载, 浅/深色双主题
      (prefers-color-scheme), 去 emoji 标题, 统一圆角体系; API 调用与确认/幂等
      逻辑零改动; JS 语法校验 + 真实 API 打桩截图验证 (桌面 1100px/移动 390px);
      已部署 dj.luozichu.ink (FileResponse 逐请求读盘, 免重启)

## 2026-09-16：用户清仓对账与受限品种替代复核

> 会话开始的首个历史未完成项为 P2-004；本次按用户明确要求优先处理账户与策略复核。

- [x] CASH-0916-01 核对生产 V4、服务器账本与用户 9/14 卖豆粕事实；默认 dry-run、
      幂等和原子对账脚本+测试，服务器现金 97684.45 / 零持仓，归档旧 pending，不伪造买入。
- [x] CASH-0916-02 执行服务器行情同步并逐品种验证至 9/15；抓取 513350 历史与产品信息，
      候选仅进入服务器研究目录，不覆盖生产池；四核心文件哈希一致。
- [x] CASH-0916-03 冻结参数和生产网格、统一 546 日历比较原池/删原油/替换513350，
      加入 T 收盘信号→下一公共交易日开盘、成本 1/2/3x、因果性测试；本地/服务器结果完全一致。
- [x] CASH-0916-04 服务器 9/16 dry-run 验证状态不变；9:58 三池预演、旧审计 BLOCK 复核；
      完整报告 reports/2026-09-16_账户对账与轮动复核.md 已更新。
- [x] CASH-0916-05 检查结果如实归档：474 passed / 2 基线失败 / 19 skipped；新增10测试通过，
      全仓 Ruff 通过、定向 mypy 通过。用户原有 test_live_signal.py 改动未修改。
- [ ] CASH-0916-06 后续：修复既有真实成交记账两项失败、恢复发布清单、核实交割单差额和
      513350 券商适当性；正式资产池变更及部署闸门另行决策，本次不标作已上线。

## 2026-09-16：原油替代品扩展筛选与腾讯复跑

- [x] OIL-0916-01 核验159697/561360发行人风险证据，明确中高风险/R4，不宣称账户可买；
      159930/159945未证实低风险优势，券商与测评等级缺失记为blocker。
- [x] OIL-0916-02 冻结行情和参数，对比原池/删除原油/513350/561360/159697，输出成本压力、
      相关性与基础排名；全部五方案腾讯复跑与本地精确一致，相关系数仅5.55e-17浮点差。
- [x] OIL-0916-03 专用测试12通过、全量476通过/2既有失败/19跳过，全仓Ruff及定向mypy通过；
      报告与证据已生成，生产状态与核心脚本哈希未变、trade-web active。
- [x] OIL-0916-04 最终报告、完整研究档案和账本已同步腾讯staging，80个文件SHA256校验通过。
- [ ] OIL-0916-05 券商已确认广发易淘金，仍待明确风险等级及具体代码准入提示；完成适当性确认、既有BLOCK
      门禁与发布证据问题处理后，另行决定生产准入过滤/替换方案；当前不修改生产池、不下单。

- [x] PROFILE-0916-01 持久记录用户券商广发证券/易淘金、风险等级原话“等级原油的下一级”；
      精确等级保持未知，补充docs/account_constraints.md与独立准入JSON，不改生产策略。

## 2026-09-16：用户入市以来，不含个人操作的当前策略回放

- [x] ENTRY-0916-01 核验服务器最早记录7/28、初始本金10万元和当前V4代码，冻结8品种至9/15；
      从独立空仓重跑原池及删原油对照，不读个人交易填入回测，保留固定网格与次开盘因果性。
- [x] ENTRY-0916-02 新增逐笔损益核算失败测试→最小实现，与每日权益独立核对；定向4通过，
      全量477通过/2既有失败/19跳过，Ruff通过、定向mypy通过；报告如实披露亏损及局限。
- [x] ENTRY-0916-03 腾讯staging已复跑并归档报告及账本，两方案三成本结果完全一致，
      生产代码/模式/状态哈希不变；冻结输入与归档按清单核验。

- [x] NOW-0916-01 14:30服务器同快照运行原池/删原油完整V4只读预演，保留原网格、关闭推送
      副作用；实测状态/模式哈希不变。原池501018，删原油161226；不是正式14:50指令。
      7/28至9/15回测重跑无变化；行情新鲜度、完整性、排除目标及副作用断言通过。

## 2026-09-16：网页信号可信度与账户可买池修复

- [x] WEB-0916-01 生产候选池剔除501018，历史资产目录/数据/调仓公共日历不变，不擅自加入替代品。
- [x] WEB-0916-02 快照/待确认单记录pool_version；旧日期/旧池快照只读失效，不冒充当前官方目标。
- [x] WEB-0916-03 正式排行读取持久化因子；无有效快照仅显示带日期参考排行，不把V3预览标成V4。
- [x] WEB-0916-04 新买入底层拒绝被移除资产；历史持仓仍可卖；过期网页确认在状态锁内拒绝。
- [x] WEB-0916-05 网页区分现金/货币持仓、历史原池回测/当前生产池，显示市场时段与准入/研究门禁警示。
- [x] WEB-0916-06 腾讯生产备份、部署、接口/移动布局验收并留回滚凭据；14:50新池正式信号验证通过。

## 2026-09-16：15轮跨角色苏格拉底策略审议

> 按用户要求，主持策略师、金融风控、程序员、设计师四个AI角色讨论；不调整生产策略、不交易。

- [x] DIALOG-0916-01 完成15轮交叉提问，区分事实/假设/反例，形成优化优先级及验证/否决标准。
- [x] DIALOG-0916-02 报告、讨论范围与任务/决策/问题账本归档腾讯研究目录，校验SHA256；不覆盖生产状态。
- [x] DIALOG-FOLLOW-01 先写失败测试修复池版本guard与研究回放确认状态不兼容；固定代码/数据版本复跑，勿以新缺陷解释旧报告。
- [x] DIALOG-FOLLOW-02 同窗口、同成交/成本、同初始状态比较此前版本/当前V4/删油V4；区分当时部署实绩与当前规则追溯。
- [ ] DIALOG-FOLLOW-03 经事前登记后才评估简单仓位预算/覆盖层消融；先验证同标的降仓确实执行，暂不加新因子或自动替代品。

## 2026-09-16：用户批准实施，C5进取型

- [x] IMPLEMENT-0916-01 更新账户资料为用户自报C5/进取型，原油仍禁入、风险预算未知。
- [x] IMPLEMENT-0916-02 TDD修复实际费用/退出量/加仓与部分卖出/净值重建/幂等保护及计划事实分离。
- [x] IMPLEMENT-0916-03 修复回放保存契约、36日同口径版本桥接；原参数/网格/数据冻结，明确锁起算局限。
- [x] IMPLEMENT-0916-04 Web提供空白事实输入、费用、核对确认、历史目录、C5提示；独立测试与390px布局验证。
- [x] IMPLEMENT-0916-05 腾讯隔离复跑、生产备份和代码级部署、API只读验收与档案校验；不写真实账本、不下单。
- [ ] IMPLEMENT-0916-06 用户明确损失预算/期限后才预登记仓位实验，先实现和验证同标的自动调量；本轮不擅定数字。

  验收补充：548通过/19跳过，Ruff/JS通过；mypy166项未通过。腾讯收益/曲线/交易一致，因子尾差1.11e-16已披露；17:57首次上线、18:05补充历史账本647.94元差额拦截，状态/模式SHA256不变、现金97684.45/零持仓。

## 2026-09-16：补齐V3/V3-G/V4入市以来回测

- [x] VERSIONS-0916-01 记录用户自报>10年期限/愿承受本笔全部亏损，区分意愿与核实财务能力，不改生产风险限额。
- [x] VERSIONS-0916-02 Git恢复7/28初始V3与7/31修改版，冻结源码/日期/数据/成本，比较V3/V3-G/V4原池及删油池8组。
- [x] VERSIONS-0916-03 TDD、未来扰动/账务对账/输入哈希测试；554通过19跳过，Ruff与新脚本定向mypy通过。
- [x] VERSIONS-0916-04 腾讯隔离复跑与档案生产更新；资金/模式/生产代码不变，收益曲线逐笔一致。
- [x] VERSIONS-0916-05 研究报告、证据与任务账本归档腾讯，清单哈希验收。

期限与损失意愿已补充，不再作为IMPLEMENT-0916-06的未回答问题；具体可执行仓位实验仍须预登记及先验证同标的调量，不因“可接受全损”自动完成。

## 2026-09-17：V4实时双池只读预演

- [x] LIVE-0917-01 使用腾讯14:14实时报价运行生产V4只读双池预演；绝对交易日网格核实下次计划调仓9/23。正式池领先159985，对照恢复原油池领先501018，今日均不触发；状态/源码/日线哈希保持一致，结果归档腾讯研究目录。

## 2026-10-02：真实记账修复验收与基线回归

- [x] FIX-1002-01 补齐新函数类型注解 (select_target/pending_block_reason/_execution_date)，
      mypy 从 73 回到基线 69 零新增；Ruff 全仓通过；全量单测零失败/19 跳过。
- [x] FIX-1002-02 关闭 I-V4A-05 / I-CASH-0916-04：真实成交按毛额±申报费用记账，
      整单确认以实际持仓全清为准，未报费用标 unreported，账本不可倒序补录。

## 2026-10-02：结构交易 (隔日/缠论) 预注册全量回测

- [x] STRUCT-1002-01 按 `.cursor/skills/a-share-structure-trade` 实现两套互斥模式的研究回测：
      缠论流式几何引擎 (在线包含合并/右邻定型后确认分型/确认笔只增不改, 无未来函数) +
      隔日日线代理 (突破跟随/强势回踩; 二次放量无数据不测, 飞刀默认关闭)。
- [x] STRUCT-1002-02 14 项单元测试覆盖几何正确性与因果性 (信号日不得成交/不追高/熊市关突破)；
      全量单测零失败，Ruff 通过。
- [x] STRUCT-1002-03 全量回测 1x/2x/3x 成本：H1 隔日 1x 证伪 (PF 0.966)；
      H2 缠论 1x PF 1.339 但 2x 归零，利润集中白银/原油趋势段、一买负贡献。
      处置：均不上线；缠论冻结为影子候选，隔日除非补个股+热点数据否则不重测。
      报告 reports/2026-10-02_结构交易回测.md，结果 data/qixing_results/structure_trade_20261002.json。

## 2026-10-02：V4 vs 缠论入市窗口 (7/28-9/30) 同口径对比

- [x] CMP-1002-01 同步服务器 9/30 行情快照 (8 标的 SHA256 记录), 同一输入/执行/成本口径:
      V4 生产池追溯回放 100,558.96 (+0.56%, 9 腿); 缠论 93,029.36 (-6.97%, 4 笔);
      真实账户 97,684.45 (-2.32%) 作参照。缠论亏损集中在创业板一买单笔 -8.52%,
      与全历史"一买负贡献"一致。窗口仅 2 个月, 排序不构成方法优劣证据。
      结果 artifacts/entry_compare_20261002/compare.json。

## 2026-10-03：ETF动量轮动+RSRS择时预注册回测 (H3)

- [x] RSRS-1003-01 预注册写入 EXPERIMENTS.md (参数表+证伪标准+披露)。
- [x] RSRS-1003-02 TDD: tests/unit/test_rsrs_momentum.py 覆盖 OLS 斜率正确性、
      z 分数因果性 (只用 ≤T 数据)、T信号T+1成交、滞回行为、动量选股、成本记账; 10 项全绿。
- [x] RSRS-1003-03 实现 scripts/exp_rsrs_momentum.py, 全量回测 1x/2x/3x 成本。
- [x] RSRS-1003-04 入市窗口 (7/28-9/30) 与 V4 同口径对比, 报告+JSON 归档。
- [x] RSRS-1003-05 H3 未证伪 (1x PF 1.375, 93 腿) 但被 V4 全面支配 (CAGR 16.6% vs 92.5%,
      MDD -48.5% vs -28.6%): 不上线不进影子, 归档 reports/2026-10-03_rsrs动量轮动回测.md,
      结果 data/qixing_results/rsrs_momentum_20261003.json。

## 2026-10-03：账户级熔断保护层 (H4) 预注册影子回测

- [x] CB-1003-01 预注册写入 EXPERIMENTS.md (R1/R2/R3 规则表+裁决标准+前置证据披露)。
- [x] CB-1003-02 TDD: tests/unit/test_circuit_breaker.py 覆盖三规则触发/不触发、
      冷却期屏蔽、换仓峰值重置、两种口径成交因果 (next-open T+1 / same-close 当日)、
      protector=None 基线回归、金标准复现归档 V4; 11 项全绿。
- [x] CB-1003-03 replay_next_open 与 run_full_pool_strategy 双引擎加 protector 注入点
      (默认 None 零行为变化); 口径修正: 主口径切 same-close (与实盘 14:50 执行一致),
      金标准精确复现归档 5,587,053; 全量 1x/2x/3x + 回吐事件分析完成。
- [x] CB-1003-04 H4 证伪 (1x CAGR 保留 -13%, MDD 恶化 42.6pp): 熔断与 V4 信号层根本
      冲突; 报告 reports/2026-10-03_熔断保护层回测.md, 结果 circuit_breaker_20261003.json。
