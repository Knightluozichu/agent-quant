# FIX_PLAN.md - 工程化修复计划 (v2)

> 基于 2026-07-31 全面审计 + 八维度工程化分析 + 查漏补缺审阅
> v2 修正：5 处根本性风险纠正 + P0 补强 + 部署/回测/记账/运维/数据源修正 + 分阶段验收门

## 优先级定义

| 层级 | 含义 | 时间窗口 |
|---|---|---|
| P0 | 安全封锁 — 阻断公网暴露和秘密泄露 | 今天 |
| P1 | 回测正确性 + 交易安全 + 实时急跌保护 — 修复未来函数和记账漏洞 | 本周 |
| P2 | 生产运维 + CI — 消除静默失败和权限问题 | 两周内 |
| P3 | 工程化 + 进化治理 — 架构统一和可解释性 | 持续 |

## 三项最关键修改（不可妥协）

1. **不能把已污染的 2023–2024 当 Locked Test** — 这段数据已多次参与参数扫描，只能标记为"已污染验证集"，真正未观察的 forward/shadow 集从当前日期之后建立。
2. **数据异常必须暂停信号而不是切防御** — fail-closed：不生成新信号、保持原持仓、返回 `DATA_UNAVAILABLE`、Bark 告警。只有独立风控触发时才能切防御。
3. **删除任何未单独授权的自动实盘交易设计** — Phase 14 当前未授权，`ALLOW_LIVE_TRADING=false` 时绝不自动创建券商订单。回撤 >10% 自动切防御、Kelly 仓位等均降为告警/研究实验。

---

## P0: 安全封锁（今天）

> P0 的 S1–S6 已于 2026-07-31 执行，以下标注已完成状态 + 待补强项。

### S1: 关闭 8090 公网入口 [审计#7] — ✅ 已完成

**已完成**：
- trade-web 改监听 `127.0.0.1:8090`
- Nginx 8443 端口 HTTPS 反代（自签证书）

**待补强**：
- [ ] 腾讯云安全组关闭公网 8090 入站规则（第一动作，不等 Nginx）
- [ ] 公网正式访问不建议自签证书。无域名时优先 SSH 隧道 / VPN；有域名再用可信 CA（Let's Encrypt）
- [ ] 当前 8443 端口需在腾讯云安全组放行，或改为仅 SSH 隧道访问

### S2: 密钥文件权限收紧 [审计#8] — ✅ 已完成

**已完成**：
- `.env` / `config.json` / `state.json` → `0600`
- `data/live/` → `0700`
- `quant.key` → `0600`，`quant.crt` → `0644`
- `/root/.ssh/` → `0700`，`authorized_keys` → `0600`

### S3: 轮换全部秘密 [审计#8] — ⚠️ 部分完成

**已完成**：
- ✅ Web 密码已用 `scripts/trade_server.py --set-password` 重置（注意：原计划误写为 `live_signal.py`，已纠正）
- ✅ 旧 token 全部清除

**待补强**：
- [ ] Bark Key 轮换（用户在 Bark App 生成新 Key，更新服务器 `.env` 和 `config.json`）
- [ ] JQData 密码更改（用户在聚宽平台操作，更新服务器 `.env`）
- [ ] Tushare Token 重新生成（用户在 Tushare 平台操作，更新服务器 `.env`）
- [ ] CLI 传递密码/Key 改为安全交互输入、环境变量或 stdin（不进 shell history / 进程参数）

### S4: 修复 sync.sh [审计#9] — ✅ 已完成（合并原 O2）

**已完成**：
- 排除 `.env` / `.env.*`
- 同步后自动 `chmod +x deploy/*.sh`
- 排除 `data/stock_cache`

**待补强**：
- [ ] `rsync --chmod` 范围过宽：改为 Git 中将 cron 脚本提交为 `100755`，部署时用 `install -m 0755`
- [ ] 排除项补充：运行状态、备份文件、本地索引
- [ ] rsync 前强制 `--dry-run --checksum` 预检
- [ ] 部署后保存 `DEPLOYED_REVISION`（git hash）、代码 hash、参数 hash、依赖锁 hash
- [ ] 部署后比较本地/远端 manifest，而非只打印"同步成功"

### S5: trade-web 降权 + systemd 安全选项 [审计#7] — ⚠️ 部分完成

**已完成**：
- systemd 安全选项：`NoNewPrivileges` / `PrivateTmp` / `ProtectHome=read-only` / `ProtectSystem=full` / `ProtectKernelTunables` / `ProtectKernelModules` / `ProtectControlGroups` / `RestrictSUIDSGID` / `LockPersonality`
- 注意：`ProtectHome=yes` 会导致 venv 软链接失效（指向 `/root/.local/`），已改用 `read-only`

**待补强**：
- [ ] 创建 `quant` 用户，服务以非 root 运行
- [ ] `chown` 数据目录到 `quant` 用户
- [ ] systemd 添加 `UMask=0077`，否则新文件仍可能成为 `0644`
- [ ] 如改用 `ProtectSystem=strict`，需额外配置 `ReadWritePaths=/opt/quant/data/live`
- [ ] `Restart=always` 改为 `Restart=on-failure` + `StartLimitIntervalSec=60` + `StartLimitBurst=3`，崩溃后不无限重启

### S6: SSH 加固 + 安全更新 [审计#11] — ✅ 已完成

**已完成**：
- `PermitRootLogin prohibit-password`（仅密钥）
- `PasswordAuthentication no`
- `MaxAuthTries 3`、`X11Forwarding no`
- 修改前已验证密钥登录 + `sshd -t` + 保留现有会话
- 172 个安全更新已安装

**待补强**：
- [ ] 安全更新应拆分独立任务：快照/备份 → 升级 → 重启 → 服务和 cron 验证 → 回滚方案
- [ ] 配置 `unattended-upgrades` 仅自动安装安全更新

### S7: Token 安全增强 [审计补充] — ✅ 已完成

- [x] Token 增加过期时间（24h），服务端自动清理过期 token
- [x] 支持服务端注销（logout 接口）
- [x] 密码修改后全部 token 失效
- [x] 删除 query-string token 支持（仅允许 `Authorization: Bearer` 和 HttpOnly Cookie）
- [x] 登录接口限流：5 次/分钟，超出返回 429
- [x] 浏览器 token 改用 `Secure + HttpOnly + SameSite=Lax` cookie，不存 `localStorage`
- [x] `/api/health` 不暴露持仓、token 或凭证状态，只返回安全运行指标

### S8: 实时急跌保护修复 + 腾讯 HTTPS [审计补充→P0] — ✅ 已完成

- [x] 修复实时急跌保护使用昨收价计算的 bug（改用腾讯返回的 prev_close）
- [x] 腾讯行情接口改用 HTTPS（`https://qt.gtimg.cn`）
- [x] 生产脚本读取中央安全配置 — 降级 P3/E1（P0 关键项 HTTPS+密钥已覆盖）

---

## P1: 回测正确性 + 交易安全（本周）

### R1: 修复回测未来函数 [审计#1] ★最高优先

**问题**：当日收盘选目标 → 当日收盘价买卖 = lookahead bias

**修复**：
- 信号生成用 T 日收盘数据
- T 日信号先进入 pending execution queue（带唯一 ID + 状态版本）
- 交易执行改 T+1 日开盘价
- T+1 停牌、无开盘价、涨跌停或数据缺失时不得成交
- 卖出失败时不得继续买入另一只 ETF
- 最后一个交易日尚未执行的信号不能计入成交
- 净值曲线改为每日采样（含非调仓日）
- 重新报告所有指标

**输出要求**：
- 每笔成交记录：信号时间、执行时间、行情版本 hash、参数 hash
- 成本压力测试：基础成本、2 倍成本、3 倍成本三档

**诊断基线**（非预期结果）：当前 +2786% / -18.5% 是 lookahead bias 下的数字；修复后预计下降，具体值需实际验证，不预设"预期结果"。

### R2: 参数隔离 — 已污染验证集标记 [审计#2] ★根本性纠正

**问题**：2023–2024 数据已多次参与参数扫描（DROP_LOOKBACK=5、A_SHARE_MA=15 均在此基础上选定），不能再称为 Locked Test。

**修复**：
- 2020–2026 历史数据只用于 nested walk-forward / 稳健性估计
- 明确标记 2023–2024 为"已污染验证集"
- 从当前日期（2026-07-31）之后建立真正未观察的 forward/shadow 集
- 建立不可修改的 manifest（数据 hash + 参数 hash + 代码版本 + 尝试时间戳）
- 登记所有参数尝试（成功/失败均记录）
- `DROP_LOOKBACK=5` 和 `A_SHARE_MA=15` 需在 walk-forward 框架下重新评估，不预设结论

### R3: Walk-forward 滚动验证 [审计#2]

**固定边界**（不可含糊）：
- 6 年数据分 4 段滚动，每段边界固定写入配置
- 每段 train / validation / test 边界明确
- 明确是否 expanding window（默认 yes）
- 每段之间设置 purge（1 交易日）+ embargo（2 交易日）消除重叠
- 每段最小交易数 ≥ 20 笔，不足则该段标记不可用
- 所有失败尝试均登记到 manifest

**验证内容**：
- 报告 OOS 夏普和收益
- 参数扰动 ±20% 扫描确认稳定性
- 计算 PBO（Probability of Backtest Overfitting）

### R4: 交易日历校验 [审计#3]

**问题**：`date.today()` 不检查交易日，周末/节假日会注入假数据

**修复**：
- `inject_realtime` 前先查交易日历（akshare `tool_trade_date_hist_sina`）
- 非交易日跳过信号生成
- 防止调仓网格错位和虚假成交
- 交易日历版本写入数据 manifest

### R5: 记账接口全面校验 [审计#5]

**问题**：trade_server 可接受任意代码、负价格、重复确认

**修复**：
- 请求模型改用 Pydantic，`extra="forbid"`（拒绝未知字段）
- 拒绝 NaN、Infinity、超大金额和非法日期
- 买入校验：代码在 ETF_POOL / 数量 > 0 且满足交易单位 / 价格 > 0 且为有限数 / 现金充足 / 无其他持仓
- 卖出校验：代码匹配持仓 / 数量 ≤ 持仓 / 要么全额卖出，要么正确维护部分剩余持仓
- 手续费扣除后现金必须非负
- 确认校验：pending_order 存在且 status=pending / 请求与原单一致
- 防重复确认：确认后 status 改为 confirmed，拒绝再次确认
- pending 订单带唯一 ID 和状态版本号
- API 支持 idempotency key 防重复提交
- 手工纠错和正常成交拆成不同接口，纠错需记录原因

**账本改造**：
- 采用追加式不可变事件日志（append-only event log），而非只保存最终状态
- 每笔事件包含：时间、类型、代码、数量、价格、手续费、操作人/系统、idempotency key

**测试要求**：
- property/Hypothesis 测试验证不变量："现金永不为负"、"股数永不为负"、"事件重放结果一致"

### R6: 状态文件原子写 + 事务锁 + 备份 [审计#6]

**问题**：仅在 `save_state()` 上锁不够，两个进程可能同时读取旧状态后依次覆盖。

**修复**（完整事务流程）：
```text
获取独立 lockfile 锁 (fcntl.flock, /tmp/quant_state.lock)
  → 读取 state.json
  → 校验版本号和订单状态
  → 修改
  → 写临时文件 state.json.tmp
  → flush + fsync
  → os.replace(state.json.tmp, state.json)  # 原子替换
  → fsync 父目录
  → 解锁
```

- 锁文件不能是被 rename 替换的 `state.json` 本身
- 备份文件权限 `0600`，可恢复验证（加载校验余额/持仓一致性）
- 限制备份数量（最多 7 个）和总大小
- 版本号单调递增，并发写入时版本不匹配则拒绝

### R7: 数据缺失 fail-closed [审计#4] ★根本性纠正

**问题**：缺数据时静默缩池继续跑，原计划"缺失则返回 DEFENSE"仍会基于坏数据产生指令。

**修复**（fail-closed）：
- `select_target` 检查全部 ETF_POOL 代码是否都有当日数据
- 缺失时不生成新信号
- 保持原持仓和 pending 状态不变
- 返回明确的 `DATA_UNAVAILABLE` 状态码
- Bark 紧急告警（含缺失代码列表）
- 不允许静默缩小候选池
- 只有独立风控模块触发时才能切防御（不能由数据缺失触发）

### R8: 实时数据双通道 [审计#4] ★根本性纠正

**问题**：原计划一方面说缺失代码用昨收填充，另一方面说不允许部分注入，前后矛盾。

**修复**（区分两条通道）：

**网页展示通道**：
- 可以显示昨收，但必须标记 `stale=true`
- 前端展示 `stale` 徽章

**策略信号通道**：
- 8 只 ETF 全部有效、时间戳正确才允许计算
- 校验：行情时间、市场状态、价格是否有限数、腾讯返回日期是否为当天
- 任何一项不满足 → fail-closed（不生成信号）
- 不允许用昨收伪造实时数据进入策略

---

## P2: 生产运维 + CI（两周内）

### O1: cron 防重入 + 退出码 [审计#10]

- 三个脚本加 `flock -n /tmp/quant_daily.lock`（或分别使用任务锁，若全局串行则明确标注目的）
- 修复退出码：显式保存 `rc=$?`，完成日志后 `exit "$rc"`（不依赖可能未定义的 `$EXIT_CODE`）
- Python 层的数据错误也必须返回非零，不能只打印告警
- 日志带任务名前缀：`[daily]` / `[calibrate]` / `[sync]`

### O2: 部署权限 + manifest [审计#9]（已合并原 S4/O2）

- Git 中将 cron 脚本正式提交为 `100755`
- 部署时用 `install -m 0755` 精确设置权限（不用 `--chmod=+x` 范围过宽）
- 排除 `.env` / `.env.*` / 运行状态 / 备份 / 本地索引
- rsync 前强制 `--dry-run --checksum` 预检
- 部署后保存 `DEPLOYED_REVISION`、代码 hash、参数 hash、依赖锁 hash
- 部署后比较本地/远端 manifest，不一致则告警

### O3: 日志归档 + 告警闭环 [审计#10,12]

- 日志保留改为 30 天滚动（非 300/500 行）
- 结构化日志：时间 / 任务名 / 退出码 / 耗时 / 数据日期
- 失败时推送 Bark（含任务名和错误摘要）
- 每日健康检查脚本：`/api/health`（仅安全指标）+ 数据新鲜度 + 持仓一致性

### O4: CI 修复 [审计#14]

- JQData 测试：默认改为 `pytest -m "not integration"`，另设手工/定时凭证集成任务
- CI 使用 all-extras 时 SDK 存在但无凭证，不能仅按"SDK 是否安装"跳过
- secret 测试：不白名单整个 fixture 文件，改用不匹配真实密钥格式的测试值
- CI 在 push 时自动运行 ruff + mypy + pytest
- 覆盖率：当前只统计 `src/`，给 `scripts/` 补测试不直接提高。需先移动生产逻辑到 `src/`，或扩展 coverage source
- `mypy src` 不检查生产脚本；迁移前显式加入 `scripts/trade_server.py` 等文件

### O5: Ruff/Mypy 清零 [审计#13]

- 分两步：先调整规则（CLI print 豁免等），再清零真实问题
- 核心 4 个生产脚本优先（165 项 → 0）
- `run_factor_v61.py` 未闭合字符串修复
- 设定基线：新代码不允许引入新警告

### O6: 依赖锁固定 [审计#10]

- 本地服务器 uv 版本对齐（0.6.3 → 统一）
- `uv lock --check` 必须通过
- cron 改为直接调用 `.venv/bin/python`，避免每次 `uv run` 检查或改变环境
- `requests` 显式加入依赖
- uv 安装方式改为固定版本

### O7: 数据源交叉校验 [审计#4,8] ★修正

**问题**：不能直接比较"新浪历史收盘 vs 腾讯盘中实时价"（盘中涨跌会触发误报）。

**正确比较**：
- 腾讯 `prev_close` 对新浪上一交易日 close
- 收盘后再比较两个来源的同日 close
- QDII / LOF / 商品 ETF 设置不同阈值（非统一 0.5%）
- 来源冲突时暂停信号（fail-closed），不任意选择"备选源"

**manifest 扩展**：
- 来源、复权方式、schema 版本、代码版本、参数版本、交易日历版本

**JQData 决策**：
- 若线上不用 JQData：将 `DATA_PROVIDER` 改为真实值，移除服务器 JQ 凭证
- 若使用 JQData：安装 SDK，health check 做真实连接和额度检查

### O8: systemd 重启限流 [审计#11]

- `StartLimitIntervalSec=60` + `StartLimitBurst=3`
- 启动前检查端口占用
- 崩溃后不无限重启，推 Bark 告警

---

## P3: 工程化 + 进化治理（持续）

### E1: 架构统一 [维度1]

- `run_qixing_v3.py` 核心逻辑 → `src/a_share_quant/strategies/qixing_v3.py`
- `live_signal.py` → `src/a_share_quant/signals/live.py`
- `trade_server.py` → `src/a_share_quant/monitoring/server.py`
- 参数从硬编码常量 → YAML + pydantic 校验
- 实盘 `import a_share_quant` 而非 `sys.path.insert`
- 核心实盘脚本测试提前到 P1（见 E5 已提级）

### E2: 参数版本化 + 回滚 [维度8]

- 每次参数变更生成 `params_v{N}.yaml` + changelog
- Champion/Challenger 框架：新参数先 shadow 运行
- **shadow 周期修正**：5 日策略两周只有约两次决策，远低于 30 笔样本要求。shadow 至少覆盖 30 笔交易或 3 个月
- 晋升条件：最小样本数 ≥ 30 笔 + OOS 夏普置信区间下界 > Champion + 成本压力测试通过 + 回撤不增 + 人工审批
- 不能只看 "Sharpe 1.2x"
- 回滚：晋升后回撤超阈值自动回退

### E3: 盈亏归因引擎 [维度7]

- 因子贡献：10日/20日动量分量对收益的贡献
- 择时贡献：换仓 vs 持有不动的超额收益
- 成本归因：手续费 + 滑点拖累
- MFE/MAE：每笔交易最大盈利/亏损幅度
- 月度归因 HTML 报告自动生成

### E4: 风控告警（非自动交易）[维度5] ★根本性纠正

**授权边界**：Phase 14 当前未授权，以下均为告警/建议，不自动执行交易。

- 盘中回撤 >5% 推 Bark 告警
- 周回撤 >3% 推告警 + 生成待确认减仓建议
- 月回撤 >8% 推紧急告警 + 生成待确认全切货币建议
- 单标的敞口 <80% 告警
- `ALLOW_LIVE_TRADING=false` 时绝不自动创建券商订单
- 自动交易必须作为独立授权项目，单独评审通过后才能启用
- Kelly 仓位降为研究实验（`experiments/`），不作为常规工程修复项

### E5: 实盘脚本测试 [维度6] — 已从 P3 提级到 P1

- `select_target` 参数快照断言：固定输入 → 固定输出
- 前后一致性：同参数回测结果 hash 不变
- Golden 场景：历史关键调仓日决策不变
- 核心策略覆盖率 >80%

### E6: 策略文档对齐 [建议#1]

- 修复 `run_qixing_v3.py` 文件头注释：
  - 动量周期 20/60/120 → 10/20
  - 日频调仓 → 每5日
  - MA20 → MA15
  - 标注哪些过滤已关闭
- 修复 `SWITCH_THRESHOLD` 未使用（建议#2）
- 实时急跌保护修复 → 已提级到 P0 (S8)
- 腾讯行情改 HTTPS → 已提级到 P0 (S8)
- 生产脚本读取中央安全配置 → 已提级到 P0 (S8)

---

## 执行顺序

```
P0 安全封锁 (今天)  [S1-S6 已执行，S7-S8 待补强]
  S7 (Token安全) → S8 (急跌保护+HTTPS+中央配置)
  S1补强 (安全组/证书) → S5补强 (quant用户/UMask)
    ↓
P1 回测正确性 (本周)
  R1 (未来函数+pending queue) → R2 (已污染验证集标记)
  R3 (Walk-forward固定边界) → R4 (交易日历)
  R5 (记账Pydantic+事件日志) → R6 (事务锁+原子写)
  R7 (fail-closed) → R8 (双通道)
  E5 (实盘脚本测试) ← 从P3提级
    ↓
P2 运维+CI (两周)
  O1 (cron退出码) → O2 (部署manifest) → O3 (日志)
  O4 (CI-m "not integration") → O5 (Ruff/Mypy) → O6 (依赖)
  O7 (数据源交叉校验修正) → O8 (systemd限流)
    ↓
P3 工程化 (持续)
  E1 (架构统一) → E2 (版本化+30笔shadow)
  E3 (归因) → E4 (风控告警非自动) → E6 (文档)
```

---

## 分阶段验收门

### P0 验收
- [x] 公网 8090 不可访问（服务绑定 127.0.0.1，仅本地可达）— ⚠️ 腾讯云安全组仍需关闭公网 8090 入站
- [x] HTTPS 或 SSH 隧道可用（Nginx 8443 HTTPS + SSH 隧道 localhost:8090）
- [ ] 服务非 root 运行（当前 User=root，S5补强待创建 quant 用户）— 🔲 服务器端任务
- [x] 秘密文件均为 `0600`，新写文件 `UMask=0077` 生效（已验证）
- [x] 旧 token 全部失效（无 token 请求返回 401，已验证）
- [x] Token 有过期时间（24h TTL），密码修改后全部失效（set_password 清空 tokens）
- [x] 登录接口限流生效（5次/分钟后 429，已验证）
- [x] `/api/health` 不暴露持仓/token/凭证（仅返回 status/service/version/has_state/data_files）
- [x] Token 改用 HttpOnly+SameSite=Lax Cookie（HTTPS 时自动加 Secure，已端到端验证）
- [x] CLI 密码/Key 安全输入（stdin/getpass/env，不进 shell history）

### P1 验收
- [ ] 回测：T 日信号进 pending queue，T+1 开盘成交
- [ ] 回测：T+1 停牌/无开盘价/涨跌停时不成交
- [ ] 回测：卖出失败时不继续买入
- [ ] 回测：最后交易日未执行信号不计入成交
- [ ] 回测：每日净值（含非调仓日）
- [ ] 回测：输出信号时间、执行时间、行情版本、参数 hash
- [ ] 回测：成本压力测试（基础/2x/3x）
- [ ] Locked 数据不可读保护：OOS manifest 通过
- [ ] 2023-2024 标记为"已污染验证集"
- [ ] 周末/节假日不生成信号
- [ ] 实时源部分失败 → fail-closed，不生成信号
- [ ] 实时源全失败 → fail-closed，Bark 告警
- [ ] 旧行情（非当天）→ fail-closed
- [ ] 重复确认 → 拒绝
- [ ] 负数/NaN/Infinity/超大金额 → 拒绝
- [ ] 超额卖出 → 拒绝
- [ ] 手续费后现金为负 → 拒绝
- [ ] 状态文件并发写 → 版本冲突拒绝
- [ ] property 测试：现金/股数永不为负

### P2 验收
- [ ] 三个 cron 在最小环境中成功执行
- [ ] 故意制造失败时返回非零并触发 Bark 告警
- [ ] cron 直接调用 `.venv/bin/python`（非 `uv run`）
- [ ] 部署后本地/服务器代码 hash、参数 hash、lock hash 一致
- [ ] 有可验证的回滚版本
- [ ] CI: Ruff / 格式 / mypy / unit / 非凭证集成 / 覆盖率全部绿色
- [ ] 数据源冲突 → 暂停信号

### 恢复验收
- [ ] 从备份恢复 state.json
- [ ] 恢复后通过余额/持仓校验
- [ ] 事件日志可重放且结果一致

### 通用要求
- 每个修复任务必须遵循"先写失败测试 → 实现 → 服务器 canary"顺序
- 每个任务完成后：代码测试通过 → 同步服务器 → 提交 Git → 更新 TASKS.md / ISSUES.md → 无新 ruff/mypy 警告
