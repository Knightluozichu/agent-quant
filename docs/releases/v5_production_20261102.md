# 2026-11-02：V5 生产、V4 影子发布

## 授权与范围

- 用户明确要求：**服务器上 V5 为生产策略，V4 改为影子策略；涉及的文档和 Web 全部更新，并 push 到 Git。**
- 本轮只切换策略模式、信号与展示；**不连接券商、不下单、不改账户、不改仓位、不改 V4/V5 选品参数**。
- V5 仍处于研究前瞻期（未完成 252 个交易日真样本外）；本次上线是**用户明确风险偏好下的生产切换**，不是研究结论证明 V5 必然优于 V4。回滚路径必须保留。

## 策略身份

| 项 | V5（生产） | V4（影子） |
| --- | --- | --- |
| 策略名 | 七星V5 | 七星V4 |
| strategy_id | `QIXING_V5_FULL_POOL_CONSENSUS_CB_RESET_20261102` | `QIXING_V4_FULL_POOL_CONSENSUS_20260811` |
| config_hash | `29469b65f42fdc74fe9937e32c724ed71fe9f245d4f5cfdb0452ae939584bb38` | `c4c921f0c4f04a17eed17879598fd161d05265bd5db5a3f1869f29285027cf67` |
| 信号层 | V4 全池快慢共振 | 同左 |
| 风险层 | −30% 组合熔断退出后重置风险高水位 | 原 −30% 熔断（保留旧高水位） |
| 池子 | 7 个轮动候选 + 511880 防御（8 个） | 同左 |

## 代码变更

- `scripts/qixing_v5.py`：V5 纯风险适配器；只在 `ACTION_EMERGENCY` 时重置 `state["peak_equity"]`，不改目标/暴露/动作/冷却/信号。
- `scripts/live_signal.py`：
  - `get_strategy_mode` / `set_strategy_mode` 支持 `V5`。
  - V5 模式信号层按 V4 计算，风控调用 `qixing_v5.assess_v5`。
  - V5 模式下同时计算 V4 影子风控（深拷贝 state），写入 `last_decision.shadow_v4`；影子不产生订单。
  - V4↔V5 切换保留同一份 V4 信号确认历史，仅迁移 `strategy_id/config_hash`。
  - pending order 的 `strategy_id/config_hash` 改用快照身份，不再硬编码 V4。
- `scripts/trade_server.py`：服务名/版本/策略 ID/配置哈希改为当前模式；`/api/signal` 返回 `shadow_v4`。
- `scripts/web/index.html`：标题与页头改为七星V5；显示当前模式/策略 ID；信号行显示“V4 影子”。
- `deploy/trade-web.service`、`deploy/run_daily.sh`：描述改为 V5，行为不变。
- 文档：README、DECISIONS、TASKS、EXPERIMENTS、SERVER_OPERATIONS 与本发布说明。

## 验证

- `pytest tests/unit/test_live_signal.py tests/unit/test_trade_server.py tests/unit/test_qixing_v5.py` 全绿。
- V5 单测：熔断重置高水位、非熔断不动、决策字段不变、模式身份 V5。
- 腾讯/新浪全量回测：V5 1x ≈ CAGR 70%，MDD −32.45%，熔断 1 次；V4 当前 269 次触发、MDD −60% 左右。详见 `reports/2026-11-02_V5改造与腾讯数据全量回测.md`。
- 服务器部署后：`live_signal.py --dry-run` 必须显示 V5、无 `pending_order` 写入、state 哈希不变；`trade-web` active；`/api/health` 返回 `strategy_mode=V5`。
- 回滚：`strategy_mode.json` 改回 V4；恢复 `live_signal.py/trade_server.py/web/index.html` 备份；`systemctl restart trade-web`；用备份 `state.json` 复核。

## 风险与边界

- 252 日真样本外未完成；2023–2024 已被全历史回放覆盖，不再是未触碰封存样本。
- V5 的 peak reset 第二次熔断门槛约原高点 −51%，不是免费午餐；recovery gate 尚未并行验证。
- 14:50 实际成交价、LOF 溢价/流动性、5 个缺行日仍未解决；V5 历史收益不是账户实绩，不构成收益承诺。
- 本轮不修改账户/订单；V5 首日信号若要求卖出黄金/买入原油，由用户自行决定是否按网页事实登记，不得自动成交。

## 部署回执（2026-11-02 11:28–11:50 CST）

- 服务器：`root@106.52.243.51:/opt/quant`
- 同步文件 SHA256（关键）：
  - `scripts/live_signal.py` = `325e5f6fbf407d48dd2b9244f9a0bb6974e6568dbc65684ce11bb53125bec5e8`
  - `scripts/qixing_v5.py` = `26bcbbb83ed42bdce954bbe172d14cecf34add5e347e672bbc7399e527a56719`
  - `scripts/trade_server.py` = `c0390810fb9f37b9c9938e43e94b228439e4b647e778c0867e65c033e4322435`
  - `scripts/web/index.html` = `1c3b9317a2e67ccec40e142ca3222de898e2cfc1bb9707a0785063f6803d815a`
- strategy_mode：`V5`（原子写入，旧文件备份 `strategy_mode.json.bak_20261102_1145`）
- state 迁移：`v4_state.strategy_id/config_hash` 从 V4 更新为 V5；信号确认历史保留；备份 `state.json.pre_v5_20261102`
- trade-web：active；systemd Description 已更新为七星V5；/ 返回 200
- dry-run：通过。输出显示 `策略模式: V5`、目标南方原油、当前非调仓日；因 2026-10-22 pending 订单，新信号被抑制（未确认/未跳过）
- 回滚备份目录：`/opt/quant/deploy/backups/20261102_1135/`（含 live_signal/trade_server/index/state/mode/unit/run_daily）
- 2026-10-22 pending：用户确认**未成交**；2026-11-02 13:42:57 通过 `live_signal.skip_pending()` 标记为 `skipped`，未补造成交、未改 trade_log（仍 8 笔）、现金/持仓不变；跳过前备份 `state.json.pre_skip_20261102`。
- 文件属主修复：迁移后 `state.json`/`strategy_mode.json` 曾为 root:root 0600，trade-web 以 quant 运行无法读写；已恢复 `quant:quant 0600`，并验证 quant 可读取 state。
- 跳过 pending 后 dry-run：无 pending 抑制告警，V5 目标南方原油，今日非调仓日、无操作。
- 未执行：任何买卖、账户/仓位修改

## 半自动执行 Phase 1/2（2026-11-02 同日上线）

- 新增 `scripts/ticket.py`、`scripts/remind_pending.py`、`deploy/run_reminder.sh`；7 条提醒 cron：14:45 / 14:55 / 15:02 / 15:10 / 次日 09:15 / 09:25 / 09:35。
- 每个信号有双窗口 ticket、参考价/限价带、复制指令、snooze、`status_history`；`expired_unresolved` 只标记漏单，不自动跳过。
- 网页新增执行票卡和 `/api/ticket`、`/api/ticket/snooze`；确认/跳过仍只登记事实。
- 不接券商、不存密码、不下单；`ALLOW_LIVE_TRADING=false` 不变。
- 设计：`docs/design/semi_auto_execution_20261102.md`；测试：`tests/unit/test_ticket.py`、`tests/unit/test_remind_pending.py`。

## 网页策略全景：V3-G / V4 / V5 三线对比（2026-11-02）

- `/api/backtest` 从单一 V3-G 基线改为三策略同数据源、同起止日回测：V3-G 旧池基线、V4 原熔断、V5 熔断重置。
- 网页策略全景改为三线净值图 + 三线回撤 + 三策略指标表 + 年度收益表；V5 持仓时间线保留。
- 三策略口径：V3-G 用冻结旧池；V4/V5 用当前 8 标的池，次日开盘、1x 成本；V5 仅替换风险层。
- 服务器实测（2026-10-30 数据）：V3-G +1193.5% / MDD -40.7%；V4 +25.0% / -64.5%；V5 +2471.7% / -32.5%。仅历史对照，不代表未来。
