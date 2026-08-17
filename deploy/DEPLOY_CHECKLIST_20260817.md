# 七星 V3-G/V4 实盘系统 Bug 修复部署检查清单

> **日期**: 2026-08-17
> **修复版本**: 13-bug-fix-20260817
> **影响文件**: `scripts/live_signal.py`, `scripts/trade_server.py`, `scripts/notify.py`
> **重启方式**: systemd `trade-web` 服务 (unit: `deploy/trade-web.service`, 127.0.0.1:8090, User=quant)
> **冷启动风险**: 低 (状态文件 schema 不变)
> **配套变更**: `deploy/health_check.sh` 已适配 `/api/health` 鉴权, 需一并部署

---

## 部署模式说明 (与旧流程的区别)

- 新文件 **先 scp 到 staging 目录** (`deploy/staging/20260817/`), **绝不直接覆盖 `scripts/`**
- 服务器上执行 `deploy/deploy_20260817.sh`, 由脚本原子完成:
  **校验 staging → 备份旧文件(带时间戳) → systemctl stop → 安装 → 冷启动校验 → systemctl start → 探活验证 → 写 .DEPLOYED_MANIFEST**
- 任一步失败, 脚本 **trap 自动回滚** (恢复备份 + `systemctl start trade-web` + is-active 校验)
- 服务器上 **禁止 `uv run`** (SERVER_OPERATIONS.md R2), 脚本一律使用 `/opt/quant/.venv/bin/python`
- 服务日志归 journalctl (`journalctl -u trade-web -f`), 不再有 nohup/日志重定向

---

## ⚠️ 部署前 — 服务器环境确认 (必须逐项打勾)

- [ ] **1. 服务器时区确认**: `timedatectl | grep "Time zone"` 输出应为 `Asia/Shanghai`
  - 如不是, 本次修复的 `_SH_TZ` 仍会正确工作 (脚本内时间判断也已强制 Asia/Shanghai), 但需知会
- [ ] **2. 确认无活跃待确认订单**: 服务器上
  `/opt/quant/.venv/bin/python -c "import json; print(json.load(open('data/live/state.json')).get('pending_order', {}).get('status'))"`,
  仅活跃状态 `pending` 阻塞; `confirmed`/`skipped`/`superseded` 为终态审计记录, 不影响部署
  - 如有活跃 pending, 先确认或取消, 避免新旧代码对 confirm_order 校验口径差异
  - **脚本 [0/6] 会自动检查并阻塞活跃 pending**, 此项为人工双保险
- [ ] **3. 确认非 14:30-15:00 窗口期**: 此期间 cron 可能自动运行 `live_signal.py` 生成信号
  - **脚本 [0/6] 会自动检查并阻塞** (含 15:00 整点)
- [ ] **4. 检查磁盘空间**: `df -h /opt/quant` 确保 > 100MB 可用
  - 新增 `idempotency.json` 文件, deepcopy 临时占用内存
- [ ] **5. 检查文件属主**: `scripts/*.py` 与 `data/live/` 应为 `quant:quant`
  - 脚本 [4/6] 安装时会自动 `chown quant:quant` 归一化属主 (含历史遗留 root 属主文件)
- [ ] **6. 确认服务现状**: `systemctl is-active trade-web` 应为 `active`;
  `ss -tln | grep 8090` 应只有 127.0.0.1:8090 (无 0.0.0.0:8090, 无游离 nohup 进程)

---

## 🚀 部署步骤 (必须按顺序)

### Step 1: 本地 — 上传新文件到 staging 目录

```bash
# 本地执行 (项目根目录; 服务器信息以 SERVER_OPERATIONS.md 为准)
ssh root@106.52.243.51 'mkdir -p /opt/quant/deploy/staging/20260817'
scp scripts/live_signal.py scripts/trade_server.py scripts/notify.py \
    root@106.52.243.51:/opt/quant/deploy/staging/20260817/

# 部署脚本本身 + health_check.sh 鉴权适配 (部署后健康检查才不会误报)
scp deploy/deploy_20260817.sh deploy/health_check.sh \
    root@106.52.243.51:/opt/quant/deploy/
ssh root@106.52.243.51 'chmod 0755 /opt/quant/deploy/health_check.sh'
```

### Step 2: 校验 staging 文件完整性 (hash 比对, 不用行数)

```bash
# 本地计算
sha256sum scripts/live_signal.py scripts/trade_server.py scripts/notify.py

# 服务器端计算, 两者必须逐一相同
ssh root@106.52.243.51 \
  'cd /opt/quant/deploy/staging/20260817 && sha256sum live_signal.py trade_server.py notify.py'
```

### Step 3: 服务器端执行部署脚本

```bash
ssh root@106.52.243.51
cd /opt/quant
bash deploy/deploy_20260817.sh
```

脚本自动完成 (人工只需观察输出):
- [0/6] 环境检查 (服务器护栏/root/pending_order/时间窗口)
- [1/6] staging 校验 (存在性/关键修改 grep/语法编译)
- [2/6] 备份旧文件到 `deploy/backup/<时间戳>/`
- [3/6] `systemctl stop trade-web` + 确认 8090 端口释放
- [4/6] 安装新文件 + 生产 venv 冷启动校验
- [5/6] `systemctl start trade-web` + is-active + 无 token 必须 401
- [6/6] 带 token 验证 + 更新 `deploy/.DEPLOYED_MANIFEST`

**任何一步失败 → 脚本自动回滚 (恢复备份 + 重启服务), 输出实际备份目录路径。**

---

## ✅ 部署后验证 (脚本已自动执行, 以下为人工复核)

### 验证 A: 鉴权正常 (用 -w 看状态码, 不要 grep body)

```bash
# /api/health 无 token 必须 401
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8090/api/health
# 预期输出: 401
```

带 token 复核 (token 经 stdin 传入, 不进命令行/终端):

```bash
# 服务器上执行, 从 config.json 取未过期 token 写入 mktemp 0600 临时配置
CONF=$(mktemp)
/opt/quant/.venv/bin/python - "$CONF" <<'EOF'
import json, sys, time
cfg = json.load(open('/opt/quant/data/live/config.json'))
now = time.time()
for t in cfg.get('web_tokens', []):
    if isinstance(t, dict) and t.get('expires', 0) > now:
        open(sys.argv[1], 'w').write(f'header = "Cookie: qx_token={t["token"]}"\n')
        break
EOF
curl -s -K "$CONF" -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8090/api/health   # 预期 200
curl -s -K "$CONF" -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8090/api/status   # 预期 200
curl -s -K "$CONF" -X POST http://127.0.0.1:8090/api/refresh | grep -q realtime_ok && echo "✓ refresh OK"
rm -f "$CONF"
```

### 验证 B: 服务与日志

```bash
systemctl is-active trade-web        # 预期: active
journalctl -u trade-web -n 30        # 无 traceback
ss -tln | grep 8090                  # 预期仅 127.0.0.1:8090 (不得出现 0.0.0.0)
```

### 验证 C: 幂等性文件 (等真实交易, 禁止主动触发确认)

```bash
# 不主动触发确认 — 那会向 state.json 写入假成交
# 待下一次真实调仓确认后检查:
ls -la /opt/quant/data/live/idempotency.json
/opt/quant/.venv/bin/python -m json.tool /opt/quant/data/live/idempotency.json
```

### 验证 D: Bark 推送重试 (可选)

```bash
cd /opt/quant && /opt/quant/.venv/bin/python scripts/live_signal.py --notify-test
# 应收到测试通知; 失败时观察日志是否有 10s 超时 × 3 次 + 2s→4s 退避重试
```

### 验证 E: 部署清单已更新

```bash
cat /opt/quant/deploy/.DEPLOYED_MANIFEST
# deploy_time 应为本次部署时间, git_hash 对应修复 commit
```

---

## 🔙 回滚方案

### 自动回滚 (部署中途失败)

脚本 trap 已自动执行: 从 `deploy/backup/<时间戳>/` 恢复 3 个文件 → `systemctl start trade-web` → is-active 校验。
确认输出中 "✓ trade-web 已恢复运行 (旧版本)" 即可; 若显示未恢复, 立即人工介入 (`journalctl -u trade-web -n 50`)。

### 手动回滚 (部署成功后, 次交易日发现问题)

```bash
ssh root@106.52.243.51
cd /opt/quant

# 找到本次部署的备份目录 (脚本部署完成时已打印; 也可取最新一个)
BACKUP_DIR=$(ls -dt deploy/backup/*/ | head -1)
ls "$BACKUP_DIR"   # 应含 live_signal.py trade_server.py notify.py state.json

systemctl stop trade-web
cp "$BACKUP_DIR"/{live_signal.py,trade_server.py,notify.py} scripts/
systemctl start trade-web
systemctl is-active --quiet trade-web && echo "✓ 回滚完成"

# 复核鉴权仍生效
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8090/api/health   # 预期 401
```

### 状态回滚 (仅当 state.json 被意外修改)

```bash
cp "$BACKUP_DIR/state.json" /opt/quant/data/live/state.json
```

> 注: `data/live/idempotency.json` 为本次新增, 旧代码不读取该文件, 回滚后保留无害;
> 如介意可 `rm -f data/live/idempotency.json`。

---

## 🚨 已知风险和注意事项

| 风险 | 影响 | 缓解 |
|---|---|---|
| `copy.deepcopy(_DATA_CACHE)` 内存占用 | 低 | DataFrame 约 2-5MB, deepcopy 临时加倍, 现代服务器无压力 |
| `/api/health` 加鉴权后健康检查失效 | ~~中~~ 已解决 | 本包 `deploy/health_check.sh` 已适配 (取未过期 token, 取不到则跳过不误报); **必须随本次一并部署** (Step 1) |
| 外部监控调用 `/api/health` | 中 | 如有本仓库以外的监控, 需更新为带 token 或改用其他端点 |
| `idempotency.json` 写入权限 | 低 | 首次真实确认时创建; unit `ReadWritePaths=/opt/quant/data/live` 保证可写; 失败静默降级 (OSError pass) |
| `_today_sh()` 与服务器时区绑定 | 低 | 服务器已确认 Asia/Shanghai; 如迁移时区需同步修改 `_SH_TZ` |
| `inject_realtime` 全部检查 | 中 | 首次运行若某 ETF 无今天数据, 会正确触发 fail-closed, 不会错误跳过 |

---

## 📋 部署后观察清单 (次交易日)

- [ ] **14:50 信号生成**: 观察 `data/live/cron.log`, `inject_realtime` 是否正确注入
- [ ] **调仓日确认**: 如有换仓信号, 网页确认成交是否正常写入 `state.json`; 确认后 `idempotency.json` 自然生成
- [ ] **Bark 推送**: 确认收到推送, 如失败观察是否有退避重试日志
- [ ] **trade_web 日志**: `journalctl -u trade-web --since today` 有无异常 traceback
- [ ] **健康检查**: 手动跑 `bash /opt/quant/deploy/health_check.sh` 应 7/7 通过 (API 项不再误报)
- [ ] **回撤计算**: 如持仓停牌, 观察 `account_value` 是否正确使用最后已知价

---

## 📝 变更摘要

```
本包在 3a20ac2 的 13 项修复之上, 叠加专家团审核后的 P0-P2 修复 (55f22cb):
  部署层: 全程 systemd/staging+替换前备份/trap 回滚/.DEPLOYED_MANIFEST/--server 护栏
  trade_server.py: 幂等原子写+锁、refresh 真注入、关闭 /docs、时区统一
  live_signal.py: 卖出零股口径、停牌回退防未来函数、inject 空集守卫
  notify.py: 退避定版 10s×3 + 2s→4s、4xx 不重试、失败落盘 notify_failures.log
配套运维变更:
  deploy/health_check.sh: /api/health 鉴权适配 (取未过期 web_token, 取不到跳过)
```

---

**部署人**: _________________  **日期**: _________________  **验证通过**: _________________
