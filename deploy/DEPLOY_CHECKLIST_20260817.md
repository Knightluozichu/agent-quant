# 七星 V3-G/V4 实盘系统 Bug 修复部署检查清单

> **日期**: 2026-08-17
> **修复版本**: 13-bug-fix-20260817
> **影响文件**: `scripts/live_signal.py`, `scripts/trade_server.py`, `scripts/notify.py`
> **必须重启**: trade_server (幂等性持久化逻辑变更)
> **冷启动风险**: 低 (状态文件 schema 不变)

---

## ⚠️ 部署前 — 服务器环境确认 (必须逐项打勾)

- [ ] **1. 服务器时区确认**: `timedatectl | grep "Time zone"` 输出应为 `Asia/Shanghai`
  - 如不是，本次修复的 `_SH_TZ` 仍会正确工作，但需知会
- [ ] **2. 当前状态备份**: 部署前备份 `data/live/state.json`
  ```bash
  cp data/live/state.json data/live/state.json.bak.$(date +%Y%m%d_%H%M%S)
  ```
- [ ] **3. 当前代码备份**: 部署前备份 3 个文件
  ```bash
  mkdir -p deploy/backup/20260817
  cp scripts/live_signal.py scripts/trade_server.py scripts/notify.py deploy/backup/20260817/
  ```
- [ ] **4. 确认无待确认订单**: `uv run python scripts/live_signal.py --status` 输出中 `pending_order` 应为 `null`
  - 如有 pending，先确认或取消，避免新旧代码对 confirm_order 校验口径差异
- [ ] **5. 确认非 14:50 窗口期**: 部署时间应避开 14:30-15:00
  - 此期间 cron 可能自动运行 `live_signal.py` 生成信号
- [ ] **6. 检查磁盘空间**: `df -h .` 确保 > 100MB 可用
  - 新增 `idempotency.json` 文件，deepcopy 临时占用内存
- [ ] **7. 检查文件属主**: `data/live/` 目录应为 `quant:quant` 或运行用户可写
  - `_IDEMPOTENCY_FILE` 需要写入权限

---

## 🚀 部署步骤 (必须按顺序)

### Step 1: 停止 trade_server
```bash
# 找到 trade_server 进程并 graceful 停止
ps aux | grep trade_server
kill -TERM <PID>  # 或 kill -INT <PID>
# 确认已停止
curl http://127.0.0.1:8090/api/health  # 应无响应
```

### Step 2: 传输文件
```bash
# 本地执行 (假设已在项目根目录)
scp scripts/live_signal.py quant@server:/path/to/scripts/
scp scripts/trade_server.py quant@server:/path/to/scripts/
scp scripts/notify.py quant@server:/path/to/scripts/
```

### Step 3: 服务器端校验文件完整性
```bash
cd /path/to/project
# 校验行数 (应大致匹配)
wc -l scripts/live_signal.py scripts/trade_server.py scripts/notify.py
# 预期: live_signal ~1897, trade_server ~783, notify.py ~181

# 校验关键修改存在
grep -q "_today_sh()" scripts/live_signal.py && echo "✓ _today_sh 存在"
grep -q "copy.deepcopy" scripts/trade_server.py && echo "✓ deepcopy 存在"
grep -q "_load_idempotency" scripts/trade_server.py && echo "✓ 幂等持久化存在"
grep -q "max_retries = 3" scripts/notify.py && echo "✓ Bark 重试存在"
```

### Step 4: 验证配置文件未变
```bash
# 确认 web_tokens 和 bark_key 仍在
python3 -c "import json; c=json.load(open('data/live/config.json')); print('web_tokens' in c, 'bark_key' in c)"
# 应输出: True True
```

### Step 5: 冷启动验证 (不连接生产状态)
```bash
# 测试 live_signal.py 能否正常加载 (dry-run)
uv run python scripts/live_signal.py --dry-run
# 预期: 正常输出今日状态，无 traceback

# 测试 trade_server 能否正常导入
uv run python -c "import scripts.trade_server as ts; print('trade_server import OK')"
```

### Step 6: 启动 trade_server
```bash
nohup uv run python scripts/trade_server.py --port 8090 --host 0.0.0.0 > logs/trade_server.log 2>&1 &
sleep 2
curl -s http://127.0.0.1:8090/api/health  # 应返回 401 (已加鉴权)
```

---

## ✅ 部署后验证 (必须逐项通过)

### 验证 A: 鉴权正常
```bash
# /api/health 无 token 应 401
curl -s http://127.0.0.1:8090/api/health | grep -q "401" && echo "✓ health 鉴权生效"

# /api/health 有 token 应正常
TOKEN=$(cat data/live/config.json | python3 -c "import sys,json; print(json.load(sys.stdin)['web_tokens'][0])")
curl -s -H "Cookie: qx_token=$TOKEN" http://127.0.0.1:8090/api/health | python3 -m json.tool
# 应返回 JSON 含 status/strategy_mode 等
```

### 验证 B: 网页状态正常
```bash
# 登录后访问 /api/status，确认有 used_realtime 标记
# 浏览器或 curl:
curl -s -H "Cookie: qx_token=$TOKEN" http://127.0.0.1:8090/api/status | python3 -m json.tool
# 检查: holding_info.price 应 > 0，used_realtime 应为 true/false
```

### 验证 C: 信号接口正常 (fail-closed)
```bash
curl -s -H "Cookie: qx_token=$TOKEN" http://127.0.0.1:8090/api/signal | python3 -m json.tool
# 预期之一:
#   - status=OK, official=false (盘中)
#   - status=DATA_UNAVAILABLE (数据未就绪)
# 不应出现 500 错误
```

### 验证 D: 刷新接口
```bash
curl -s -X POST -H "Cookie: qx_token=$TOKEN" http://127.0.0.1:8090/api/refresh | python3 -m json.tool
# 应返回 {"ok": true, "realtime_ok": true/false, "realtime_reason": ...}
```

### 验证 E: 幂等性文件写入
```bash
# 触发一次确认 (或等待真实确认)
ls -la data/live/idempotency.json
# 应存在且可读取
cat data/live/idempotency.json | python3 -m json.tool
```

### 验证 F: Bark 推送重试 (可选，非必须)
```bash
uv run python scripts/live_signal.py --notify-test
# 应收到测试通知
```

---

## 🔙 回滚方案 (如果任何验证失败)

### 快速回滚 (30 秒内)
```bash
cd /path/to/project
# 停止 trade_server
pkill -f trade_server

# 恢复备份
cp deploy/backup/20260817/live_signal.py scripts/
cp deploy/backup/20260817/trade_server.py scripts/
cp deploy/backup/20260817/notify.py scripts/

# 删除新增文件 (如果已创建)
rm -f data/live/idempotency.json

# 重启 trade_server
nohup uv run python scripts/trade_server.py --port 8090 --host 0.0.0.0 > logs/trade_server.log 2>&1 &
```

### 状态回滚 (如果 state.json 被意外修改)
```bash
# 从部署前备份恢复
cp data/live/state.json.bak.20260817_XXXXXX data/live/state.json
```

---

## 🚨 已知风险和注意事项

| 风险 | 影响 | 缓解 |
|---|---|---|
| `copy.deepcopy(_DATA_CACHE)` 内存占用 | 低 | DataFrame 约 2-5MB，deepcopy 临时加倍，现代服务器无压力 |
| `/api/health` 加鉴权后监控脚本失效 | 中 | 如有外部监控调用 `/api/health`，需更新为带 token 或使用 `/api/status` |
| `idempotency.json` 写入权限 | 低 | 首次确认时创建，如失败会静默降级（OSError pass） |
| `_today_sh()` 与服务器时区绑定 | 低 | 服务器已确认 Asia/Shanghai；如迁移时区需同步修改 `_SH_TZ` |
| `inject_realtime` 全部检查 | 中 | 首次运行若某 ETF 无今天数据，会正确触发 fail-closed，不会错误跳过 |

---

## 📋 部署后观察清单 (次交易日)

- [ ] **14:50 信号生成**: 观察 `live_signal.py` 日志，`inject_realtime` 是否正确注入
- [ ] **调仓日确认**: 如有换仓信号，网页确认成交是否正常写入 `state.json`
- [ ] **Bark 推送**: 确认收到推送，如失败观察是否有重试日志
- [ ] **trade_server 日志**: 检查 `logs/trade_server.log` 有无异常 traceback
- [ ] **回撤计算**: 如持仓停牌，观察 `account_value` 是否正确使用最后已知价

---

## 📝 变更摘要

```
修复 13 个 bug:
  live_signal.py: 时区统一、inject_realtime 全部检查、停牌回退、100股校验
  trade_server.py: 幂等性持久化、health 鉴权、缓存文件数检查、deepcopy 防污染
  notify.py: Bark 指数退避重试 (3次)
```

---

**部署人**: _________________  **日期**: _________________  **验证通过**: _________________
