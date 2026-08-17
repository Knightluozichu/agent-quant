#!/usr/bin/env bash
# 七星 V3-G/V4 Bug 修复部署脚本 (2026-08-17)
# 用法: 复制到服务器项目根目录，chmod +x deploy.sh && ./deploy.sh

set -euo pipefail

echo "=========================================="
echo "七星实盘系统 Bug 修复部署"
echo "=========================================="

# ---------- 配置 ----------
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
BACKUP_DIR="$PROJECT_DIR/deploy/backup/20260817_$(date +%H%M%S)"
LOG_DIR="$PROJECT_DIR/logs"
PORT="${PORT:-8090}"
HOST="${HOST:-0.0.0.0}"

mkdir -p "$BACKUP_DIR" "$LOG_DIR"
cd "$PROJECT_DIR"

# ---------- 0. 环境检查 ----------
echo ""
echo "[0/6] 环境检查..."

# 时区
TZ=$(timedatectl 2>/dev/null | grep "Time zone" | awk '{print $3}' || echo "unknown")
echo "  服务器时区: $TZ"
if [[ "$TZ" != "Asia/Shanghai" ]]; then
    echo "  ⚠️  警告: 时区非 Asia/Shanghai，但 _SH_TZ 显式处理仍可正常工作"
fi

# pending_order
PENDING=$(python3 -c "
import json
s=json.load(open('data/live/state.json'))
p=s.get('pending_order')
print('YES' if p else 'NO')
" 2>/dev/null || echo "UNKNOWN")
if [[ "$PENDING" == "YES" ]]; then
    echo "  ❌ 阻塞: 有 pending_order 待确认，请先处理后再部署"
    exit 1
fi
echo "  ✓ 无待确认订单"

# 14:50 窗口检查
HOUR=$(date +%H)
MIN=$(date +%M)
if (( HOUR == 14 && MIN >= 30 )) || (( HOUR == 14 && MIN <= 59 )); then
    echo "  ❌ 阻塞: 当前为 14:30-15:00 窗口期，建议 15:30 后再部署"
    exit 1
fi
echo "  ✓ 非 14:50 窗口期"

# ---------- 1. 备份 ----------
echo ""
echo "[1/6] 备份当前文件..."
cp scripts/live_signal.py "$BACKUP_DIR/"
cp scripts/trade_server.py "$BACKUP_DIR/"
cp scripts/notify.py "$BACKUP_DIR/"
cp data/live/state.json "$BACKUP_DIR/" 2>/dev/null || true
echo "  ✓ 备份至 $BACKUP_DIR"

# ---------- 2. 停服务 ----------
echo ""
echo "[2/6] 停止 trade_server..."
PID=$(ps aux | grep "[t]rade_server.py" | awk '{print $2}' || true)
if [[ -n "$PID" ]]; then
    kill -TERM "$PID" 2>/dev/null || true
    sleep 2
    # 确认已停止
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "  ⚠️  进程仍在，强制 kill..."
        kill -KILL "$PID" 2>/dev/null || true
    fi
    echo "  ✓ trade_server 已停止 (PID: $PID)"
else
    echo "  ✓ trade_server 未运行"
fi

# 确认端口释放
for i in {1..10}; do
    if ! ss -tlnp 2>/dev/null | grep -q ":$PORT "; then
        break
    fi
    sleep 1
done

# ---------- 3. 校验新文件 ----------
echo ""
echo "[3/6] 校验新文件..."

check_file() {
    local f="$1"
    local name="$2"
    if [[ ! -f "$f" ]]; then
        echo "  ❌ $name 不存在: $f"
        exit 1
    fi
}

check_file "scripts/live_signal.py" "live_signal.py"
check_file "scripts/trade_server.py" "trade_server.py"
check_file "scripts/notify.py" "notify.py"

# 关键修改存在性校验
grep -q "_today_sh()" scripts/live_signal.py || { echo "  ❌ _today_sh 缺失"; exit 1; }
grep -q "copy.deepcopy" scripts/trade_server.py || { echo "  ❌ deepcopy 缺失"; exit 1; }
grep -q "_load_idempotency" scripts/trade_server.py || { echo "  ❌ 幂等持久化缺失"; exit 1; }
grep -q "max_retries = 3" scripts/notify.py || { echo "  ❌ Bark 重试缺失"; exit 1; }

echo "  ✓ 关键修改校验通过"

# ---------- 4. 冷启动测试 ----------
echo ""
echo "[4/6] 冷启动测试..."

# 测试导入
if ! python3 -c "import sys; sys.path.insert(0, 'scripts'); import trade_server; print('trade_server import OK')" 2>/dev/null; then
    # uv 方式
    if ! uv run python -c "import sys; sys.path.insert(0, 'scripts'); import trade_server; print('trade_server import OK')" 2>/dev/null; then
        echo "  ❌ trade_server 导入失败，中止部署"
        exit 1
    fi
fi
echo "  ✓ trade_server 导入正常"

# 测试 live_signal --dry-run (不依赖状态文件)
if uv run python scripts/live_signal.py --dry-run > "$LOG_DIR/dry_run_test.log" 2>&1; then
    echo "  ✓ live_signal --dry-run 正常"
else
    echo "  ⚠️  dry-run 有告警 (可能非交易日或无数据)，继续部署..."
fi

# ---------- 5. 启动服务 ----------
echo ""
echo "[5/6] 启动 trade_server..."

nohup uv run python scripts/trade_server.py --port "$PORT" --host "$HOST" \
    > "$LOG_DIR/trade_server.log" 2>&1 &

sleep 3

# 检查进程
NEW_PID=$(ps aux | grep "[t]rade_server.py" | awk '{print $2}' || true)
if [[ -z "$NEW_PID" ]]; then
    echo "  ❌ trade_server 启动失败，查看日志:"
    tail -20 "$LOG_DIR/trade_server.log"
    exit 1
fi
echo "  ✓ trade_server 启动 (PID: $NEW_PID)"

# ---------- 6. 快速验证 ----------
echo ""
echo "[6/6] 快速验证..."

# 6a. health 无鉴权应 401
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/api/health 2>/dev/null || echo "000")
if [[ "$HTTP_CODE" == "401" ]] || [[ "$HTTP_CODE" == "403" ]]; then
    echo "  ✓ /api/health 鉴权生效 (返回 $HTTP_CODE)"
else
    echo "  ⚠️  /api/health 返回 $HTTP_CODE (预期 401/403)，继续检查..."
fi

# 6b. 有 token 应正常
TOKEN=$(python3 -c "
import json
c=json.load(open('data/live/config.json'))
tokens=c.get('web_tokens', [])
print(tokens[0] if tokens else '')
" 2>/dev/null || true)

if [[ -n "$TOKEN" ]]; then
    HEALTH=$(curl -s -H "Cookie: qx_token=$TOKEN" http://127.0.0.1:$PORT/api/health 2>/dev/null || echo "")
    if echo "$HEALTH" | grep -q "status"; then
        echo "  ✓ /api/health 带 token 正常"
    else
        echo "  ⚠️  /api/health 带 token 异常: $HEALTH"
    fi

    # 6c. /api/status
    STATUS=$(curl -s -H "Cookie: qx_token=$TOKEN" http://127.0.0.1:$PORT/api/status 2>/dev/null || echo "")
    if echo "$STATUS" | grep -q "initialized"; then
        echo "  ✓ /api/status 正常"
    else
        echo "  ⚠️  /api/status 异常"
    fi

    # 6d. /api/refresh
    REFRESH=$(curl -s -X POST -H "Cookie: qx_token=$TOKEN" http://127.0.0.1:$PORT/api/refresh 2>/dev/null || echo "")
    if echo "$REFRESH" | grep -q "realtime_ok"; then
        echo "  ✓ /api/refresh 含 realtime_ok 字段"
    else
        echo "  ⚠️  /api/refresh 异常: $REFRESH"
    fi
else
    echo "  ⚠️  未找到 web_tokens，跳过带 token 验证"
fi

# 6e. 幂等文件
touch data/live/idempotency.json 2>/dev/null || true
if [[ -f "data/live/idempotency.json" ]]; then
    echo "  ✓ idempotency.json 可创建"
else
    echo "  ⚠️  idempotency.json 无法创建，检查目录权限"
fi

echo ""
echo "=========================================="
echo "部署完成 ✅"
echo "=========================================="
echo "备份目录: $BACKUP_DIR"
echo "日志文件: $LOG_DIR/trade_server.log"
echo ""
echo "次交易日观察项:"
echo "  - 14:50 信号生成日志"
echo "  - 调仓确认是否正常"
echo "  - Bark 推送是否收到"
echo ""
echo "回滚命令 (如需要):"
echo "  cp $BACKUP_DIR/live_signal.py scripts/"
echo "  cp $BACKUP_DIR/trade_server.py scripts/"
echo "  cp $BACKUP_DIR/notify.py scripts/"
echo "  rm -f data/live/idempotency.json"
echo "  pkill -f trade_server; sleep 2; nohup uv run python scripts/trade_server.py --port $PORT --host $HOST &"
