#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 七星 V3-G/V4 Bug 修复部署脚本 (2026-08-17)
#
# 部署流程 (生产服务器 = root@106.52.243.51:/opt/quant, 见 SERVER_OPERATIONS.md):
#   1. 本地: 新文件 scp 到 staging 目录 (绝不直接覆盖 scripts/)
#        ssh  root@106.52.243.51 'mkdir -p /opt/quant/deploy/staging/20260817'
#        scp  scripts/{live_signal,trade_server,notify}.py \
#             root@106.52.243.51:/opt/quant/deploy/staging/20260817/
#   2. 服务器: cd /opt/quant && bash deploy/deploy_20260817.sh
#
# 脚本行为 (任一步失败 → trap 自动回滚: 恢复备份 + systemctl start + is-active 校验):
#   [0] 环境检查 (服务器护栏/root/pending_order/14:30-15:00 窗口)
#   [1] staging 文件校验 (存在性/关键修改/语法编译)
#   [2] 备份旧文件 (带时间戳, 在替换之前)
#   [3] systemctl stop trade-web + 确认 8090 端口释放
#   [4] 安装新文件 + 冷启动校验 (.venv/bin/python)
#   [5] systemctl start trade-web + is-active + 无 token 应 401
#   [6] 带 token 验证 (token 走 stdin/0600 临时文件, 不进命令行) + 写 .DEPLOYED_MANIFEST
#
# 纪律:
#   - 服务器禁止 uv run (SERVER_OPERATIONS.md R2), 一律 .venv/bin/python
#   - 服务由 systemd unit 管理 (127.0.0.1:8090, User=quant), 本脚本不传 --host/--port
#   - 日志归 journalctl (journalctl -u trade-web), 不做 nohup/日志重定向
# ---------------------------------------------------------------------------
set -euo pipefail
export TZ=Asia/Shanghai

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

STAGING_DIR="$PROJECT_ROOT/deploy/staging/20260817"
BACKUP_DIR="$PROJECT_ROOT/deploy/backup/$(date +%Y%m%d_%H%M%S)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
PORT=8090
FILES=(live_signal.py trade_server.py notify.py)
DEPLOY_OK=0

echo "=========================================="
echo "七星实盘系统 Bug 修复部署 (2026-08-17)"
echo "  项目目录: $PROJECT_ROOT"
echo "  备份目录: $BACKUP_DIR"
echo "=========================================="

# ---------- 服务器护栏: 防止本地/开发机误执行 ----------
FORCE=0
for arg in "$@"; do
    [[ "$arg" == "--server" ]] && FORCE=1
done
if [[ "$FORCE" != "1" ]]; then
    if [[ ! -d /opt/quant || ! -f /etc/systemd/system/trade-web.service ]]; then
        echo "❌ 未检测到生产服务器环境 (/opt/quant + trade-web.service)"
        echo "   本脚本会停止/重启实盘服务, 为防止误执行已中止"
        echo "   如确认在生产服务器上, 请加 --server 参数"
        exit 1
    fi
fi

# ---------- 自动回滚 (备份完成后才 arm) ----------
rollback() {
    set +e  # 回滚过程中尽力而为, 不因单步失败中断
    echo ""
    echo "!!!!!! 部署失败, 自动回滚 !!!!!!"
    for f in "${FILES[@]}"; do
        if [[ -f "$BACKUP_DIR/$f" ]]; then
            cp "$BACKUP_DIR/$f" "scripts/$f"
        fi
    done
    echo "  ✓ 已从 $BACKUP_DIR 恢复旧文件"
    # 清理可能残留的 token 临时文件 (0600, mktemp 生成)
    [[ -n "${CURL_CONF:-}" ]] && rm -f "$CURL_CONF"
    systemctl start trade-web 2>/dev/null || true
    if systemctl is-active --quiet trade-web; then
        echo "  ✓ trade-web 已恢复运行 (旧版本)"
    else
        echo "  ❌ 回滚后 trade-web 仍未 active, 立即人工介入:"
        echo "     journalctl -u trade-web -n 50 --no-pager"
    fi
}

# ---------- [0/6] 环境检查 ----------
echo ""
echo "[0/6] 环境检查..."

# 需 root: systemctl + 保持文件属主
if [[ "$(id -u)" != "0" ]]; then
    echo "  ❌ 需 root 运行 (systemctl 管理及文件属主保持), 当前: $(id -un)"
    exit 1
fi

# 生产 Python (禁止 uv run, 见 SERVER_OPERATIONS.md R2)
if [[ ! -x "$PYTHON" ]]; then
    echo "  ❌ 生产 venv 不存在: $PYTHON"
    exit 1
fi
echo "  ✓ Python: $PYTHON"

# 时区 (仅提示; _SH_TZ 显式处理, 脚本内已 export TZ=Asia/Shanghai)
SYS_TZ=$(timedatectl 2>/dev/null | grep "Time zone" | awk '{print $3}' || echo "unknown")
echo "  服务器时区: $SYS_TZ"
if [[ "$SYS_TZ" != "Asia/Shanghai" ]]; then
    echo "  ⚠️  警告: 系统时区非 Asia/Shanghai (脚本内时间判断已强制 Asia/Shanghai)"
fi

# pending_order 阻塞
PENDING=$("$PYTHON" -c "
import json
s = json.load(open('data/live/state.json'))
print('YES' if s.get('pending_order') else 'NO')
" 2>/dev/null || echo "UNKNOWN")
if [[ "$PENDING" == "YES" ]]; then
    echo "  ❌ 阻塞: 有 pending_order 待确认, 请先处理后再部署"
    exit 1
fi
echo "  ✓ 无待确认订单 (pending_order=$PENDING)"

# 14:30-15:00 窗口阻塞 (10# 消除 08/09 八进制坑; TZ 已 export 为 Asia/Shanghai)
HOUR=$(date +%H)
MIN=$(date +%M)
if (( 10#$HOUR == 14 && 10#$MIN >= 30 )) || (( 10#$HOUR == 15 && 10#$MIN == 0 )); then
    echo "  ❌ 阻塞: 当前为 14:30-15:00 窗口期, 建议 15:30 后再部署"
    exit 1
fi
echo "  ✓ 非 14:30-15:00 窗口期 (当前 $(date '+%H:%M'))"

# ---------- [1/6] staging 文件校验 ----------
echo ""
echo "[1/6] 校验 staging 文件 ($STAGING_DIR)..."

for f in "${FILES[@]}"; do
    if [[ ! -f "$STAGING_DIR/$f" ]]; then
        echo "  ❌ staging 缺文件: $STAGING_DIR/$f"
        echo "     请先从本地 scp: scp scripts/$f root@106.52.243.51:$STAGING_DIR/"
        exit 1
    fi
done

# 关键修改存在性校验 (在 staging 上, 不碰线上文件)
grep -q "_today_sh()"        "$STAGING_DIR/live_signal.py"  || { echo "  ❌ _today_sh 缺失"; exit 1; }
grep -q "copy.deepcopy"      "$STAGING_DIR/trade_server.py" || { echo "  ❌ deepcopy 缺失"; exit 1; }
grep -q "_load_idempotency"  "$STAGING_DIR/trade_server.py" || { echo "  ❌ 幂等持久化缺失"; exit 1; }
grep -q "max_retries = 3"    "$STAGING_DIR/notify.py"       || { echo "  ❌ Bark 重试缺失"; exit 1; }

# 语法编译检查 (不产生 __pycache__)
"$PYTHON" - "$STAGING_DIR" <<'PYEOF'
import pathlib, sys
d = pathlib.Path(sys.argv[1])
for name in ("live_signal.py", "trade_server.py", "notify.py"):
    compile((d / name).read_text(), name, "exec")
PYEOF
echo "  ✓ staging 文件校验通过 (存在性/关键修改/语法)"

# ---------- [2/6] 备份旧文件 (替换之前, 带时间戳) ----------
echo ""
echo "[2/6] 备份当前文件..."
mkdir -p "$BACKUP_DIR"
for f in "${FILES[@]}"; do
    cp "scripts/$f" "$BACKUP_DIR/$f"
done
cp data/live/state.json "$BACKUP_DIR/" 2>/dev/null || true
echo "  ✓ 备份至 $BACKUP_DIR"

# 备份完成后 arm 回滚 trap: 此后任何非零退出 (含显式 exit 1) 自动恢复旧文件并拉起服务
trap 'rc=$?; if [[ "$rc" != "0" && "$DEPLOY_OK" != "1" ]]; then rollback; fi' EXIT

# ---------- [3/6] 停服务 (systemd) ----------
echo ""
echo "[3/6] 停止 trade-web..."
echo "  ⚠️  即将执行: systemctl stop trade-web"
systemctl stop trade-web

# 确认端口释放 (unit 的 ExecStartPre 要求 8090 空闲)
PORT_FREE=0
for _ in {1..10}; do
    if ! ss -tln 2>/dev/null | grep -q ":$PORT "; then
        PORT_FREE=1
        break
    fi
    sleep 1
done
if [[ "$PORT_FREE" != "1" ]]; then
    echo "  ❌ 端口 $PORT 10 秒未释放, 中止 (检查是否有非 systemd 进程占用)"
    exit 1
fi
echo "  ✓ trade-web 已停止, 端口 $PORT 已释放"

# ---------- [4/6] 安装新文件 + 冷启动校验 ----------
echo ""
echo "[4/6] 安装新文件..."
for f in "${FILES[@]}"; do
    # cp 覆盖已存在文件时保留原 inode, 属主/权限不变 (quant)
    cp "$STAGING_DIR/$f" "scripts/$f"
    echo "  ✓ 已安装 scripts/$f"
done

# 冷启动校验 1: trade_server 可导入 (生产 venv, 禁止 uv run)
"$PYTHON" -c "import sys; sys.path.insert(0, 'scripts'); import trade_server" \
    || { echo "  ❌ trade_server 导入失败"; exit 1; }
echo "  ✓ trade_server 导入正常"

# 冷启动校验 2: live_signal --dry-run (非交易日/无数据可能告警, 不阻塞)
if "$PYTHON" scripts/live_signal.py --dry-run > "$PROJECT_ROOT/data/live/deploy_dryrun.log" 2>&1; then
    echo "  ✓ live_signal --dry-run 正常"
else
    echo "  ⚠️  dry-run 有告警 (可能非交易日或无数据), 日志: data/live/deploy_dryrun.log"
fi

# ---------- [5/6] 启动服务 (systemd) ----------
echo ""
echo "[5/6] 启动 trade-web..."
echo "  ⚠️  即将执行: systemctl start trade-web"
systemctl start trade-web

# 启动成功判定: systemctl is-active (不再 ps grep)
ACTIVE=0
for _ in {1..15}; do
    if systemctl is-active --quiet trade-web; then
        ACTIVE=1
        break
    fi
    sleep 1
done
if [[ "$ACTIVE" != "1" ]]; then
    echo "  ❌ trade-web 启动失败, 最近日志:"
    journalctl -u trade-web -n 30 --no-pager 2>/dev/null || true
    exit 1
fi
echo "  ✓ trade-web is-active"

# 无 token 必须 401/403 (鉴权生效; 只看状态码, 不取 body)
# is-active 到端口可连接有短暂竞态, 允许重试; 但返回码必须最终是 401/403
HTTP_CODE="000"
for _ in {1..10}; do
    HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/api/health" 2>/dev/null || echo "000")
    [[ "$HTTP_CODE" != "000" ]] && break
    sleep 1
done
if [[ "$HTTP_CODE" == "401" || "$HTTP_CODE" == "403" ]]; then
    echo "  ✓ /api/health 鉴权生效 (HTTP $HTTP_CODE)"
else
    echo "  ❌ /api/health 无 token 返回 HTTP $HTTP_CODE (预期 401/403)"
    exit 1
fi

# ---------- [6/6] 带 token 验证 + manifest ----------
echo ""
echo "[6/6] 带 token 验证..."

# token 由 python 直接写入 mktemp 0600 文件, 全程不进命令行/终端
CURL_CONF=$(mktemp)
if "$PYTHON" - "$CURL_CONF" <<'PYEOF'
import json, sys, time
try:
    cfg = json.load(open("data/live/config.json"))
except Exception:
    sys.exit(1)
now = time.time()
token = ""
for t in cfg.get("web_tokens", []):
    if isinstance(t, dict) and t.get("expires", 0) > now:
        token = t["token"]
        break
    elif isinstance(t, str):  # 兼容旧格式 (下次请求时服务端自动迁移)
        token = t
        break
if not token:
    sys.exit(1)
with open(sys.argv[1], "w") as f:
    f.write(f'header = "Cookie: qx_token={token}"\n')
PYEOF
then
    CODE=$(curl -s -K "$CURL_CONF" -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/api/health" 2>/dev/null || echo "000")
    if [[ "$CODE" != "200" ]]; then
        rm -f "$CURL_CONF"
        echo "  ❌ /api/health 带 token 返回 HTTP $CODE (预期 200)"
        exit 1
    fi
    echo "  ✓ /api/health 带 token 200"

    # 以下为软检查 (非交易时段/数据未就绪时可能异常, 不阻断)
    CODE=$(curl -s -K "$CURL_CONF" -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/api/status" 2>/dev/null || echo "000")
    if [[ "$CODE" == "200" ]]; then
        echo "  ✓ /api/status 200"
    else
        echo "  ⚠️  /api/status 返回 HTTP $CODE (非阻断, 可能数据未就绪)"
    fi
    if curl -s -K "$CURL_CONF" -X POST "http://127.0.0.1:$PORT/api/refresh" 2>/dev/null | grep -q "realtime_ok"; then
        echo "  ✓ /api/refresh 含 realtime_ok 字段"
    else
        echo "  ⚠️  /api/refresh 未返回 realtime_ok (非阻断, 可能非交易时段)"
    fi
else
    echo "  ⚠️  config.json 无未过期 web_token, 跳过带 token 验证 (401 检查已通过)"
fi
rm -f "$CURL_CONF"

# 更新部署清单 (格式与 deploy/sync.sh 一致)
GIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "no-git")
CODE_HASH=$(find scripts -name '*.py' | LC_ALL=C sort | xargs cat 2>/dev/null | sha256sum | cut -c1-16)
LOCK_HASH=$(sha256sum uv.lock 2>/dev/null | cut -c1-16 || echo "no-lock")
PARAM_HASH=$(grep -E '^(DROP_LOOKBACK|A_SHARE_MA|MOM_PERIODS|MOM_WEIGHTS|REBALANCE_DAYS) ' \
    scripts/run_qixing_v3.py 2>/dev/null | sha256sum | cut -c1-16 || echo "no-params")
cat > deploy/.DEPLOYED_MANIFEST <<EOF
git_hash=$GIT_HASH
code_hash=$CODE_HASH
lock_hash=$LOCK_HASH
param_hash=$PARAM_HASH
deploy_time=$(date '+%Y-%m-%d %H:%M:%S')
EOF
echo "  ✓ .DEPLOYED_MANIFEST 已更新 (git=${GIT_HASH:0:8} code=$CODE_HASH)"

DEPLOY_OK=1

echo ""
echo "=========================================="
echo "部署完成 ✅"
echo "=========================================="
echo "备份目录: $BACKUP_DIR"
echo "服务日志: journalctl -u trade-web -f"
echo ""
echo "次交易日观察项:"
echo "  - 14:50 信号生成日志 (data/live/cron.log)"
echo "  - 调仓确认是否正常 (观察 idempotency.json 自然生成)"
echo "  - Bark 推送是否收到"
echo ""
echo "手动回滚 (如次交易日发现问题):"
echo "  systemctl stop trade-web"
echo "  cp $BACKUP_DIR/{live_signal.py,trade_server.py,notify.py} scripts/"
echo "  systemctl start trade-web && systemctl is-active --quiet trade-web"
