#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 七星V3 每日定时任务 (由 cron 调用)
#
# 每个交易日 14:50 自动运行:
#   - 更新最新行情
#   - 判断是否调仓日, 生成信号
#   - 调仓日推送买卖指令到 iPhone (Bark)
#
# 防重入: flock -n 确保同一时间只有一个实例运行
# 退出码: 显式保存 rc, 失败时非零退出 + Bark 告警
# 结构化日志: SUMMARY 行含 start/end/exit/duration/data_date
# ---------------------------------------------------------------------------
set -uo pipefail
export TZ=Asia/Shanghai

# 专用锁目录，避免共享 /tmp 的 fs.protected_regular 导致 root cron 静默失败。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/data/live"
LOG="${LOG_DIR}/cron.log"
LOCK_DIR="${QIXING_LOCK_DIR:-/run/lock/qixing}"
mkdir -p "$LOG_DIR" "$LOCK_DIR" 2>>"$LOG"
LOCK_FILE="${LOCK_DIR}/daily.lock"
exec 9>"$LOCK_FILE" 2>>"$LOG"
if ! flock -n 9; then
    echo "[daily] $(date '+%Y-%m-%d %H:%M:%S') 已有实例运行, 跳过" >> "$LOG"
    exit 0
fi

cd "$PROJECT_ROOT"

# 直接使用 venv python (避免 uv run 每次检查环境, cron 环境精简)
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    # 回退: 本地开发环境可能没有 .venv
    PYTHON="uv run python"
fi

TASK_NAME="[daily]"
RC=0
START_TS=$(date '+%s')
START_ISO=$(date '+%Y-%m-%dT%H:%M:%S')
DATA_DATE=""

{
    echo ""
    echo "$TASK_NAME ========== $(date '+%Y-%m-%d %H:%M:%S %Z') =========="
    $PYTHON scripts/live_signal.py
    RC=$?
    # 提取数据日期 (从最新 parquet 文件)
    DATA_DATE=$($PYTHON -c "
import pandas as pd, glob
dates = []
for f in glob.glob('data/cross_asset/*.parquet'):
    try:
        df = pd.read_parquet(f, columns=['trade_date'])
        if len(df): dates.append(str(df['trade_date'].max())[:10])
    except: pass
print(max(dates) if dates else 'unknown')
" 2>/dev/null || echo "unknown")

    END_TS=$(date '+%s')
    DURATION=$((END_TS - START_TS))

    if [ $RC -ne 0 ]; then
        echo "$TASK_NAME ❌ 信号生成失败 (exit=$RC), 请检查上方日志"
        echo "$TASK_NAME SUMMARY start=${START_ISO} end=$(date '+%Y-%m-%dT%H:%M:%S') exit=${RC} duration=${DURATION}s data_date=${DATA_DATE}"
        # Bark 告警 (失败时推送)
        $PYTHON -c "
import sys; sys.path.insert(0, 'scripts')
from notify import push_bark
push_bark('❌ 信号生成失败', f'exit=${RC}, 请检查 cron.log')
" 2>&1 || echo "$TASK_NAME ⚠️ Bark 推送失败"
    else
        echo "$TASK_NAME ✅ 信号生成完成"
        echo "$TASK_NAME SUMMARY start=${START_ISO} end=$(date '+%Y-%m-%dT%H:%M:%S') exit=0 duration=${DURATION}s data_date=${DATA_DATE}"
    fi
} >> "$LOG" 2>&1

# 30天日志滚动 (按修改时间清理)
find "$LOG_DIR" -name "*.log" -mtime +30 -delete 2>/dev/null || true

exit "$RC"
