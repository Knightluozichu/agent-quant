#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 七星V3 夜间数据补齐 (由 cron 调用)
#
# 每个交易日 21:30 自动运行:
#   - 新浪当日K线在收盘后才陆续发布, 14:50/16:30 任务拿不到当日数据
#   - 本任务在夜间补齐当日K线, 避免"数据日期"滞后
#   - 仅更新数据, 不生成信号/不改账户状态 (幂等, 可安全重跑)
#
# 防重入: flock -n 确保同一时间只有一个实例运行
# 退出码: 显式保存 rc, 失败时非零退出 + Bark 告警
# 结构化日志: SUMMARY 行含 start/end/exit/duration/data_date
# ---------------------------------------------------------------------------
set -uo pipefail
export TZ=Asia/Shanghai

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/data/live"
LOG="${LOG_DIR}/sync.log"
LOCK_DIR="${QIXING_LOCK_DIR:-/run/lock/qixing}"
mkdir -p "$LOG_DIR" "$LOCK_DIR" 2>>"$LOG"
LOCK_FILE="${LOCK_DIR}/sync.lock"
exec 9>"$LOCK_FILE" 2>>"$LOG"
if ! flock -n 9; then
    echo "[sync] $(date '+%Y-%m-%d %H:%M:%S') 已有实例运行, 跳过" >> "$LOG"
    exit 0
fi

cd "$PROJECT_ROOT"

# crontab 自愈守护: 检查三个定时任务是否被外部覆盖丢失 (幂等, 缺失时自动补回+告警)
bash "${PROJECT_ROOT}/deploy/ensure_cron.sh" 2>/dev/null || true

# 直接使用 venv python (避免 uv run 每次检查环境, cron 环境精简)
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    PYTHON="uv run python"
fi

TASK_NAME="[sync]"
RC=0
START_TS=$(date '+%s')
START_ISO=$(date '+%Y-%m-%dT%H:%M:%S')
DATA_DATE=""

{
    echo ""
    echo "$TASK_NAME ========== $(date '+%Y-%m-%d %H:%M:%S %Z') =========="
    $PYTHON scripts/live_signal.py --sync-only
    RC=$?
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
        echo "$TASK_NAME ❌ 数据同步失败 (exit=$RC), 请检查上方日志"
        echo "$TASK_NAME SUMMARY start=${START_ISO} end=$(date '+%Y-%m-%dT%H:%M:%S') exit=${RC} duration=${DURATION}s data_date=${DATA_DATE}"
        $PYTHON -c "
import sys; sys.path.insert(0, 'scripts')
from notify import push_bark
push_bark('七星V3 数据同步失败', f'exit=${RC}, 请检查 sync.log')
" 2>&1 || echo "$TASK_NAME ⚠️ Bark 推送失败"
    else
        echo "$TASK_NAME ✅ 数据同步完成"
        echo "$TASK_NAME SUMMARY start=${START_ISO} end=$(date '+%Y-%m-%dT%H:%M:%S') exit=0 duration=${DURATION}s data_date=${DATA_DATE}"
    fi
} >> "$LOG" 2>&1

# 30天日志滚动
find "$LOG_DIR" -name "*.log" -mtime +30 -delete 2>/dev/null || true

exit "$RC"
