#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 七星V3 crontab 自愈守护 (幂等)
#
# 背景: 2026-07-31 crontab 被外部进程覆盖, 三个 quant 定时任务静默丢失.
# 本脚本检查三个任务是否存在, 缺失时自动补回并 Bark 告警.
#
# 调用方式: 由 run_data_sync.sh (每晚21:30) 自动调用, 也可手动执行.
# ---------------------------------------------------------------------------
set -uo pipefail
export TZ=Asia/Shanghai

# 只允许 root 管理 crontab, 防止非 root 用户给自己写入重复 crontab
# 背景: root 和 quant 各有一套 crontab 导致 16:30 校准任务因 state.json 权限冲突失败
if [ "$(id -u)" -ne 0 ]; then
    echo "ensure_cron.sh: 必须以 root 运行, 当前用户 $(id -un), 跳过"
    exit 0
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    PYTHON="uv run python"
fi

# 三个必须存在的定时任务 (与生产 crontab 一致)
REQUIRED_CRONS=(
    "50 14 * * 1-5 /opt/quant/deploy/run_daily.sh"
    "30 16 * * 1-5 /opt/quant/deploy/run_calibrate.sh"
    "30 21 * * 1-5 /opt/quant/deploy/run_data_sync.sh"
)

CURRENT=$(crontab -l 2>/dev/null || echo "")
MISSING=()

for entry in "${REQUIRED_CRONS[@]}"; do
    # 按脚本路径匹配 (容忍时间字段被调整)
    script_path=$(echo "$entry" | awk '{print $NF}')
    if ! echo "$CURRENT" | grep -q "$script_path"; then
        MISSING+=("$entry")
    fi
done

if [ ${#MISSING[@]} -eq 0 ]; then
    exit 0
fi

# 补回缺失任务
{
    echo "$CURRENT"
    for entry in "${MISSING[@]}"; do
        echo "$entry"
    done
} | crontab -

LOG="$PROJECT_ROOT/data/live/cron_guard.log"
mkdir -p "$(dirname "$LOG")"
{
    echo "[cron-guard] $(date '+%Y-%m-%d %H:%M:%S %Z') 检测到 ${#MISSING[@]} 个定时任务丢失, 已自动恢复:"
    for entry in "${MISSING[@]}"; do
        echo "  + $entry"
    done
} >> "$LOG"

# Bark 告警
cd "$PROJECT_ROOT"
$PYTHON -c "
import sys; sys.path.insert(0, 'scripts')
from notify import push_bark
push_bark('⚠️ 七星V3 crontab曾被覆盖', f'${#MISSING[@]}个定时任务已自动恢复, 详见cron_guard.log')
" 2>/dev/null || true

exit 0
