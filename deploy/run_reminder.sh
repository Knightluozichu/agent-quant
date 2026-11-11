#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 七星V5 半自动执行提醒 (由 cron 调用)
#
# 只读 ticket/状态并推送 Bark；不生成信号、不下单、不改账户。
# 用法: run_reminder.sh pre|same_day_warn|next_open_final|expired
# ---------------------------------------------------------------------------
set -uo pipefail
export TZ=Asia/Shanghai

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    PYTHON="uv run python"
fi

STAGE="${1:-}"
case "$STAGE" in
    pre|same_day_warn|next_open_final|expired) ;;
    *)
        echo "run_reminder.sh: 未知 stage: ${STAGE:-<empty>}" >&2
        exit 2
        ;;
esac

LOG_DIR="${PROJECT_ROOT}/data/live"
LOG="${LOG_DIR}/reminder.log"
mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"
{
    echo "[reminder] $(date '+%Y-%m-%d %H:%M:%S %Z') stage=${STAGE}"
    "$PYTHON" scripts/remind_pending.py --stage "$STAGE"
    RC=$?
    echo "[reminder] stage=${STAGE} exit=${RC}"
    exit "$RC"
} >> "$LOG" 2>&1
