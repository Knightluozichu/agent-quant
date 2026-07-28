#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 七星V3 每日校准任务 (由 cron 调用)
#
# 每个交易日 15:10 (收盘后) 自动运行:
#   - 更新数据到最终收盘
#   - 校准下个调仓日 + 预览调仓数据
#   - 推送每日校准状态到 iPhone (Bark)
# ---------------------------------------------------------------------------
set -euo pipefail
export TZ=Asia/Shanghai

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"

LOG_DIR="data/live"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/calibrate.log"

{
    echo ""
    echo "========== 校准 $(date '+%Y-%m-%d %H:%M:%S %Z') =========="
    uv run python scripts/daily_calibrate.py
} >> "$LOG" 2>&1

# 日志只保留最近 300 行
tail -n 300 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
