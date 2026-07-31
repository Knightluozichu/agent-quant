#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 七星V3 夜间数据补齐 (由 cron 调用)
#
# 每个交易日 21:30 自动运行:
#   - 新浪当日K线在收盘后才陆续发布, 14:50/16:30 任务拿不到当日数据
#   - 本任务在夜间补齐当日K线, 避免"数据日期"滞后
#   - 仅更新数据, 不生成信号/不改账户状态 (幂等, 可安全重跑)
# ---------------------------------------------------------------------------
set -uo pipefail
export TZ=Asia/Shanghai

# 定位项目根目录 (本脚本在 deploy/ 下)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 确保 uv 在 PATH 里 (cron 环境精简, 需手动加)
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"

LOG_DIR="data/live"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/sync.log"

{
    echo ""
    echo "========== 数据补齐 $(date '+%Y-%m-%d %H:%M:%S %Z') =========="
    uv run python scripts/live_signal.py --sync-only
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "❌ 数据同步失败 (exit=$EXIT_CODE), 请检查上方日志"
    fi
} >> "$LOG" 2>&1

# 日志只保留最近 300 行, 防止无限增长
tail -n 300 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
