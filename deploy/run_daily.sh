#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 七星V3 每日定时任务 (由 cron 调用)
#
# 每个交易日 14:50 自动运行:
#   - 更新最新行情
#   - 判断是否调仓日, 生成信号
#   - 调仓日推送买卖指令到 iPhone (Bark)
# ---------------------------------------------------------------------------
set -euo pipefail
export TZ=Asia/Shanghai

# 定位项目根目录 (本脚本在 deploy/ 下)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 确保 uv 在 PATH 里 (cron 环境精简, 需手动加)
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"

LOG_DIR="data/live"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/cron.log"

{
    echo ""
    echo "========== $(date '+%Y-%m-%d %H:%M:%S %Z') =========="
    uv run python scripts/live_signal.py
} >> "$LOG" 2>&1

# 日志只保留最近 500 行, 防止无限增长
tail -n 500 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
