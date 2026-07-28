#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 七星V3 实盘服务器一键部署 (Ubuntu/Debian)
#
# 用法: 把项目放到服务器后, 在项目根目录运行
#   bash deploy/setup_server.sh
#
# 本脚本会:
#   1. 安装 uv (Python 包管理器)
#   2. 安装项目依赖 (含 akshare 数据源)
#   3. 初始化历史数据 (若无缓存, 全量拉取+复权)
#   4. 设置时区为 Asia/Shanghai
#   5. 安装 cron 定时任务 (每个交易日 14:50)
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
echo "=================================================="
echo "  七星V3 实盘部署"
echo "  项目目录: $PROJECT_ROOT"
echo "=================================================="

# 1. 安装 uv ---------------------------------------------------------------
if ! command -v uv &> /dev/null; then
    echo ""
    echo ">>> [1/5] 安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo ">>> [1/5] uv 已安装: $(uv --version)"
fi

# 2. 安装依赖 (含 akshare) --------------------------------------------------
echo ""
echo ">>> [2/5] 安装 Python 依赖 (含 akshare 数据源)..."
uv sync --extra data-akshare

# 3. 初始化历史数据 ---------------------------------------------------------
echo ""
if [ -f data/cross_asset/518880.parquet ]; then
    echo ">>> [3/5] 数据缓存已存在, 跳过 bootstrap"
else
    echo ">>> [3/5] 初始化历史数据 (全量拉取 + 份额拆分复权, 约1分钟)..."
    uv run python scripts/live_signal.py --bootstrap
fi

# 4. 设置时区 ---------------------------------------------------------------
echo ""
echo ">>> [4/5] 设置时区为 Asia/Shanghai..."
if command -v timedatectl &> /dev/null && sudo -n true 2>/dev/null; then
    sudo timedatectl set-timezone Asia/Shanghai
    echo "    系统时区: $(timedatectl show -p Timezone --value)"
else
    echo "    (无 sudo 权限, 跳过系统时区设置; run_daily.sh 已内置 TZ=Asia/Shanghai)"
fi

# 5. 安装 cron 定时任务 -----------------------------------------------------
echo ""
echo ">>> [5/5] 安装定时任务 (每个交易日 14:50)..."
chmod +x "$PROJECT_ROOT/deploy/run_daily.sh"
CRON_CMD="50 14 * * 1-5 $PROJECT_ROOT/deploy/run_daily.sh"
# 去重后写入 (避免重复安装)
( crontab -l 2>/dev/null | grep -v "run_daily.sh" ; echo "$CRON_CMD" ) | crontab -
echo "    已安装:"
crontab -l | grep run_daily.sh | sed 's/^/    /'

# 完成 ---------------------------------------------------------------------
echo ""
echo "=================================================="
echo "  ✅ 部署完成! 还需手动完成 2 步:"
echo ""
echo "  1) 配置 Bark 手机推送 (iPhone 装 Bark App 获取 Key):"
echo "     uv run python scripts/live_signal.py --set-bark <你的KEY>"
echo "     uv run python scripts/live_signal.py --notify-test"
echo ""
echo "  2) 初始化账户 (首次):"
echo "     uv run python scripts/live_signal.py --init 100000"
echo ""
echo "  之后每个交易日 14:50 会自动运行并推送信号到手机。"
echo "  查看运行日志: cat data/live/cron.log"
echo "=================================================="
