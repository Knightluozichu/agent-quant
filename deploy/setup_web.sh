#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 七星V3 记账网页 一键部署 (在生产服务器上运行)
#
# 用法:
#   bash deploy/setup_web.sh              # 交互式设置密码
#   bash deploy/setup_web.sh <密码>       # 直接指定密码
#
# 本脚本会:
#   1. 安装 web 依赖 (fastapi/uvicorn)
#   2. 设置网页访问密码 (若尚未设置)
#   3. 安装并启动 systemd 常驻服务 (开机自启 + 崩溃自动重启)
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 确保 uv 在 PATH 里 (非交互 ssh 环境需手动加)
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"

PORT=8090

echo "=================================================="
echo "  七星V3 记账网页部署"
echo "=================================================="

# 1. 安装依赖 ----------------------------------------------------------------
echo ""
echo ">>> [1/3] 安装 web 依赖 (fastapi/uvicorn)..."
uv sync --extra server --extra data-akshare

# 2. 设置访问密码 ------------------------------------------------------------
echo ""
HAS_PWD=$(python3 -c "import json;print(1 if json.load(open('data/live/config.json')).get('web_password') else 0)" 2>/dev/null || echo 0)
if [ "$HAS_PWD" = "1" ]; then
    echo ">>> [2/3] 访问密码已设置, 跳过"
else
    if [ -n "${1:-}" ]; then
        WEB_PWD="$1"
    else
        read -r -s -p ">>> [2/3] 设置网页访问密码: " WEB_PWD
        echo ""
    fi
    uv run python scripts/trade_server.py --set-password "$WEB_PWD"
    echo "    ✓ 访问密码已设置"
fi

# 3. 安装 systemd 服务 -------------------------------------------------------
echo ""
echo ">>> [3/3] 安装 systemd 常驻服务..."
cp deploy/trade-web.service /etc/systemd/system/trade-web.service
systemctl daemon-reload
systemctl enable trade-web
systemctl restart trade-web
sleep 3

if systemctl is-active --quiet trade-web; then
    IP=$(hostname -I | awk '{print $1}')
    echo ""
    echo "=================================================="
    echo "  ✅ 记账网页已启动!"
    echo ""
    echo "  手机浏览器访问:  http://${IP}:${PORT}"
    echo "  (外网访问需在腾讯云安全组放行 ${PORT} 端口)"
    echo ""
    echo "  查看状态:  systemctl status trade-web"
    echo "  查看日志:  journalctl -u trade-web -f"
    echo "=================================================="
else
    echo "  ❌ 服务启动失败, 请查看日志:"
    echo "     journalctl -u trade-web -n 30"
    exit 1
fi
