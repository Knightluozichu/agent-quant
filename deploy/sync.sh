#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 本地代码 → 生产服务器 一键同步
#
# 用法: ./deploy/sync.sh
#
# 说明:
#   - 只同步代码 (scripts/ deploy/ pyproject.toml 等)
#   - 自动排除: 虚拟环境 / git / 服务器数据缓存 / 实盘状态(含Bark Key)
#   - 无需重启服务: cron 每天14:50会自动使用新代码
# ---------------------------------------------------------------------------
set -euo pipefail

SERVER="root@106.52.243.51"
REMOTE_DIR="/opt/quant"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "  同步本地代码 → ${SERVER}:${REMOTE_DIR}"
rsync -avz --delete \
    --exclude '.venv' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.pytest_cache' \
    --exclude '.env' \
    --exclude '.env.*' \
    --exclude 'data/live' \
    --exclude 'data/cross_asset' \
    --exclude 'data/qixing_results' \
    --exclude 'data/stock_cache' \
    --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    --perms \
    "$PROJECT_ROOT/" "${SERVER}:${REMOTE_DIR}/"

# 确保 deploy 脚本有执行权限
echo "  确保部署脚本执行权限..."
ssh "${SERVER}" "chmod +x ${REMOTE_DIR}/deploy/*.sh"

echo ""
echo "  ✓ 代码已同步到生产服务器"
echo "  ✓ cron 无需重启 — 明天14:50 自动用新代码"
echo ""
echo "  ⚠️  若本次改了 scripts/ 下的代码, 记账网页(常驻服务)需重启才生效:"
echo "      ssh ${SERVER} 'systemctl restart trade-web'"
echo ""
echo "  ⚠️  若本次改了 pyproject.toml/uv.lock (动了依赖), 需在服务器重跑:"
echo "      ssh ${SERVER} 'cd ${REMOTE_DIR} && uv sync --extra data-akshare --extra server'"
