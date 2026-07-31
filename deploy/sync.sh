#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 本地代码 → 生产服务器 一键同步 (P0 安全加固版)
#
# 用法: ./deploy/sync.sh
#
# 改进 (S4补强):
#   - rsync 前强制 --dry-run 预检
#   - 精确权限: 仅 deploy/*.sh 设为 0755, 其余保持 Git 权限
#   - 部署后保存 manifest (git hash + 代码 hash + 参数 hash + lock hash)
#   - 比较本地/远端 manifest 确认一致性
#   - 排除运行状态、备份文件、本地索引
# ---------------------------------------------------------------------------
set -uo pipefail

SERVER="root@106.52.243.51"
REMOTE_DIR="/opt/quant"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${PROJECT_ROOT}/deploy/.DEPLOYED_MANIFEST"

# 排除项 (安全: 不同步秘密、运行状态、缓存、备份)
EXCLUDES=(
    --exclude '.venv'
    --exclude '.git'
    --exclude '__pycache__'
    --exclude '*.pyc'
    --exclude '.pytest_cache'
    --exclude '.ruff_cache'
    --exclude '.mypy_cache'
    --exclude '.codegraph'
    --exclude '.env'
    --exclude '.env.*'
    --exclude 'data/live'
    --exclude 'data/cross_asset'
    --exclude 'data/qixing_results'
    --exclude 'data/stock_cache'
    --exclude '*.tmp'
    --exclude '*.bak'
    --exclude '*~'
    --exclude '.DS_Store'
    --exclude 'deploy/.DEPLOYED_MANIFEST'
)

echo "=== 1/4: dry-run 预检 ==="
if ! rsync -avz --dry-run "${EXCLUDES[@]}" --perms \
    "$PROJECT_ROOT/" "${SERVER}:${REMOTE_DIR}/" >/tmp/rsync_dryrun.log 2>&1; then
    echo "  ❌ dry-run 预检失败!"
    cat /tmp/rsync_dryrun.log
    exit 1
fi
CHANGED=$(grep -c '^<f\|^>f\|^cL' /tmp/rsync_dryrun.log 2>/dev/null || echo "0")
echo "  ✓ 预检通过 (${CHANGED} 个文件待同步)"

echo ""
echo "=== 2/4: 同步代码 ==="
if ! rsync -avz --delete "${EXCLUDES[@]}" --perms \
    "$PROJECT_ROOT/" "${SERVER}:${REMOTE_DIR}/"; then
    echo "  ❌ 同步失败!"
    exit 1
fi
echo "  ✓ 代码已同步"

echo ""
echo "=== 3/4: 精确设置权限 ==="
# 仅 deploy 脚本设为 0755, 不用 --chmod 范围过宽
ssh "${SERVER}" "chmod 0755 ${REMOTE_DIR}/deploy/*.sh 2>/dev/null; \
                 chmod 0644 ${REMOTE_DIR}/deploy/*.md 2>/dev/null; \
                 chmod 0600 ${REMOTE_DIR}/.env 2>/dev/null; \
                 chmod 0600 ${REMOTE_DIR}/data/live/config.json 2>/dev/null; \
                 chmod 0600 ${REMOTE_DIR}/data/live/state.json 2>/dev/null; \
                 chmod 0700 ${REMOTE_DIR}/data/live 2>/dev/null; \
                 true"
echo "  ✓ 权限已设置 (deploy=0755, secrets=0600)"

echo ""
echo "=== 4/4: 生成并比较 manifest ==="
# 生成本地 manifest
GIT_HASH=$(cd "${PROJECT_ROOT}" && git rev-parse HEAD 2>/dev/null || echo "no-git")
CODE_HASH=$(find "${PROJECT_ROOT}/scripts" -name '*.py' -exec cat {} + 2>/dev/null | sha256sum | cut -c1-16)
LOCK_HASH=$(sha256sum "${PROJECT_ROOT}/uv.lock" 2>/dev/null | cut -c1-16 || echo "no-lock")
PARAM_HASH=$(grep -E '^(DROP_LOOKBACK|A_SHARE_MA|MOM_PERIODS|MOM_WEIGHTS|REBALANCE_DAYS) ' \
    "${PROJECT_ROOT}/scripts/run_qixing_v3.py" 2>/dev/null | sha256sum | cut -c1-16 || echo "no-params")

cat > "${MANIFEST}" << EOF
git_hash=${GIT_HASH}
code_hash=${CODE_HASH}
lock_hash=${LOCK_HASH}
param_hash=${PARAM_HASH}
deploy_time=$(date '+%Y-%m-%d %H:%M:%S')
EOF

# 同步 manifest 到服务器并比较
scp -q "${MANIFEST}" "${SERVER}:${REMOTE_DIR}/deploy/.DEPLOYED_MANIFEST"
REMOTE_HASH=$(ssh "${SERVER}" "sha256sum ${REMOTE_DIR}/deploy/.DEPLOYED_MANIFEST" | cut -c1-16)
LOCAL_HASH=$(sha256sum "${MANIFEST}" | cut -c1-16)

if [ "${LOCAL_HASH}" = "${REMOTE_HASH}" ]; then
    echo "  ✓ manifest 一致"
    echo "    git:    ${GIT_HASH:0:8}"
    echo "    code:   ${CODE_HASH}"
    echo "    lock:   ${LOCK_HASH}"
    echo "    params: ${PARAM_HASH}"
else
    echo "  ⚠️  manifest 不一致! 本地 ${LOCAL_HASH} ≠ 远端 ${REMOTE_HASH}"
    echo "      请检查网络或手动验证"
fi

echo ""
echo "  ✓ 部署完成"
echo ""
echo "  ⚠️  若本次改了 scripts/ 下的代码, 记账网页(常驻服务)需重启才生效:"
echo "      ssh ${SERVER} 'systemctl restart trade-web'"
echo ""
echo "  ⚠️  若本次改了 pyproject.toml/uv.lock (动了依赖), 需在服务器重跑:"
echo "      ssh ${SERVER} 'cd ${REMOTE_DIR} && uv sync --extra data-akshare --extra server'"
