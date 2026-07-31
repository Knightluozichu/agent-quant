#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 七星V3 每日健康检查 (可由 cron 或手动运行)
#
# 检查项:
#   1. trade-web 服务状态 (systemd)
#   2. /api/health 端点响应
#   3. 数据新鲜度 (parquet 最新日期 vs 今天)
#   4. 持仓一致性 (state.json 非空且字段完整)
#   5. cron 日志最近一次 SUMMARY 行 (exit code)
#
# 用法: ./deploy/health_check.sh [--bark]
#   --bark: 检查失败时推送 Bark 告警
# ---------------------------------------------------------------------------
set -uo pipefail
export TZ=Asia/Shanghai

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    PYTHON="uv run python"
fi

SEND_BARK="${1:-}"
FAILURES=0
REPORT=""

check() {
    local name="$1"
    local result="$2"
    local detail="$3"
    if [ "$result" = "ok" ]; then
        REPORT="${REPORT}✅ ${name}: ${detail}\n"
    else
        REPORT="${REPORT}❌ ${name}: ${detail}\n"
        FAILURES=$((FAILURES + 1))
    fi
}

# 1. trade-web 服务状态
if systemctl is-active --quiet trade-web 2>/dev/null; then
    check "trade-web服务" "ok" "active"
else
    check "trade-web服务" "fail" "inactive或未安装"
fi

# 2. /api/health 端点
HEALTH_RESP=$(curl -sf http://127.0.0.1:8090/api/health 2>/dev/null || echo "")
if echo "$HEALTH_RESP" | grep -q '"status":"ok"' 2>/dev/null; then
    check "API健康" "ok" "status=ok"
else
    check "API健康" "fail" "无响应或status≠ok"
fi

# 3. 数据新鲜度 + 4. 持仓一致性 (Python 脚本)
HEALTH_JSON=$($PYTHON -c "
import json, glob, pandas as pd
from datetime import date, timedelta
from pathlib import Path

results = {}

# 数据新鲜度
dates = []
for f in glob.glob('data/cross_asset/*.parquet'):
    try:
        df = pd.read_parquet(f, columns=['trade_date'])
        if len(df): dates.append(str(df['trade_date'].max())[:10])
    except: pass
latest = max(dates) if dates else 'unknown'
today = date.today().isoformat()
# 允许数据滞后1天 (盘后更新)
yesterday = (date.today() - timedelta(days=1)).isoformat()
results['data_date'] = latest
results['data_fresh'] = latest >= yesterday
results['today'] = today

# 持仓一致性
state_file = Path('data/live/state.json')
if state_file.exists():
    state = json.loads(state_file.read_text())
    required = ['initial_capital', 'cash', 'holding', 'shares', 'trade_log']
    missing = [k for k in required if k not in state]
    results['state_ok'] = len(missing) == 0
    results['state_missing'] = missing
    results['holding'] = state.get('holding')
    results['cash'] = state.get('cash')
else:
    results['state_ok'] = False
    results['state_missing'] = ['file not found']

print(json.dumps(results, ensure_ascii=False))
" 2>/dev/null || echo '{"error": "python check failed"}')

DATA_DATE=$(echo "$HEALTH_JSON" | $PYTHON -c "import sys,json; d=json.load(sys.stdin); print(d.get('data_date','unknown'))" 2>/dev/null || echo "unknown")
DATA_FRESH=$(echo "$HEALTH_JSON" | $PYTHON -c "import sys,json; d=json.load(sys.stdin); print('ok' if d.get('data_fresh') else 'fail')" 2>/dev/null || echo "fail")
STATE_OK=$(echo "$HEALTH_JSON" | $PYTHON -c "import sys,json; d=json.load(sys.stdin); print('ok' if d.get('state_ok') else 'fail')" 2>/dev/null || echo "fail")

if [ "$DATA_FRESH" = "ok" ]; then
    check "数据新鲜度" "ok" "最新数据=${DATA_DATE}"
else
    check "数据新鲜度" "fail" "最新数据=${DATA_DATE} (期望≥昨天)"
fi

if [ "$STATE_OK" = "ok" ]; then
    HOLDING=$(echo "$HEALTH_JSON" | $PYTHON -c "import sys,json; d=json.load(sys.stdin); print(d.get('holding','空仓'))" 2>/dev/null || echo "unknown")
    check "持仓状态" "ok" "holding=${HOLDING}"
else
    check "持仓状态" "fail" "state.json 字段缺失或文件不存在"
fi

# 5. 最近 cron SUMMARY
LATEST_SUMMARY=$(grep 'SUMMARY' data/live/cron.log 2>/dev/null | tail -1 || echo "")
if [ -n "$LATEST_SUMMARY" ]; then
    LAST_EXIT=$(echo "$LATEST_SUMMARY" | grep -oP 'exit=\K[0-9]+' || echo "?")
    if [ "$LAST_EXIT" = "0" ]; then
        check "最近cron" "ok" "exit=0"
    else
        check "最近cron" "fail" "exit=${LAST_EXIT}"
    fi
else
    check "最近cron" "fail" "无SUMMARY记录"
fi

# 输出报告
echo ""
echo "=== 七星V3 健康检查 $(date '+%Y-%m-%d %H:%M:%S') ==="
echo ""
echo -e "$REPORT"
TOTAL_CHECKS=5
echo "总计: $((TOTAL_CHECKS - FAILURES))/${TOTAL_CHECKS} 通过, $FAILURES 失败"

# 失败时推送 Bark
if [ $FAILURES -gt 0 ] && [ "$SEND_BARK" = "--bark" ]; then
    $PYTHON -c "
import sys; sys.path.insert(0, 'scripts')
from notify import push_bark
push_bark('七星V3 健康检查告警', '${FAILURES}项检查失败, 请查看 health_check 输出')
" 2>&1 || echo "⚠️ Bark 推送失败"
fi

exit $FAILURES
