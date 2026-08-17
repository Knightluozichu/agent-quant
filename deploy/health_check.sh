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
#   7. crontab 三任务完整性 (run_daily/run_calibrate/run_data_sync, 缺失时自动恢复)
#   8. V32 风控状态字段 (risk_exposure/cooldown_until/risk_log)
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

# 2. /api/health 端点 (2026-08-17 起需鉴权: 从 config.json 取未过期 token;
#    取不到有效 token 时标记跳过, 不误报 fail)
TOKEN=$($PYTHON -c "
import json, time
try:
    cfg = json.load(open('data/live/config.json'))
except Exception:
    cfg = {}
now = time.time()
for t in cfg.get('web_tokens', []):
    if isinstance(t, dict) and t.get('expires', 0) > now:
        print(t['token']); break
    elif isinstance(t, str):  # 兼容旧格式
        print(t); break
" 2>/dev/null || echo "")

if [ -z "$TOKEN" ]; then
    REPORT="${REPORT}⚠️ API健康: config.json 无未过期 web_token, 跳过端点检查\n"
else
    # token 经 stdin 传给 curl (-K -), 不出现在命令行
    HEALTH_RESP=$(printf 'header = "Cookie: qx_token=%s"\n' "$TOKEN" | curl -sf -K - http://127.0.0.1:8090/api/health 2>/dev/null || echo "")
    if echo "$HEALTH_RESP" | grep -q '"status":"ok"' 2>/dev/null; then
        check "API健康" "ok" "status=ok"
    else
        check "API健康" "fail" "无响应或status≠ok"
    fi
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

# 5. 最近 cron SUMMARY (cron.log 无 SUMMARY 行, 实际记录在 sync.log/calibrate.log)
LATEST_SUMMARY=$(grep -h 'SUMMARY' data/live/*.log 2>/dev/null | LC_ALL=C sort | tail -1 || echo "")
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

# 7. crontab 三任务校验 (C1: 防止 crontab 被外部覆盖导致定时任务静默丢失)
CRON_CURRENT=$(crontab -l 2>/dev/null || echo "")
CRON_REQUIRED_SCRIPTS=("run_daily.sh" "run_calibrate.sh" "run_data_sync.sh")
CRON_MISSING=()
for script in "${CRON_REQUIRED_SCRIPTS[@]}"; do
    if ! echo "$CRON_CURRENT" | grep -q "/opt/quant/deploy/${script}"; then
        CRON_MISSING+=("$script")
    fi
done

if [ ${#CRON_MISSING[@]} -eq 0 ]; then
    check "crontab三任务" "ok" "run_daily/run_calibrate/run_data_sync 均在crontab"
else
    # 自动调用 ensure_cron.sh 恢复, 恢复后复查是否真正补齐
    RESTORE_NOTE=""
    if [ -x "${PROJECT_ROOT}/deploy/ensure_cron.sh" ]; then
        bash "${PROJECT_ROOT}/deploy/ensure_cron.sh" >/dev/null 2>&1 || true
        CRON_AFTER=$(crontab -l 2>/dev/null || echo "")
        STILL_MISSING=()
        for script in "${CRON_MISSING[@]}"; do
            if ! echo "$CRON_AFTER" | grep -q "/opt/quant/deploy/${script}"; then
                STILL_MISSING+=("$script")
            fi
        done
        if [ ${#STILL_MISSING[@]} -eq 0 ]; then
            RESTORE_NOTE="已调用ensure_cron.sh自动恢复"
        else
            RESTORE_NOTE="ensure_cron.sh恢复后仍缺失: ${STILL_MISSING[*]}"
        fi
    else
        RESTORE_NOTE="ensure_cron.sh不可用, 需手动恢复"
    fi
    check "crontab三任务" "fail" "缺失: ${CRON_MISSING[*]} (${RESTORE_NOTE})"
fi

# 8. V32 风控状态字段校验 (A2/A6: 防止 risk_exposure/cooldown_until/risk_log 腐坏)
RISK_JSON=$($PYTHON -c "
import json
from pathlib import Path
from datetime import datetime

result = {}
state_file = Path('data/live/state.json')
if not state_file.exists():
    # 文件不存在 = 未初始化, 不视为异常
    result['risk_state'] = 'ok'
    result['detail'] = 'state.json不存在, 视为未初始化'
else:
    try:
        state = json.loads(state_file.read_text())
    except Exception as e:
        result['risk_state'] = 'fail'
        result['detail'] = f'state.json解析失败: {e}'
        print(json.dumps(result, ensure_ascii=False))
        raise SystemExit(0)

    # risk_exposure: 缺失=未初始化(正常); 值须 ∈ {0.7, 0.8, 1.0}
    re_ = state.get('risk_exposure')
    if re_ is None:
        re_ok, re_txt = True, '未初始化'
    elif isinstance(re_, bool) or not isinstance(re_, (int, float)):
        re_ok, re_txt = False, f'类型异常({type(re_).__name__})'
    elif any(abs(re_ - v) < 1e-9 for v in (0.7, 0.8, 1.0)):
        re_ok, re_txt = True, str(re_)
    else:
        re_ok, re_txt = False, str(re_)

    # cooldown_until: null 或合法日期 (兼容 ISO 格式及 Z 后缀)
    cu = state.get('cooldown_until')
    if cu is None:
        cu_ok, cu_txt = True, 'null'
    else:
        try:
            datetime.fromisoformat(str(cu).replace('Z', '+00:00'))
            cu_ok, cu_txt = True, str(cu)
        except ValueError:
            cu_ok, cu_txt = False, f'非法日期({cu})'

    # risk_log: 数组
    rl = state.get('risk_log')
    if rl is None:
        rl_ok, rl_txt = True, '未初始化'
    elif isinstance(rl, list):
        rl_ok, rl_txt = True, f'len={len(rl)}'
    else:
        rl_ok, rl_txt = False, f'类型异常({type(rl).__name__})'

    result['risk_state'] = 'ok' if (re_ok and cu_ok and rl_ok) else 'fail'
    result['detail'] = f'risk_exposure={re_txt} cooldown_until={cu_txt} risk_log={rl_txt}'
    print(json.dumps(result, ensure_ascii=False))
" 2>/dev/null || echo '{"risk_state":"fail","detail":"风控检查脚本异常"}')

RISK_STATE=$(echo "$RISK_JSON" | $PYTHON -c "import sys,json; d=json.load(sys.stdin); print(d.get('risk_state','fail'))" 2>/dev/null || echo "fail")
RISK_DETAIL=$(echo "$RISK_JSON" | $PYTHON -c "import sys,json; d=json.load(sys.stdin); print(d.get('detail','风控检查异常'))" 2>/dev/null || echo "风控检查异常")

if [ "$RISK_STATE" = "ok" ]; then
    check "风控状态字段" "ok" "$RISK_DETAIL"
else
    check "风控状态字段" "fail" "$RISK_DETAIL"
    # 风控字段腐坏 → 立即 Bark 提示 (不依赖 --bark 参数, 通过环境变量传参避免引号问题)
    RISK_DETAIL="$RISK_DETAIL" $PYTHON -c "
import os, sys; sys.path.insert(0, 'scripts')
from notify import push_bark
push_bark('七星V3 风控状态异常', os.environ['RISK_DETAIL'])
" 2>&1 || echo "⚠️ 风控Bark推送失败"
fi

# 输出报告
echo ""
echo "=== 七星V3 健康检查 $(date '+%Y-%m-%d %H:%M:%S') ==="
echo ""
echo -e "$REPORT"
TOTAL_CHECKS=7
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
