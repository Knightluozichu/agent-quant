# 生产服务器运维手册 (V32二期 R2: 依赖与部署治理)

> 服务器: root@106.52.243.51:/opt/quant | 2026-08-07

## 依赖治理 (R2, 重要)

- **禁止在服务器执行 `uv sync` / `uv add`**: 本地 uv.lock 已含研究依赖
  (torch/torchvision/xgboost/lightgbm, ~200MB+), 服务器 .venv 为生产精简集
  (numpy/pandas/requests/akshare), 一旦 sync 会拉取全部研究依赖且可能破坏现有环境
- 若确需更新生产依赖: 本地 `uv export --no-dev -o requirements-prod.txt` 导出子集,
  服务器 `pip install -r requirements-prod.txt` (逐包验证, 禁止批量)
- 生产脚本运行一律用 `/opt/quant/.venv/bin/python` (cron 已配置)

## 部署流程 (V32 定向 scp)

1. 本地 git commit (留痕)
2. `scp scripts/xxx.py root@106.52.243.51:/opt/quant/scripts/`
3. 若改动 live_signal.py (trade_server import 它) → `systemctl restart trade-web`
4. cron 脚本 (live_signal/audit_risk) 无需重启, 下次运行自动加载
5. 更新 `/opt/quant/deploy/.DEPLOYED_MANIFEST` (git_hash/code_hash/deploy_time)
6. 验证: `live_signal.py --dry-run` + `health_check.sh` 第7/8项

## 定时任务 (每日核实)

- 14:50 run_daily (信号) | 16:30 run_calibrate | 21:30 run_data_sync | 每月1日16:00 audit_risk
- health_check.sh 第7项自动校验三任务存在性, 缺失自动 ensure_cron 恢复

## 风控状态 (V32)

- state.json: peak_equity/risk_exposure/cooldown_until/risk_log (risk_log 保留最近500条)
- risk_exposure 合法值 {0.7, 0.8, 1.0}; health_check 第8项校验
- 误杀审计: audit_risk.py (月度 cron 自动 + Bark)

## 公网访问 (2026-08-20 起)

- 记账网页: https://dj.luozichu.ink (nginx → 127.0.0.1:8090, 证书 certbot 自动续期)
- nginx 站点: /etc/nginx/sites-available/qixing (独立 vhost, 勿动 gfx 等既有站点)
- 公网前置安全项已全部就位: 密码+token 24h / 64KB 请求体上限 / 15s 全局超时
