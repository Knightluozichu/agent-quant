# A股家庭量化活体

可解释、可复盘、受控进化的 A 股日频量化系统。

## 特性

- 📊 **9 状态市场识别**: 方向 × 震荡强度的完整市场状态模型
- 🎯 **5 种策略**: 趋势持有、回调波段、区间均值回归、熊市反弹、现金防御
- 🛡️ **完整风控**: T+1、涨跌停、停牌、频率限制、组合熔断
- 🔍 **盈亏归因**: 每笔交易可解释市场、行业、因子、策略、退出贡献
- 🧬 **受控进化**: Champion/Challenger 机制，样本外验证，Shadow/Canary 晋升
- 📝 **完整审计**: 信号→计划→订单→成交→归因全链路可追溯

## 快速开始

### 前置要求

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) 包管理器
- Git

### 安装

```bash
# 克隆仓库
git clone <repo-url>
cd quant

# 安装依赖
uv sync --locked

# 或安装开发依赖
uv sync --locked --all-extras
```

### 配置

```bash
# 复制环境配置模板
cp .env.example .env

# 编辑 .env 填入你的配置（可选）
# 默认使用 mock 数据源，无需任何密钥即可运行
```

### 运行健康检查

```bash
uv run quant doctor
```

### 同步服务器实盘状态并做盘中预演

```bash
# 只读拉取服务器持仓、现金、策略模式和 V4 状态，然后更新本地日线并 dry-run
uv run python scripts/sync_server_preview.py

# 仅同步状态，不获取行情、不运行策略
uv run python scripts/sync_server_preview.py --state-only
```

脚本只读取服务器的 `state.json`、`strategy_mode.json` 和 V4 发布清单，不读取
`.env`/`config.json`，也不会写服务器或连接券商。同步前会验证本地与服务器生产核心
文件哈希；本地旧状态保存在 `data/live/server_sync_backups/`。盘中结果仅供观察，正式
信号仍以服务器交易日 14:50 官方快照为准。

### 运行测试

```bash
# 全部测试
uv run pytest

# 带覆盖率
uv run pytest --cov=src/a_share_quant --cov-report=term-missing

# 只运行单元测试
uv run pytest tests/unit
```

### 代码质量

```bash
# Lint
uv run ruff check .

# 格式化
uv run ruff format .

# 类型检查
uv run mypy src
```

## 项目结构

```
quant/
├── src/a_share_quant/     # 源代码
│   ├── domain/            # 领域模型
│   ├── data/              # 数据层
│   ├── market_rules/      # 市场规则引擎
│   ├── features/          # 因子计算
│   ├── regimes/           # 市场状态识别
│   ├── strategies/        # 策略
│   ├── risk/              # 风控
│   ├── backtest/          # 回测引擎
│   ├── attribution/       # 归因
│   ├── evolution/         # 自进化
│   └── cli/               # CLI
├── tests/                 # 测试
├── configs/               # 配置文件
├── data/                  # 数据目录
├── tasks/                 # 任务跟踪
└── docs/                  # 文档
```

## CLI 命令

```bash
uv run quant doctor              # 系统健康检查
uv run quant config validate     # 验证配置
uv run quant data sync           # 同步数据
uv run quant backtest run        # 运行回测
uv run quant paper start         # 启动纸面交易
```

## 安全声明

- ⚠️ 本系统默认运行在 paper/dry-run 模式
- ⚠️ 不会自动连接真实资金
- ⚠️ 不承诺任何收益
- ⚠️ 量化交易存在风险，历史表现不代表未来收益

## 技术栈

- Python 3.12 + uv
- NumPy, Pandas, PyArrow, DuckDB
- scikit-learn, SciPy, statsmodels
- Typer, Rich, structlog
- pytest, Hypothesis, Ruff, mypy

## 许可

MIT
