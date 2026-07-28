# AGENTS.md - AI IDE 执行协议

本文件是 AI IDE（Trae、Cursor、Claude Code、Codex 等）执行本项目时的行为协议。

## 项目身份

- **项目名称**: A股家庭量化活体 (a-share-quant-life)
- **总协议文件**: `a_share_quant_ai_ide_implementation_plan.md`（位于项目根目录或用户指定位置）
- **目标**: 构建可解释、可复盘、受控进化的 A 股日频量化系统

## 执行纪律

### 每次会话开始

1. 读取本文件 (AGENTS.md)
2. 读取 `tasks/TASKS.md` 找到第一个未完成任务
3. 读取 `tasks/DECISIONS.md` 和 `tasks/ISSUES.md` 了解上下文
4. 检查 Git 状态
5. 声明当前任务的输入、输出和验收标准
6. 开始实施

### 每个任务的 SOP

```
理解需求 → 检查现有代码 → 写失败测试 → 最小实现 → 单元测试 →
集成测试 → 静态检查 → 自审代码 → 自审研究正确性 → 更新文档 →
更新任务账本 → 进入下一任务
```

### 任务完成标准

- [ ] 测试通过
- [ ] 文档更新
- [ ] 没有引入未来函数
- [ ] 没有硬编码秘密
- [ ] 没有破坏已有回归
- [ ] 输出符合领域 Schema
- [ ] 异常路径已测试
- [ ] TASKS.md 勾选
- [ ] DECISIONS.md 或 ISSUES.md 必要时更新

## 硬性约束

### 安全

- **绝不** 连接真实资金，除非用户单独明确授权
- **绝不** 在代码、日志、报告中打印 Token、密码或完整账号
- **绝不** 将 .env 或任何秘密提交 Git
- **绝不** 伪造 Token 或账户
- ALLOW_LIVE_TRADING 默认为 false
- 所有危险命令默认 dry-run

### 研究正确性

- **绝不** 使用未来数据（lookahead bias）
- **绝不** 为了提高回测结果修改封存测试集 (Locked Test)
- **绝不** 把一次盈利当作因子有效证据
- **绝不** 允许生产模型根据单笔盈亏直接修改自身
- **绝不** 承诺收益

### 工程

- 使用 uv + pyproject.toml + uv.lock 管理依赖
- 不允许只维护 requirements.txt
- 不在策略代码里直接调用第三方 SDK
- 数据源密钥只读取环境变量
- 无约束 dict 不进入核心接口

### 阻塞处理

缺少密钥、数据、接口时：
1. 在 ISSUES.md 创建 blocker
2. 创建 mock 或接口
3. 完成所有不依赖外部材料的任务
4. **不得停止整个工程**

## 技术栈

| 层 | 选择 |
|---|---|
| 语言 | Python 3.12 |
| 依赖管理 | uv + pyproject.toml + uv.lock |
| 数据计算 | NumPy, Pandas, PyArrow |
| 分析数据库 | DuckDB |
| 统计 | SciPy, statsmodels |
| 机器学习 | scikit-learn |
| 配置 | YAML + pydantic-settings |
| CLI | Typer |
| 日志 | structlog |
| 图表 | Plotly, Matplotlib, Jinja2 |
| 测试 | pytest, pytest-cov, Hypothesis |
| 代码质量 | Ruff, mypy |

## 目录结构

```
src/a_share_quant/
├── settings.py          # 应用配置
├── domain/              # 领域模型
├── data/                # 数据层
├── market_rules/        # 市场规则引擎
├── features/            # 因子计算
├── regimes/             # 市场状态识别
├── strategies/          # 策略
├── signals/             # 信号
├── frequency/           # 频率控制
├── portfolio/           # 组合
├── risk/                # 风控
├── execution/           # 执行
├── backtest/            # 回测引擎
├── attribution/         # 归因
├── evolution/           # 自进化
├── broker/              # 券商接口
├── monitoring/          # 监控
├── reporting/           # 报告
└── cli/                 # CLI
```

## 优先级

```
数据正确 > 无未来函数 > 真实可执行 > 风险可控 > 能解释 > 能复现 > 能回滚 > 收益
```
