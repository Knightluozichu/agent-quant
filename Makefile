# A股家庭量化活体 - Makefile
# 常用开发命令快捷方式

.PHONY: help install dev lint format typecheck test test-cov ci doctor clean

help: ## 显示帮助
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## 安装依赖（生产）
	uv sync --locked

dev: ## 安装依赖（开发）
	uv sync --locked --all-extras

lint: ## 运行 Ruff 检查
	uv run ruff check .

format: ## 格式化代码
	uv run ruff format .
	uv run ruff check --fix .

format-check: ## 检查格式（不修改）
	uv run ruff format --check .
	uv run ruff check .

typecheck: ## 运行 mypy 类型检查
	uv run mypy src

test: ## 运行测试
	uv run pytest -q

test-cov: ## 运行测试（带覆盖率）
	uv run pytest --cov=src/a_share_quant --cov-report=term-missing

test-unit: ## 只运行单元测试
	uv run pytest tests/unit -q

test-golden: ## 只运行 golden 场景测试
	uv run pytest tests/golden -q

ci: lint format-check typecheck test-cov ## 完整 CI 流程

doctor: ## 运行系统健康检查
	uv run quant doctor

clean: ## 清理缓存
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
