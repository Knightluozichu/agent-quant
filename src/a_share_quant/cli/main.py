"""CLI entry point for the A-share quant system.

Usage:
    uv run quant doctor
    uv run quant config validate
    uv run quant backtest run --start 2024-01-01 --end 2024-06-30
    uv run quant strategy list
    uv run quant viz --port 8050
    uv run quant --help
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from a_share_quant.settings import Settings, get_settings

app = typer.Typer(
    name="quant",
    help="A股家庭量化活体 - 日频量化系统 CLI",
    no_args_is_help=True,
)
console = Console()

# Sub-command groups
config_app = typer.Typer(help="Configuration management", no_args_is_help=True)
data_app = typer.Typer(help="Data management", no_args_is_help=True)
backtest_app = typer.Typer(help="Backtesting", no_args_is_help=True)
strategy_app = typer.Typer(help="Strategy management", no_args_is_help=True)

app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")
app.add_typer(backtest_app, name="backtest")
app.add_typer(strategy_app, name="strategy")


@app.command()
def doctor() -> None:
    """Run system health checks.

    Verifies environment, dependencies, and configuration without
    exposing any secrets.
    """
    console.print("\n[bold cyan]🩺 A股量化系统健康检查[/bold cyan]\n")

    checks_passed = 0
    checks_failed = 0
    checks_warned = 0

    def check_ok(msg: str) -> None:
        nonlocal checks_passed
        checks_passed += 1
        console.print(f"  [green]✓[/green] {msg}")

    def check_fail(msg: str) -> None:
        nonlocal checks_failed
        checks_failed += 1
        console.print(f"  [red]✗[/red] {msg}")

    def check_warn(msg: str) -> None:
        nonlocal checks_warned
        checks_warned += 1
        console.print(f"  [yellow]⚠[/yellow] {msg}")

    # --- Python version ---
    console.print("[bold]Python 环境[/bold]")
    py_version = sys.version_info
    if py_version >= (3, 12):
        check_ok(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        check_fail(f"Python {py_version.major}.{py_version.minor} < 3.12")

    # --- Settings ---
    console.print("\n[bold]配置[/bold]")
    try:
        settings = get_settings()
        check_ok("Settings 加载成功")
    except Exception as e:  # noqa: BLE001
        check_fail(f"Settings 加载失败: {e}")
        console.print("\n[red]配置加载失败，无法继续检查[/red]")
        raise typer.Exit(code=1) from e

    # --- Environment file ---
    env_file = Path(".env")
    env_example = Path(".env.example")
    if env_file.exists():
        check_ok(".env 文件存在")
    elif env_example.exists():
        check_warn(".env 不存在，使用默认配置（可从 .env.example 复制）")
    else:
        check_warn(".env 和 .env.example 均不存在")

    # --- Data provider ---
    console.print("\n[bold]数据源[/bold]")
    check_ok(f"数据提供者: {settings.data_provider.value}")

    if settings.data_provider.value == "joinquant":
        if settings.joinquant_username and settings.joinquant_password:
            check_ok("聚宽凭证已配置")
        else:
            check_fail("聚宽凭证未配置")
    elif settings.data_provider.value == "tushare":
        if settings.tushare_token:
            check_ok("Tushare Token 已配置")
        else:
            check_fail("Tushare Token 未配置")
    elif settings.data_provider.value == "mock":
        check_ok("Mock 模式，无需外部凭证")

    # --- Secret status (never print values) ---
    console.print("\n[bold]密钥状态[/bold]")
    secret_status = settings.get_secret_status()
    for key, status in secret_status.items():
        if status == "configured":
            check_ok(f"{key}: 已配置")
        else:
            check_warn(f"{key}: 未配置")

    # --- Directories ---
    console.print("\n[bold]目录[/bold]")
    dirs_to_check = [
        ("data", settings.data_root),
        ("artifacts", settings.artifact_root),
        ("reports", settings.report_root),
    ]
    for name, path in dirs_to_check:
        if path.exists():
            check_ok(f"{name}: {path}")
        else:
            check_warn(f"{name}: {path} 不存在（将自动创建）")

    # --- Broker ---
    console.print("\n[bold]券商/交易[/bold]")
    check_ok(f"Broker: {settings.broker_provider.value}")
    check_ok(f"Dry-run: {settings.dry_run}")
    check_ok(f"Live trading 允许: {settings.allow_live_trading}")

    if settings.is_live_trading_allowed():
        check_warn("⚠️  Live trading 已启用！请确认这是预期行为。")
    else:
        check_ok("Live trading 安全锁已启用")

    # --- Summary ---
    console.print("\n[bold]检查摘要[/bold]")
    table = Table(show_header=False)
    table.add_row("[green]通过[/green]", str(checks_passed))
    table.add_row("[yellow]警告[/yellow]", str(checks_warned))
    table.add_row("[red]失败[/red]", str(checks_failed))
    console.print(table)

    if checks_failed > 0:
        console.print("\n[red]存在失败项，请修复后重试[/red]")
        raise typer.Exit(code=1)

    console.print("\n[green]✓ 系统健康检查通过[/green]\n")


@config_app.command("validate")
def config_validate() -> None:
    """Validate configuration files."""
    console.print("[cyan]验证配置文件...[/cyan]")

    settings = get_settings()
    console.print(f"  数据提供者: {settings.data_provider.value}")
    console.print(f"  初始资金: {settings.initial_capital:,.0f} {settings.base_currency}")
    console.print(f"  回测引擎: {settings.backtest_engine}")
    console.print(f"  模拟 T+1: {settings.simulate_t_plus_one}")
    console.print(f"  模拟涨跌停: {settings.simulate_price_limits}")
    console.print(f"  模拟停牌: {settings.simulate_suspensions}")

    # Check market rules file
    if settings.market_rules_path.exists():
        console.print(f"  [green]✓[/green] 市场规则文件: {settings.market_rules_path}")
    else:
        console.print(f"  [yellow]⚠[/yellow] 市场规则文件不存在: {settings.market_rules_path}")

    console.print("\n[green]配置验证完成[/green]")


@backtest_app.command("run")
def backtest_run(
    start: str = typer.Option(..., "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., "--end", "-e", help="End date (YYYY-MM-DD)"),
    symbols: str = typer.Option("510300.SSE", "--symbols", help="Comma-separated symbols"),
    capital: float = typer.Option(1_000_000, "--capital", "-c", help="Initial capital"),
    strategy: str = typer.Option("TREND_HOLD", "--strategy", help="Strategy name"),
) -> None:
    """Run a backtest."""
    from a_share_quant.backtest import BacktestConfig, BacktestEngine, OrderEvent
    from a_share_quant.data.providers.mock import MockProvider
    from a_share_quant.regime import RegimeDetector
    from a_share_quant.strategies import get_strategy

    console.print(f"\n[bold cyan]📊 运行回测[/bold cyan]")
    console.print(f"  期间: {start} → {end}")
    console.print(f"  标的: {symbols}")
    console.print(f"  资金: {capital:,.0f}")
    console.print(f"  策略: {strategy}\n")

    # Parse inputs
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    symbol_list = [s.strip() for s in symbols.split(",")]

    # Setup
    provider = MockProvider()
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=capital,
        symbols=symbol_list,
    )
    engine = BacktestEngine(provider, config)
    strat = get_strategy(strategy)
    detector = RegimeDetector()

    # Strategy function
    def strategy_fn(trade_date, market_data, positions):
        orders = []
        for symbol in symbol_list:
            if symbol not in market_data:
                continue

            # Get historical data
            df = provider.get_daily_bars(symbol, start_date, trade_date)
            if df.empty:
                continue

            # Detect regime
            regime = detector.detect(df, trade_date)

            # Get position info
            pos_info = None
            if symbol in positions:
                from a_share_quant.strategies import PositionInfo

                pos = positions[symbol]
                pos_info = PositionInfo(
                    symbol=symbol,
                    quantity=pos["quantity"],
                    avg_cost=pos["avg_cost"],
                    sellable=pos["sellable"],
                    current_price=market_data[symbol]["close"],
                    unrealized_pnl=0,
                    holding_days=0,
                )

            # Generate signal
            signal = strat.generate_signal(symbol, df, regime, pos_info, trade_date)

            if signal and signal.action == "BUY" and pos_info is None:
                # Calculate position size (simple: 10% of capital)
                quantity = int(capital * 0.1 / signal.entry_price)
                quantity = (quantity // 100) * 100
                if quantity > 0:
                    orders.append(
                        OrderEvent(
                            trade_date=trade_date,
                            symbol=symbol,
                            side="BUY",
                            quantity=quantity,
                            strategy_name=strategy,
                        )
                    )

            elif signal and signal.action == "SELL" and pos_info:
                orders.append(
                    OrderEvent(
                        trade_date=trade_date,
                        symbol=symbol,
                        side="SELL",
                        quantity=pos_info.sellable,
                        strategy_name=strategy,
                    )
                )

            # Check exit conditions
            if pos_info:
                should_exit, reason = strat.should_exit(pos_info, df, regime, trade_date)
                if should_exit and pos_info.sellable > 0:
                    orders.append(
                        OrderEvent(
                            trade_date=trade_date,
                            symbol=symbol,
                            side="SELL",
                            quantity=pos_info.sellable,
                            strategy_name=strategy,
                        )
                    )

        return orders

    # Run backtest
    with console.status("[bold green]回测运行中..."):
        result = engine.run(strategy_fn)

    # Display results
    console.print("\n[bold]回测结果[/bold]\n")

    metrics_table = Table(show_header=False, title="绩效指标")
    metrics_table.add_column("指标", style="cyan")
    metrics_table.add_column("数值", style="green")

    m = result.metrics
    metrics_table.add_row("总收益率", f"{m.total_return:.2%}")
    metrics_table.add_row("年化收益率", f"{m.annual_return:.2%}")
    metrics_table.add_row("波动率", f"{m.volatility:.2%}")
    metrics_table.add_row("夏普比率", f"{m.sharpe_ratio:.2f}")
    metrics_table.add_row("最大回撤", f"{m.max_drawdown:.2%}")
    metrics_table.add_row("胜率", f"{m.win_rate:.2%}")
    metrics_table.add_row("盈亏比", f"{m.profit_factor:.2f}")
    metrics_table.add_row("交易次数", str(m.total_trades))
    metrics_table.add_row("平均持有天数", f"{m.avg_holding_days:.1f}")

    console.print(metrics_table)

    # Final equity
    final_equity = result.equity_curve.iloc[-1] if len(result.equity_curve) > 0 else capital
    console.print(
        f"\n[bold]期末资金:[/bold] {final_equity:,.0f} ({(final_equity / capital - 1):.2%})"
    )


@strategy_app.command("list")
def strategy_list() -> None:
    """List available strategies."""
    from a_share_quant.strategies import STRATEGY_REGISTRY

    console.print("\n[bold cyan]📋 可用策略[/bold cyan]\n")

    table = Table(title="策略列表")
    table.add_column("名称", style="cyan")
    table.add_column("描述", style="green")
    table.add_column("适用状态", style="yellow")

    regime_map = {
        "TREND_HOLD": "UP_LOW, UP_MEDIUM",
        "PULLBACK_SWING": "UP_MEDIUM, UP_HIGH, FLAT_MEDIUM",
        "RANGE_MEAN_REVERSION": "FLAT_LOW, FLAT_MEDIUM",
        "BEAR_REBOUND": "DOWN_LOW, DOWN_MEDIUM",
        "CASH_DEFENSE": "DOWN_HIGH",
    }

    for name, cls in STRATEGY_REGISTRY.items():
        table.add_row(name, cls.description, regime_map.get(name, ""))

    console.print(table)


@strategy_app.command("info")
def strategy_info(name: str = typer.Argument(..., help="Strategy name")) -> None:
    """Show strategy details."""
    from a_share_quant.strategies import get_strategy, STRATEGY_REGISTRY

    if name not in STRATEGY_REGISTRY:
        console.print(f"[red]未知策略: {name}[/red]")
        raise typer.Exit(code=1)

    strat = get_strategy(name)

    console.print(f"\n[bold cyan]📊 {name}[/bold cyan]")
    console.print(f"  描述: {strat.description}")
    console.print(f"\n[bold]默认参数:[/bold]")
    for key, value in strat.params.items():
        console.print(f"  {key}: {value}")
    console.print()


@data_app.command("test")
def data_test(
    provider: str = typer.Option("mock", "--provider", "-p", help="Provider: mock, joinquant"),
) -> None:
    """Test data provider connection."""
    console.print(f"\n[bold cyan]🔌 测试数据源: {provider}[/bold cyan]\n")

    if provider == "mock":
        from a_share_quant.data.providers.mock import MockProvider

        p = MockProvider()
        console.print("  [green]✓[/green] MockProvider 初始化成功")

        # Test calendar
        cal = p.get_trading_calendar("SSE", date(2024, 1, 1), date(2024, 1, 31))
        console.print(f"  [green]✓[/green] 交易日历: {len(cal)} 天")

        # Test daily bars
        bars = p.get_daily_bars("510300.SSE", date(2024, 1, 1), date(2024, 1, 31))
        console.print(f"  [green]✓[/green] 日线数据: {len(bars)} 条")

    elif provider == "joinquant":
        try:
            from a_share_quant.data.providers.joinquant import JoinQuantProvider

            p = JoinQuantProvider()
            console.print("  认证中...")
            p._ensure_auth()
            console.print("  [green]✓[/green] JQData 认证成功")

            # Quota info
            quota = p.get_quota_info()
            console.print(f"  今日已用: {quota.get('tracked_used', 0):,}")
            console.print(f"  剩余配额: {quota.get('tracked_remaining', 0):,}")

            # Test calendar (small request)
            console.print("  测试交易日历...")
            cal = p.get_trading_calendar("SSE", date(2025, 6, 1), date(2025, 6, 30))
            console.print(f"  [green]✓[/green] 交易日历: {len(cal)} 天")

            # Test daily bars (small request)
            # Note: Trial account data range: 2025-04-11 to 2026-04-18
            console.print("  测试日线数据...")
            bars = p.get_daily_bars("510300.SSE", date(2025, 6, 1), date(2025, 6, 15))
            console.print(f"  [green]✓[/green] 日线数据: {len(bars)} 条")

            console.print("\n[green]✓ JQData 连接测试通过[/green]")

        except ImportError:
            console.print("  [red]✗[/red] jqdatasdk 未安装")
            console.print("  运行: uv pip install jqdatasdk")
            raise typer.Exit(code=1)
        except Exception as e:
            console.print(f"  [red]✗[/red] 错误: {e}")
            raise typer.Exit(code=1) from e

    else:
        console.print(f"[red]未知数据源: {provider}[/red]")
        raise typer.Exit(code=1)

    console.print()


@app.command()
def viz(
    provider: str = typer.Option("mock", "--provider", "-p", help="Data provider: mock, joinquant"),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Server host"),
    port: int = typer.Option(8050, "--port", "-P", help="Server port"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
) -> None:
    """Launch the visualization dashboard.

    Starts a Dash web server for monitoring market state, positions,
    trade plans, and strategy evolution.

    Example:
        uv run quant viz --provider mock --port 8050
    """
    try:
        from a_share_quant.viz import run_dashboard
    except ImportError as e:
        console.print("[red]✗ 可视化模块未安装[/red]")
        console.print("  运行: uv sync --all-extras")
        console.print(f"  错误: {e}")
        raise typer.Exit(code=1) from e

    console.print(f"\n[bold cyan]🚀 启动可视化仪表盘[/bold cyan]")
    console.print(f"  地址: http://{host}:{port}")
    console.print(f"  数据源: {provider}")
    console.print(f"  调试模式: {'开启' if debug else '关闭'}")
    console.print("  按 Ctrl+C 停止\n")

    run_dashboard(provider_type=provider, host=host, port=port, debug=debug)


if __name__ == "__main__":
    app()
