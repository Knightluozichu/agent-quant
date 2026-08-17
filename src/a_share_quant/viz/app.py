"""Dash application for visualization dashboard."""

from __future__ import annotations

from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

from a_share_quant.viz.data_loader import DashboardDataLoader
from a_share_quant.viz.layouts import (
    create_market_state_layout,
    create_positions_layout,
    create_trade_plan_layout,
    create_evolution_layout,
)


def create_app(provider_type: str = "mock") -> Dash:
    """Create and configure the Dash application.

    Args:
        provider_type: Data provider type ("mock", "joinquant", etc.)

    Returns:
        Configured Dash application
    """
    # Initialize data loader
    loader = DashboardDataLoader(provider_type)

    # Create Dash app with Bootstrap theme
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title="A股量化仪表盘",
        suppress_callback_exceptions=True,
    )

    # Navigation tabs
    tabs = dbc.Tabs(
        [
            dbc.Tab(label="市场状态", tab_id="market-state"),
            dbc.Tab(label="持仓管理", tab_id="positions"),
            dbc.Tab(label="交易计划", tab_id="trade-plan"),
            dbc.Tab(label="策略进化", tab_id="evolution"),
        ],
        id="tabs",
        active_tab="market-state",
        className="mb-4",
    )

    # Main layout
    app.layout = dbc.Container(
        [
            # Header
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2(
                                "A股家庭量化活体",
                                className="text-primary",
                            ),
                            html.P(
                                "日频量化系统监控仪表盘",
                                className="text-muted",
                            ),
                        ],
                        width=12,
                    ),
                ],
                className="my-4",
            ),
            # Navigation tabs
            tabs,
            # Tab content
            html.Div(id="tab-content"),
            # Store for provider type
            dcc.Store(id="provider-type", data=provider_type),
            # Footer
            html.Hr(),
            html.Footer(
                html.P(
                    "仅供研究使用，不构成投资建议",
                    className="text-muted text-center small",
                ),
                className="my-4",
            ),
        ],
        fluid=True,
        className="px-4",
    )

    # Register callbacks
    _register_callbacks(app, loader)

    return app


def _register_callbacks(app: Dash, loader: DashboardDataLoader) -> None:
    """Register Dash callbacks.

    Args:
        app: Dash application
        loader: Data loader instance
    """
    from dash import Input, Output

    @app.callback(
        Output("tab-content", "children"),
        Input("tabs", "active_tab"),
    )
    def render_tab_content(active_tab: str) -> html.Div:
        """Render content based on active tab.

        Args:
            active_tab: Active tab ID

        Returns:
            Layout component for the active tab
        """
        if active_tab == "market-state":
            return create_market_state_layout(loader)
        elif active_tab == "positions":
            return create_positions_layout(loader)
        elif active_tab == "trade-plan":
            return create_trade_plan_layout(loader)
        elif active_tab == "evolution":
            return create_evolution_layout(loader)
        else:
            return html.Div("未知页面", className="text-danger")


def run_dashboard(
    provider_type: str = "mock",
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
) -> None:
    """Run the dashboard server.

    Args:
        provider_type: Data provider type
        host: Server host address
        port: Server port
        debug: Enable debug mode
    """
    app = create_app(provider_type)
    print(f"\n🚀 启动仪表盘: http://{host}:{port}")
    print(f"   数据源: {provider_type}")
    print("   按 Ctrl+C 停止\n")
    app.run(host=host, port=port, debug=debug)
