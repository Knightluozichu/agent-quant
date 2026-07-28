"""Positions page layout with stop-loss/take-profit visualization."""

from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc

from a_share_quant.viz.components.cards import MetricCard
from a_share_quant.viz.components.charts import create_drawdown_chart
from a_share_quant.viz.data_loader import DashboardDataLoader, PositionData


def create_positions_layout(loader: DashboardDataLoader) -> html.Div:
    """Create positions page layout.

    Args:
        loader: Data loader instance

    Returns:
        Dash layout component
    """
    positions = loader.get_positions()
    summary = loader.get_portfolio_summary()

    return html.Div(
        [
            # Page header
            html.H4("持仓管理", className="mb-4"),

            # Portfolio summary cards
            dbc.Row(
                [
                    dbc.Col(
                        MetricCard.render(
                            "总资产",
                            f"¥{summary['total_assets']:,.0f}",
                            "含现金",
                            color="primary",
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        MetricCard.render(
                            "持仓市值",
                            f"¥{summary['total_market_value']:,.0f}",
                            f"{summary['position_count']} 只标的",
                            color="info",
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        MetricCard.render(
                            "浮动盈亏",
                            f"¥{summary['total_unrealized_pnl']:+,.0f}",
                            f"{summary['total_unrealized_pnl_pct']:+.2%}",
                            color="success" if summary['total_unrealized_pnl'] >= 0 else "danger",
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        MetricCard.render(
                            "可用现金",
                            f"¥{summary['cash']:,.0f}",
                            "可买入",
                            color="secondary",
                        ),
                        width=3,
                    ),
                ],
                className="mb-4",
            ),

            # Positions table
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("持仓明细", className="card-title mb-3"),
                                    _create_positions_table(positions),
                                ]
                            ),
                            className="shadow-sm",
                        ),
                        width=12,
                    ),
                ],
                className="mb-4",
            ),

            # Position detail cards with stop-loss/take-profit
            dbc.Row(
                [
                    dbc.Col(
                        _create_position_detail_card(pos),
                        width=6,
                        className="mb-3",
                    )
                    for pos in positions
                ],
            ),

            # Drawdown chart (mock data)
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("组合回撤", className="card-title"),
                                    create_drawdown_chart(
                                        dates=["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"],
                                        drawdowns=[0, -0.02, -0.05, -0.03, -0.08, -0.04],
                                    ),
                                ]
                            ),
                            className="shadow-sm",
                        ),
                        width=12,
                    ),
                ],
            ),
        ]
    )


def _create_positions_table(positions: list[PositionData]) -> dbc.Table:
    """Create positions table.

    Args:
        positions: List of position data

    Returns:
        Bootstrap table component
    """
    header = html.Thead(
        html.Tr(
            [
                html.Th("代码"),
                html.Th("数量"),
                html.Th("成本价"),
                html.Th("现价"),
                html.Th("市值"),
                html.Th("盈亏"),
                html.Th("盈亏%"),
                html.Th("止损"),
                html.Th("止盈"),
                html.Th("持有天数"),
            ]
        )
    )

    rows = []
    for pos in positions:
        pnl_color = "success" if pos.unrealized_pnl >= 0 else "danger"
        rows.append(
            html.Tr(
                [
                    html.Td(html.Code(pos.symbol)),
                    html.Td(f"{pos.quantity:,}"),
                    html.Td(f"¥{pos.avg_cost:.3f}"),
                    html.Td(f"¥{pos.current_price:.3f}"),
                    html.Td(f"¥{pos.market_value:,.0f}"),
                    html.Td(
                        html.Span(f"¥{pos.unrealized_pnl:+,.0f}", className=f"text-{pnl_color}")
                    ),
                    html.Td(
                        html.Span(f"{pos.unrealized_pnl_pct:+.2%}", className=f"text-{pnl_color}")
                    ),
                    html.Td(f"¥{pos.stop_loss:.3f}" if pos.stop_loss else "-"),
                    html.Td(f"¥{pos.take_profit:.3f}" if pos.take_profit else "-"),
                    html.Td(f"{pos.days_held}天"),
                ]
            )
        )

    return dbc.Table(
        [header, html.Tbody(rows)],
        bordered=True,
        hover=True,
        responsive=True,
        size="sm",
    )


def _create_position_detail_card(pos: PositionData) -> dbc.Card:
    """Create detailed position card with stop-loss/take-profit levels.

    Args:
        pos: Position data

    Returns:
        Bootstrap card component
    """
    pnl_color = "success" if pos.unrealized_pnl >= 0 else "danger"

    # Calculate price position relative to stop-loss and take-profit
    if pos.stop_loss and pos.take_profit:
        price_range = pos.take_profit - pos.stop_loss
        if price_range > 0:
            price_position = (pos.current_price - pos.stop_loss) / price_range * 100
            price_position = max(0, min(100, price_position))
        else:
            price_position = 50
    else:
        price_position = 50

    return dbc.Card(
        dbc.CardBody(
            [
                # Header
                html.Div(
                    [
                        html.H6(html.Code(pos.symbol), className="d-inline"),
                        dbc.Badge(
                            f"{pos.unrealized_pnl_pct:+.2%}",
                            color=pnl_color,
                            className="ms-2",
                        ),
                    ],
                    className="mb-3",
                ),

                # Price info
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Small("成本价", className="text-muted d-block"),
                                html.Span(f"¥{pos.avg_cost:.3f}"),
                            ],
                            width=4,
                        ),
                        dbc.Col(
                            [
                                html.Small("现价", className="text-muted d-block"),
                                html.Span(f"¥{pos.current_price:.3f}", className=f"text-{pnl_color}"),
                            ],
                            width=4,
                        ),
                        dbc.Col(
                            [
                                html.Small("盈亏", className="text-muted d-block"),
                                html.Span(f"¥{pos.unrealized_pnl:+,.0f}", className=f"text-{pnl_color}"),
                            ],
                            width=4,
                        ),
                    ],
                    className="mb-3",
                ),

                # Stop-loss / Take-profit progress bar
                html.Div(
                    [
                        html.Div(
                            [
                                html.Small(f"止损 ¥{pos.stop_loss:.3f}", className="text-danger"),
                                html.Small(f"止盈 ¥{pos.take_profit:.3f}", className="text-success float-end"),
                            ],
                            className="mb-1",
                        ),
                        dbc.Progress(
                            value=price_position,
                            color=pnl_color,
                            striped=True,
                            style={"height": "20px"},
                        ),
                        html.Small(
                            f"现价位置: {price_position:.0f}%",
                            className="text-muted",
                        ),
                    ],
                ),
            ]
        ),
        className="shadow-sm",
    )
