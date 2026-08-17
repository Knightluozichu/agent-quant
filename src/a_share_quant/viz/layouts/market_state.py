"""Market state page layout."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dash_bootstrap_components as dbc
from dash import html

from a_share_quant.viz.components.cards import MetricCard, StateCard
from a_share_quant.viz.components.charts import create_heatmap, create_pie_chart

if TYPE_CHECKING:
    from a_share_quant.viz.data_loader import DashboardDataLoader


def create_market_state_layout(loader: DashboardDataLoader) -> html.Div:
    """Create market state page layout.

    Args:
        loader: Data loader instance

    Returns:
        Dash layout component
    """
    # Get market state data
    state_data = loader.get_market_state()

    # Build 9-state heatmap data
    directions = ["UP", "FLAT", "DOWN"]
    volatilities = ["LOW", "MEDIUM", "HIGH"]

    # Create z matrix (3x3)
    z_matrix = []
    text_matrix = []
    for d in directions:
        row = []
        text_row = []
        for v in volatilities:
            state_key = f"{d}_{v}"
            count = state_data.state_distribution.get(state_key, 0)
            row.append(count)
            text_row.append(str(count))
        z_matrix.append(row)
        text_matrix.append(text_row)

    # State distribution for pie chart
    pie_labels = list(state_data.state_distribution.keys())
    pie_values = list(state_data.state_distribution.values())

    # State colors
    state_colors = [
        "#2ca02c",
        "#17becf",
        "#ff7f0e",  # UP: green, cyan, orange
        "#7f7f7f",
        "#bcbd22",
        "#ff7f0e",  # FLAT: gray, yellow-green, orange
        "#d62728",
        "#d62728",
        "#1f1f1f",  # DOWN: red, red, dark
    ]

    return html.Div(
        [
            # Page header
            html.H4("市场状态监控", className="mb-4"),
            # Current state card (full width)
            dbc.Row(
                [
                    dbc.Col(
                        StateCard.render(state_data.current_state, state_data.confidence),
                        width=12,
                    ),
                ],
                className="mb-4",
            ),
            # Key metrics row
            dbc.Row(
                [
                    dbc.Col(
                        MetricCard.render(
                            "推荐策略",
                            state_data.recommended_strategy,
                            "基于当前状态",
                            color="info",
                        ),
                        width=4,
                    ),
                    dbc.Col(
                        MetricCard.render(
                            "分析样本",
                            str(len(state_data.state_history)),
                            "历史状态点",
                            color="secondary",
                        ),
                        width=4,
                    ),
                    dbc.Col(
                        MetricCard.render(
                            "数据源",
                            "510300.SSE",
                            "沪深300ETF",
                            color="primary",
                        ),
                        width=4,
                    ),
                ],
                className="mb-4",
            ),
            # Charts row
            dbc.Row(
                [
                    # 9-state heatmap
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("9状态分布热力图", className="card-title"),
                                    create_heatmap(
                                        z=z_matrix,
                                        x=volatilities,
                                        y=directions,
                                        title="",
                                        colorscale="YlOrRd",
                                        text=text_matrix,
                                    ),
                                ]
                            ),
                            className="shadow-sm",
                        ),
                        width=6,
                    ),
                    # State distribution pie
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("状态分布", className="card-title"),
                                    create_pie_chart(
                                        labels=pie_labels,
                                        values=pie_values,
                                        title="",
                                        colors=state_colors[: len(pie_labels)],
                                    ),
                                ]
                            ),
                            className="shadow-sm",
                        ),
                        width=6,
                    ),
                ],
                className="mb-4",
            ),
            # State history timeline
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("状态转换历史", className="card-title"),
                                    _create_state_timeline(state_data.state_history),
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


def _create_state_timeline(history: list[dict]) -> dbc.Table:
    """Create state history timeline table.

    Args:
        history: List of state history records

    Returns:
        Bootstrap table component
    """
    if not history:
        return html.P("暂无历史数据", className="text-muted")

    # Show last 20 records
    recent = history[-20:][::-1]

    header = html.Thead(
        html.Tr(
            [
                html.Th("日期"),
                html.Th("状态"),
                html.Th("方向"),
                html.Th("震荡"),
            ]
        )
    )

    rows = []
    for record in recent:
        state_badge = _get_state_badge(record["state"])
        rows.append(
            html.Tr(
                [
                    html.Td(str(record["date"])),
                    html.Td(state_badge),
                    html.Td(record["direction"]),
                    html.Td(record["oscillation"]),
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


def _get_state_badge(state: str) -> dbc.Badge:
    """Get colored badge for state."""
    color_map = {
        "UP_LOW": "success",
        "UP_MEDIUM": "info",
        "UP_HIGH": "warning",
        "FLAT_LOW": "secondary",
        "FLAT_MEDIUM": "secondary",
        "FLAT_HIGH": "warning",
        "DOWN_LOW": "danger",
        "DOWN_MEDIUM": "danger",
        "DOWN_HIGH": "dark",
    }
    color = color_map.get(state, "secondary")
    return dbc.Badge(state, color=color, className="me-1")
