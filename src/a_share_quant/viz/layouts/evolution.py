"""Champion/Challenger evolution page layout."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dash_bootstrap_components as dbc
from dash import html

from a_share_quant.viz.components.charts import create_equity_curve

if TYPE_CHECKING:
    from a_share_quant.viz.data_loader import DashboardDataLoader, EvolutionData


def create_evolution_layout(loader: DashboardDataLoader) -> html.Div:
    """Create evolution page layout.

    Args:
        loader: Data loader instance

    Returns:
        Dash layout component
    """
    evo = loader.get_evolution_data()

    return html.Div(
        [
            # Page header
            html.H4("策略进化监控", className="mb-4"),
            # Strategy names
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("Champion (当前冠军)", className="text-muted"),
                                    html.H4(evo.champion_name, className="text-success"),
                                ]
                            ),
                            className="shadow-sm border-success",
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("Challenger (挑战者)", className="text-muted"),
                                    html.H4(
                                        evo.challenger_name or "暂无挑战者",
                                        className="text-warning"
                                        if evo.challenger_name
                                        else "text-muted",
                                    ),
                                ]
                            ),
                            className="shadow-sm border-warning",
                        ),
                        width=6,
                    ),
                ],
                className="mb-4",
            ),
            # Promotion progress
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("晋升进度", className="card-title mb-3"),
                                    _create_promotion_progress(evo),
                                ]
                            ),
                            className="shadow-sm",
                        ),
                        width=12,
                    ),
                ],
                className="mb-4",
            ),
            # Metrics comparison
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("关键指标对比", className="card-title mb-3"),
                                    _create_metrics_comparison(evo),
                                ]
                            ),
                            className="shadow-sm",
                        ),
                        width=12,
                    ),
                ],
                className="mb-4",
            ),
            # Equity curve comparison
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("收益曲线对比", className="card-title"),
                                    create_equity_curve(
                                        dates=["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8"],
                                        champion=[1.0, 1.02, 1.05, 1.03, 1.08, 1.12, 1.15, 1.185],
                                        challenger=[1.0, 1.03, 1.07, 1.06, 1.12, 1.18, 1.20, 1.215],
                                        benchmark=[1.0, 1.01, 1.02, 1.01, 1.04, 1.06, 1.08, 1.10],
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


def _create_promotion_progress(evo: EvolutionData) -> html.Div:
    """Create promotion progress display.

    Args:
        evo: Evolution data

    Returns:
        Dash component
    """
    # Sample size progress
    sample_progress = (
        evo.current_trades / evo.min_trades_required if evo.min_trades_required > 0 else 0
    )
    sample_progress = min(1.0, sample_progress)

    # Overall promotion progress
    overall_progress = evo.promotion_progress

    return html.Div(
        [
            # Sample size requirement
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("最小样本量要求", className="text-muted"),
                            html.Span(
                                f"{evo.current_trades} / {evo.min_trades_required} 笔交易",
                                className="float-end",
                            ),
                        ],
                        className="mb-1",
                    ),
                    dbc.Progress(
                        value=sample_progress * 100,
                        color="info" if sample_progress < 1 else "success",
                        striped=True,
                        label=f"{sample_progress:.0%}",
                        style={"height": "25px"},
                    ),
                ],
                className="mb-3",
            ),
            # Performance threshold
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("表现阈值达标", className="text-muted"),
                            html.Span(
                                "Sharpe > 1.0 且 MaxDD < 15%",
                                className="float-end text-muted",
                            ),
                        ],
                        className="mb-1",
                    ),
                    dbc.Progress(
                        value=100 if evo.challenger_metrics.get("sharpe", 0) > 1.0 else 50,
                        color="success"
                        if evo.challenger_metrics.get("sharpe", 0) > 1.0
                        else "warning",
                        striped=True,
                        style={"height": "25px"},
                    ),
                ],
                className="mb-3",
            ),
            # Overall promotion readiness
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("综合晋升就绪度", className="fw-bold"),
                            html.Span(
                                f"{overall_progress:.0%}",
                                className="float-end fw-bold",
                            ),
                        ],
                        className="mb-1",
                    ),
                    dbc.Progress(
                        value=overall_progress * 100,
                        color="success" if overall_progress >= 0.8 else "warning",
                        animated=True,
                        style={"height": "30px"},
                    ),
                ],
            ),
            # Status message
            html.Div(
                _get_promotion_status_message(evo),
                className="mt-3 text-center",
            ),
        ]
    )


def _get_promotion_status_message(evo: EvolutionData) -> dbc.Alert:
    """Get promotion status message.

    Args:
        evo: Evolution data

    Returns:
        Alert component
    """
    if evo.promotion_progress >= 0.9:
        return dbc.Alert(
            "挑战者即将达到晋升条件，请密切关注！",
            color="success",
            className="mb-0",
        )
    if evo.promotion_progress >= 0.7:
        return dbc.Alert(
            "挑战者表现良好，继续积累样本。",
            color="info",
            className="mb-0",
        )
    if evo.current_trades < evo.min_trades_required:
        remaining = evo.min_trades_required - evo.current_trades
        return dbc.Alert(
            f"样本量不足，还需 {remaining} 笔交易。",
            color="warning",
            className="mb-0",
        )
    return dbc.Alert(
        "挑战者表现未达标，继续观察。",
        color="secondary",
        className="mb-0",
    )


def _create_metrics_comparison(evo: EvolutionData) -> dbc.Table:
    """Create metrics comparison table.

    Args:
        evo: Evolution data

    Returns:
        Bootstrap table component
    """
    metrics = [
        ("Sharpe Ratio", "sharpe", "{:.2f}", True),
        ("最大回撤", "max_drawdown", "{:.1%}", False),  # Lower is better
        ("胜率", "win_rate", "{:.1%}", True),
        ("总收益", "total_return", "{:.1%}", True),
        ("交易次数", "trades", "{:.0f}", False),
    ]

    header = html.Thead(
        html.Tr(
            [
                html.Th("指标"),
                html.Th("Champion", className="text-success"),
                html.Th("Challenger", className="text-warning"),
                html.Th("差异"),
                html.Th("优胜"),
            ]
        )
    )

    rows = []
    for label, key, fmt, higher_better in metrics:
        champ_val = evo.champion_metrics.get(key, 0)
        chall_val = evo.challenger_metrics.get(key, 0)
        diff = chall_val - champ_val

        # Determine winner
        if higher_better:
            challenger_wins = chall_val > champ_val
        else:
            # For max_drawdown, less negative is better
            if key == "max_drawdown":
                challenger_wins = chall_val > champ_val  # -0.09 > -0.12
            else:
                challenger_wins = chall_val < champ_val

        winner_badge = (
            dbc.Badge(
                "Challenger",
                color="warning",
            )
            if challenger_wins
            else dbc.Badge(
                "Champion",
                color="success",
            )
        )

        diff_color = "success" if challenger_wins else "danger"

        rows.append(
            html.Tr(
                [
                    html.Td(label),
                    html.Td(fmt.format(champ_val)),
                    html.Td(fmt.format(chall_val)),
                    html.Td(
                        html.Span(
                            f"{diff:+.2f}" if "d" not in fmt else f"{diff:+.0f}",
                            className=f"text-{diff_color}",
                        )
                    ),
                    html.Td(winner_badge),
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
