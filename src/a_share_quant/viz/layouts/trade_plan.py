"""Trade plan page layout with signals and rejected orders."""

from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc

from a_share_quant.viz.components.cards import MetricCard
from a_share_quant.viz.data_loader import DashboardDataLoader, TradeSignalData


def create_trade_plan_layout(loader: DashboardDataLoader) -> html.Div:
    """Create trade plan page layout.

    Args:
        loader: Data loader instance

    Returns:
        Dash layout component
    """
    signals = loader.get_trade_signals()

    # Separate active and rejected signals
    active_signals = [s for s in signals if not s.rejected]
    rejected_signals = [s for s in signals if s.rejected]

    # Count by action
    buy_count = sum(1 for s in active_signals if s.action == "BUY")
    sell_count = sum(1 for s in active_signals if s.action == "SELL")
    hold_count = sum(1 for s in active_signals if s.action == "HOLD")

    return html.Div(
        [
            # Page header
            html.H4("今日交易计划", className="mb-4"),

            # Summary cards
            dbc.Row(
                [
                    dbc.Col(
                        MetricCard.render(
                            "买入信号",
                            str(buy_count),
                            "待执行",
                            color="success",
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        MetricCard.render(
                            "卖出信号",
                            str(sell_count),
                            "待执行",
                            color="danger",
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        MetricCard.render(
                            "持有信号",
                            str(hold_count),
                            "继续持有",
                            color="info",
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        MetricCard.render(
                            "拒绝信号",
                            str(len(rejected_signals)),
                            "风控拦截",
                            color="warning",
                        ),
                        width=3,
                    ),
                ],
                className="mb-4",
            ),

            # Active signals
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("交易信号", className="card-title mb-3"),
                                    _create_signals_list(active_signals),
                                ]
                            ),
                            className="shadow-sm",
                        ),
                        width=12,
                    ),
                ],
                className="mb-4",
            ),

            # Rejected signals
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("拒绝信号 (风控拦截)", className="card-title mb-3"),
                                    _create_rejected_table(rejected_signals),
                                ]
                            ),
                            className="shadow-sm border-warning",
                        ),
                        width=12,
                    ),
                ],
            ),
        ]
    )


def _create_signals_list(signals: list[TradeSignalData]) -> html.Div:
    """Create signals list with cards.

    Args:
        signals: List of trade signals

    Returns:
        Dash component
    """
    if not signals:
        return html.P("今日无交易信号", className="text-muted")

    cards = []
    for signal in signals:
        cards.append(_create_signal_card(signal))

    return html.Div(cards)


def _create_signal_card(signal: TradeSignalData) -> dbc.Card:
    """Create individual signal card.

    Args:
        signal: Trade signal data

    Returns:
        Bootstrap card component
    """
    # Action color mapping
    action_colors = {
        "BUY": "success",
        "SELL": "danger",
        "HOLD": "info",
    }
    action_labels = {
        "BUY": "买入",
        "SELL": "卖出",
        "HOLD": "持有",
    }

    color = action_colors.get(signal.action, "secondary")
    label = action_labels.get(signal.action, signal.action)

    return dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        # Action badge
                        dbc.Col(
                            dbc.Badge(
                                label,
                                color=color,
                                className="px-3 py-2",
                                style={"fontSize": "1rem"},
                            ),
                            width="auto",
                            className="d-flex align-items-center",
                        ),

                        # Symbol and strategy
                        dbc.Col(
                            [
                                html.H6(html.Code(signal.symbol), className="mb-1"),
                                html.Small(
                                    f"策略: {signal.strategy}",
                                    className="text-muted",
                                ),
                            ],
                            width=3,
                        ),

                        # Reason
                        dbc.Col(
                            [
                                html.Small("信号原因", className="text-muted d-block"),
                                html.Span(signal.reason),
                            ],
                            width=4,
                        ),

                        # Confidence
                        dbc.Col(
                            [
                                html.Small("置信度", className="text-muted d-block"),
                                dbc.Progress(
                                    value=signal.confidence * 100,
                                    color=color,
                                    label=f"{signal.confidence:.0%}",
                                    style={"height": "20px"},
                                ),
                            ],
                            width=3,
                        ),
                    ],
                    className="align-items-center",
                ),

                # Target and stop-loss (if available)
                html.Hr(className="my-2") if (signal.target_price or signal.stop_loss) else None,
                dbc.Row(
                    [
                        dbc.Col(
                            html.Small(f"目标价: ¥{signal.target_price:.3f}")
                            if signal.target_price else None,
                            width="auto",
                        ),
                        dbc.Col(
                            html.Small(f"止损价: ¥{signal.stop_loss:.3f}", className="text-danger")
                            if signal.stop_loss else None,
                            width="auto",
                        ),
                    ],
                    className="mt-2",
                ) if (signal.target_price or signal.stop_loss) else None,
            ]
        ),
        className="mb-2 shadow-sm",
        style={"borderLeft": f"4px solid var(--bs-{color})"},
    )


def _create_rejected_table(signals: list[TradeSignalData]) -> dbc.Table:
    """Create rejected signals table.

    Args:
        signals: List of rejected signals

    Returns:
        Bootstrap table component
    """
    if not signals:
        return html.P("无拒绝信号", className="text-muted")

    header = html.Thead(
        html.Tr(
            [
                html.Th("代码"),
                html.Th("动作"),
                html.Th("策略"),
                html.Th("拒绝原因"),
                html.Th("置信度"),
            ]
        )
    )

    rows = []
    for signal in signals:
        rows.append(
            html.Tr(
                [
                    html.Td(html.Code(signal.symbol)),
                    html.Td(
                        dbc.Badge(
                            signal.action,
                            color="warning",
                        )
                    ),
                    html.Td(signal.strategy),
                    html.Td(
                        dbc.Badge(
                            signal.reject_reason,
                            color="danger",
                            className="me-1",
                        )
                    ),
                    html.Td(f"{signal.confidence:.0%}"),
                ],
                className="table-warning",
            )
        )

    return dbc.Table(
        [header, html.Tbody(rows)],
        bordered=True,
        hover=True,
        responsive=True,
        size="sm",
    )
