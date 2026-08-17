"""Card components for dashboard metrics."""

from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc


class MetricCard:
    """Metric display card."""

    @staticmethod
    def render(
        title: str,
        value: str,
        subtitle: str = "",
        color: str = "primary",
        icon: str = "",
    ) -> dbc.Card:
        """Render a metric card.

        Args:
            title: Card title
            value: Main value to display
            subtitle: Optional subtitle
            color: Bootstrap color (primary, success, danger, warning, info)
            icon: Optional icon name
        """
        return dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.H6(title, className="card-subtitle mb-2 text-muted"),
                            html.H3(value, className=f"card-title text-{color}"),
                            html.Small(subtitle, className="text-muted") if subtitle else None,
                        ]
                    )
                ]
            ),
            className="mb-3 shadow-sm",
        )


class StateCard:
    """Market state display card with color coding."""

    STATE_COLORS = {
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

    STATE_LABELS = {
        "UP_LOW": "上涨 + 低波动",
        "UP_MEDIUM": "上涨 + 中波动",
        "UP_HIGH": "上涨 + 高波动",
        "FLAT_LOW": "震荡 + 低波动",
        "FLAT_MEDIUM": "震荡 + 中波动",
        "FLAT_HIGH": "震荡 + 高波动",
        "DOWN_LOW": "下跌 + 低波动",
        "DOWN_MEDIUM": "下跌 + 中波动",
        "DOWN_HIGH": "下跌 + 高波动",
    }

    @classmethod
    def render(cls, state: str, confidence: float = 0.0) -> dbc.Card:
        """Render a market state card.

        Args:
            state: Market state (e.g., "UP_LOW")
            confidence: Detection confidence (0-1)
        """
        color = cls.STATE_COLORS.get(state, "secondary")
        label = cls.STATE_LABELS.get(state, state)

        return dbc.Card(
            dbc.CardBody(
                [
                    html.H6("当前市场状态", className="card-subtitle mb-2 text-muted"),
                    html.H2(
                        label,
                        className=f"text-{color} text-center mb-3",
                        style={"fontWeight": "bold"},
                    ),
                    html.Div(
                        [
                            html.Span("状态码: ", className="text-muted"),
                            html.Code(state),
                        ],
                        className="text-center mb-2",
                    ),
                    dbc.Progress(
                        value=confidence * 100,
                        color=color,
                        striped=True,
                        label=f"置信度 {confidence:.0%}",
                    )
                    if confidence > 0
                    else None,
                ]
            ),
            className="mb-3 shadow-sm border-2",
            style={"borderColor": f"var(--bs-{color})"} if color != "secondary" else {},
        )
