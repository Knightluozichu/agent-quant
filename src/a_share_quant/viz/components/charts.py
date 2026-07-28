"""Chart components using Plotly."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import plotly.express as px
from dash import dcc


def create_heatmap(
    z: list[list[float]],
    x: list[str],
    y: list[str],
    title: str = "",
    colorscale: str = "RdYlGn",
    text: list[list[str]] | None = None,
) -> dcc.Graph:
    """Create a heatmap chart.

    Args:
        z: 2D array of values
        x: X-axis labels
        y: Y-axis labels
        title: Chart title
        colorscale: Plotly colorscale
        text: Optional text annotations for cells
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=colorscale,
            text=text,
            texttemplate="%{text}" if text else None,
            hovertemplate="x: %{x}<br>y: %{y}<br>value: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return dcc.Graph(figure=fig)


def create_pie_chart(
    labels: list[str],
    values: list[float],
    title: str = "",
    colors: list[str] | None = None,
) -> dcc.Graph:
    """Create a pie chart.

    Args:
        labels: Category labels
        values: Category values
        title: Chart title
        colors: Optional custom colors
    """
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker_colors=colors,
            )
        ]
    )
    fig.update_layout(
        title=title,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True,
    )
    return dcc.Graph(figure=fig)


def create_time_series(
    x: list[Any],
    y: list[float],
    title: str = "",
    yaxis_title: str = "",
    color: str = "#1f77b4",
    fill: bool = False,
) -> dcc.Graph:
    """Create a time series line chart.

    Args:
        x: X-axis values (dates)
        y: Y-axis values
        title: Chart title
        yaxis_title: Y-axis label
        color: Line color
        fill: Whether to fill area under curve
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color=color, width=2),
            fill="tozeroy" if fill else None,
        )
    )
    fig.update_layout(
        title=title,
        yaxis_title=yaxis_title,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
    )
    return dcc.Graph(figure=fig)


def create_equity_curve(
    dates: list[Any],
    champion: list[float],
    challenger: list[float] | None = None,
    benchmark: list[float] | None = None,
    title: str = "收益曲线对比",
) -> dcc.Graph:
    """Create equity curve comparison chart.

    Args:
        dates: Date axis
        champion: Champion strategy returns
        challenger: Optional challenger returns
        benchmark: Optional benchmark returns
        title: Chart title
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=champion,
            mode="lines",
            name="Champion",
            line=dict(color="#2ca02c", width=2),
        )
    )

    if challenger:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=challenger,
                mode="lines",
                name="Challenger",
                line=dict(color="#ff7f0e", width=2, dash="dash"),
            )
        )

    if benchmark:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=benchmark,
                mode="lines",
                name="Benchmark",
                line=dict(color="#7f7f7f", width=1, dash="dot"),
            )
        )

    fig.update_layout(
        title=title,
        yaxis_title="累计收益",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return dcc.Graph(figure=fig)


def create_candlestick(
    dates: list[Any],
    open_prices: list[float],
    high_prices: list[float],
    low_prices: list[float],
    close_prices: list[float],
    title: str = "K线图",
) -> dcc.Graph:
    """Create a candlestick chart.

    Args:
        dates: Date axis
        open_prices: Open prices
        high_prices: High prices
        low_prices: Low prices
        close_prices: Close prices
        title: Chart title
    """
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=dates,
                open=open_prices,
                high=high_prices,
                low=low_prices,
                close=close_prices,
            )
        ]
    )
    fig.update_layout(
        title=title,
        yaxis_title="价格",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_rangeslider_visible=False,
    )
    return dcc.Graph(figure=fig)


def create_drawdown_chart(
    dates: list[Any],
    drawdowns: list[float],
    title: str = "回撤曲线",
) -> dcc.Graph:
    """Create a drawdown chart.

    Args:
        dates: Date axis
        drawdowns: Drawdown values (negative)
        title: Chart title
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=drawdowns,
            mode="lines",
            fill="tozeroy",
            line=dict(color="#d62728", width=1),
            fillcolor="rgba(214, 39, 40, 0.3)",
        )
    )
    fig.update_layout(
        title=title,
        yaxis_title="回撤",
        yaxis_tickformat=".1%",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
    )
    return dcc.Graph(figure=fig)
