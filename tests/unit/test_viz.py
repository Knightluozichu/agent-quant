"""Tests for visualization module."""

from __future__ import annotations

import pytest


class TestDataLoader:
    """Tests for DashboardDataLoader."""

    def test_loader_init(self):
        """Test data loader initialization."""
        from a_share_quant.viz.data_loader import DashboardDataLoader

        loader = DashboardDataLoader("mock")
        assert loader.provider is not None

    def test_get_market_state(self):
        """Test market state data loading."""
        from a_share_quant.viz.data_loader import DashboardDataLoader

        loader = DashboardDataLoader("mock")
        state = loader.get_market_state()

        assert state.current_state is not None
        assert state.current_state_label is not None
        assert 0 <= state.confidence <= 1
        assert isinstance(state.state_distribution, dict)
        assert isinstance(state.state_history, list)

    def test_get_positions(self):
        """Test positions data loading."""
        from a_share_quant.viz.data_loader import DashboardDataLoader

        loader = DashboardDataLoader("mock")
        positions = loader.get_positions()

        assert isinstance(positions, list)
        assert len(positions) > 0

        pos = positions[0]
        assert pos.symbol is not None
        assert pos.quantity > 0
        assert pos.avg_cost > 0

    def test_get_trade_signals(self):
        """Test trade signals data loading."""
        from a_share_quant.viz.data_loader import DashboardDataLoader

        loader = DashboardDataLoader("mock")
        signals = loader.get_trade_signals()

        assert isinstance(signals, list)
        assert len(signals) > 0

        signal = signals[0]
        assert signal.symbol is not None
        assert signal.action in ["BUY", "SELL", "HOLD"]

    def test_get_evolution_data(self):
        """Test evolution data loading."""
        from a_share_quant.viz.data_loader import DashboardDataLoader

        loader = DashboardDataLoader("mock")
        evo = loader.get_evolution_data()

        assert evo.champion_name is not None
        assert isinstance(evo.champion_metrics, dict)
        assert 0 <= evo.promotion_progress <= 1

    def test_get_portfolio_summary(self):
        """Test portfolio summary."""
        from a_share_quant.viz.data_loader import DashboardDataLoader

        loader = DashboardDataLoader("mock")
        summary = loader.get_portfolio_summary()

        assert "total_assets" in summary
        assert "total_market_value" in summary
        assert "position_count" in summary


class TestComponents:
    """Tests for UI components."""

    def test_metric_card(self):
        """Test MetricCard rendering."""
        from a_share_quant.viz.components.cards import MetricCard

        card = MetricCard.render("Test", "123", "subtitle")
        assert card is not None

    def test_state_card(self):
        """Test StateCard rendering."""
        from a_share_quant.viz.components.cards import StateCard

        card = StateCard.render("UP_LOW", 0.8)
        assert card is not None

    def test_create_heatmap(self):
        """Test heatmap creation."""
        from a_share_quant.viz.components.charts import create_heatmap

        chart = create_heatmap(
            z=[[1, 2], [3, 4]],
            x=["A", "B"],
            y=["X", "Y"],
        )
        assert chart is not None

    def test_create_pie_chart(self):
        """Test pie chart creation."""
        from a_share_quant.viz.components.charts import create_pie_chart

        chart = create_pie_chart(
            labels=["A", "B"],
            values=[60, 40],
        )
        assert chart is not None


class TestLayouts:
    """Tests for page layouts."""

    def test_market_state_layout(self):
        """Test market state layout creation."""
        from a_share_quant.viz.data_loader import DashboardDataLoader
        from a_share_quant.viz.layouts.market_state import create_market_state_layout

        loader = DashboardDataLoader("mock")
        layout = create_market_state_layout(loader)
        assert layout is not None

    def test_positions_layout(self):
        """Test positions layout creation."""
        from a_share_quant.viz.data_loader import DashboardDataLoader
        from a_share_quant.viz.layouts.positions import create_positions_layout

        loader = DashboardDataLoader("mock")
        layout = create_positions_layout(loader)
        assert layout is not None

    def test_trade_plan_layout(self):
        """Test trade plan layout creation."""
        from a_share_quant.viz.data_loader import DashboardDataLoader
        from a_share_quant.viz.layouts.trade_plan import create_trade_plan_layout

        loader = DashboardDataLoader("mock")
        layout = create_trade_plan_layout(loader)
        assert layout is not None

    def test_evolution_layout(self):
        """Test evolution layout creation."""
        from a_share_quant.viz.data_loader import DashboardDataLoader
        from a_share_quant.viz.layouts.evolution import create_evolution_layout

        loader = DashboardDataLoader("mock")
        layout = create_evolution_layout(loader)
        assert layout is not None


class TestApp:
    """Tests for Dash app."""

    def test_create_app(self):
        """Test app creation."""
        from a_share_quant.viz.app import create_app

        app = create_app("mock")
        assert app is not None
        assert app.title == "A股量化仪表盘"
