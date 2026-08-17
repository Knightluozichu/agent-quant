"""Layout components for dashboard pages."""

from a_share_quant.viz.layouts.evolution import create_evolution_layout
from a_share_quant.viz.layouts.market_state import create_market_state_layout
from a_share_quant.viz.layouts.positions import create_positions_layout
from a_share_quant.viz.layouts.trade_plan import create_trade_plan_layout

__all__ = [
    "create_evolution_layout",
    "create_market_state_layout",
    "create_positions_layout",
    "create_trade_plan_layout",
]
