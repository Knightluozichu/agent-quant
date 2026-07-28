"""Visualization module for A-share quant system.

Provides Dash-based dashboard for monitoring:
- Market regime state (9-state model)
- Daily trade plans and signals
- Position management with stop-loss/take-profit
- Champion/Challenger strategy evolution
"""

from a_share_quant.viz.app import create_app, run_dashboard

__all__ = ["create_app", "run_dashboard"]
