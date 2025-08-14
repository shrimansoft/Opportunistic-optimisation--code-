"""Reward calculation system."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from warehouse_sim.warehouse import Warehouse


class RewardCalculator:
    """Handles reward calculation strategies."""
    
    def __init__(self, strategy: str = "completed_orders"):
        self.strategy = strategy
    
    def calculate_reward(self, warehouse: 'Warehouse') -> float:
        """Calculate reward based on warehouse state."""
        strategies = {
            "completed_orders": self._completed_orders,
            "efficiency": self._efficiency,
            "combined": self._combined
        }
        return strategies.get(self.strategy, self._completed_orders)(warehouse)
    
    def _completed_orders(self, warehouse: 'Warehouse') -> float:
        return float(len(warehouse.order_compleated))
    
    def _efficiency(self, warehouse: 'Warehouse') -> float:
        busy_robots = len([r for r in warehouse.robots if not r.available])
        utilization = (busy_robots / len(warehouse.robots)) if warehouse.robots else 0
        delay_penalty = max(0, 1.0 - (warehouse.average_delay() / 100.0))
        return (utilization * 0.6) + (delay_penalty * 0.4)
    
    def _combined(self, warehouse: 'Warehouse') -> float:
        return (self._completed_orders(warehouse) * 0.7) + (self._efficiency(warehouse) * 0.3)
