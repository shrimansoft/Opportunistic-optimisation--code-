"""
Reward calculation system for the warehouse environment.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from warehouse_sim.warehouse import Warehouse


class RewardCalculator:
    """
    Handles reward calculation for the warehouse environment.
    Encapsulates different reward strategies and metrics.
    """
    
    def __init__(self, reward_strategy: str = "completed_orders"):
        """
        Initialize the reward calculator.
        
        Args:
            reward_strategy: The strategy to use for reward calculation.
                           Options: "completed_orders", "efficiency", "combined"
        """
        self.reward_strategy = reward_strategy
    
    def calculate_reward(self, warehouse: 'Warehouse') -> float:
        """
        Calculate reward based on the current warehouse state.
        
        Args:
            warehouse: The warehouse instance to calculate reward for
            
        Returns:
            float: The calculated reward value
        """
        if self.reward_strategy == "completed_orders":
            return self._completed_orders_reward(warehouse)
        elif self.reward_strategy == "efficiency":
            return self._efficiency_reward(warehouse)
        elif self.reward_strategy == "combined":
            return self._combined_reward(warehouse)
        else:
            raise ValueError(f"Unknown reward strategy: {self.reward_strategy}")
    
    def _completed_orders_reward(self, warehouse: 'Warehouse') -> float:
        """
        Reward based on the number of completed orders.
        
        Args:
            warehouse: The warehouse instance
            
        Returns:
            float: Reward based on completed orders
        """
        return float(len(warehouse.order_compleated))
    
    def _efficiency_reward(self, warehouse: 'Warehouse') -> float:
        """
        Reward based on efficiency metrics (robot utilization, delay, etc.).
        
        Args:
            warehouse: The warehouse instance
            
        Returns:
            float: Reward based on efficiency
        """
        # Calculate robot utilization
        busy_robots = len([r for r in warehouse.robots if not r.available])
        robot_utilization = (busy_robots / len(warehouse.robots)) if warehouse.robots else 0
        
        # Calculate delay penalty (lower delay is better)
        avg_delay = warehouse.average_delay()
        delay_penalty = max(0, 1.0 - (avg_delay / 100.0))  # Normalize delay
        
        # Combine utilization and delay
        efficiency_score = (robot_utilization * 0.6) + (delay_penalty * 0.4)
        return efficiency_score
    
    def _combined_reward(self, warehouse: 'Warehouse') -> float:
        """
        Combined reward using both completed orders and efficiency.
        
        Args:
            warehouse: The warehouse instance
            
        Returns:
            float: Combined reward score
        """
        orders_reward = self._completed_orders_reward(warehouse)
        efficiency_reward = self._efficiency_reward(warehouse)
        
        # Weight completed orders more heavily than efficiency
        return (orders_reward * 0.7) + (efficiency_reward * 0.3)
