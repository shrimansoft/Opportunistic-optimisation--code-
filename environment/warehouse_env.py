"""Warehouse Gymnasium Environment."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import TYPE_CHECKING, Tuple, Any, Dict

from .reward_system import RewardCalculator
from .observation_handler import ObservationHandler
from .metrics_display import MetricsDisplay

if TYPE_CHECKING:
    from warehouse_sim.warehouse import Warehouse


class WarehouseEnv(gym.Env):
    """Custom Gymnasium Environment for Warehouse Simulation."""

    def __init__(self, warehouse: 'Warehouse', reward_strategy: str = "completed_orders",
                 observation_type: str = "shelf_counts", max_steps: int = 10000):
        super().__init__()
        
        self.warehouse = warehouse
        self.max_steps = max_steps
        self.current_step = 0

        # Initialize components
        self.reward_calculator = RewardCalculator(reward_strategy)
        self.observation_handler = ObservationHandler(observation_type)
        self.metrics_display = MetricsDisplay()

        # Define spaces
        self.action_space = spaces.Discrete(4)
        obs_shape = self.observation_handler.get_observation_space_shape(warehouse)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=obs_shape, dtype=np.int32)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            from utils.random_utils import set_seed
            set_seed(seed)
            
        self.warehouse.reset()
        self.current_step = 0
        obs = self.observation_handler.get_observation(self.warehouse)
        return obs, {}

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        self._execute_warehouse_step()
        
        reward = self.reward_calculator.calculate_reward(self.warehouse)
        obs = self.observation_handler.get_observation(self.warehouse)
        terminated = self.warehouse.stock.sum() == 0
        truncated = self.current_step >= self.max_steps
        
        info = {
            'step': self.current_step,
            'completed_orders': len(self.warehouse.order_compleated),
            'pending_orders': len(self.warehouse.order_buffer),
            'total_stock': int(self.warehouse.stock.sum()),
            'average_delay': self.warehouse.average_delay(),
        }
        
        return obs, reward, terminated, truncated, info

    def render(self, mode="human", pause_time=0.05) -> None:
        try:
            from warehouse_sim.renderer import render_warehouse
            render_warehouse(self.warehouse, mode=mode, pause_time=pause_time)
        except Exception as e:
            print(f"Visual render error: {e}")

        self.metrics_display.display_step_metrics(self.warehouse, self.current_step)
        self.metrics_display.display_robot_status(self.warehouse)
        self.metrics_display.display_color_legend()

    def _execute_warehouse_step(self) -> None:
        """Execute warehouse simulation step."""
        self.warehouse.order_step()
        self.warehouse.robot_step_with_collision_avoidance()
        self.warehouse.robot_assigner()
