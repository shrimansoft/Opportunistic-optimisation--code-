"""
Warehouse Gymnasium Environment.
"""

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
    """
    Custom Gymnasium Environment for the Warehouse Simulation.
    
    This environment wraps the warehouse simulation and provides
    a standard Gymnasium interface for reinforcement learning.
    """

    def __init__(self, 
                 warehouse: 'Warehouse',
                 reward_strategy: str = "completed_orders",
                 observation_type: str = "shelf_counts",
                 max_steps: int = 10000):
        """
        Initialize the warehouse environment.
        
        Args:
            warehouse: The warehouse simulation instance
            reward_strategy: Strategy for reward calculation
            observation_type: Type of observation to generate
            max_steps: Maximum steps per episode
        """
        super(WarehouseEnv, self).__init__()

        self.warehouse: 'Warehouse' = warehouse
        self.max_steps = max_steps
        self.current_step = 0

        # Initialize modular components
        self.reward_calculator = RewardCalculator(reward_strategy)
        self.observation_handler = ObservationHandler(observation_type)
        self.metrics_display = MetricsDisplay()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)
        
        # Get observation space shape from handler
        obs_shape = self.observation_handler.get_observation_space_shape(warehouse)
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf,  # Changed from 1 to inf to accommodate various observation types
            shape=obs_shape, 
            dtype=np.int32
        )

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to the initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            Tuple of (observation, info)
        """
        if seed is not None:
            from utils.random_utils import set_seed
            set_seed(seed)
            
        self.warehouse.reset()
        self.current_step = 0
        obs = self.observation_handler.get_observation(self.warehouse)
        info = {}
        return obs, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment based on the action taken by the agent.
        
        Args:
            action: Action to take in the environment
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1

        # Execute warehouse simulation steps
        self._execute_warehouse_step()

        # Calculate reward using the reward calculator
        reward = self.reward_calculator.calculate_reward(self.warehouse)

        # Get observation using the observation handler
        obs = self.observation_handler.get_observation(self.warehouse)

        # Determine episode termination
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        
        info = self._get_step_info()

        return obs, reward, terminated, truncated, info

    def render(self, mode="human", pause_time=0.05) -> None:
        """
        Render the current state of the environment for visualization.
        
        Args:
            mode: Rendering mode
            pause_time: Pause time for visualization
        """
        try:
            # Use the warehouse renderer for visual display
            from warehouse_sim.renderer import render_warehouse
            render_warehouse(self.warehouse, mode=mode, pause_time=pause_time)
        except Exception as e:
            print(f"Visual render error: {e}")

        # Display metrics using the metrics display handler
        self.metrics_display.display_step_metrics(self.warehouse, self.current_step)
        self.metrics_display.display_robot_status(self.warehouse)
        self.metrics_display.display_color_legend()

    def _execute_warehouse_step(self) -> None:
        """Execute the core warehouse simulation step."""
        # Simulate the warehouse process (order creation and robot work)
        self.warehouse.order_step()

        # Perform robot movements with collision avoidance
        self.warehouse.robot_step_with_collision_avoidance()

        # Assign robots to tasks
        self.warehouse.robot_assigner()

    def _is_terminated(self) -> bool:
        """
        Check if the episode should be terminated.
        
        Returns:
            bool: True if episode should be terminated
        """
        # Episode ends when out of stock
        return self.warehouse.stock.sum() == 0

    def _is_truncated(self) -> bool:
        """
        Check if the episode should be truncated.
        
        Returns:
            bool: True if episode should be truncated
        """
        # Episode truncated at max steps
        return self.current_step >= self.max_steps

    def _get_step_info(self) -> Dict[str, Any]:
        """
        Get additional information about the current step.
        
        Returns:
            Dict containing step information
        """
        return {
            'step': self.current_step,
            'completed_orders': len(self.warehouse.order_compleated),
            'pending_orders': len(self.warehouse.order_buffer),
            'total_stock': int(self.warehouse.stock.sum()),
            'average_delay': self.warehouse.average_delay(),
        }
