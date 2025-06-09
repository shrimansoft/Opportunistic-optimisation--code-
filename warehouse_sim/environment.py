"""
Gymnasium Environment for Warehouse Simulation

This module provides a Gymnasium-compatible environment wrapper for the warehouse
simulation, enabling reinforcement learning experimentation.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional

from .warehouse import Warehouse


class WarehouseEnv(gym.Env):
    """
    Custom Gymnasium Environment for the Warehouse Simulation.

    This environment wraps the warehouse simulation to provide a standard
    reinforcement learning interface.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, warehouse: Warehouse, max_steps: int = 500):
        """
        Initialize the warehouse environment.

        Args:
            warehouse: The warehouse simulation instance
            max_steps: Maximum number of steps per episode
        """
        super(WarehouseEnv, self).__init__()

        self.warehouse = warehouse
        self.max_steps = max_steps
        self.current_step = 0

        # Define action space
        # Actions: 0=Idle, 1=Optimize robot assignment, 2=Adjust buffer strategy, 3=Emergency mode
        self.action_space = spaces.Discrete(4)

        # Define observation space
        # Observation includes: shelf states, robot states, order queue info, buffer states
        obs_size = (
            400
            + len(warehouse.robots) * 4  # shelf item counts (400 shelves)
            + 50  # robot states (position x, y, mode, availability)
            + 10  # item availability (50 item types)  # additional metrics (pending orders, average delay, etc.)
        )

        self.observation_space = spaces.Box(
            low=0, high=100, shape=(obs_size,), dtype=np.float32
        )

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        self.warehouse.reset()
        self.current_step = 0

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: The action to take

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1

        # Execute action (this is where RL agent decisions would be implemented)
        self._execute_action(action)

        # Run warehouse simulation step
        self.warehouse.order_step()

        # Update robot states
        for robot in self.warehouse.robots:
            robot.step()

        # Assign robots to pending orders
        self.warehouse.robot_assigner()

        # Calculate reward
        reward = self._calculate_reward()

        # Get new observation
        obs = self._get_observation()

        # Check termination conditions
        terminated = self.warehouse.stock.sum() == 0  # Out of stock
        truncated = self.current_step >= self.max_steps  # Max steps reached

        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human"):
        """
        Render the environment.

        Args:
            mode: Rendering mode ('human' or 'rgb_array')
        """
        if mode == "human":
            # Generate visualization frame
            self.warehouse.shelf_plot("temp_frames")
            print(
                f"Step {self.current_step}: "
                f"Completed Orders: {len(self.warehouse.order_compleated)}, "
                f"Pending: {len(self.warehouse.order_buffer)}, "
                f"Avg Delay: {self.warehouse.average_delay():.2f}"
            )
        elif mode == "rgb_array":
            # This would return an RGB array for programmatic use
            # For now, just return a placeholder
            return np.zeros((600, 800, 3), dtype=np.uint8)

    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation state.

        Returns:
            Observation array
        """
        obs_parts = []

        # Shelf states (400 values)
        shelf_counts = [len(shelf) for shelf in self.warehouse.shelfs]
        obs_parts.extend(shelf_counts)

        # Robot states (4 values per robot: x, y, mode, availability)
        for robot in self.warehouse.robots:
            x, y = robot.current_location
            obs_parts.extend([x, y, robot.mode, 1.0 if robot.available else 0.0])

        # Item availability (50 values)
        availability = self.warehouse.available()
        obs_parts.extend([1.0 if avail else 0.0 for avail in availability])

        # Additional metrics (10 values)
        obs_parts.extend(
            [
                len(self.warehouse.order_buffer),  # pending orders
                len(self.warehouse.order_compleated),  # completed orders
                self.warehouse.average_delay(),  # average delay
                self.warehouse.stock.sum(),  # total stock
                self.current_step,  # current time step
                len(
                    [r for r in self.warehouse.robots if not r.available]
                ),  # busy robots
                # Buffer-related metrics (if buffers enabled)
                sum(len(station.buffer) for station in self.warehouse.picking_stations),
                sum(station.buffer_size for station in self.warehouse.picking_stations),
                # Additional padding to reach 10 values
                0.0,
                0.0,
            ]
        )

        return np.array(obs_parts, dtype=np.float32)

    def _execute_action(self, action: int):
        """
        Execute the given action in the environment.

        Args:
            action: Action to execute
        """
        # Action interpretation:
        # 0: Idle - no special action
        # 1: Optimize robot assignment - prioritize closest robots
        # 2: Adjust buffer strategy - not implemented in base warehouse
        # 3: Emergency mode - prioritize oldest orders

        if action == 1:
            # Optimize robot assignment (this would require modifying warehouse logic)
            pass
        elif action == 2:
            # Adjust buffer strategy
            pass
        elif action == 3:
            # Emergency mode - could modify order processing priority
            pass
        # action == 0 is idle, no special behavior

    def _calculate_reward(self) -> float:
        """
        Calculate the reward for the current state.

        Returns:
            Reward value
        """
        # Reward components:
        # 1. Completed orders (positive)
        # 2. Penalty for high average delay (negative)
        # 3. Penalty for pending orders (negative)
        # 4. Bonus for efficient robot utilization (positive)

        completed_orders = len(self.warehouse.order_compleated)
        pending_orders = len(self.warehouse.order_buffer)
        avg_delay = self.warehouse.average_delay()

        # Calculate robot utilization
        busy_robots = len([r for r in self.warehouse.robots if not r.available])
        total_robots = len(self.warehouse.robots)
        utilization = busy_robots / total_robots if total_robots > 0 else 0

        # Reward calculation
        reward = (
            completed_orders * 1.0
            + -avg_delay * 0.1  # Positive for completions
            + -pending_orders * 0.05  # Penalty for delays
            + utilization  # Penalty for queue buildup
            * 0.5  # Bonus for keeping robots busy
        )

        return reward

    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information about the current state.

        Returns:
            Info dictionary
        """
        return {
            "current_step": self.current_step,
            "completed_orders": len(self.warehouse.order_compleated),
            "pending_orders": len(self.warehouse.order_buffer),
            "average_delay": self.warehouse.average_delay(),
            "total_stock": int(self.warehouse.stock.sum()),
            "robot_utilization": len(
                [r for r in self.warehouse.robots if not r.available]
            )
            / len(self.warehouse.robots),
            "buffer_status": [
                {
                    "station_id": i,
                    "buffer_size": station.buffer_size,
                    "buffer_count": len(station.buffer),
                    "buffer_enabled": station.buffer_enabled,
                }
                for i, station in enumerate(self.warehouse.picking_stations)
            ],
        }

    def close(self):
        """Clean up the environment."""
        pass
