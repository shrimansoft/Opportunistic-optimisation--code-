"""Factory for creating warehouse environments."""

from typing import Dict, Any
from environment.warehouse_env import WarehouseEnv
from warehouse_sim.warehouse import Warehouse
from simulation.config import SimulationConfig


class EnvironmentFactory:
    """Factory for creating warehouse environments."""
    
    @staticmethod
    def create_basic_env(layout_size: str = "medium", seed: int = 42, **kwargs) -> WarehouseEnv:
        """Create basic warehouse environment."""
        warehouse = Warehouse(seed=seed, layout_size=layout_size, buffer_enabled=True,
                            column_height=8, difficulty="normal", **kwargs)
        return WarehouseEnv(warehouse)
    
    @staticmethod
    def create_research_env(layout_size: str = "large", reward_strategy: str = "combined",
                           observation_type: str = "grid_state", max_steps: int = 5000,
                           seed: int = 42) -> WarehouseEnv:
        """Create research-optimized environment."""
        warehouse = Warehouse(seed=seed, layout_size=layout_size, buffer_enabled=True,
                            column_height=10, difficulty="hard")
        return WarehouseEnv(warehouse, reward_strategy, observation_type, max_steps)
    
    @staticmethod
    def create_demo_env(layout_size: str = "medium", seed: int = 42) -> WarehouseEnv:
        """Create demo-optimized environment."""
        warehouse = Warehouse(seed=seed, layout_size=layout_size, buffer_enabled=True,
                            column_height=6, difficulty="normal")
        return WarehouseEnv(warehouse, "completed_orders", "shelf_counts", 1000)
    
    @staticmethod
    def create_from_config(config: SimulationConfig, layout_size: str, seed: int) -> WarehouseEnv:
        """Create environment from configuration."""
        warehouse = Warehouse(seed=seed, layout_size=layout_size, 
                            buffer_enabled=config.buffer_enabled,
                            column_height=config.column_height,
                            difficulty=config.difficulty)
        return WarehouseEnv(warehouse, config.reward_strategy, 
                          config.observation_type, config.max_steps)
    
    @staticmethod
    def create_custom_env(warehouse_params: Dict[str, Any], 
                         env_params: Dict[str, Any]) -> WarehouseEnv:
        """Create custom environment."""
        warehouse = Warehouse(**warehouse_params)
        return WarehouseEnv(warehouse=warehouse, **env_params)
