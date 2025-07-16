"""
Factory for creating different types of warehouse environments.
"""

from typing import Optional, Dict, Any
from environment.warehouse_env import WarehouseEnv
from warehouse_sim.warehouse import Warehouse
from simulation.config import SimulationConfig


class EnvironmentFactory:
    """
    Factory class for creating warehouse environments with different configurations.
    Provides a clean interface for environment creation with presets.
    """
    
    @staticmethod
    def create_basic_env(layout_size: str = "medium", 
                        seed: int = 42,
                        **kwargs) -> WarehouseEnv:
        """
        Create a basic warehouse environment with default settings.
        
        Args:
            layout_size: Size of the warehouse layout
            seed: Random seed
            **kwargs: Additional warehouse parameters
            
        Returns:
            WarehouseEnv: Configured environment
        """
        warehouse = Warehouse(
            seed=seed,
            layout_size=layout_size,
            buffer_enabled=True,
            column_height=8,
            difficulty="normal",
            **kwargs
        )
        
        return WarehouseEnv(warehouse)
    
    @staticmethod
    def create_research_env(layout_size: str = "large",
                           reward_strategy: str = "combined",
                           observation_type: str = "grid_state",
                           max_steps: int = 5000,
                           seed: int = 42) -> WarehouseEnv:
        """
        Create an environment optimized for research and experimentation.
        
        Args:
            layout_size: Size of the warehouse layout
            reward_strategy: Reward calculation strategy
            observation_type: Type of observations
            max_steps: Maximum steps per episode
            seed: Random seed
            
        Returns:
            WarehouseEnv: Research-optimized environment
        """
        warehouse = Warehouse(
            seed=seed,
            layout_size=layout_size,
            buffer_enabled=True,
            column_height=10,
            difficulty="hard"
        )
        
        return WarehouseEnv(
            warehouse=warehouse,
            reward_strategy=reward_strategy,
            observation_type=observation_type,
            max_steps=max_steps
        )
    
    @staticmethod
    def create_demo_env(layout_size: str = "medium",
                       seed: int = 42) -> WarehouseEnv:
        """
        Create an environment optimized for demonstrations.
        
        Args:
            layout_size: Size of the warehouse layout
            seed: Random seed
            
        Returns:
            WarehouseEnv: Demo-optimized environment
        """
        warehouse = Warehouse(
            seed=seed,
            layout_size=layout_size,
            buffer_enabled=True,
            column_height=6,
            difficulty="normal"
        )
        
        return WarehouseEnv(
            warehouse=warehouse,
            reward_strategy="completed_orders",
            observation_type="shelf_counts",
            max_steps=1000
        )
    
    @staticmethod
    def create_from_config(config: SimulationConfig, 
                          layout_size: str,
                          seed: int) -> WarehouseEnv:
        """
        Create an environment from a simulation configuration.
        
        Args:
            config: Simulation configuration
            layout_size: Size of the warehouse layout
            seed: Random seed
            
        Returns:
            WarehouseEnv: Configured environment
        """
        warehouse = Warehouse(
            seed=seed,
            layout_size=layout_size,
            buffer_enabled=config.buffer_enabled,
            column_height=config.column_height,
            difficulty=config.difficulty
        )
        
        return WarehouseEnv(
            warehouse=warehouse,
            reward_strategy=config.reward_strategy,
            observation_type=config.observation_type,
            max_steps=config.max_steps
        )
    
    @staticmethod
    def create_custom_env(warehouse_params: Dict[str, Any],
                         env_params: Dict[str, Any]) -> WarehouseEnv:
        """
        Create a custom environment with specific parameters.
        
        Args:
            warehouse_params: Parameters for warehouse creation
            env_params: Parameters for environment creation
            
        Returns:
            WarehouseEnv: Custom configured environment
        """
        warehouse = Warehouse(**warehouse_params)
        return WarehouseEnv(warehouse=warehouse, **env_params)
