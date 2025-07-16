"""
Configuration management for warehouse simulation.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SimulationConfig:
    """
    Configuration class for warehouse simulation parameters.
    Centralizes all configurable values.
    """
    
    # Seed configuration
    default_seed: int = 42
    
    # Environment configuration
    max_steps: int = 10000
    reward_strategy: str = "completed_orders"
    observation_type: str = "shelf_counts"
    
    # Warehouse configuration
    buffer_enabled: bool = True
    column_height: int = 8
    difficulty: str = "normal"
    
    # Demo configuration
    demo_steps: int = 500
    render_frequency: int = 1  # Render every N steps
    pause_time: float = 0.05
    
    # Layout configurations
    available_layouts: Optional[List[str]] = None
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.available_layouts is None:
            self.available_layouts = ["tiny", "small", "medium", "large"]
    
    @classmethod
    def default(cls) -> 'SimulationConfig':
        """
        Create a default configuration.
        
        Returns:
            SimulationConfig: Default configuration instance
        """
        return cls()
    
    @classmethod
    def quick_demo(cls) -> 'SimulationConfig':
        """
        Create a configuration optimized for quick demos.
        
        Returns:
            SimulationConfig: Quick demo configuration
        """
        return cls(
            demo_steps=100,
            render_frequency=5,
            pause_time=0.01,
            available_layouts=["medium"]
        )
    
    @classmethod
    def comprehensive_demo(cls) -> 'SimulationConfig':
        """
        Create a configuration for comprehensive demos.
        
        Returns:
            SimulationConfig: Comprehensive demo configuration
        """
        return cls(
            demo_steps=1000,
            render_frequency=10,
            pause_time=0.1,
            available_layouts=["tiny", "small", "medium", "large"]
        )
