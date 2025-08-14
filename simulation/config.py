"""Configuration management for warehouse simulation."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SimulationConfig:
    """Centralized simulation configuration."""
    
    default_seed: int = 42
    max_steps: int = 10000
    reward_strategy: str = "completed_orders"
    observation_type: str = "shelf_counts"
    buffer_enabled: bool = True
    column_height: int = 8
    difficulty: str = "normal"
    demo_steps: int = 500
    render_frequency: int = 1
    pause_time: float = 0.05
    available_layouts: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.available_layouts is None:
            self.available_layouts = ["tiny", "small", "medium", "large"]
    
    @classmethod
    def quick_demo(cls) -> 'SimulationConfig':
        return cls(demo_steps=100, render_frequency=5, pause_time=0.01, available_layouts=["medium"])
    
    @classmethod
    def comprehensive_demo(cls) -> 'SimulationConfig':
        return cls(demo_steps=1000, render_frequency=10, pause_time=0.1)
