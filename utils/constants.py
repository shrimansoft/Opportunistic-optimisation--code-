"""
Constants and enumerations for the warehouse simulation.
"""

from enum import IntEnum


class RobotMode(IntEnum):
    """Robot operational modes."""
    IDLE = 0
    TO_SHELF = 1
    TO_STATION = 2
    RETURNING = 3


class SimulationDefaults:
    """Default values for simulation parameters."""
    
    # Environment defaults
    MAX_STEPS = 10000
    DEFAULT_SEED = 42
    
    # Rendering defaults
    DEFAULT_PAUSE_TIME = 0.05
    DEFAULT_RENDER_FREQUENCY = 1
    
    # Demo defaults
    DEMO_STEPS = 500
    
    # Warehouse defaults
    COLUMN_HEIGHT = 8
    DEFAULT_DIFFICULTY = "normal"
    
    # Layout sizes
    LAYOUT_SIZES = ["tiny", "small", "medium", "large"]
    DEFAULT_LAYOUT = "medium"
