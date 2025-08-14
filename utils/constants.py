"""Constants and enumerations."""

from enum import IntEnum


class RobotMode(IntEnum):
    """Robot operational modes."""
    IDLE = 0
    TO_SHELF = 1
    TO_STATION = 2
    RETURNING = 3


class SimulationDefaults:
    """Default simulation parameters."""
    MAX_STEPS = 10000
    DEFAULT_SEED = 42
    DEFAULT_PAUSE_TIME = 0.05
    DEMO_STEPS = 500
    COLUMN_HEIGHT = 8
    LAYOUT_SIZES = ["tiny", "small", "medium", "large"]
