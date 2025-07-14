"""
Core enums and constants for RWare-style warehouse simulation.
"""

from enum import Enum


class Action(Enum):
    """Robot actions (RWare compatible)."""
    NOOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    TOGGLE_LOAD = 4


class Direction(Enum):
    """Movement directions (RWare compatible)."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class RewardType(Enum):
    """Reward distribution types (RWare compatible)."""
    GLOBAL = 0
    INDIVIDUAL = 1
    TWO_STAGE = 2


class ObservationType(Enum):
    """Observation types (RWare compatible)."""
    DICT = 0
    FLATTENED = 1
    IMAGE = 2
    IMAGE_DICT = 3


class ImageLayer(Enum):
    """Image observation layers (RWare compatible)."""
    SHELVES = 0
    REQUESTS = 1
    AGENTS = 2
    AGENT_DIRECTION = 3
    AGENT_LOAD = 4
    GOALS = 5
    ACCESSIBLE = 6


# RWare collision layer constants
COLLISION_LAYERS = 2
LAYER_AGENTS = 0
LAYER_SHELFS = 1

# RWare standard layout sizes
RWARE_SIZES = {
    "tiny": (1, 3),      # (shelf_rows, shelf_columns)
    "small": (2, 3),
    "medium": (2, 5),
    "large": (3, 5),
}

# RWare difficulty levels (affects request queue size)
RWARE_DIFFICULTY = {
    "easy": 2,
    "normal": 1,
    "hard": 0.5
}
