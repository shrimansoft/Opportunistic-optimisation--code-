"""Core enums for the warehouse simulation system."""

from enum import Enum


class Direction(Enum):
    """Robot facing directions."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Action(Enum):
    """Robot actions."""
    NOOP = 0      # No operation
    FORWARD = 1   # Move forward in current direction
    LEFT = 2      # Turn left
    RIGHT = 3     # Turn right
    TOGGLE_LOAD = 4  # Pick up or drop shelf


class RewardType(Enum):
    """Types of rewards in the environment."""
    MOVEMENT = 0
    COLLISION = 1
    DELIVERY = 2
    REQUEST_COMPLETED = 3
    INVALID_ACTION = 4
