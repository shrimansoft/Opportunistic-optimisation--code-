"""Coordinate conversion utilities for the warehouse simulation."""

from typing import Tuple


def flat_to_grid(index: int, width: int) -> Tuple[int, int]:
    """Convert flat array index to grid coordinates.
    
    Args:
        index: Flat array index
        width: Width of the grid
        
    Returns:
        Tuple of (x, y) coordinates
    """
    x = index % width
    y = index // width
    return x, y


def grid_to_flat(x: int, y: int, width: int) -> int:
    """Convert grid coordinates to flat array index.
    
    Args:
        x: X coordinate
        y: Y coordinate
        width: Width of the grid
        
    Returns:
        Flat array index
    """
    return y * width + x


def validate_coordinates(x: int, y: int, grid_size: Tuple[int, int]) -> bool:
    """Validate that coordinates are within grid bounds.
    
    Args:
        x: X coordinate
        y: Y coordinate
        grid_size: Grid size as (height, width)
        
    Returns:
        True if coordinates are valid, False otherwise
    """
    height, width = grid_size
    return 0 <= x < width and 0 <= y < height
