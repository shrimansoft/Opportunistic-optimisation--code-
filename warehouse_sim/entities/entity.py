"""Base entity class for all warehouse entities."""

from typing import Optional


class Entity:
    """Base class for all entities in the warehouse (agents, shelves, etc.)."""
    
    def __init__(self, id_: int, x: int, y: int):
        """Initialize entity with ID and position.
        
        Args:
            id_: Unique identifier for this entity
            x: X coordinate
            y: Y coordinate
        """
        self.id = id_
        self.prev_x: Optional[int] = None
        self.prev_y: Optional[int] = None
        self.x = x
        self.y = y
