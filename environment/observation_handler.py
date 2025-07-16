"""Observation handling for warehouse environment."""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from warehouse_sim.warehouse import Warehouse


class ObservationHandler:
    """Handles observation processing for warehouse environment."""
    
    def __init__(self, observation_type: str = "shelf_counts"):
        self.observation_type = observation_type
    
    def get_observation(self, warehouse: 'Warehouse') -> np.ndarray:
        """Get observation from warehouse state."""
        handlers = {
            "shelf_counts": self._shelf_counts,
            "grid_state": self._grid_state,
            "robot_positions": self._robot_positions
        }
        return handlers.get(self.observation_type, self._shelf_counts)(warehouse)
    
    def get_observation_space_shape(self, warehouse: 'Warehouse') -> tuple:
        """Get observation space shape."""
        shapes = {
            "shelf_counts": (len(warehouse.shelfs),),
            "grid_state": warehouse.grid_size,
            "robot_positions": (len(warehouse.robots) * 2 + len(warehouse.shelfs),)
        }
        return shapes.get(self.observation_type, shapes["shelf_counts"])
    
    def _shelf_counts(self, warehouse: 'Warehouse') -> np.ndarray:
        return np.array([len(shelf) for shelf in warehouse.shelfs], dtype=np.int32)
    
    def _grid_state(self, warehouse: 'Warehouse') -> np.ndarray:
        grid = np.zeros(warehouse.grid_size, dtype=np.int32)
        
        # Mark robots
        for robot in warehouse.robots:
            x, y = robot.current_location
            if 0 <= x < warehouse.grid_size[0] and 0 <= y < warehouse.grid_size[1]:
                grid[x, y] = 2
        
        return grid.flatten()
    
    def _robot_positions(self, warehouse: 'Warehouse') -> np.ndarray:
        positions = []
        for robot in warehouse.robots:
            positions.extend(robot.current_location)
        shelf_counts = [len(shelf) for shelf in warehouse.shelfs]
        return np.array(positions + shelf_counts, dtype=np.int32)
