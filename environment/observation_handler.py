"""
Observation handling for the warehouse environment.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from warehouse_sim.warehouse import Warehouse


class ObservationHandler:
    """
    Handles observation processing for the warehouse environment.
    Manages different observation types and formats.
    """
    
    def __init__(self, observation_type: str = "shelf_counts"):
        """
        Initialize the observation handler.
        
        Args:
            observation_type: Type of observation to generate.
                            Options: "shelf_counts", "grid_state", "robot_positions"
        """
        self.observation_type = observation_type
    
    def get_observation(self, warehouse: 'Warehouse') -> np.ndarray:
        """
        Get the current observation from the warehouse state.
        
        Args:
            warehouse: The warehouse instance to observe
            
        Returns:
            np.ndarray: The observation array
        """
        if self.observation_type == "shelf_counts":
            return self._get_shelf_counts_observation(warehouse)
        elif self.observation_type == "grid_state":
            return self._get_grid_state_observation(warehouse)
        elif self.observation_type == "robot_positions":
            return self._get_robot_positions_observation(warehouse)
        else:
            raise ValueError(f"Unknown observation type: {self.observation_type}")
    
    def get_observation_space_shape(self, warehouse: 'Warehouse') -> tuple:
        """
        Get the shape of the observation space.
        
        Args:
            warehouse: The warehouse instance
            
        Returns:
            tuple: Shape of the observation space
        """
        if self.observation_type == "shelf_counts":
            return (len(warehouse.shelfs),)
        elif self.observation_type == "grid_state":
            return warehouse.grid_size
        elif self.observation_type == "robot_positions":
            # Robot positions (x, y) for each robot + shelf states
            return (len(warehouse.robots) * 2 + len(warehouse.shelfs),)
        else:
            raise ValueError(f"Unknown observation type: {self.observation_type}")
    
    def _get_shelf_counts_observation(self, warehouse: 'Warehouse') -> np.ndarray:
        """
        Get observation based on shelf item counts.
        
        Args:
            warehouse: The warehouse instance
            
        Returns:
            np.ndarray: Array of shelf counts
        """
        shelf_counts = [len(shelf) for shelf in warehouse.shelfs]
        return np.array(shelf_counts, dtype=np.int32)
    
    def _get_grid_state_observation(self, warehouse: 'Warehouse') -> np.ndarray:
        """
        Get observation based on the full grid state.
        
        Args:
            warehouse: The warehouse instance
            
        Returns:
            np.ndarray: Grid state observation
        """
        # Create a grid representation of the warehouse
        grid = np.zeros(warehouse.grid_size, dtype=np.int32)
        
        # Mark shelves (value 1)
        for shelf in warehouse.shelfs:
            if hasattr(shelf, 'x') and hasattr(shelf, 'y'):
                grid[shelf.x, shelf.y] = 1
        
        # Mark robots (value 2)
        for robot in warehouse.robots:
            x, y = robot.current_location
            if 0 <= x < warehouse.grid_size[0] and 0 <= y < warehouse.grid_size[1]:
                grid[x, y] = 2
        
        # Mark picking stations (value 3)
        for station in warehouse.picking_stations:
            if hasattr(station, 'x') and hasattr(station, 'y'):
                grid[station.x, station.y] = 3
        
        return grid.flatten()
    
    def _get_robot_positions_observation(self, warehouse: 'Warehouse') -> np.ndarray:
        """
        Get observation combining robot positions and shelf states.
        
        Args:
            warehouse: The warehouse instance
            
        Returns:
            np.ndarray: Combined robot and shelf observation
        """
        # Robot positions
        robot_positions = []
        for robot in warehouse.robots:
            robot_positions.extend(robot.current_location)
        
        # Shelf counts
        shelf_counts = [len(shelf) for shelf in warehouse.shelfs]
        
        # Combine both
        observation = robot_positions + shelf_counts
        return np.array(observation, dtype=np.int32)
