"""
Advanced collision avoidance system based on RWare implementation.
Uses NetworkX graph-based approach for multi-robot coordination.
"""

import networkx as nx
from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .enums import Action, Direction


class MovementStatus(Enum):
    """Status of movement planning."""
    SUCCESS = 0
    BLOCKED = 1
    INVALID_MOVE = 2
    COLLISION = 3


@dataclass
class MovementRequest:
    """Represents a robot's movement request."""
    robot_id: int
    start_pos: Tuple[int, int]
    target_pos: Tuple[int, int]
    action: Action
    is_carrying_shelf: bool = False


class CollisionAvoidanceSystem:
    """
    RWare-style collision avoidance system using NetworkX graphs.
    
    This system handles multi-robot coordination by:
    1. Creating a directed graph of movement requests
    2. Finding weakly connected components
    3. Resolving conflicts using cycle detection and longest paths
    4. Ensuring collision-free movement for all robots
    """
    
    def __init__(self, warehouse):
        self.warehouse = warehouse
        self.grid_size = warehouse.grid_size
        self.committed_robots = set()
        self.failed_robots = set()
        
    def can_move_to_position(self, x: int, y: int, robot_id: int, is_carrying_shelf: bool = False) -> bool:
        """
        Check if a robot can move to a specific position.
        
        Args:
            x, y: Target position coordinates
            robot_id: ID of the robot requesting movement
            is_carrying_shelf: Whether the robot is carrying a shelf
            
        Returns:
            True if the position is valid and accessible
        """
        # Check grid bounds
        if not (0 <= x < self.grid_size[1] and 0 <= y < self.grid_size[0]):
            return False
            
        # If carrying shelf, must stay on highways (except when returning to original shelf location)
        if is_carrying_shelf and hasattr(self.warehouse, '_is_highway'):
            if not self.warehouse._is_highway(x, y):
                return False
                
        # Check for existing shelf at target location if robot is carrying one
        if (is_carrying_shelf and 
            hasattr(self.warehouse, 'grid') and 
            len(self.warehouse.grid.shape) >= 3):
            
            # RWare collision layer constants
            LAYER_AGENTS = 0
            LAYER_SHELFS = 1
            
            shelf_at_target = self.warehouse.grid[LAYER_SHELFS, y, x]
            if shelf_at_target > 0:
                # Check if there's an agent at the target carrying a shelf
                agent_at_target = self.warehouse.grid[LAYER_AGENTS, y, x]
                if agent_at_target > 0:
                    # Get the agent and check if it's carrying a shelf
                    target_robot = self.warehouse.robots[agent_at_target - 1]
                    if hasattr(target_robot, 'mode') and target_robot.mode in [2, 3]:
                        return False  # Can't move to position with shelf+carrying robot
                        
                return False  # Can't move to position with shelf when carrying one
                
        return True
        
    def get_requested_location(self, robot, action: Action) -> Tuple[int, int]:
        """
        Calculate the target location based on robot's current position and action.
        
        Args:
            robot: Robot object with current_location and direction
            action: Action to be performed
            
        Returns:
            Tuple of (x, y) coordinates for the target location
        """
        if action != Action.FORWARD:
            return robot.current_location
            
        x, y = robot.current_location
        
        # Get robot direction (assuming robots have a direction attribute)
        if hasattr(robot, 'direction'):
            direction = robot.direction
        elif hasattr(robot, 'dir'):
            direction = robot.dir
        else:
            # Default direction handling
            direction = Direction.UP
            
        # Calculate new position based on direction
        if direction == Direction.UP:
            return x, max(0, y - 1)
        elif direction == Direction.DOWN:
            return x, min(self.grid_size[0] - 1, y + 1)
        elif direction == Direction.LEFT:
            return max(0, x - 1), y
        elif direction == Direction.RIGHT:
            return min(self.grid_size[1] - 1, x + 1), y
        else:
            return x, y
            
    def resolve_movement_conflicts(self, movement_requests: List[MovementRequest]) -> Dict[int, MovementStatus]:
        """
        Resolve movement conflicts using RWare's graph-based approach.
        
        Args:
            movement_requests: List of movement requests from all robots
            
        Returns:
            Dictionary mapping robot_id to movement status
        """
        self.committed_robots = set()
        self.failed_robots = set()
        
        # Create directed graph of movement requests
        G = nx.DiGraph()
        
        # Add edges for each movement request
        for request in movement_requests:
            start = request.start_pos
            target = request.target_pos
            
            # Validate movement before adding to graph
            if (request.is_carrying_shelf and 
                start != target and 
                hasattr(self.warehouse, 'grid') and
                len(self.warehouse.grid.shape) >= 3):
                
                LAYER_SHELFS = 1
                LAYER_AGENTS = 0
                
                # Check for shelf collision when carrying
                if self.warehouse.grid[LAYER_SHELFS, target[1], target[0]] > 0:
                    agent_at_target = self.warehouse.grid[LAYER_AGENTS, target[1], target[0]]
                    if agent_at_target > 0:
                        target_robot = self.warehouse.robots[agent_at_target - 1]
                        if (hasattr(target_robot, 'mode') and 
                            target_robot.mode in [2, 3]):  # Robot carrying shelf
                            # Invalid movement, cancel it
                            G.add_edge(start, start)
                            continue
                            
            G.add_edge(start, target)
            
        # Find weakly connected components
        weakly_connected_components = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]
        
        # Resolve conflicts in each component
        for component in weakly_connected_components:
            try:
                # Look for cycles in the component
                cycle = nx.algorithms.find_cycle(component)
                
                # Skip 2-node cycles (impossible movements like A<->B)
                if len(cycle) == 2:
                    continue
                    
                # Commit all nodes in the cycle
                for edge in cycle:
                    start_node = edge[0]
                    if hasattr(self.warehouse, 'grid') and len(self.warehouse.grid.shape) >= 3:
                        LAYER_AGENTS = 0
                        agent_id = self.warehouse.grid[LAYER_AGENTS, start_node[1], start_node[0]]
                        if agent_id > 0:
                            self.committed_robots.add(agent_id)
                            
            except nx.NetworkXNoCycle:
                # No cycle found, use longest path
                try:
                    longest_path = nx.algorithms.dag_longest_path(component)
                    # Ensure longest_path is iterable and contains position tuples
                    if isinstance(longest_path, (list, tuple)) and longest_path:
                        for pos in longest_path:
                            try:
                                if hasattr(self.warehouse, 'grid') and len(self.warehouse.grid.shape) >= 3:
                                    LAYER_AGENTS = 0
                                    # Type ignore for NetworkX return type complexity
                                    x, y = pos  # type: ignore
                                    agent_id = self.warehouse.grid[LAYER_AGENTS, y, x]
                                    if agent_id > 0:
                                        self.committed_robots.add(agent_id)
                            except (ValueError, TypeError, IndexError):
                                continue
                except (nx.NetworkXError, nx.NetworkXUnfeasible, ValueError, TypeError):
                    # Handle cases where longest path cannot be computed or is invalid
                    pass
                    
        # Determine failed robots
        for request in movement_requests:
            if request.action == Action.FORWARD and request.robot_id not in self.committed_robots:
                self.failed_robots.add(request.robot_id)
                
        # Create result dictionary
        result = {}
        for request in movement_requests:
            if request.robot_id in self.failed_robots:
                result[request.robot_id] = MovementStatus.BLOCKED
            elif request.robot_id in self.committed_robots or request.action != Action.FORWARD:
                result[request.robot_id] = MovementStatus.SUCCESS
            else:
                result[request.robot_id] = MovementStatus.SUCCESS
                
        return result
        
    def plan_robot_movements(self, robots: List) -> Dict[int, MovementStatus]:
        """
        Plan movements for all robots with collision avoidance.
        
        Args:
            robots: List of robot objects
            
        Returns:
            Dictionary mapping robot_id to movement status
        """
        movement_requests = []
        
        for robot in robots:
            # Skip robots that are not moving or don't have a target
            if (not hasattr(robot, 'target_location') or 
                robot.target_location is None or
                getattr(robot, 'available', True)):
                continue
                
            current_pos = robot.current_location
            target_pos = robot.target_location
            
            # If robot is already at target, no movement needed
            if current_pos == target_pos:
                continue
                
            # Calculate next step towards target (simple Manhattan movement)
            next_pos = self._calculate_next_position(current_pos, target_pos)
            
            # Check if robot is carrying a shelf
            is_carrying = False
            if hasattr(robot, 'mode'):
                is_carrying = robot.mode in [2, 3]  # Carrying shelf modes
            elif hasattr(robot, 'carrying_shelf'):
                is_carrying = robot.carrying_shelf is not None
                
            # Create movement request
            from .enums import Action
            action = Action.FORWARD if next_pos != current_pos else Action.NOOP
            
            request = MovementRequest(
                robot_id=getattr(robot, 'robot_id', getattr(robot, 'id', robots.index(robot) + 1)),
                start_pos=current_pos,
                target_pos=next_pos,
                action=action,
                is_carrying_shelf=is_carrying
            )
            movement_requests.append(request)
            
        # Resolve conflicts and return results
        return self.resolve_movement_conflicts(movement_requests)
        
    def execute_movements(self, robots: List, movement_results: Dict[int, MovementStatus]):
        """
        Execute the planned movements for robots.
        
        Args:
            robots: List of robot objects
            movement_results: Results from collision avoidance planning
        """
        for robot in robots:
            robot_id = getattr(robot, 'robot_id', getattr(robot, 'id', robots.index(robot) + 1))
            
            if robot_id in movement_results:
                status = movement_results[robot_id]
                
                if status == MovementStatus.SUCCESS:
                    # Only move if robot has a target and isn't already there
                    if (hasattr(robot, 'target_location') and 
                        robot.target_location and 
                        robot.current_location != robot.target_location and
                        not getattr(robot, 'available', True)):  # Only move busy robots
                        
                        # Calculate next position towards target using A* pathfinding
                        current_pos = robot.current_location
                        target_pos = robot.target_location
                        next_pos = self._calculate_next_position(current_pos, target_pos, robot_id)
                        
                        # Validate the move is safe and makes sense
                        if (next_pos != current_pos and 
                            self.can_move_to_position(next_pos[0], next_pos[1], robot_id)):
                            robot.current_location = next_pos
                            
                elif status == MovementStatus.BLOCKED:
                    # Robot stays in place (NOOP)
                    pass
                    
    def _calculate_next_position(self, current_pos: Tuple[int, int], target_pos: Tuple[int, int], robot_id: Optional[int] = None) -> Tuple[int, int]:
        """Calculate the next position towards target using A* pathfinding."""
        # Use A* pathfinding for robust navigation
        path = self._astar_pathfind(current_pos, target_pos, robot_id)
        
        if path and len(path) > 1:
            # Return the next step in the path
            return path[1]
        else:
            # Fallback to simple greedy movement if A* fails
            return self._greedy_next_position(current_pos, target_pos)
    
    def _astar_pathfind(self, start: Tuple[int, int], goal: Tuple[int, int], robot_id: Optional[int] = None) -> List[Tuple[int, int]]:
        """
        A* pathfinding algorithm that considers obstacles and other robots.
        Returns a list of positions from start to goal, or empty list if no path.
        """
        import heapq
        
        if start == goal:
            return [start]
        
        # Define the heuristic function (Manhattan distance)
        def heuristic(pos):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        # Priority queue: (f_score, g_score, position, path)
        open_set = [(heuristic(start), 0, start, [start])]
        closed_set = set()
        g_scores = {start: 0}
        
        max_iterations = self.grid_size[0] * self.grid_size[1]  # Prevent infinite loops
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            f_score, g_score, current, path = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
                
            closed_set.add(current)
            
            # Check if we reached the goal
            if current == goal:
                return path
            
            # Explore neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check bounds
                if not (0 <= neighbor[0] < self.grid_size[1] and 0 <= neighbor[1] < self.grid_size[0]):
                    continue
                
                # Check if position is passable
                if not self._is_position_passable(neighbor, robot_id):
                    continue
                
                # Skip if already processed
                if neighbor in closed_set:
                    continue
                
                tentative_g_score = g_score + 1
                
                # If we found a better path to this neighbor
                if neighbor not in g_scores or tentative_g_score < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor)
                    new_path = path + [neighbor]
                    heapq.heappush(open_set, (f_score, tentative_g_score, neighbor, new_path))
        
        # No path found
        return []
    
    def _is_position_passable(self, pos: Tuple[int, int], robot_id: Optional[int] = None) -> bool:
        """
        Check if a position is passable for a robot.
        Considers shelves, other robots, and warehouse layout.
        """
        x, y = pos
        
        # Check warehouse grid for obstacles
        if hasattr(self.warehouse, 'grid') and self.warehouse.grid is not None:
            # Warehouse grid is 3D: (layers, height, width)
            # Layer 0 is for shelves, Layer 1 is for robots
            # Check if there's a shelf at this position (layer 0)
            if self.warehouse.grid[0, y, x] == 1:  # Shelf obstacle in layer 0
                return False
        
        # Check for other robots at this position (but allow own position)
        if hasattr(self.warehouse, 'robots'):
            for robot in self.warehouse.robots:
                other_robot_id = getattr(robot, 'robot_id', getattr(robot, 'id', None))
                if (other_robot_id != robot_id and 
                    hasattr(robot, 'current_location') and 
                    robot.current_location == pos):
                    return False
        
        return True
    
    def _greedy_next_position(self, current_pos: Tuple[int, int], target_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Fallback greedy movement towards target."""
        x, y = current_pos
        target_x, target_y = target_pos
        
        # If already at target, don't move
        if (x, y) == (target_x, target_y):
            return current_pos
        
        # Try to move in the direction that reduces Manhattan distance most
        possible_moves = [
            (x + 1, y),  # Right
            (x - 1, y),  # Left
            (x, y + 1),  # Down
            (x, y - 1),  # Up
        ]
        
        best_move = current_pos
        best_distance = abs(x - target_x) + abs(y - target_y)
        
        for new_x, new_y in possible_moves:
            # Check bounds
            if (0 <= new_x < self.grid_size[1] and 0 <= new_y < self.grid_size[0]):
                # Check if position is passable
                if self._is_position_passable((new_x, new_y)):
                    # Calculate distance to target from this position
                    distance = abs(new_x - target_x) + abs(new_y - target_y)
                    if distance < best_distance:
                        best_distance = distance
                        best_move = (new_x, new_y)
        
        return best_move


class PathfindingCoordinator:
    """
    High-level coordinator that integrates collision avoidance with the warehouse system.
    """
    
    def __init__(self, warehouse):
        self.warehouse = warehouse
        self.collision_system = CollisionAvoidanceSystem(warehouse)
        
    def step(self):
        """
        Execute one step of collision-aware robot movement.
        This should be called from the warehouse's step method.
        """
        if not hasattr(self.warehouse, 'robots') or not self.warehouse.robots:
            return
            
        # Plan movements for all robots
        movement_results = self.collision_system.plan_robot_movements(self.warehouse.robots)
        
        # Execute the planned movements
        self.collision_system.execute_movements(self.warehouse.robots, movement_results)
        
        # Update the warehouse grid if needed
        if hasattr(self.warehouse, '_recalc_grid'):
            self.warehouse._recalc_grid()
