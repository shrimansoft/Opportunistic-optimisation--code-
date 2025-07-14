import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")  # its for intteractive plots.

from typing import List, Tuple, Optional, Dict

from .order import OrderItem
from .picking_station import PickingStation
from .robot import Robot
from .core.enums import (COLLISION_LAYERS, LAYER_AGENTS, LAYER_SHELFS, 
                         RWARE_SIZES, RWARE_DIFFICULTY, Direction)
from .utils.coordinates import grid_to_flat, flat_to_grid

matplotlib.use("TkAgg")  # its for intteractive plots.


from typing import List, Tuple, Optional, Dict

from .order import OrderItem
from .picking_station import PickingStation
from .robot import Robot

# RWare-compatible constants
_COLLISION_LAYERS = 2

class Warehouse:
    def __init__(self, 
                 seed=None, 
                 buffer_enabled=True,
                 # RWare-style parameters
                 layout_size="medium",  # "tiny", "small", "medium", "large"
                 column_height=8,       # Standard RWare column height
                 difficulty="normal",   # "easy", "normal", "hard"
                 custom_layout=None):   # Custom layout string (overrides other params)
        """
        Initialize warehouse with RWare-compatible layout system.
        
        Args:
            seed: Random seed for reproducibility
            buffer_enabled: Enable picking station buffers
            layout_size: Standard size ("tiny", "small", "medium", "large")
            column_height: Height of shelf columns (default: 8)
            difficulty: Difficulty level affecting request queue size
            custom_layout: Custom layout string (X=shelf, .=corridor, g=goal)
        """
        if seed is not None:
            np.random.seed(seed)

        self.time = 0
        self.buffer_enabled = buffer_enabled
        
        # Store layout parameters
        self.layout_size = layout_size
        self.column_height = column_height
        self.difficulty = difficulty
        
        # Initialize grid system with RWare layout
        self._setup_rware_layout(layout_size, column_height, custom_layout)
        
        # Setup shelves list for RWare compatibility
        self._setup_shelf_entities()
        
        # Calculate warehouse parameters based on grid size
        self._setup_warehouse_content()
        
        # Initialize robots and picking stations
        self._setup_entities()

        self.fig = None
        self.interactive_mode = False
    
    def _setup_rware_layout(self, layout_size: str, column_height: int, custom_layout: Optional[str] = None):
        """Setup warehouse layout using RWare standard configurations."""
        if custom_layout:
            self._make_layout_from_str(custom_layout)
        else:
            if layout_size not in RWARE_SIZES:
                raise ValueError(f"Layout size '{layout_size}' not supported. Use: {list(RWARE_SIZES.keys())}")
            
            shelf_rows, shelf_columns = RWARE_SIZES[layout_size]
            self._make_layout_from_params(shelf_columns, shelf_rows, column_height)
    
    def _make_layout_from_params(self, shelf_columns: int, shelf_rows: int, column_height: int):
        """Create RWare-style layout from parameters."""
        assert shelf_columns % 2 == 1, "Only odd number of shelf columns is supported"

        # Calculate grid size using RWare formula
        self.grid_size = (
            (column_height + 1) * shelf_rows + 2,  # height
            (2 + 1) * shelf_columns + 1,           # width
        )
        
        self.shelf_rows = shelf_rows
        self.shelf_columns = shelf_columns
        self.column_height = column_height
        
        # Initialize collision grid
        self.grid = np.zeros((COLLISION_LAYERS, *self.grid_size), dtype=np.int32)
        
        # Setup goals (delivery locations) at bottom center
        self.goals = [
            (self.grid_size[1] // 2 - 1, self.grid_size[0] - 1),
            (self.grid_size[1] // 2, self.grid_size[0] - 1),
        ]
        
        # Initialize highways
        self.highways = np.zeros(self.grid_size, dtype=np.uint8)
        self._setup_rware_highways(column_height)
    
    def _setup_rware_highways(self, column_height: int):
        """Setup highway system using RWare algorithm."""
        def highway_func(x, y):
            # Vertical highways every 3 columns
            is_on_vertical_highway = x % 3 == 0
            # Horizontal highways every (column_height + 1) rows
            is_on_horizontal_highway = y % (column_height + 1) == 0
            # Bottom row for delivery
            is_on_delivery_row = y == self.grid_size[0] - 1
            # Queue area near goals
            is_on_queue = (y > self.grid_size[0] - (column_height + 3)) and (
                x == self.grid_size[1] // 2 - 1 or x == self.grid_size[1] // 2
            )
            
            return (
                is_on_vertical_highway or 
                is_on_horizontal_highway or 
                is_on_delivery_row or 
                is_on_queue
            )
        
        for x in range(self.grid_size[1]):
            for y in range(self.grid_size[0]):
                self.highways[y, x] = int(highway_func(x, y))
    
    def _make_layout_from_str(self, layout: str):
        """Create layout from custom string (RWare format)."""
        layout = layout.strip()
        layout = layout.replace(" ", "")
        grid_height = layout.count("\n") + 1
        lines = layout.split("\n")
        grid_width = len(lines[0])
        
        for line in lines:
            assert len(line) == grid_width, "Layout must be rectangular"

        self.grid_size = (grid_height, grid_width)
        self.grid = np.zeros((COLLISION_LAYERS, *self.grid_size), dtype=np.int32)
        self.highways = np.zeros(self.grid_size, dtype=np.uint8)
        self.goals = []

        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                assert char.lower() in "gx.", f"Invalid character '{char}' in layout"
                if char.lower() == "g":
                    self.goals.append((x, y))
                    self.highways[y, x] = 1
                elif char.lower() == ".":
                    self.highways[y, x] = 1
                # 'x' represents shelf locations (non-highway)

        assert len(self.goals) >= 1, "At least one goal is required"
    
    def _setup_warehouse_content(self):
        """Setup warehouse content based on grid size."""
        # Calculate number of shelf positions (non-highway cells)
        shelf_positions = []
        for y in range(self.grid_size[0]):
            for x in range(self.grid_size[1]):
                if not self._is_highway(x, y):
                    shelf_positions.append(y * self.grid_size[1] + x)
        
        num_shelf_positions = len(shelf_positions)
        
        # Calculate item types to perfectly match shelf capacity
        # Formula: (item_types × 12 items_per_type) = (shelves × 6 items_per_shelf)
        # Therefore: item_types = (shelves × 6) ÷ 12 = shelves ÷ 2
        items_per_type = 12
        items_per_shelf = 6
        total_shelf_capacity = num_shelf_positions * items_per_shelf
        
        self.n_item_types = total_shelf_capacity // items_per_type
        
        # Ensure we have at least 1 item type
        if self.n_item_types < 1:
            self.n_item_types = 1
        
        # Setup probabilistic distribution for items
        self.probabilities = np.random.dirichlet(np.ones(self.n_item_types), size=1)[0]
        
        # Setup distance calculation (Manhattan distance from goals)
        self.distance = np.zeros(self.grid_size[0] * self.grid_size[1])
        goal_x, goal_y = self.goals[0]  # Use first goal as reference
        
        for i in range(len(self.distance)):
            x, y = i % self.grid_size[1], i // self.grid_size[1]
            # Manhattan distance to goal, with highway bonus
            base_distance = abs(x - goal_x) + abs(y - goal_y) + 2
            if self._is_highway(x, y):
                self.distance[i] = max(1, base_distance * 0.8)  # Highway bonus
            else:
                self.distance[i] = base_distance
        
        # Initialize stock and shelves
        # IMPORTANT: Stock must match what's actually placed on shelves
        items_per_shelf = 6
        total_items_needed = num_shelf_positions * items_per_shelf
        
        # Create items array based on actual shelf capacity
        items = np.repeat(np.arange(0, self.n_item_types), items_per_type)
        np.random.shuffle(items)
        
        if len(items) < total_items_needed:
            # Repeat items if we don't have enough
            items = np.tile(items, (total_items_needed // len(items)) + 1)[:total_items_needed]
        
        # Use only the items we can actually place
        items_to_place = items[:total_items_needed]
        
        # Initialize stock based on what's actually being placed
        self.stock = np.zeros(self.n_item_types)
        for item_type in items_to_place:
            self.stock[item_type] += 1
        
        shelfs_array = items_to_place.reshape(num_shelf_positions, items_per_shelf)
        self.shelfs = [[] for _ in range(self.grid_size[0] * self.grid_size[1])]
        
        # Place items only in valid shelf positions
        for i, shelf_pos in enumerate(shelf_positions):
            self.shelfs[shelf_pos] = shelfs_array[i].tolist()
        
        # Initialize order tracking
        self.order_buffer: List[OrderItem] = []
        self.order_compleated: List[OrderItem] = []
        self.itemShelfsBufferSet = set()
    
    def _setup_entities(self):
        """Setup robots and picking stations for the layout."""
        # Setup picking stations at goal locations
        self.picking_stations = []
        for goal_x, goal_y in self.goals:
            self.picking_stations.append(
                PickingStation(self, (goal_x, goal_y), buffer_enabled=self.buffer_enabled)
            )
        
        # Setup robots (scale with warehouse size)
        grid_area = self.grid_size[0] * self.grid_size[1]
        if grid_area < 100:
            n_robots = 3
        elif grid_area < 200:
            n_robots = 4
        elif grid_area < 400:
            n_robots = 6
        else:
            n_robots = 8
            
        self.robots = [Robot(self, i+1) for i in range(n_robots)]

    def _is_highway(self, x: int, y: int) -> bool:
        """Check if given coordinates are on a highway."""
        if not (0 <= x < self.grid_size[1] and 0 <= y < self.grid_size[0]):
            return False
        return bool(self.highways[y, x])

    def _recalc_grid(self):
        """Recalculate collision grid based on current entity positions (RWare style)."""
        # Clear the grid
        self.grid[:] = 0
        
        # Place shelf entities on the shelf layer
        if hasattr(self, 'shelf_entities'):
            for shelf in self.shelf_entities:
                self.grid[LAYER_SHELFS, shelf.y, shelf.x] = shelf.id
        
        # Place robots on the agent layer (when robot system is migrated)
        # For now, this is a placeholder for future robot entity integration
        for i, robot in enumerate(self.robots):
            if hasattr(robot, 'current_location'):
                x, y = robot.current_location
                if 0 <= x < self.grid_size[1] and 0 <= y < self.grid_size[0]:
                    self.grid[LAYER_AGENTS, y, x] = i + 1

    def buffer_update(self, shelf, picking_station: PickingStation):
        """
        Redistributes items between shelf and buffer based on demand probabilities.
        Most demanded items go to buffer, rest to shelf. Only works if buffers are enabled.
        """

        # Skip buffer operations if buffers are disabled
        if not self.buffer_enabled or not picking_station.buffer_enabled:
            return

        # Combine and sort items by demand probability (descending)
        all_items = self.shelfs[shelf] + picking_station.buffer
        if not all_items:
            return

        sorted_items = sorted(
            all_items, key=lambda item: self.probabilities[item], reverse=True
        )

        # Clear and redistribute based on max buffer capacity
        self.shelfs[shelf].clear()
        picking_station.buffer.clear()

        max_capacity = picking_station.buffer_size
        picking_station.buffer.extend(sorted_items[:max_capacity])
        self.shelfs[shelf].extend(sorted_items[max_capacity : max_capacity * 2])

    def reset(self):
        """Reset the warehouse to initial state for a new simulation."""
        self.time = 0

        # Get valid shelf positions
        shelf_positions = []
        for y in range(self.grid_size[0]):
            for x in range(self.grid_size[1]):
                if not self._is_highway(x, y):
                    shelf_positions.append(y * self.grid_size[1] + x)
        
        # Calculate items based on actual shelf capacity
        items_per_shelf = 6
        items_per_type = 12
        total_items_needed = len(shelf_positions) * items_per_shelf
        
        # Create items array - should perfectly match with our formula
        # (n_item_types × 12) should equal (shelf_positions × 6)
        items = np.repeat(np.arange(0, self.n_item_types), items_per_type)
        
        # Verify the math matches (should be exact)
        if len(items) != total_items_needed:
            print(f"Warning: Item count mismatch! Generated {len(items)}, needed {total_items_needed}")
            # If there's still a mismatch, adjust
            if len(items) < total_items_needed:
                extra_items_needed = total_items_needed - len(items)
                extra_items = np.random.choice(self.n_item_types, extra_items_needed)
                items = np.concatenate([items, extra_items])
            else:
                items = items[:total_items_needed]
        
        np.random.shuffle(items)
        items_to_place = items[:total_items_needed]
        
        # Reset stock based on what's actually being placed
        self.stock = np.zeros(self.n_item_types)
        for item_type in items_to_place:
            self.stock[item_type] += 1
        
        # Clear all shelves first
        self.shelfs = [[] for _ in range(self.grid_size[0] * self.grid_size[1])]
        
        # Redistribute items
        shelfs_array = items_to_place.reshape(len(shelf_positions), items_per_shelf)
        for i, shelf_pos in enumerate(shelf_positions):
            self.shelfs[shelf_pos] = shelfs_array[i].tolist()

        # Clear order queues
        self.order_buffer.clear()
        self.order_compleated.clear()
        self.itemShelfsBufferSet.clear()

        # Reset robots to initial state
        for robot in self.robots:
            robot.available = True
            robot.mode = 0
            robot.time_left = 0
            robot.shelf = None
            # Start robots at a highway position near the goals
            if self.goals:
                goal_x, goal_y = self.goals[0]
                # Find a nearby highway position
                start_x, start_y = goal_x, max(0, goal_y - 1)
                if self._is_highway(start_x, start_y):
                    robot.current_location = (start_x, start_y)
                else:
                    robot.current_location = (goal_x, goal_y)  # Fallback to goal position
            else:
                robot.current_location = (0, 0)  # Fallback
            robot.target_location = robot.current_location
            robot.shelf_location = None

        # Reset picking station buffers
        for station in self.picking_stations:
            station.buffer.clear()

    def sample(self):
        """Sample an item type based on probability distribution."""
        return int(np.random.choice(np.arange(self.n_item_types), size=1, p=self.probabilities).item())

    def available(self):
        # Check both shelf stock and picking station buffers
        availability = list(map(bool, self.stock))

        # Only check buffers if they are enabled
        if self.buffer_enabled:
            # Also check if items are available in any picking station buffer
            for i in range(len(availability)):
                if not availability[
                    i
                ]:  # Only check buffers if item not available on shelves
                    # Check all picking stations for this item
                    for picking_station in self.picking_stations:
                        if i in picking_station.buffer:
                            availability[i] = True
                            break

        return availability

    def itemInShelfs(self, n):
        """Get count of item n in each shelf position."""
        return list(map(lambda x: sum([1 for i in x if i == n]), self.shelfs))

    def nearestShelf(self, n):
        availableInShelfs = list(map(bool, self.itemInShelfs(n)))
        
        # Updated distance calculation to consider highways
        distance = []
        for i in range(len(self.distance)):
            if i in self.itemShelfsBufferSet:
                distance.append(0)
            else:
                # Use RWare-style distance calculation
                base_distance = self.distance[i]
                x, y = i % self.grid_size[1], i // self.grid_size[1]
                # Highway bonus - reduce distance if on highway
                if self._is_highway(x, y):
                    distance.append(max(1, base_distance * 0.8))  # 20% reduction for highway locations
                else:
                    distance.append(base_distance)
        
        filteredList = [
            (i, v) for i, (v, l) in enumerate(zip(distance, availableInShelfs)) if l
        ]
        shelf, distance = min(filteredList, key=lambda x: x[1])
        return shelf, distance

    def order_step(self):
        self.time += 1
        # if self.time % 1000 == 0:
        #     print("Total average_dealy: ", self.average_delay())
        if np.random.random() < 0.3:
            available = self.available()
            samples = self.sample()
            if available[samples]:
                # Check if the item is available in any picking station buffer (only if buffers are enabled)
                item_found_in_buffer = False
                if self.buffer_enabled:
                    for station in self.picking_stations:
                        if samples in station.buffer:
                            # Item found in buffer - create order and fulfill immediately
                            order = OrderItem(
                                samples, self.time, None
                            )  # No shelf needed
                            order.done(
                                self.time, None
                            )  # Completed immediately, no robot needed
                            self.order_compleated.append(order)
                            station.buffer.remove(samples)
                            self.stock[samples] -= 1
                            item_found_in_buffer = True
                            print(
                                f"Order for item {samples} fulfilled immediately from picking station buffer"
                            )
                            # print("Total stock >> ", self.stock.sum())
                            break

                # If item not found in buffer, create regular order
                if not item_found_in_buffer:
                    shelf, distence = self.nearestShelf(samples)
                    self.itemShelfsBufferSet.add(shelf)
                    self.order_buffer.append(OrderItem(samples, self.time, shelf))
                    self.shelfs[shelf].remove(samples)
                    # print("Total stock >> ", self.stock.sum())
                    self.stock[samples] -= 1

    def robot_assigner(self):
        itemShelfsBufferSet = self.itemShelfsBufferSet

        if len(itemShelfsBufferSet) > 0:
            for robot in self.robots:
                if robot.available:
                    if len(self.itemShelfsBufferSet) > 0:
                        shelf_to_move = self.itemShelfsBufferSet.pop()
                        # Convert flat shelf index to grid coordinates
                        shelf_x = shelf_to_move % self.grid_size[1]
                        shelf_y = shelf_to_move // self.grid_size[1]
                        robot.assigne(
                            shelf_to_move,
                            2 * self.distance[shelf_to_move],
                            (shelf_x, shelf_y),  # Use 0-based coordinates consistently
                        )

    def enable_interactive_plot(self):
        """Enable interactive plotting mode."""
        self.interactive_mode = True
        plt.ion()

    def disable_interactive_plot(self):
        """Disable interactive plotting mode."""
        self.interactive_mode = False
        if self.fig:
            plt.close(self.fig)
            self.fig = None

    def average_delay(self):
        delay = 0
        order_count = 0
        for order in self.order_compleated:
            # print(
            #     "order \t",
            #     order.item_type,
            #     " Delay: \t",
            #     order.delay,
            #     "robot: \t",
            #     order.robot_id,
            # )
            delay += order.delay
            order_count += 1
        for order in self.order_buffer:
            delay += self.time - order.creation_time
            order_count += 1
        if order_count == 0:
            return 0
        else:
            return delay / order_count

    def enhanced_plot(self, frame_dir=None, step_number=None, pause_time=0.1):
        """Enhanced plotting using the new RWare-style renderer."""
        from .renderer import render_warehouse
        return render_warehouse(self, mode="human", pause_time=pause_time)

    def set_buffer_enabled(self, enabled):
        """Enable or disable all picking station buffers."""
        self.buffer_enabled = enabled
        for station in self.picking_stations:
            station.set_buffer_enabled(enabled)

    def set_station_buffer_enabled(self, station_index, enabled):
        """Enable or disable a specific picking station buffer."""
        if 0 <= station_index < len(self.picking_stations):
            self.picking_stations[station_index].set_buffer_enabled(enabled)
        else:
            raise IndexError(f"Station index {station_index} out of range")
    
    def _setup_shelf_entities(self):
        """Setup shelf entities for RWare compatibility."""
        # Create shelf entities at non-highway positions
        self.shelf_entities = []
        shelf_id = 1
        
        for y in range(self.grid_size[0]):
            for x in range(self.grid_size[1]):
                if not self._is_highway(x, y):
                    # Create a simple shelf entity (placeholder for full RWare migration)
                    shelf = type('Shelf', (), {
                        'id': shelf_id,
                        'x': x,
                        'y': y,
                        'prev_x': None,
                        'prev_y': None
                    })()
                    self.shelf_entities.append(shelf)
                    shelf_id += 1
