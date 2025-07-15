"""
Simplified warehouse renderer using pyglet.
"""

import numpy as np
from typing import Optional, TYPE_CHECKING
import pyglet
from pyglet import gl
from pyglet.graphics import Batch
from pyglet.text import Label
from pyglet.shapes import Rectangle, Circle, Line
import time

if TYPE_CHECKING:
    from .warehouse import Warehouse

# Color scheme (RGB values 0-255)
COLORS = {
    'background': (255, 255, 255),
    'grid': (200, 200, 200),
    'highway': (232, 232, 232),
    'shelf': (72, 77, 109),
    'shelf_requested': (0, 128, 128),
    'robot_idle': (255, 140, 0),
    'robot_busy': (255, 0, 0),
    'goal': (60, 60, 60),
    'text': (0, 0, 0),
    'info_bg': (240, 240, 240)
}


class WarehouseRenderer(pyglet.window.Window):
    """Simplified warehouse renderer using pyglet."""
    
    def __init__(self, warehouse: 'Warehouse', cell_size: int = 40):
        self.warehouse = warehouse
        self.cell_size = cell_size
        self.rows, self.cols = warehouse.grid_size
        
        # Calculate window size
        width = self.cols * cell_size + 100
        height = self.rows * cell_size + 100
        
        super().__init__(width=width, height=height, caption="Warehouse Simulation")
        
        # Set background color
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)  # White background
        
        self.batch = Batch()
        self.shapes = []
        self.labels = []
    
    def on_draw(self):
        """Draw the warehouse."""
        self.clear()
        
        # Clear previous shapes and labels
        for shape in self.shapes:
            shape.delete()
        for label in self.labels:
            label.delete()
        self.shapes.clear()
        self.labels.clear()
        
        self._draw_grid()
        self._draw_shelves()
        self._draw_robots()
        self._draw_picking_stations()
        self._draw_info()
    
    def _draw_grid(self):
        """Draw grid lines."""
        color = COLORS['grid']
        
        # Draw vertical and horizontal grid lines
        for x in range(self.cols + 1):
            x_pos = x * self.cell_size + 50
            line = Line(x_pos, 50, x_pos, self.rows * self.cell_size + 50, 
                       color=color, batch=self.batch)
            self.shapes.append(line)
        
        for y in range(self.rows + 1):
            y_pos = y * self.cell_size + 50
            line = Line(50, y_pos, self.cols * self.cell_size + 50, y_pos, 
                       color=color, batch=self.batch)
            self.shapes.append(line)
        
        self.batch.draw()
    
    def _draw_shelves(self):
        """Draw shelves with items."""
        for shelf_idx, shelf_items in enumerate(self.warehouse.shelfs):
            y = shelf_idx // self.cols
            x = shelf_idx % self.cols
            
            if x >= self.cols or y >= self.rows or self.warehouse._is_highway(x, y):
                continue
            
            num_items = len(shelf_items)
            is_requested = shelf_idx in self.warehouse.itemShelfsBufferSet
            
            # Check if any robot is assigned to this shelf
            assigned_robot = None
            for robot in self.warehouse.robots:
                if robot.shelf == shelf_idx:
                    assigned_robot = robot
                    break
            
            # Choose color based on state
            if assigned_robot:
                color = (255, 215, 0)  # Gold for assigned shelf
            elif is_requested:
                color = COLORS['shelf_requested']
            elif num_items > 0:
                color = COLORS['shelf']
            else:
                color = COLORS['highway']
            
            # Calculate position
            x_pos = x * self.cell_size + 52
            y_pos = (self.rows - 1 - y) * self.cell_size + 52
            size = self.cell_size - 4
            
            # Draw shelf rectangle
            rect = Rectangle(x_pos, y_pos, size, size, color=color)
            self.shapes.append(rect)
            
            # Add shelf ID label (top-left corner)
            shelf_id_x = x_pos + 2
            shelf_id_y = y_pos + size - 2
            shelf_label = Label(f"S{shelf_idx}", x=shelf_id_x, y=shelf_id_y,
                              anchor_x='left', anchor_y='top',
                              font_size=7, color=(*COLORS['text'], 255))
            self.labels.append(shelf_label)
            
            # Add item count label (bottom-right corner)
            if num_items > 0:
                item_x = x_pos + size - 2
                item_y = y_pos + 2
                item_label = Label(str(num_items), x=item_x, y=item_y,
                                 anchor_x='right', anchor_y='bottom',
                                 font_size=8, color=(*COLORS['text'], 255))
                self.labels.append(item_label)
            
            # Add assigned robot ID label (top-right corner)
            if assigned_robot:
                robot_x = x_pos +  2
                robot_y = y_pos +  2
                robot_label = Label(f"R{assigned_robot.robot_id}", x=robot_x, y=robot_y,
                                  anchor_x='left', anchor_y='bottom',
                                  font_size=7, color=(255, 0, 0, 255))
                self.labels.append(robot_label)
    
    def _draw_robots(self):
        """Draw robots."""
        robot_colors = {
            0: (255, 140, 0),   # IDLE - Orange
            1: (65, 105, 225),  # Going to shelf - Royal Blue
            2: (255, 140, 0),   # Going to station - Dark Orange
            3: (220, 20, 60)    # Returning shelf - Crimson
        }
        
        for robot in self.warehouse.robots:
            x, y = robot.current_location
            
            if not (0 <= x < self.cols and 0 <= y < self.rows):
                continue
            
            color = robot_colors.get(robot.mode, COLORS['robot_idle'])
            
            # Calculate position
            center_x = x * self.cell_size + 50 + self.cell_size // 2
            center_y = (self.rows - 1 - y) * self.cell_size + 50 + self.cell_size // 2
            radius = self.cell_size // 4
            
            # Draw robot circle
            circle = Circle(center_x, center_y, radius, color=color)
            self.shapes.append(circle)
            
            # Add robot ID label
            label = Label(str(robot.robot_id), x=center_x, y=center_y,
                         anchor_x='center', anchor_y='center',
                         font_size=8, color=(255, 255, 255, 255))
            self.labels.append(label)
    
    def _draw_picking_stations(self):
        """Draw picking stations with buffer information."""
        for i, station in enumerate(self.warehouse.picking_stations):
            x, y = station.location
            x_pos = x * self.cell_size + 50
            y_pos = (self.rows - 1 - y) * self.cell_size + 50
            
            # Choose color based on buffer status
            if station.buffer_enabled:
            # Green tint if buffer has space, red tint if full
                if station.buffer_available():
                    station_color = (0, 120, 0)  # Dark green
                else:
                    station_color = (120, 0, 0)  # Dark red
            else:
                station_color = COLORS['goal']  # Default gray when buffer disabled
            
            # Draw station rectangle
            rect = Rectangle(x_pos, y_pos, self.cell_size, self.cell_size, 
                   color=station_color)
            self.shapes.append(rect)
            
            # Add station number label (S1, S2, etc.)
            center_x = x_pos + self.cell_size // 2
            center_y = y_pos + self.cell_size // 2 + 6
            label = Label(f'S{i+1}', x=center_x, y=center_y,
                 anchor_x='center', anchor_y='center',
                 font_size=12, color=(255, 255, 255, 255))
            self.labels.append(label)
            
            # Add buffer information
            if station.buffer_enabled:
                # Buffer status label (current/max)
                buffer_text = f"{len(station.buffer)}/{station.buffer_size}"
                buffer_x = center_x
                buffer_y = center_y - 12
                buffer_label = Label(buffer_text, x=buffer_x, y=buffer_y- 2,
                                   anchor_x='center', anchor_y='center',
                                   font_size=8, color=(255, 255, 255, 255))
                self.labels.append(buffer_label)
            else:
                # Show "NO BUF" when buffer is disabled
                no_buffer_label = Label('NO BUF', x=center_x, y=center_y - 12,
                                      anchor_x='center', anchor_y='center',
                                      font_size=6, color=(255, 255, 255, 255))
                self.labels.append(no_buffer_label)
    
    def _draw_info(self):
        """Draw information panel."""
        # Calculate metrics
        completed = len(self.warehouse.order_compleated)
        pending = len(self.warehouse.order_buffer)
        busy_robots = len([r for r in self.warehouse.robots if not r.available])
        
        # Info text
        info_text = f"Step: {self.warehouse.time} | Orders: {pending}P/{completed}C | Busy Robots: {busy_robots}"
        
        # Main info
        label = Label(info_text, x=10, y=self.height - 20,
                     font_size=10, color=(*COLORS['text'], 255))
        self.labels.append(label)
        
        # Individual picking station information
        y_offset = 40
        for i, station in enumerate(self.warehouse.picking_stations):
            if station.buffer_enabled:
                buffer_info = f"{len(station.buffer)}/{station.buffer_size}"
                status = "FULL" if not station.buffer_available() else "OK"
                station_text = f"Station {i}: Buffer {buffer_info} ({status})"
            else:
                station_text = f"Station {i}: Buffer DISABLED"
            
            station_label = Label(station_text, x=10, y=self.height - y_offset,
                                font_size=9, color=(*COLORS['text'], 255))
            self.labels.append(station_label)
            y_offset += 20
        
        # Draw all shapes and labels
        for shape in self.shapes:
            shape.draw()
        for label in self.labels:
            label.draw()


class Renderer:
    """Main renderer class for warehouse visualization."""
    _instance = None
    
    def __init__(self, warehouse: 'Warehouse', cell_size: int = 40, visible: bool = True):
        self.warehouse = warehouse
        self.cell_size = cell_size
        self.visible = visible
        self.window = None
        
        if visible:
            self.window = WarehouseRenderer(warehouse, cell_size)
    
    @classmethod
    def get_instance(cls, warehouse: 'Warehouse', cell_size: int = 40, visible: bool = True):
        """Get or create renderer instance (singleton pattern)."""
        if cls._instance is None or cls._instance.window is None:
            cls._instance = cls(warehouse, cell_size, visible)
        else:
            # Update warehouse reference for existing instance
            cls._instance.warehouse = warehouse
            if cls._instance.window:
                cls._instance.window.warehouse = warehouse
        return cls._instance
    
    def render(self, mode: str = "human", pause_time: float = 0.05) -> Optional[np.ndarray]:
        """Render the warehouse."""
        if mode == "human" and self.window:
            self.window.switch_to()
            self.window.dispatch_events()
            self.window.on_draw()
            self.window.flip()
            
            # Simple pause
            time.sleep(pause_time)
            
        elif mode == "rgb_array":
            # Return placeholder for RGB array mode
            rows, cols = self.warehouse.grid_size
            return np.zeros((rows * self.cell_size + 100, cols * self.cell_size + 100, 3), dtype=np.uint8)
        
        return None
    
    def close(self):
        """Close the renderer."""
        if self.window:
            self.window.close()
            self.window = None
        Renderer._instance = None


def render_warehouse(warehouse: 'Warehouse', mode: str = "human", pause_time: float = 0.05) -> Optional[np.ndarray]:
    """Convenience function to render warehouse."""
    renderer = Renderer.get_instance(warehouse)
    return renderer.render(mode, pause_time)


def debug_render_warehouse(warehouse: 'Warehouse') -> None:
    """Simple text-based debug renderer."""
    print("=== WAREHOUSE DEBUG ===")
    print(f"Time: {warehouse.time}")
    print(f"Grid: {warehouse.grid_size}")
    
    # Robot status
    mode_names = {0: "IDLE", 1: "→SHELF", 2: "→STATION", 3: "→RETURN"}
    print("\nRobots:")
    for robot in warehouse.robots:
        mode = mode_names.get(robot.mode, "UNKNOWN")
        loc = f"({robot.current_location[0]},{robot.current_location[1]})"
        shelf = f"S{robot.shelf}" if robot.shelf is not None else "None"
        print(f"  R{robot.robot_id}: {mode} @ {loc}, Shelf: {shelf}")
    
    # Picking stations buffer status with individual details
    print("\nPicking Stations:")
    for i, station in enumerate(warehouse.picking_stations):
        loc = f"({station.location[0]},{station.location[1]})"
        if station.buffer_enabled:
            buffer_current = len(station.buffer)
            buffer_max = station.buffer_size
            buffer_info = f"{buffer_current}/{buffer_max}"
            status = "FULL" if not station.buffer_available() else "AVAILABLE"
            print(f"  Station {i}: @ {loc}, Buffer: {buffer_info} ({status})")
            if buffer_current > 0:
                print(f"    Buffer contents: {list(station.buffer)}")
        else:
            print(f"  Station {i}: @ {loc}, Buffer: DISABLED")
    
    # Orders
    print(f"\nOrders: {len(warehouse.order_buffer)} pending, {len(warehouse.order_compleated)} completed")
    print(f"Requested shelves: {warehouse.itemShelfsBufferSet}")
    print("=" * 30)
