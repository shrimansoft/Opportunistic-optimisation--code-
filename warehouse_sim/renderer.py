"""
RWare-style visual renderer for the warehouse simulation.
This renderer uses pyglet for fast, hardware-accelerated graphics.
"""

import numpy as np
from typing import Optional, TYPE_CHECKING, Any, Literal, Tuple
import pyglet
from pyglet import gl
from pyglet.graphics import Batch
from pyglet.text import Label
from pyglet.shapes import Rectangle, Circle, Line
import datetime

if TYPE_CHECKING:
    from .warehouse import Warehouse

# RWare color scheme (RGB colors for pyglet)
RWARE_COLORS = {
    'background': (255, 255, 255),      # White
    'grid_lines': (0, 0, 0),            # Black
    'highway': (232, 232, 232),         # Light gray
    'shelf': (72, 77, 109),             # Dark slate blue
    'shelf_requested': (0, 128, 128),   # Teal
    'robot_idle': (255, 140, 0),        # Dark orange
    'robot_busy': (255, 0, 0),          # Red
    'robot_direction': (0, 0, 0),       # Black
    'goal': (60, 60, 60),               # Dark gray
    'picking_station': (128, 0, 128),   # Purple
    'text_primary': (255, 255, 255),    # White
    'text_secondary': (0, 0, 0),        # Black
    'info_panel': (240, 240, 240)       # Light gray
}

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    rgb_tuple = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb_tuple[0], rgb_tuple[1], rgb_tuple[2])

def rgb_to_gl(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert RGB tuple to OpenGL color values (0.0-1.0)."""
    return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)


class PygletWindow(pyglet.window.Window):
    """Pyglet window for warehouse rendering."""
    
    def __init__(self, warehouse: 'Warehouse', cell_size: int = 40, **kwargs):
        self.warehouse = warehouse
        self.cell_size = cell_size
        self.rows, self.cols = warehouse.grid_size
        
        # Calculate window size
        window_width = self.cols * cell_size + 100  # Extra space for padding
        window_height = self.rows * cell_size + 200  # Extra space for info panel
        
        super().__init__(width=window_width, height=window_height, **kwargs)
        
        # Initialize rendering components
        self.batch = Batch()
        self.labels = []
        
        # Set background color
        gl.glClearColor(*rgb_to_gl(RWARE_COLORS['background']), 1.0)
        
        # Store shape objects for drawing
        self.shapes = []
    
    def on_draw(self):
        """Draw the warehouse."""
        self.clear()
        
        # Clear previous shapes and labels
        self.shapes.clear()
        for label in self.labels:
            label.delete()
        self.labels.clear()
        
        # Draw components
        self._draw_grid()
        self._draw_highways()
        self._draw_shelves()
        self._draw_goals()
        self._draw_robots()
        self._draw_info_panel()
        
        # Draw all shapes
        for shape in self.shapes:
            shape.draw()
        
        # Draw all labels
        for label in self.labels:
            label.draw()
    
    def _draw_grid(self):
        """Draw grid lines."""
        color = rgb_to_gl(RWARE_COLORS['grid_lines'])
        
        # Vertical lines
        for x in range(self.cols + 1):
            x_pos = x * self.cell_size + 50  # 50px padding
            y_start = 50
            y_end = self.rows * self.cell_size + 50
            self._draw_line(x_pos, y_start, x_pos, y_end, color, alpha=0.2)
        
        # Horizontal lines
        for y in range(self.rows + 1):
            y_pos = y * self.cell_size + 50
            x_start = 50
            x_end = self.cols * self.cell_size + 50
            self._draw_line(x_start, y_pos, x_end, y_pos, color, alpha=0.2)
    
    def _draw_highways(self):
        """Draw highway cells."""
        color = rgb_to_gl(RWARE_COLORS['highway'])
        
        for y in range(self.rows):
            for x in range(self.cols):
                if self.warehouse._is_highway(x, y):
                    x_pos = x * self.cell_size + 50
                    y_pos = (self.rows - 1 - y) * self.cell_size + 50  # Flip Y
                    self._draw_rect(x_pos, y_pos, self.cell_size, self.cell_size, color, alpha=0.6)
    
    def _draw_shelves(self):
        """Draw shelves with items and robot assignments."""
        # Create mapping of shelf to assigned robot
        shelf_to_robot = {}
        for robot in self.warehouse.robots:
            if robot.shelf is not None:
                shelf_to_robot[robot.shelf] = robot.robot_id
        
        for shelf_idx, shelf_items in enumerate(self.warehouse.shelfs):
            # Convert flat index to grid coordinates
            y = shelf_idx // self.cols
            x = shelf_idx % self.cols
            
            # Skip highways and out-of-bounds
            if x >= self.cols or y >= self.rows or self.warehouse._is_highway(x, y):
                continue
            
            if not self._is_shelf_position(x, y):
                continue
            
            num_items = len(shelf_items)
            is_requested = shelf_idx in self.warehouse.itemShelfsBufferSet
            has_robot = shelf_idx in shelf_to_robot
            
            # Determine colors
            if is_requested and has_robot:
                color = hex_to_rgb('#00CED1')  # Dark turquoise
                border_color = hex_to_rgb('#FFD700')  # Gold
            elif is_requested:
                color = RWARE_COLORS['shelf_requested']  # Teal
                border_color = hex_to_rgb('#FF6347')  # Tomato
            elif has_robot:
                color = hex_to_rgb('#8A2BE2')  # Blue violet
                border_color = hex_to_rgb('#32CD32')  # Lime
            elif num_items > 0:
                color = RWARE_COLORS['shelf']  # Dark blue
                border_color = hex_to_rgb('#708090')  # Slate gray
            else:
                color = hex_to_rgb('#D3D3D3')  # Light gray
                border_color = hex_to_rgb('#A9A9A9')  # Dark gray
            
            # Calculate position
            x_pos = x * self.cell_size + 50 + 3  # Small padding
            y_pos = (self.rows - 1 - y) * self.cell_size + 50 + 3  # Flip Y + padding
            size = self.cell_size - 6  # Account for padding
            
            # Draw shelf rectangle
            self._draw_rect(x_pos, y_pos, size, size, rgb_to_gl(color), alpha=0.9)
            self._draw_rect_border(x_pos, y_pos, size, size, rgb_to_gl(border_color), 
                                 line_width=2 if (is_requested or has_robot) else 1)
            
            # Add text labels
            center_x = x_pos + size // 2
            center_y = y_pos + size // 2
            
            # Shelf ID (top-left)
            id_label = Label(str(shelf_idx), 
                           font_size=8, 
                           color=(*RWARE_COLORS['text_primary'], 255),
                           x=x_pos + 5, y=y_pos + size - 5,
                           anchor_x='left', anchor_y='top')
            self.labels.append(id_label)
            
            # Item count (center)
            item_text = str(num_items) if num_items > 0 else "∅"
            item_color = RWARE_COLORS['text_primary'] if color != hex_to_rgb('#D3D3D3') else RWARE_COLORS['text_secondary']
            item_label = Label(item_text,
                             font_size=12,
                             color=(*item_color, 255),
                             x=center_x, y=center_y,
                             anchor_x='center', anchor_y='center')
            self.labels.append(item_label)
            
            # Robot assignment (bottom-right)
            if shelf_idx in shelf_to_robot:
                robot_id = shelf_to_robot[shelf_idx]
                robot_label = Label(f"R{robot_id}",
                                  font_size=8,
                                  color=(*RWARE_COLORS['text_primary'], 255),
                                  x=x_pos + size - 5, y=y_pos + 5,
                                  anchor_x='right', anchor_y='bottom')
                self.labels.append(robot_label)
            
            # Priority indicator for requested shelves
            if is_requested:
                priority_x = x_pos + 8
                priority_y = y_pos + size - 8
                self._draw_circle(priority_x, priority_y, 4, rgb_to_gl((255, 0, 0)), alpha=0.9)
                priority_label = Label("!",
                                     font_size=8,
                                     color=(255, 255, 255, 255),
                                     x=priority_x, y=priority_y,
                                     anchor_x='center', anchor_y='center')
                self.labels.append(priority_label)
    
    def _draw_goals(self):
        """Draw picking stations as goals."""
        color = rgb_to_gl(RWARE_COLORS['goal'])
        
        for station in self.warehouse.picking_stations:
            x, y = station.location
            x_pos = x * self.cell_size + 50
            y_pos = (self.rows - 1 - y) * self.cell_size + 50  # Flip Y
            
            # Draw goal rectangle
            self._draw_rect(x_pos, y_pos, self.cell_size, self.cell_size, color, alpha=0.9)
            self._draw_rect_border(x_pos, y_pos, self.cell_size, self.cell_size, 
                                 rgb_to_gl(RWARE_COLORS['grid_lines']), line_width=2)
            
            # Add 'G' label
            center_x = x_pos + self.cell_size // 2
            center_y = y_pos + self.cell_size // 2
            goal_label = Label('G',
                             font_size=16,
                             color=(*RWARE_COLORS['text_primary'], 255),
                             x=center_x, y=center_y,
                             anchor_x='center', anchor_y='center')
            self.labels.append(goal_label)
    
    def _draw_robots(self):
        """Draw robots with mode colors."""
        mode_colors = {
            0: hex_to_rgb('#32CD32'),  # Lime Green - IDLE
            1: hex_to_rgb('#4169E1'),  # Royal Blue - Going to shelf
            2: hex_to_rgb('#FF8C00'),  # Dark Orange - Going to station
            3: hex_to_rgb('#DC143C')   # Crimson - Returning shelf
        }
        
        for robot in self.warehouse.robots:
            x, y = robot.current_location
            
            # Bounds check
            if not (0 <= x < self.cols and 0 <= y < self.rows):
                continue
            
            color = mode_colors.get(robot.mode, RWARE_COLORS['robot_idle'])
            
            # Calculate position
            center_x = x * self.cell_size + 50 + self.cell_size // 2
            center_y = (self.rows - 1 - y) * self.cell_size + 50 + self.cell_size // 2  # Flip Y
            radius = self.cell_size // 3
            
            # Draw robot circle
            self._draw_circle(center_x, center_y, radius, rgb_to_gl(color), alpha=0.75)
            
            # Add glow effect for active robots
            if robot.mode != 0:  # Not idle
                self._draw_circle(center_x, center_y, radius + 5, rgb_to_gl(color), alpha=0.2)
            
            # Robot ID label
            robot_label = Label(str(robot.robot_id),
                              font_size=10,
                              color=(255, 255, 255, 255),
                              x=center_x, y=center_y,
                              anchor_x='center', anchor_y='center')
            self.labels.append(robot_label)
            
            # Direction indicator
            if robot.target_location != robot.current_location:
                target_x, target_y = robot.target_location
                dx = target_x - x
                dy = target_y - y
                
                if dx != 0 or dy != 0:
                    length = (dx**2 + dy**2)**0.5
                    dx_norm = dx / length * 10  # Arrow length
                    dy_norm = dy / length * 10
                    
                    arrow_end_x = center_x + dx_norm
                    arrow_end_y = center_y - dy_norm  # Flip Y for arrow
                    
                    self._draw_line(center_x, center_y, arrow_end_x, arrow_end_y, 
                                  rgb_to_gl((255, 255, 255)), line_width=2, alpha=0.9)
    
    def _draw_info_panel(self):
        """Draw information panel at bottom."""
        panel_y = 10
        panel_height = 30
        panel_width = self.width - 20
        
        # Draw background panel
        self._draw_rect(10, panel_y, panel_width, panel_height, 
                       rgb_to_gl(RWARE_COLORS['info_panel']), alpha=0.9)
        
        # Calculate metrics
        avg_delay = self.warehouse.average_delay()
        completed = len(self.warehouse.order_compleated)
        pending = len(self.warehouse.order_buffer)
        stock = int(self.warehouse.stock.sum())
        busy_robots = len([r for r in self.warehouse.robots if not r.available])
        total_robots = len(self.warehouse.robots)
        efficiency = (busy_robots/total_robots*100) if total_robots > 0 else 0
        
        # Main info text
        info_text = f"Step: {self.warehouse.time} | Stock: {stock} | Orders: {pending}P/{completed}C | Efficiency: {efficiency:.0f}% | Avg Delay: {avg_delay:.1f}s"
        
        info_label = Label(info_text,
                         font_size=10,
                         color=(*RWARE_COLORS['text_secondary'], 255),
                         x=self.width // 2, y=panel_y + panel_height // 2,
                         anchor_x='center', anchor_y='center')
        self.labels.append(info_label)
        
        # Timestamp
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        time_label = Label(f"Time: {timestamp}",
                         font_size=8,
                         color=(128, 128, 128, 255),
                         x=self.width - 10, y=panel_y + 5,
                         anchor_x='right', anchor_y='bottom')
        self.labels.append(time_label)
    
    def _is_shelf_position(self, x: int, y: int) -> bool:
        """Check if position should have a shelf."""
        return not self.warehouse._is_highway(x, y)
    
    def _draw_rect(self, x: float, y: float, width: float, height: float, 
                   color: Tuple[float, float, float], alpha: float = 1.0):
        """Draw a rectangle using pyglet shapes."""
        # Convert normalized color to 0-255 range
        color_255 = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        alpha_255 = int(alpha * 255)
        
        rect = Rectangle(x, y, width, height, color=(*color_255, alpha_255))
        self.shapes.append(rect)
    
    def _draw_rect_border(self, x: float, y: float, width: float, height: float,
                         color: Tuple[float, float, float], line_width: float = 1.0):
        """Draw rectangle border using lines."""
        # Convert normalized color to 0-255 range
        color_255 = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        
        # Create four lines for the border (pyglet Line doesn't support width parameter)
        lines = [
            Line(x, y, x + width, y, color=color_255),  # bottom
            Line(x + width, y, x + width, y + height, color=color_255),  # right
            Line(x + width, y + height, x, y + height, color=color_255),  # top
            Line(x, y + height, x, y, color=color_255),  # left
        ]
        self.shapes.extend(lines)
    
    def _draw_line(self, x1: float, y1: float, x2: float, y2: float,
                   color: Tuple[float, float, float], line_width: float = 1.0, alpha: float = 1.0):
        """Draw a line."""
        # Convert normalized color to 0-255 range
        color_255 = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        
        line = Line(x1, y1, x2, y2, color=color_255)
        self.shapes.append(line)
    
    def _draw_circle(self, center_x: float, center_y: float, radius: float,
                     color: Tuple[float, float, float], alpha: float = 1.0, segments: int = 20):
        """Draw a circle using pyglet shapes."""
        # Convert normalized color to 0-255 range
        color_255 = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), int(alpha * 255))
        
        circle = Circle(center_x, center_y, radius, color=color_255[:3])
        self.shapes.append(circle)


class RWareStyleRenderer:
    """Pyglet-based renderer that closely mimics RWare's visual style."""
    
    def __init__(self, warehouse: 'Warehouse', cell_size: int = 40, window_visible: bool = True):
        """
        Initialize the RWare-style renderer.
        
        Args:
            warehouse: Warehouse instance to render
            cell_size: Size of each grid cell in pixels
            window_visible: Whether to show the window
        """
        self.warehouse = warehouse
        self.cell_size = cell_size
        self.window_visible = window_visible
        self.window = None
        
        if window_visible:
            self._setup_window()
    
    def _setup_window(self):
        """Setup pyglet window."""
        if self.window is None:
            self.window = PygletWindow(
                self.warehouse, 
                cell_size=self.cell_size,
                caption="Warehouse Simulation - RWare Style",
                resizable=False,
                visible=self.window_visible
            )
    
    def render(self, mode: str = "human", pause_time: float = 0.05) -> Optional[np.ndarray]:
        """
        Render warehouse with pyglet.
        
        Args:
            mode: 'human' for display, 'rgb_array' for array output
            pause_time: Pause duration between frames
            
        Returns:
            RGB array if mode='rgb_array', None otherwise
        """
        if mode == "human":
            if not self.window_visible:
                self._setup_window()
            
            if self.window is None:
                return None
                
            # Draw and display
            self.window.switch_to()
            self.window.dispatch_events()
            self.window.on_draw()
            self.window.flip()
            
            # Simple pause implementation
            import time
            time.sleep(pause_time)
            
            return None
            
        elif mode == "rgb_array":
            # For rgb_array mode, we need to render to a buffer
            # This is more complex in pyglet and would require FBO setup
            # For now, return a placeholder
            rows, cols = self.warehouse.grid_size
            return np.zeros((rows * self.cell_size + 200, cols * self.cell_size + 100, 3), dtype=np.uint8)
        
        return None
    
    def close(self):
        """Close the renderer."""
        if self.window is not None:
            self.window.close()
            self.window = None


# Global renderer cache to avoid creating multiple instances
_renderer_cache = {}

# Convenience function
def render_warehouse(warehouse: 'Warehouse', mode: str = "human", pause_time: float = 0.05) -> Optional[np.ndarray]:
    """
    Render warehouse with enhanced RWare-style visuals.
    
    Args:
        warehouse: Warehouse instance
        mode: 'human' or 'rgb_array'
        pause_time: Pause between frames
        
    Returns:
        RGB array if mode='rgb_array', None otherwise
    """
    warehouse_id = id(warehouse)
    if warehouse_id not in _renderer_cache:
        _renderer_cache[warehouse_id] = RWareStyleRenderer(warehouse)
    
    return _renderer_cache[warehouse_id].render(mode, pause_time)

def debug_render_warehouse(warehouse: 'Warehouse') -> None:
    """
    Simple text-based debug renderer that shows robot assignments and shelf status.
    """
    print("=== WAREHOUSE DEBUG RENDER ===")
    print(f"Grid size: {warehouse.grid_size}")
    print(f"Time: {warehouse.time}")
    
    # Create a mapping of shelf to assigned robot
    shelf_to_robot = {}
    for robot in warehouse.robots:
        if robot.shelf is not None:
            shelf_to_robot[robot.shelf] = robot.robot_id
    
    print("\nRobot Status:")
    mode_names = {0: "IDLE", 1: "→SHELF", 2: "→STATION", 3: "→RETURN"}
    for robot in warehouse.robots:
        mode_name = mode_names.get(robot.mode, "UNKNOWN")
        shelf_info = f"Shelf{robot.shelf}" if robot.shelf is not None else "No Assignment"
        location_info = f"@({robot.current_location[0]},{robot.current_location[1]})"
        target_info = f"→({robot.target_location[0]},{robot.target_location[1]})" if robot.target_location != robot.current_location else "@ Target"
        time_left = f"T:{robot.time_left}" if robot.time_left > 0 else "Ready"
        print(f"  R{robot.robot_id}: {mode_name:<10} | {shelf_info:<12} | {location_info} {target_info} | {time_left}")
    
    print(f"\nRequested Shelves: {warehouse.itemShelfsBufferSet}")
    print(f"Orders in buffer: {len(warehouse.order_buffer)}")
    print(f"Completed orders: {len(warehouse.order_compleated)}")
    
    # Show shelves with robot assignments
    print("\nShelf Assignments:")
    for shelf_idx, robot_id in shelf_to_robot.items():
        rows, cols = warehouse.grid_size
        y = shelf_idx // cols
        x = shelf_idx % cols
        items = len(warehouse.shelfs[shelf_idx])
        print(f"  Shelf {shelf_idx} @({x},{y}): R{robot_id} assigned, {items} items")
    
    print("=" * 40)
