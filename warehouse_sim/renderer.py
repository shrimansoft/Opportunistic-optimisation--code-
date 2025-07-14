"""
RWare-style visual renderer for the warehouse simulation.
This renderer provides multiple backends: matplotlib (stable), pygame (performance), and pyglet (future).
"""

import numpy as np
from typing import Optional, TYPE_CHECKING, Any, Literal
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import matplotlib.patheffects as pe

if TYPE_CHECKING:
    from .warehouse import Warehouse

# RWare color scheme (hex colors for matplotlib)
RWARE_COLORS = {
    'background': '#FFFFFF',
    'grid_lines': '#000000', 
    'highway': '#E8E8E8',
    'shelf': '#484D6D',  # Dark slate blue
    'shelf_requested': '#008080',  # Teal
    'robot_idle': '#FF8C00',  # Dark orange
    'robot_busy': '#FF0000',  # Red
    'robot_direction': '#000000',
    'goal': '#3C3C3C',  # Dark gray
    'picking_station': '#800080',  # Purple
    'text_primary': '#FFFFFF',
    'text_secondary': '#000000',
    'info_panel': '#F0F0F0'
}


class RWareStyleRenderer:
    """Enhanced matplotlib renderer that closely mimics RWare's visual style."""
    
    def __init__(self, warehouse: 'Warehouse', cell_size: float = 0.8, fig_scale: float = 0.8):
        """
        Initialize the RWare-style renderer.
        
        Args:
            warehouse: Warehouse instance to render
            cell_size: Size of cells relative to grid (0.0-1.0) 
            fig_scale: Scale factor for figure size
        """
        self.warehouse = warehouse
        self.cell_size = cell_size
        self.fig_scale = fig_scale
        self.fig = None
        self.ax = None
        self.setup_plot()
    
    def setup_plot(self):
        """Setup matplotlib figure with RWare styling."""
        rows, cols = self.warehouse.grid_size
        
        # Calculate figure size for square cells
        fig_width = cols * self.fig_scale + 2
        fig_height = rows * self.fig_scale + 3  # Extra space for info panel
        
        self.fig, self.ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Set grid bounds with padding
        self.ax.set_xlim(-0.5, cols - 0.5)
        self.ax.set_ylim(-1.5, rows - 0.5)  # Extra space at bottom for info
        self.ax.set_aspect('equal')
        
        # RWare uses inverted Y (0,0 at top-left)
        self.ax.invert_yaxis()
        
        # Clean styling
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False) 
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        
        # Set background
        self.fig.patch.set_facecolor(RWARE_COLORS['background'])
        self.ax.set_facecolor(RWARE_COLORS['background'])
        
        # Enable interactive mode
        plt.ion()
        plt.tight_layout()
    
    def render(self, mode: str = "human", pause_time: float = 0.05) -> Optional[np.ndarray]:
        """
        Render warehouse with RWare-style visuals.
        
        Args:
            mode: 'human' for display, 'rgb_array' for array output
            pause_time: Pause duration between frames
            
        Returns:
            RGB array if mode='rgb_array', None otherwise
        """
        if self.fig is None:
            self.setup_plot()
        
        # Clear and reset 
        self.ax.clear()
        self.ax.set_facecolor(RWARE_COLORS['background'])
        rows, cols = self.warehouse.grid_size
        self.ax.set_xlim(-0.5, cols - 0.5)
        self.ax.set_ylim(-1.5, rows - 0.5)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Hide spines
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        
        # Draw components in RWare order
        self._draw_grid()
        self._draw_highways()
        self._draw_shelves()
        self._draw_goals()
        self._draw_robots()
        self._draw_info_panel()
        
        # Handle output
        if mode == "human":
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(pause_time)
            return None
        elif mode == "rgb_array":
            import io
            self.fig.canvas.draw()
            buf = io.BytesIO()
            self.fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            from PIL import Image
            img = Image.open(buf)
            return np.array(img)[:, :, :3]  # Remove alpha if present
        
        return None
    
    def _draw_grid(self):
        """Draw subtle grid lines like RWare."""
        rows, cols = self.warehouse.grid_size
        
        # Vertical lines
        for x in range(cols + 1):
            self.ax.axvline(x - 0.5, color=RWARE_COLORS['grid_lines'], 
                           linewidth=0.5, alpha=0.2, zorder=0)
        
        # Horizontal lines
        for y in range(rows + 1):
            self.ax.axhline(y - 0.5, color=RWARE_COLORS['grid_lines'], 
                           linewidth=0.5, alpha=0.2, zorder=0)
    
    def _draw_highways(self):
        """Draw highway cells with light gray background."""
        rows, cols = self.warehouse.grid_size
        
        for y in range(rows):
            for x in range(cols):
                if self.warehouse._is_highway(x, y):
                    rect = Rectangle(
                        (x - 0.5, y - 0.5), 1, 1,
                        facecolor=RWARE_COLORS['highway'],
                        edgecolor='none',
                        alpha=0.6,
                        zorder=1
                    )
                    self.ax.add_patch(rect)
    
    def _draw_shelves(self):
        """Draw shelves with RWare styling and robot assignments."""
        rows, cols = self.warehouse.grid_size
        
        # Create a mapping of shelf to assigned robot
        shelf_to_robot = {}
        for robot in self.warehouse.robots:
            if robot.shelf is not None:
                shelf_to_robot[robot.shelf] = robot.robot_id
        
        for shelf_idx, shelf_items in enumerate(self.warehouse.shelfs):
            # Convert flat index to grid coordinates
            # The shelfs array is indexed by flat coordinates (row-major order)
            y = shelf_idx // cols  # Row (y-coordinate)
            x = shelf_idx % cols   # Column (x-coordinate)
            
            # Skip highways and out-of-bounds positions
            if x >= cols or y >= rows or self.warehouse._is_highway(x, y):
                continue
            
            # Only draw shelves that are supposed to be shelf locations
            # Check if this position should have a shelf based on warehouse layout
            if not self._is_shelf_position(x, y):
                continue
                
            num_items = len(shelf_items)
            
            # Choose color based on request status and content
            is_requested = shelf_idx in self.warehouse.itemShelfsBufferSet
            has_items = num_items > 0
            
            if is_requested:
                color = RWARE_COLORS['shelf_requested']  # Teal for requested shelves
            elif has_items:
                color = RWARE_COLORS['shelf']  # Dark blue for shelves with items
            else:
                color = '#A0A0A0'  # Light gray for empty shelves
            
            # Draw shelf with padding (RWare style)
            padding = 0.1
            rect = Rectangle(
                (x - 0.5 + padding, y - 0.5 + padding), 
                1 - 2*padding, 1 - 2*padding,
                facecolor=color,
                edgecolor=RWARE_COLORS['grid_lines'],
                linewidth=1,
                alpha=0.9,
                zorder=2
            )
            self.ax.add_patch(rect)
            
            # Add shelf information: ID and item count
            # Top number = shelf_idx (unique shelf identifier)
            # Bottom number = number of items currently stored in this shelf
            self.ax.text(x, y - 0.2, str(shelf_idx), 
                       ha='center', va='center',
                       fontsize=6, fontweight='bold',
                       color=RWARE_COLORS['text_primary'],
                       zorder=3)
            
            # Display item count
            self.ax.text(x, y, f"({num_items})", 
                       ha='center', va='center',
                       fontsize=6, fontweight='normal',
                       color=RWARE_COLORS['text_primary'],
                       zorder=3)
            
            # Display assigned robot ID if any
            if shelf_idx in shelf_to_robot:
                robot_id = shelf_to_robot[shelf_idx]
                self.ax.text(x, y + 0.2, f"R{robot_id}", 
                           ha='center', va='center',
                           fontsize=7, fontweight='bold',
                           color='yellow',  # Bright color for visibility
                           zorder=3,
                           bbox=dict(boxstyle="round,pad=0.1", facecolor='black', alpha=0.7))
    
    def _is_shelf_position(self, x: int, y: int) -> bool:
        """Check if a position should have a shelf based on warehouse layout."""
        # This is a simplified check - you might need to adjust based on your specific layout
        # For now, assume non-highway positions can have shelves
        return not self.warehouse._is_highway(x, y)
    
    def _draw_goals(self):
        """Draw picking stations as goals."""
        for station in self.warehouse.picking_stations:
            x, y = station.location
            
            # Draw goal rectangle
            rect = Rectangle(
                (x - 0.5, y - 0.5), 1, 1,
                facecolor=RWARE_COLORS['goal'],
                edgecolor=RWARE_COLORS['grid_lines'],
                linewidth=2,
                alpha=0.9,
                zorder=2
            )
            self.ax.add_patch(rect)
            
            # Add 'G' label with enhanced styling
            text = self.ax.text(x, y, 'G', 
                              ha='center', va='center',
                              fontsize=16, fontweight='bold',
                              color=RWARE_COLORS['text_primary'],
                              zorder=3)
            # Add text outline for better visibility
            text.set_path_effects([pe.withStroke(linewidth=2, foreground='black')])
    
    def _draw_robots(self):
        """Draw robots with RWare styling and mode-specific colors."""
        rows, cols = self.warehouse.grid_size
        
        for robot in self.warehouse.robots:
            x, y = robot.current_location
            
            # Bounds check - make sure robot is within grid
            if not (0 <= x < cols and 0 <= y < rows):
                continue
            
            # Choose color based on robot mode (like legacy plotter)
            # Mode 0: IDLE (Green), Mode 1: ->SHELF (Blue), Mode 2: ->STATION (Orange), Mode 3: ->RETURN (Red)
            mode_colors = {
                0: '#00FF00',  # Green - IDLE
                1: '#0000FF',  # Blue - Going to shelf
                2: '#FFA500',  # Orange - Going to station
                3: '#FF0000'   # Red - Returning shelf
            }
            color = mode_colors.get(robot.mode, RWARE_COLORS['robot_idle'])
            
            # Draw robot as circle (RWare style)
            circle = Circle(
                (x, y), 0.25,
                facecolor=color,
                edgecolor=RWARE_COLORS['robot_direction'],
                linewidth=2,
                alpha=0.95,
                zorder=4
            )
            self.ax.add_patch(circle)
            
            # Add robot ID
            self.ax.text(x, y, str(robot.robot_id),
                        ha='center', va='center', 
                        fontsize=8, fontweight='bold',
                        color=RWARE_COLORS['text_primary'],
                        zorder=5)
            
            # Draw direction indicator (arrow) - point towards target
            if hasattr(robot, 'target_location') and robot.target_location != robot.current_location:
                target_x, target_y = robot.target_location
                # Calculate direction vector
                dx = target_x - x
                dy = target_y - y
                # Normalize to arrow length
                arrow_len = 0.15
                if dx != 0 or dy != 0:
                    length = (dx*dx + dy*dy) ** 0.5
                    dx = (dx / length) * arrow_len
                    dy = (dy / length) * arrow_len
                    
                    self.ax.arrow(x, y, dx, dy,
                                 head_width=0.08, head_length=0.06,
                                 fc=RWARE_COLORS['robot_direction'], 
                                 ec=RWARE_COLORS['robot_direction'],
                                 alpha=0.8, zorder=5, linewidth=1.5)
            else:
                # Default arrow pointing right if no target
                arrow_len = 0.15
                self.ax.arrow(x, y, arrow_len, 0,
                             head_width=0.08, head_length=0.06,
                             fc=RWARE_COLORS['robot_direction'], 
                             ec=RWARE_COLORS['robot_direction'],
                             alpha=0.8, zorder=5, linewidth=1.5)
    
    def _draw_info_panel(self):
        """Draw information panel at bottom with RWare styling and robot assignments."""
        rows, cols = self.warehouse.grid_size
        
        # Calculate metrics
        avg_delay = self.warehouse.average_delay()
        completed = len(self.warehouse.order_compleated)
        pending = len(self.warehouse.order_buffer)
        stock = int(self.warehouse.stock.sum())
        busy_robots = len([r for r in self.warehouse.robots if not r.available])
        total_robots = len(self.warehouse.robots)
        
        # Create info panel background
        panel = FancyBboxPatch(
            (-0.4, -1.4), cols - 0.2, 0.8,
            boxstyle="round,pad=0.1",
            facecolor=RWARE_COLORS['info_panel'],
            edgecolor=RWARE_COLORS['grid_lines'],
            linewidth=1,
            alpha=0.9,
            zorder=10
        )
        self.ax.add_patch(panel)
        
        # Title
        title = f"Warehouse Simulation - Step {self.warehouse.time}"
        self.ax.text(cols/2, -0.75, title,
                    ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    color=RWARE_COLORS['text_secondary'],
                    zorder=11)
        
        # Main metrics
        metrics = f"Stock: {stock} | Pending: {pending} | Completed: {completed} | Avg Delay: {avg_delay:.1f} | Robots: {busy_robots}/{total_robots}"
        self.ax.text(cols/2, -0.95, metrics,
                    ha='center', va='center',
                    fontsize=10,
                    color=RWARE_COLORS['text_secondary'],
                    zorder=11)
        
        # Robot assignments and color legend
        mode_names = {0: "IDLE", 1: "→SHELF", 2: "→STATION", 3: "→RETURN"}
        mode_colors = {0: "Green", 1: "Blue", 2: "Orange", 3: "Red"}
        
        robot_info = []
        for robot in self.warehouse.robots:
            mode_name = mode_names.get(robot.mode, "UNKNOWN")
            shelf_info = f"Shelf{robot.shelf}" if robot.shelf is not None else "No Shelf"
            robot_info.append(f"R{robot.robot_id}:{mode_name}({shelf_info})")
        
        # Robot assignments text (split into lines if too long)
        robot_text = " | ".join(robot_info)
        if len(robot_text) > 80:  # Split long text
            mid = len(robot_info) // 2
            line1 = " | ".join(robot_info[:mid])
            line2 = " | ".join(robot_info[mid:])
            self.ax.text(cols/2, -1.15, line1, ha='center', va='center', fontsize=8, 
                        color=RWARE_COLORS['text_secondary'], zorder=11)
            self.ax.text(cols/2, -1.25, line2, ha='center', va='center', fontsize=8,
                        color=RWARE_COLORS['text_secondary'], zorder=11)
        else:
            self.ax.text(cols/2, -1.15, robot_text, ha='center', va='center', fontsize=8,
                        color=RWARE_COLORS['text_secondary'], zorder=11)
        
        # Color legend
        legend_text = "Robot Colors: Green=IDLE | Blue=→SHELF | Orange=→STATION | Red=→RETURN"
        self.ax.text(cols/2, -1.35, legend_text, ha='center', va='center', fontsize=8,
                    color=RWARE_COLORS['text_secondary'], style='italic', zorder=11)
    
    def close(self):
        """Close the renderer."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


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
    if not hasattr(warehouse, '_rware_renderer'):
        warehouse._rware_renderer = RWareStyleRenderer(warehouse)
    
    return warehouse._rware_renderer.render(mode, pause_time)

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
