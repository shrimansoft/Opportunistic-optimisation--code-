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
        """Setup matplotlib figure with enhanced RWare styling and performance optimizations."""
        rows, cols = self.warehouse.grid_size
        
        # Calculate optimized figure size for better visual quality
        fig_width = max(8, cols * self.fig_scale + 2)
        fig_height = max(6, rows * self.fig_scale + 4.2)  # Extra space for two-row info panel
        
        # Create figure with high DPI for crisp rendering
        self.fig, self.ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
        
        # Set grid bounds with optimized padding
        self.ax.set_xlim(-0.6, cols - 0.4)
        self.ax.set_ylim(-2.5, rows - 0.4)  # Expanded space for two-row info panel
        self.ax.set_aspect('equal')
        
        # RWare uses inverted Y (0,0 at top-left)
        self.ax.invert_yaxis()
        
        # Enhanced clean styling
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        
        # Premium background setup
        self.fig.patch.set_facecolor(RWARE_COLORS['background'])
        self.ax.set_facecolor(RWARE_COLORS['background'])
        
        # Enable interactive mode with better performance
        plt.ion()
        plt.tight_layout(pad=0.5)
    
    def render(self, mode: str = "human", pause_time: float = 0.05) -> Optional[np.ndarray]:
        """
        Render warehouse with enhanced RWare-style visuals and improved performance.
        
        Args:
            mode: 'human' for display, 'rgb_array' for array output
            pause_time: Pause duration between frames
            
        Returns:
            RGB array if mode='rgb_array', None otherwise
        """
        if self.fig is None:
            self.setup_plot()
        
        # Clear and reset with enhanced bounds
        self.ax.clear()
        self.ax.set_facecolor(RWARE_COLORS['background'])
        rows, cols = self.warehouse.grid_size
        self.ax.set_xlim(-0.6, cols - 0.4)
        self.ax.set_ylim(-2.5, rows - 0.4)  # Expanded for two-row info panel
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Hide spines for clean look
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        
        # Draw components in optimized order for better layering
        self._draw_grid()
        self._draw_highways()
        self._draw_shelves()
        self._draw_goals()
        self._draw_robots()
        self._draw_info_panel()
        
        # Add subtle title/branding
        self.ax.text(cols - 0.5, rows - 0.3, "Enhanced RWare Renderer", 
                    ha='right', va='top', fontsize=8, 
                    color='#999999', style='italic', alpha=0.7, zorder=1)
        
        # Handle output with optimized rendering
        if mode == "human":
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(pause_time)
            return None
        elif mode == "rgb_array":
            import io
            self.fig.canvas.draw()
            buf = io.BytesIO()
            self.fig.savefig(buf, format='png', bbox_inches='tight', 
                           dpi=100, facecolor='white', edgecolor='none')
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
        """Draw shelves with enhanced visual styling and responsive text."""
        rows, cols = self.warehouse.grid_size
        
        # Calculate responsive font sizes for shelf labels
        shelf_font_scale = min(cols, rows) * self.fig_scale * 0.12
        shelf_id_font = max(5, min(8, shelf_font_scale * 0.8))
        shelf_item_font = max(8, min(14, shelf_font_scale * 1.4))
        shelf_robot_font = max(6, min(10, shelf_font_scale * 1.0))
        
        # Create a mapping of shelf to assigned robot
        shelf_to_robot = {}
        for robot in self.warehouse.robots:
            if robot.shelf is not None:
                shelf_to_robot[robot.shelf] = robot.robot_id
        
        for shelf_idx, shelf_items in enumerate(self.warehouse.shelfs):
            # Convert flat index to grid coordinates
            y = shelf_idx // cols  # Row (y-coordinate)
            x = shelf_idx % cols   # Column (x-coordinate)
            
            # Skip highways and out-of-bounds positions
            if x >= cols or y >= rows or self.warehouse._is_highway(x, y):
                continue
            
            # Only draw shelves that are supposed to be shelf locations
            if not self._is_shelf_position(x, y):
                continue
                
            num_items = len(shelf_items)
            
            # Enhanced color scheme with better visual hierarchy
            is_requested = shelf_idx in self.warehouse.itemShelfsBufferSet
            has_items = num_items > 0
            has_robot = shelf_idx in shelf_to_robot
            
            if is_requested and has_robot:
                color = '#00CED1'  # Dark turquoise - requested with robot assigned
                border_color = '#FFD700'  # Gold border for active assignment
            elif is_requested:
                color = RWARE_COLORS['shelf_requested']  # Teal for requested shelves
                border_color = '#FF6347'  # Tomato border for urgent
            elif has_robot:
                color = '#8A2BE2'  # Blue violet - robot assigned but not requested
                border_color = '#32CD32'  # Lime border for assigned
            elif has_items:
                color = RWARE_COLORS['shelf']  # Dark blue for shelves with items
                border_color = '#708090'  # Slate gray
            else:
                color = '#D3D3D3'  # Light gray for empty shelves
                border_color = '#A9A9A9'  # Dark gray
            
            # Draw shelf with enhanced styling
            padding = 0.08
            rect = Rectangle(
                (x - 0.5 + padding, y - 0.5 + padding), 
                1 - 2*padding, 1 - 2*padding,
                facecolor=color,
                edgecolor=border_color,
                linewidth=2 if (is_requested or has_robot) else 1,
                alpha=0.9,
                zorder=2
            )
            self.ax.add_patch(rect)
            
            # Add subtle gradient effect for active shelves
            if is_requested or has_robot:
                highlight = Rectangle(
                    (x - 0.5 + padding, y - 0.5 + padding), 
                    (1 - 2*padding) * 0.3, 1 - 2*padding,
                    facecolor='white',
                    alpha=0.2,
                    zorder=2.5
                )
                self.ax.add_patch(highlight)
            
            # Enhanced text layout with better readability
            # Shelf ID (top-left, smaller)
            self.ax.text(x - 0.3, y - 0.25, f"{shelf_idx}", 
                       ha='left', va='center',
                       fontsize=shelf_id_font, fontweight='bold',
                       color='white' if color != '#D3D3D3' else 'black',
                       zorder=3,
                       bbox=dict(boxstyle="round,pad=0.1", 
                                facecolor='black' if color != '#D3D3D3' else 'white', 
                                alpha=0.6))
            
            # Item count (center, prominent)
            item_text = f"{num_items}" if num_items > 0 else "∅"
            item_color = 'white' if color != '#D3D3D3' else 'black'
            self.ax.text(x, y, item_text, 
                       ha='center', va='center',
                       fontsize=shelf_item_font if num_items > 0 else shelf_item_font * 0.8, 
                       fontweight='bold',
                       color=item_color,
                       zorder=3)
            
            # Robot assignment indicator (bottom-right, highlighted)
            if shelf_idx in shelf_to_robot:
                robot_id = shelf_to_robot[shelf_idx]
                robot_text = self.ax.text(x + 0.25, y + 0.25, f"R{robot_id}", 
                           ha='center', va='center',
                           fontsize=shelf_robot_font, fontweight='bold',
                           color='white',
                           zorder=4,
                           bbox=dict(boxstyle="round,pad=0.15", 
                                    facecolor='#FF4500', 
                                    edgecolor='white',
                                    linewidth=1,
                                    alpha=0.95))
                # Add glow effect for robot assignment
                robot_text.set_path_effects([pe.withStroke(linewidth=2, foreground='black')])
            
            # Priority indicator for urgent requests
            if is_requested:
                priority_circle = Circle(
                    (x - 0.35, y - 0.35), 0.08,
                    facecolor='red',
                    edgecolor='white',
                    linewidth=1.5,
                    alpha=0.9,
                    zorder=4
                )
                self.ax.add_patch(priority_circle)
                self.ax.text(x - 0.35, y - 0.35, "!", 
                           ha='center', va='center',
                           fontsize=shelf_robot_font, fontweight='bold',
                           color='white',
                           zorder=5)
    
    def _is_shelf_position(self, x: int, y: int) -> bool:
        """Check if a position should have a shelf based on warehouse layout."""
        # This is a simplified check - you might need to adjust based on your specific layout
        # For now, assume non-highway positions can have shelves
        return not self.warehouse._is_highway(x, y)
    
    def _draw_goals(self):
        """Draw picking stations as goals with responsive text."""
        # Calculate responsive font size for goal labels
        rows, cols = self.warehouse.grid_size
        goal_font_scale = min(cols, rows) * self.fig_scale * 0.15
        goal_font = max(10, min(20, goal_font_scale * 2.0))
        
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
                              fontsize=goal_font, fontweight='bold',
                              color=RWARE_COLORS['text_primary'],
                              zorder=3)
            # Add text outline for better visibility
            text.set_path_effects([pe.withStroke(linewidth=2, foreground='black')])
    
    def _draw_robots(self):
        """Draw robots with enhanced transparency and responsive text."""
        rows, cols = self.warehouse.grid_size
        
        # Calculate responsive font size for robot IDs
        robot_font_scale = min(cols, rows) * self.fig_scale * 0.12
        robot_font = max(8, min(16, robot_font_scale * 1.6))
        
        for robot in self.warehouse.robots:
            x, y = robot.current_location
            
            # Bounds check - make sure robot is within grid
            if not (0 <= x < cols and 0 <= y < rows):
                continue
            
            # Enhanced color scheme with better contrast
            mode_colors = {
                0: '#32CD32',  # Lime Green - IDLE (more vibrant)
                1: '#4169E1',  # Royal Blue - Going to shelf
                2: '#FF8C00',  # Dark Orange - Going to station
                3: '#DC143C'   # Crimson - Returning shelf
            }
            color = mode_colors.get(robot.mode, RWARE_COLORS['robot_idle'])
            
            # Draw robot with improved transparency and styling
            circle = Circle(
                (x, y), 0.3,  # Slightly larger for better visibility
                facecolor=color,
                edgecolor='#2F2F2F',  # Darker edge for better definition
                linewidth=2.5,
                alpha=0.75,  # More transparent
                zorder=4
            )
            self.ax.add_patch(circle)
            
            # Add subtle glow effect for active robots
            if robot.mode != 0:  # Not idle
                glow_circle = Circle(
                    (x, y), 0.4,
                    facecolor=color,
                    edgecolor='none',
                    alpha=0.2,
                    zorder=3
                )
                self.ax.add_patch(glow_circle)
            
            # Enhanced robot ID with better contrast and responsive font
            text = self.ax.text(x, y, str(robot.robot_id),
                               ha='center', va='center', 
                               fontsize=robot_font, fontweight='bold',
                               color='white',
                               zorder=5)
            # Add text outline for better readability
            text.set_path_effects([pe.withStroke(linewidth=3, foreground='black')])
            
            # Add direction indicator if robot is moving
            if robot.target_location != robot.current_location:
                target_x, target_y = robot.target_location
                dx = target_x - x
                dy = target_y - y
                
                # Normalize direction
                if dx != 0 or dy != 0:
                    length = (dx**2 + dy**2)**0.5
                    dx_norm = dx / length * 0.15
                    dy_norm = dy / length * 0.15
                    
                    # Draw arrow
                    self.ax.arrow(x + dx_norm*0.5, y + dy_norm*0.5, 
                                 dx_norm, dy_norm,
                                 head_width=0.08, head_length=0.08,
                                 fc='white', ec='black', linewidth=1.5,
                                 alpha=0.9, zorder=6)
    
    def _draw_info_panel(self):
        """Draw simple ASCII-only information panel with responsive font sizes."""
        rows, cols = self.warehouse.grid_size
        
        # Calculate responsive font sizes based on grid size and figure scale
        base_font_scale = min(cols, rows) * self.fig_scale * 0.15
        title_font = max(8, min(16, base_font_scale * 1.8))
        text_font = max(7, min(12, base_font_scale * 1.4))
        small_font = max(6, min(10, base_font_scale * 1.2))
        tiny_font = max(5, min(9, base_font_scale * 1.0))
        
        # Calculate metrics
        avg_delay = self.warehouse.average_delay()
        completed = len(self.warehouse.order_compleated)
        pending = len(self.warehouse.order_buffer)
        stock = int(self.warehouse.stock.sum())
        busy_robots = len([r for r in self.warehouse.robots if not r.available])
        total_robots = len(self.warehouse.robots)
        efficiency = (busy_robots/total_robots*100) if total_robots > 0 else 0
        
        # Simple background panel
        panel = Rectangle(
            (-0.5, -2.0), cols, 1.5,
            facecolor='#F5F5F5',
            edgecolor='#CCCCCC',
            linewidth=1,
            alpha=0.9,
            zorder=10
        )
        self.ax.add_patch(panel)
        
        # Title
        self.ax.text(cols/2, -0.7, "WAREHOUSE OPERATIONS DASHBOARD",
                    ha='center', va='center',
                    fontsize=title_font, fontweight='bold',
                    color='#000000',
                    zorder=11)
        
        # Step counter
        self.ax.text(cols - 0.2, -0.7, f"Step: {self.warehouse.time}",
                    ha='right', va='center',
                    fontsize=small_font,
                    color='#000000',
                    zorder=11)
        
        # Metrics row
        metrics_line = f"Stock: {stock}    Orders: {pending}P/{completed}C    Efficiency: {efficiency:.0f}%    Avg Delay: {avg_delay:.1f}s"
        self.ax.text(cols/2, -1.0, metrics_line,
                    ha='center', va='center',
                    fontsize=text_font,
                    color='#000000',
                    zorder=11)
        
        # Robot status
        mode_names = {0: "IDLE", 1: "TO_SHELF", 2: "TO_STATION", 3: "RETURNING"}
        robot_groups = {mode: [] for mode in range(4)}
        for robot in self.warehouse.robots:
            robot_groups[robot.mode].append(robot)
        
        # Robot status line
        status_parts = []
        for mode in [0, 1, 2, 3]:
            count = len(robot_groups[mode])
            if count > 0:
                robot_ids = [str(r.robot_id) for r in robot_groups[mode][:8]]
                id_str = ",".join(robot_ids)
                if len(robot_groups[mode]) > 8:
                    id_str += "..."
                status_parts.append(f"{mode_names[mode]}({count}): {id_str}")
            else:
                status_parts.append(f"{mode_names[mode]}(0)")
        
        robot_line = "  |  ".join(status_parts)
        self.ax.text(cols/2, -1.3, robot_line,
                    ha='center', va='center',
                    fontsize=small_font,
                    color='#000000',
                    zorder=11)
        
        # Performance status
        if avg_delay < 5:
            perf_status = "PERFORMANCE: EXCELLENT"
        elif avg_delay < 10:
            perf_status = "PERFORMANCE: GOOD"
        elif avg_delay < 20:
            perf_status = "PERFORMANCE: MODERATE"
        else:
            perf_status = "PERFORMANCE: POOR"
        
        self.ax.text(cols/2, -1.6, perf_status,
                    ha='center', va='center',
                    fontsize=small_font,
                    color='#000000',
                    zorder=11)
        
        # Timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.ax.text(cols - 0.2, -1.9, f"Time: {timestamp}",
                    ha='right', va='center',
                    fontsize=tiny_font,
                    color='#666666',
                    zorder=11)
    
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
