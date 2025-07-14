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
            plt.draw()
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
        """Draw shelves with RWare styling."""
        for shelf_idx, shelf_items in enumerate(self.warehouse.shelfs):
            if len(shelf_items) > 0:  # Only draw shelves with items
                # Convert flat index to grid coordinates
                x = shelf_idx % 20
                y = shelf_idx // 20
                
                # Skip highways
                if self.warehouse._is_highway(x, y):
                    continue
                
                # Choose color based on request status
                is_requested = shelf_idx in self.warehouse.itemShelfsBufferSet
                color = RWARE_COLORS['shelf_requested'] if is_requested else RWARE_COLORS['shelf']
                
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
                
                # Add shelf ID with white text
                self.ax.text(x, y, str(shelf_idx), 
                           ha='center', va='center',
                           fontsize=7, fontweight='bold',
                           color=RWARE_COLORS['text_primary'],
                           zorder=3)
    
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
        """Draw robots with RWare styling."""
        for robot in self.warehouse.robots:
            x, y = robot.current_location
            
            # Bounds check
            rows, cols = self.warehouse.grid_size
            if not (0 <= x < cols and 0 <= y < rows):
                continue
            
            # Choose color based on state
            if hasattr(robot, 'mode') and robot.mode != 0:
                color = RWARE_COLORS['robot_busy']
            else:
                color = RWARE_COLORS['robot_idle']
            
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
            
            # Draw direction indicator (arrow)
            # For now, point right (will be enhanced in Phase 2)
            arrow_len = 0.15
            self.ax.arrow(x, y, arrow_len, 0,
                         head_width=0.08, head_length=0.06,
                         fc=RWARE_COLORS['robot_direction'], 
                         ec=RWARE_COLORS['robot_direction'],
                         alpha=0.8, zorder=5, linewidth=1.5)
    
    def _draw_info_panel(self):
        """Draw information panel at bottom with RWare styling."""
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
        self.ax.text(cols/2, -0.8, title,
                    ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    color=RWARE_COLORS['text_secondary'],
                    zorder=11)
        
        # Metrics in RWare style
        metrics = f"Stock: {stock} | Pending: {pending} | Completed: {completed} | Avg Delay: {avg_delay:.1f} | Robots: {busy_robots}/{total_robots}"
        self.ax.text(cols/2, -1.2, metrics,
                    ha='center', va='center',
                    fontsize=10,
                    color=RWARE_COLORS['text_secondary'],
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
