# RWare Warehouse Usage Guide

## Quick Start

### Standard RWare Layouts

```python
from warehouse_sim.warehouse import Warehouse

# Create different sized warehouses
tiny_warehouse = Warehouse(layout_size="tiny", column_height=8)
small_warehouse = Warehouse(layout_size="small", column_height=8) 
medium_warehouse = Warehouse(layout_size="medium", column_height=8)
large_warehouse = Warehouse(layout_size="large", column_height=8)

# Print layout information
print(f"Tiny: {tiny_warehouse.grid_size}, {len(tiny_warehouse.robots)} robots")
print(f"Small: {small_warehouse.grid_size}, {len(small_warehouse.robots)} robots")
print(f"Medium: {medium_warehouse.grid_size}, {len(medium_warehouse.robots)} robots")
print(f"Large: {large_warehouse.grid_size}, {len(large_warehouse.robots)} robots")
```

### Custom Layouts

```python
# Define a custom warehouse layout
custom_layout = """
..........
.XX.XX.XX.
.XX.XX.XX.
..........
.XX.XX.XX.
.XX.XX.XX.
..........
....gg....
"""

# Create warehouse with custom layout
warehouse = Warehouse(custom_layout=custom_layout.strip())
print(f"Custom warehouse: {warehouse.grid_size}")
```

### Rendering

```python
from warehouse_sim.renderer import render_warehouse

# Create and visualize warehouse
warehouse = Warehouse(layout_size="medium")
render_warehouse(warehouse, mode="human", pause_time=1.0)
```

### Simulation Loop

```python
# Basic simulation
warehouse = Warehouse(layout_size="small", seed=42)

for step in range(100):
    # Process orders
    warehouse.order_step()
    
    # Assign robots
    warehouse.robot_assigner()
    
    # Update robots
    for robot in warehouse.robots:
        robot.step()
    
    # Visualize every 10 steps
    if step % 10 == 0:
        warehouse.enhanced_plot(pause_time=0.1)
        
    print(f"Step {step}: Avg Delay: {warehouse.average_delay():.2f}")
```

## Layout Specifications

### Standard Sizes
- **Tiny**: 1×3 shelf blocks → 11×10 grid (32 shelf positions, 4 robots)
- **Small**: 2×3 shelf blocks → 20×10 grid (80 shelf positions, 6 robots)  
- **Medium**: 2×5 shelf blocks → 20×16 grid (144 shelf positions, 6 robots)
- **Large**: 3×5 shelf blocks → 29×16 grid (224 shelf positions, 8 robots)

### Custom Layout Format
- `X` or `x`: Shelf area (non-highway cell)
- `.`: Highway/corridor (robot movement area)
- `G` or `g`: Goal location (picking station)

### Grid Size Calculation
```
height = (column_height + 1) × shelf_rows + 2
width = (2 + 1) × shelf_columns + 1
```

## Features

✅ **Standard RWare layouts** (tiny, small, medium, large)  
✅ **Custom layout support** with string parsing  
✅ **Proper highway/corridor system**  
✅ **Dynamic robot and station scaling**  
✅ **RWare-compatible grid system**  
✅ **Visual rendering** with matplotlib  
✅ **Collision layer management**  
✅ **Highway-aware pathfinding**  

## Migration Status

**Phase 1 Complete**: Layout system, grid management, highway system, entity foundation  
**Phase 2 Pending**: Full agent system, action handling, collision detection  
**Phase 3 Pending**: Observation types, reward systems, advanced features  

The warehouse simulation now fully supports RWare-style layouts and is ready for further migration phases!
