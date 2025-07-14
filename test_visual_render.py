#!/usr/bin/env python3
"""
Test visual rendering with robot assignments.
"""

import sys
sys.path.append('.')

from warehouse_sim.warehouse import Warehouse
from warehouse_sim.renderer import render_warehouse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid hanging

def test_visual_rendering():
    print("Testing Visual Rendering")
    print("=" * 50)
    
    # Create a simple warehouse
    warehouse = Warehouse(
        seed=42, 
        buffer_enabled=True,
        layout_size="tiny",
        column_height=8,
        difficulty="normal"
    )
    
    # Run a few steps to get robot assignments
    for step in range(15):
        warehouse.order_step()
        for robot in warehouse.robots:
            robot.step()
        warehouse.robot_assigner()
        
        if step % 5 == 0:
            print(f"\nStep {step}:")
            for robot in warehouse.robots:
                mode_names = {0: "IDLE", 1: "→SHELF", 2: "→STATION", 3: "→RETURN"}
                mode_name = mode_names.get(robot.mode, "UNKNOWN")
                shelf_info = f"Shelf{robot.shelf}" if robot.shelf is not None else "No Assignment"
                print(f"  R{robot.robot_id}: {mode_name} | {shelf_info} | @{robot.current_location} →{robot.target_location}")
            
            # Try to render
            try:
                result = render_warehouse(warehouse, mode="rgb_array", pause_time=0.01)
                if result is not None:
                    print(f"  ✅ Rendered successfully! Shape: {result.shape}")
                else:
                    print("  ⚠️ Render returned None")
            except Exception as e:
                print(f"  ❌ Render failed: {e}")

if __name__ == "__main__":
    test_visual_rendering()
