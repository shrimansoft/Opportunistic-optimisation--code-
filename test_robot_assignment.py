#!/usr/bin/env python3
"""
Test script to debug robot assignment issues.
"""

import sys
sys.path.append('.')

from warehouse_sim.warehouse import Warehouse
from warehouse_sim.renderer import debug_render_warehouse

def test_robot_assignment():
    print("Testing Robot Assignment Logic")
    print("=" * 50)
    
    # Create a simple warehouse
    warehouse = Warehouse(
        seed=42, 
        buffer_enabled=True,
        layout_size="tiny",
        column_height=8,
        difficulty="normal"
    )
    
    print(f"Initial state:")
    print(f"Grid size: {warehouse.grid_size}")
    print(f"Robots: {len(warehouse.robots)}")
    print(f"Picking stations: {len(warehouse.picking_stations)}")
    
    # Initial debug render
    debug_render_warehouse(warehouse)
    
    # Simulate a few steps to see robot assignments
    for step in range(10):
        print(f"\n--- STEP {step} ---")
        
        # Create orders (this should trigger robot assignments)
        warehouse.order_step()
        
        # Check what robots are assigned before they move
        print("After order_step:")
        for robot in warehouse.robots:
            print(f"  Robot {robot.robot_id}: mode={robot.mode}, shelf={robot.shelf}, target={robot.target_location}")
        
        # Move robots
        for robot in warehouse.robots:
            robot.step()
        
        # Assign robots to new tasks
        warehouse.robot_assigner()
        
        print("After robot movement and assignment:")
        debug_render_warehouse(warehouse)
        
        if step >= 5:  # Stop after a few steps for debugging
            break

if __name__ == "__main__":
    test_robot_assignment()
