"""Metrics display for warehouse environment."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from warehouse_sim.warehouse import Warehouse


class MetricsDisplay:
    """Handles metrics display and console output."""
    
    def __init__(self):
        self.mode_names = {0: "IDLE", 1: "→SHELF", 2: "→STATION", 3: "→RETURN"}
    
    def display_step_metrics(self, warehouse: 'Warehouse', step: int) -> None:
        """Display step metrics."""
        busy_robots = len([r for r in warehouse.robots if not r.available])
        utilization = (busy_robots / len(warehouse.robots)) * 100 if warehouse.robots else 0
        
        print(f"Step {step}: Stock={int(warehouse.stock.sum())} | "
              f"Pending={len(warehouse.order_buffer)} | "
              f"Completed={len(warehouse.order_compleated)} | "
              f"Delay={warehouse.average_delay():.2f} | "
              f"Util={utilization:.1f}%")
    
    def display_robot_status(self, warehouse: 'Warehouse') -> None:
        """Display robot status."""
        print("Robot Status:")
        for robot in warehouse.robots:
            mode = self.mode_names.get(robot.mode, "UNKNOWN")
            shelf = f"Shelf{robot.shelf}" if robot.shelf is not None else "No Assignment"
            loc = f"@({robot.current_location[0]},{robot.current_location[1]})"
            target = (f"→({robot.target_location[0]},{robot.target_location[1]})" 
                     if robot.target_location != robot.current_location else "@ Target")
            time_left = f"T:{robot.time_left}" if robot.time_left > 0 else "Ready"
            print(f"  R{robot.robot_id}: {mode:<10} | {shelf:<12} | {loc} {target} | {time_left}")
    
    def display_color_legend(self) -> None:
        """Display color legend."""
        print("Legend: Green=IDLE | Blue=→SHELF | Orange=→STATION | Red=→RETURN")
        print("-" * 80)
    
    def display_demo_header(self, layout_size: str, demo_num: int, total: int) -> None:
        """Display demo header."""
        print(f"\n=== Demo {demo_num}/{total}: {layout_size.upper()} layout ===")
    
    def display_warehouse_info(self, warehouse: 'Warehouse', layout_size: str) -> None:
        """Display warehouse info."""
        print(f"Grid: {warehouse.grid_size} | Layout: {layout_size} "
              f"({warehouse.shelf_rows}x{warehouse.shelf_columns}) | "
              f"Robots: {len(warehouse.robots)} | Stations: {len(warehouse.picking_stations)}")
    
    def display_completion(self, reward: float, delay: float) -> None:
        """Display completion summary."""
        print(f"Completed - Reward: {reward:.2f}, Delay: {delay:.2f}")
    
    def display_final_summary(self) -> None:
        """Display final summary."""
        print("\n" + "=" * 50)
        print("Demo completed! All layouts tested.")
        print("=" * 50)
