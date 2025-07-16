"""
Metrics display and console output for the warehouse environment.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from warehouse_sim.warehouse import Warehouse


class MetricsDisplay:
    """
    Handles metrics display and console output for the warehouse environment.
    Provides clear separation of rendering concerns.
    """
    
    def __init__(self):
        """Initialize the metrics display handler."""
        self.mode_names = {0: "IDLE", 1: "→SHELF", 2: "→STATION", 3: "→RETURN"}
    
    def display_step_metrics(self, warehouse: 'Warehouse', current_step: int) -> None:
        """
        Display step-by-step metrics for the warehouse simulation.
        
        Args:
            warehouse: The warehouse instance
            current_step: Current simulation step
        """
        # Calculate key metrics
        avg_delay = warehouse.average_delay()
        completed_orders = len(warehouse.order_compleated)
        pending_orders = len(warehouse.order_buffer)
        total_stock = warehouse.stock.sum()
        
        # Robot utilization
        busy_robots = len([r for r in warehouse.robots if not r.available])
        robot_utilization = (busy_robots / len(warehouse.robots)) * 100 if warehouse.robots else 0
        
        # Display main metrics
        print(f"Step {current_step}: Stock={int(total_stock)} | Pending={pending_orders} | "
              f"Completed={completed_orders} | Avg Delay={avg_delay:.2f} | "
              f"Robot Util={robot_utilization:.1f}%")
    
    def display_robot_status(self, warehouse: 'Warehouse') -> None:
        """
        Display detailed robot status information.
        
        Args:
            warehouse: The warehouse instance
        """
        print("Robot Status:")
        for robot in warehouse.robots:
            mode_name = self.mode_names.get(robot.mode, "UNKNOWN")
            shelf_info = f"Shelf{robot.shelf}" if robot.shelf is not None else "No Assignment"
            location_info = f"@({robot.current_location[0]},{robot.current_location[1]})"
            
            if robot.target_location != robot.current_location:
                target_info = f"→({robot.target_location[0]},{robot.target_location[1]})"
            else:
                target_info = "@ Target"
            
            time_left = f"T:{robot.time_left}" if robot.time_left > 0 else "Ready"
            
            print(f"  R{robot.robot_id}: {mode_name:<10} | {shelf_info:<12} | "
                  f"{location_info} {target_info} | {time_left}")
    
    def display_color_legend(self) -> None:
        """Display the color legend for robot visualization."""
        print("Robot Color Legend: Green=IDLE | Blue=→SHELF | Orange=→STATION | Red=→RETURN")
        print("-" * 80)
    
    def display_demo_header(self, layout_size: str, demo_num: int, total_demos: int) -> None:
        """
        Display demo section header.
        
        Args:
            layout_size: Size of the warehouse layout
            demo_num: Current demo number
            total_demos: Total number of demos
        """
        print(f"\n=== Demo {demo_num}/{total_demos}: {layout_size.upper()} warehouse layout ===")
    
    def display_warehouse_info(self, warehouse: 'Warehouse', layout_size: str) -> None:
        """
        Display warehouse configuration information.
        
        Args:
            warehouse: The warehouse instance
            layout_size: Size of the warehouse layout
        """
        print(f"Grid size: {warehouse.grid_size}")
        print(f"Layout: {layout_size} ({warehouse.shelf_rows}x{warehouse.shelf_columns} shelf blocks)")
        print(f"Goals: {warehouse.goals}")
        print(f"Robots: {len(warehouse.robots)}")
        print(f"Picking stations: {len(warehouse.picking_stations)}")
    
    def display_demo_completion(self, total_reward: float, avg_delay: float) -> None:
        """
        Display demo completion summary.
        
        Args:
            total_reward: Total reward accumulated
            avg_delay: Average delay in the simulation
        """
        print(f"Demo completed - Reward: {total_reward:.2f}, Avg Delay: {avg_delay:.2f}")
    
    def display_final_summary(self) -> None:
        """Display final summary of all demos."""
        print("\n" + "=" * 50)
        print("RWare layout demo completed!")
        print("All standard layout sizes (tiny, small, medium, large) tested.")
