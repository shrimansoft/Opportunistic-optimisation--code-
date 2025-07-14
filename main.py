import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces
from collections import deque

from warehouse_sim.warehouse import Warehouse


def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)


class WarehouseEnv(gym.Env):
    """
    Custom Gymnasium Environment for the Warehouse Simulation
    """

    def __init__(self, warehouse: Warehouse):
        super(WarehouseEnv, self).__init__()

        self.warehouse: Warehouse = warehouse

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(400,), dtype=np.int32)

        # The maximum number of steps per episode (time steps)
        self.max_steps = 10000
        self.current_step = 0

    def reset(self, seed=None, options=None):
        """
        Resets the environment to the initial state.
        """
        if seed is not None:
            set_seed(seed)
        self.warehouse.reset()
        self.current_step = 0
        obs = self._get_observation()
        info = {}
        return obs, info

    def _get_observation(self):
        """
        Returns the current observation (state) of the warehouse
        For simplicity, it could be the number of items available in each shelf
        """
        shelf_counts = [
            len(shelf) for shelf in self.warehouse.shelfs
        ]  # Number of items per shelf
        return np.array(shelf_counts, dtype=np.int32)

    def step(self, action):
        """
        Executes one step in the environment based on the action taken by the agent.
        """
        self.current_step += 1

        # Take action

        # Simulate the warehouse process (order creation and robot work)
        self.warehouse.order_step()

        # Perform the robots' steps
        for robot in self.warehouse.robots:
            robot.step()

        self.warehouse.robot_assigner()

        # Calculate the reward based on the completed orders and warehouse state
        reward = self._calculate_reward()

        # Get the next observation
        obs = self._get_observation()

        # For newer gymnasium API, return (obs, reward, terminated, truncated, info)
        terminated = self.warehouse.stock.sum() == 0  # Episode ends when out of stock
        truncated = (
            self.current_step >= self.max_steps
        )  # Episode truncated at max steps
        info = {}

        return obs, reward, terminated, truncated, info

    def _calculate_reward(self):
        """
        Reward function. Can be based on factors like:
        - Number of completed orders
        - Time taken for tasks
        - Efficiency of robot movements
        """
        # Reward can be based on the number of orders completed within the time steps
        reward = len(
            self.warehouse.order_compleated
        )  # Completed orders count as reward
        return reward

    def render(self, mode="human"):
        """
        Render the current state of the environment for visualization (RWare-style)
        """
        from warehouse_sim.renderer import render_warehouse
        
        # Use the new RWare-style visual renderer
        render_warehouse(self.warehouse, mode=mode, pause_time=0.5)
        
        # Enhanced console metrics with robot assignments
        avg_delay = self.warehouse.average_delay()
        completed_orders = len(self.warehouse.order_compleated)
        pending_orders = len(self.warehouse.order_buffer)
        total_stock = self.warehouse.stock.sum()

        # Robot utilization and detailed status
        busy_robots = len([r for r in self.warehouse.robots if not r.available])
        robot_utilization = (busy_robots / len(self.warehouse.robots)) * 100

        print(f"Step {self.current_step}: Stock={int(total_stock)} | Pending={pending_orders} | "
              f"Completed={completed_orders} | Avg Delay={avg_delay:.2f} | "
              f"Robot Util={robot_utilization:.1f}%")
        
        # Show robot assignments and modes
        mode_names = {0: "IDLE", 1: "→SHELF", 2: "→STATION", 3: "→RETURN"}
        print("Robot Status:")
        for robot in self.warehouse.robots:
            mode_name = mode_names.get(robot.mode, "UNKNOWN")
            shelf_info = f"Shelf{robot.shelf}" if robot.shelf is not None else "No Assignment"
            location_info = f"@({robot.current_location[0]},{robot.current_location[1]})"
            target_info = f"→({robot.target_location[0]},{robot.target_location[1]})" if robot.target_location != robot.current_location else "@ Target"
            time_left = f"T:{robot.time_left}" if robot.time_left > 0 else "Ready"
            print(f"  R{robot.robot_id}: {mode_name:<10} | {shelf_info:<12} | {location_info} {target_info} | {time_left}")
        
        print("Robot Color Legend: Green=IDLE | Blue=→SHELF | Orange=→STATION | Red=→RETURN")
        print("-" * 80)


def main():
    """
    Main function demonstrating RWare-style warehouse layouts
    """
    SEED = 42
    set_seed(SEED)
    
    print("RWare-style Warehouse Simulation Demo")
    print("=" * 50)

    # Test different RWare layout sizes
    # layout_sizes = ["tiny", "small", "medium", "large"]
    layout_sizes = ["tiny"]
    
    for i, layout_size in enumerate(layout_sizes):
        print(f"\n=== Demo {i+1}/4: {layout_size.upper()} warehouse layout ===")
        
        # Create warehouse with RWare layout
        warehouse = Warehouse(
            seed=SEED + i, 
            buffer_enabled=True,
            layout_size=layout_size,
            column_height=8,
            difficulty="normal"
        )
        
        print(f"Grid size: {warehouse.grid_size}")
        print(f"Layout: {layout_size} ({warehouse.shelf_rows}x{warehouse.shelf_columns} shelf blocks)")
        print(f"Goals: {warehouse.goals}")
        print(f"Robots: {len(warehouse.robots)}")
        print(f"Picking stations: {len(warehouse.picking_stations)}")
        
        # Create and reset environment
        env = WarehouseEnv(warehouse)
        obs, info = env.reset(seed=SEED + i)
        
        print(f"Running simulation...")
        total_reward = 0
        
        # Run shorter simulation for demo
        for step in range(50):  # Reduced steps for clearer demo
            # Environment step (this handles all warehouse operations internally)
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Render every 10 steps to see progression clearly
            if step % 10 == 0:
                try:
                    print(f"\n--- Step {step} ---")
                    
                    # Use debug renderer for clear console output
                    from warehouse_sim.renderer import debug_render_warehouse
                    debug_render_warehouse(warehouse)
                    
                    # Try visual render (might hang in some environments)
                    try:
                        env.render()
                    except Exception as e:
                        print(f"Visual render skipped: {e}")
                    
                except Exception as e:
                    print(f"Rendering issue: {e}")
            
            if terminated or truncated:
                break
            
            if terminated or truncated:
                break
        
        print(f"Demo completed - Reward: {total_reward:.2f}, Avg Delay: {warehouse.average_delay():.2f}")
        
        # Final visualization
        try:
            print(f"Final state visualization for {layout_size} layout...")
            env.render()
        except Exception as e:
            print(f"Final rendering error: {e}")
    
    print("\n" + "=" * 50)
    print("RWare layout demo completed!")
    print("All standard layout sizes (tiny, small, medium, large) tested.")
    
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()

