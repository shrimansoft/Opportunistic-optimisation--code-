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
    # If using torch or tensorflow, add their seeds here too
    # torch.manual_seed(seed)
    # tf.random.set_seed(seed)


class WarehouseEnv(gym.Env):
    """
    Custom Gymnasium Environment for the Warehouse Simulation
    """

    def __init__(self, warehouse: Warehouse):
        super(WarehouseEnv, self).__init__()

        # Initialize the warehouse from the existing warehouse class
        self.warehouse: Warehouse = warehouse

        # Define action and observation space
        # Action: Robot can either go to pick a shelf or pick up an item (depending on its state)
        self.action_space = spaces.Discrete(
            4
        )  # Idle, Go to Shelf, Go to Pickup Station, Return Shelf
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(400,), dtype=np.int32
        )  # Shelves' state

        # The maximum number of steps per episode (time steps)
        self.max_steps = 500
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

        # Enhanced plotting with metrics
        self.warehouse.enhanced_plot("data/frames", self.current_step)

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
        Render the current state of the environment for visualization (useful for debugging)
        """
        # Print comprehensive metrics to console
        avg_delay = self.warehouse.average_delay()
        completed_orders = len(self.warehouse.order_compleated)
        pending_orders = len(self.warehouse.order_buffer)
        total_stock = self.warehouse.stock.sum()
        
        # Robot utilization
        busy_robots = len([r for r in self.warehouse.robots if not r.available])
        robot_utilization = (busy_robots / len(self.warehouse.robots)) * 100
        
        # Buffer metrics
        total_buffer_items = sum(len(station.buffer) for station in self.warehouse.picking_stations)
        buffer_orders = len([order for order in self.warehouse.order_compleated if order.delay == 0])
        buffer_hit_rate = (buffer_orders / max(completed_orders, 1)) * 100
        
        print(f"\n{'='*60}")
        print(f"STEP {self.current_step} - WAREHOUSE METRICS")
        print(f"{'='*60}")
        print(f"📦 INVENTORY    : Stock={int(total_stock):4d} | Pending={pending_orders:3d} | Completed={completed_orders:3d}")
        print(f"⏱️  PERFORMANCE : Avg Delay={avg_delay:6.2f} | Buffer Hit Rate={buffer_hit_rate:5.1f}%")
        print(f"🤖 ROBOTS      : Utilization={robot_utilization:5.1f}% | Busy={busy_robots}/{len(self.warehouse.robots)}")
        print(f"📊 BUFFERS     : Total Items={total_buffer_items:3d}")
        
        # Show recent order delays
        if completed_orders > 0:
            recent_delays = [order.delay for order in self.warehouse.order_compleated[-5:]]
            print(f"🕒 RECENT DELAYS: {recent_delays}")
        
        # Show buffer contents
        for i, station in enumerate(self.warehouse.picking_stations):
            if station.buffer_enabled:
                buffer_items = station.buffer[:8]  # Show first 8 items
                print(f"🗂️  BUFFER PS{i}: {len(station.buffer)}/{station.buffer_size} items {buffer_items}")
            else:
                print(f"🗂️  BUFFER PS{i}: DISABLED")
        
        print(f"{'='*60}")

        # Show the interactive plot instead of saving frames
        self.warehouse.enhanced_plot(pause_time=0.5)  # Show plot with half-second pause


def main():
    # Set seed for reproducibility
    SEED = 42
    set_seed(SEED)

    warehouse = Warehouse(seed=SEED)
    # Enable interactive plotting mode for real-time visualization
    warehouse.enable_interactive_plot()
    
    env = WarehouseEnv(warehouse)  # Initialize the Gymnasium environment

    for episode in range(1):
        obs, info = env.reset(seed=SEED + episode)  # Different seed per episode
        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            action = env.action_space.sample()  # Random action
            next_obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            total_reward += reward

        print(f"Episode {episode + 1} finished with total reward: {total_reward}")
    
    # Keep the plot window open after simulation ends
    print("Simulation complete. Close the plot window to exit.")
    input("Press Enter to exit...")


if __name__ == "__main__":
    ...
    # main()

main()
