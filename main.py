import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces
from collections import deque

from warehouse_sim.warehouse import Warehouse


class WarehouseEnv(gym.Env):
    """
    Custom Gymnasium Environment for the Warehouse Simulation
    """

    def __init__(self, warehouse: Warehouse):
        super(WarehouseEnv, self).__init__()

        # Initialize the warehouse from the existing warehouse class
        self.warehouse = warehouse

        # Define action and observation space
        # Action: Robot can either go to pick a shelf or pick up an item (depending on its state)
        self.action_space = spaces.Discrete(
            4)  # Idle, Go to Shelf, Go to Pickup Station, Return Shelf
        self.observation_space = spaces.Box(low=0,
                                            high=1,
                                            shape=(400, ),
                                            dtype=np.int32)  # Shelves' state

        # The maximum number of steps per episode (time steps)
        self.max_steps = 500
        self.current_step = 0

    def reset(self):
        """
        Resets the environment to the initial state.
        """
        self.warehouse.reset()
        self.current_step = 0
        return self._get_observation()

    def _get_observation(self):
        """
        Returns the current observation (state) of the warehouse
        For simplicity, it could be the number of items available in each shelf
        """
        shelf_counts = [len(shelf) for shelf in self.warehouse.shelfs
                        ]  # Number of items per shelf
        return np.array(shelf_counts, dtype=np.int32)

    def step(self, action):
        """
        Executes one step in the environment based on the action taken by the agent.
        """
        self.current_step += 1

        # Take action


        #ploting
        self.warehouse.shelf_plot('data/frames')

        # Simulate the warehouse process (order creation and robot work)
        self.warehouse.order_step()

        # Perform the robots' steps
        for robot in self.warehouse.robots:
            robot.step()

        self.warehouse.robot_assigner()

        # Check if the warehouse is out of stock
        done = self.warehouse.stock.sum(
        ) == 0 or self.current_step >= self.max_steps

        # Calculate the reward based on the completed orders and warehouse state
        reward = self._calculate_reward()

        # Get the next observation
        obs = self._get_observation()

        return obs, reward, done, {}

    def _calculate_reward(self):
        """
        Reward function. Can be based on factors like:
        - Number of completed orders
        - Time taken for tasks
        - Efficiency of robot movements
        """
        # Reward can be based on the number of orders completed within the time steps
        reward = len(self.warehouse.order_compleated
                     )  # Completed orders count as reward
        return reward

    def render(self, mode='human'):
        """
        Render the current state of the environment for visualization (useful for debugging)
        """
        self.warehouse.shelf_plot(f"frames/{self.current_step}")
        print(
            f"Step {self.current_step}: Total Orders Completed: {len(self.warehouse.order_compleated)}"
        )


def main():

    warehouse = Warehouse()
    env = WarehouseEnv(warehouse)  # Initialize the Gymnasium environment


    for episode in range(1):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = env.action_space.sample()  # Random action
            next_obs, reward, done, _ = env.step(action)
            total_reward += reward

        print(f"Episode {episode + 1} finished with total reward: {total_reward}")





if __name__ == "__main__":
    ...
    # main()

main()
