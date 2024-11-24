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

    def __init__(self, warehouse):
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
        self.max_steps = 200
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

        # Take action based on the robot's current state
        for robot in self.warehouse.robots:
            if robot.available:
                if action == 0:  # Idle - Do nothing
                    pass
                elif action == 1:  # Go to Shelf
                    # Assign a shelf to the robot to go to
                    shelf = random.choice(self.warehouse.itemShelfsBufferSet)
                    robot.assigne(shelf, self.warehouse.distance[shelf],
                                  self.warehouse.shelfs[shelf])
                elif action == 2:  # Go to Pickup Station
                    robot.step()
                elif action == 3:  # Return shelf
                    robot.step()

        # Simulate the warehouse process (order creation and robot work)
        self.warehouse.order_step()

        # Perform the robots' steps
        for robot in self.warehouse.robots:
            robot.step()

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
    itemBuffer = warehouse.itemBuffer
    shelfs = warehouse.shelfs
    probabilities = warehouse.probabilities
    available = warehouse.available

    # while True:
    for t in range(200):

        warehouse.shelf_plot('data/frames')

        warehouse.order_step()

        for robot in warehouse.robots:
            robot.step()

        if warehouse.stock.sum() == 0:
            print(t)
            break

        warehouse.robot_assigner()

        t += 1


def expectedTime():
    total_time = 0
    count = 0
    total_prob = 0
    for i in itemShelfsBufferSet:
        dist = distance[i]
        shelf = shelfs[i]

        for item in shelf:
            prob = probabilities[item]
            time = prob * dist  # we are assuming unit speed.
            count += 1
            total_prob += prob
            total_time += time
            # print(dist)
    return total_time / total_prob


if __name__ == "__main__":
    ...
    # main()

main()
