"""
Main entry point for the warehouse simulation.

This module provides the main function to run the warehouse optimization simulation
with reinforcement learning capabilities.
"""

from .warehouse import Warehouse
from .environment import WarehouseEnv


def main():
    """
    Main function to run the warehouse simulation.
    """
    warehouse = Warehouse()
    env = WarehouseEnv(warehouse)  # Initialize the Gymnasium environment

    for episode in range(1):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = env.action_space.sample()  # Random action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Episode {episode + 1} finished with total reward: {total_reward}")


if __name__ == "__main__":
    main()
