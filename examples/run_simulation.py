#!/usr/bin/env python3
"""
Example script demonstrating how to run the warehouse simulation.

This script shows how to:
1. Initialize the warehouse environment
2. Run a simple simulation loop
3. Collect performance metrics
4. Generate visualizations
"""

import os
import sys
import numpy as np

# Add the parent directory to the path so we can import warehouse_sim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from warehouse_sim import Warehouse, WarehouseEnv


def run_basic_simulation(episodes=1, max_steps=100):
    """
    Run a basic warehouse simulation.
    
    Args:
        episodes (int): Number of episodes to run
        max_steps (int): Maximum steps per episode
    """
    print("Starting warehouse simulation...")
    
    # Initialize warehouse and environment
    warehouse = Warehouse()
    env = WarehouseEnv(warehouse)
    env.max_steps = max_steps
    
    # Track metrics
    total_rewards = []
    average_delays = []
    
    for episode in range(episodes):
        print(f"\n--- Episode {episode + 1}/{episodes} ---")
        
        # Reset environment
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Take random action (replace with your RL agent)
            action = env.action_space.sample()
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1
            
            # Print progress every 10 steps
            if step % 10 == 0:
                completed_orders = len(warehouse.order_compleated)
                pending_orders = len(warehouse.order_buffer)
                avg_delay = warehouse.average_delay()
                
                print(f"Step {step}: Completed={completed_orders}, "
                      f"Pending={pending_orders}, Avg Delay={avg_delay:.2f}")
        
        # Episode summary
        final_avg_delay = warehouse.average_delay()
        total_completed = len(warehouse.order_compleated)
        
        print(f"Episode {episode + 1} Summary:")
        print(f"  Total Reward: {episode_reward}")
        print(f"  Completed Orders: {total_completed}")
        print(f"  Average Delay: {final_avg_delay:.2f}")
        print(f"  Total Stock Remaining: {warehouse.stock.sum()}")
        
        total_rewards.append(episode_reward)
        average_delays.append(final_avg_delay)
    
    # Overall summary
    print(f"\n--- Simulation Complete ---")
    print(f"Episodes: {episodes}")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"Average Delay: {np.mean(average_delays):.2f}")
    
    return total_rewards, average_delays


def run_visualization_demo(steps=50):
    """
    Run a short simulation with visualization frames.
    
    Args:
        steps (int): Number of steps to run
    """
    print("Running visualization demo...")
    
    # Create frames directory
    frames_dir = "demo_frames"
    os.makedirs(frames_dir, exist_ok=True)
    
    # Initialize warehouse and environment
    warehouse = Warehouse()
    env = WarehouseEnv(warehouse)
    
    # Reset and run
    obs, info = env.reset()
    
    for step in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 5 == 0:
            print(f"Step {step}: Generated frame")
        
        if terminated or truncated:
            break
    
    print(f"Visualization frames saved to {frames_dir}/")
    print("To create a video, run:")
    print(f"ffmpeg -framerate 10 -i {frames_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p demo.mp4")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run warehouse simulation")
    parser.add_argument("--episodes", type=int, default=1, 
                       help="Number of episodes to run")
    parser.add_argument("--steps", type=int, default=100,
                       help="Maximum steps per episode")
    parser.add_argument("--demo", action="store_true",
                       help="Run visualization demo")
    
    args = parser.parse_args()
    
    if args.demo:
        run_visualization_demo(args.steps)
    else:
        run_basic_simulation(args.episodes, args.steps)