"""
Tests for the WarehouseEnv class
"""
import pytest
import numpy as np
from warehouse_sim.warehouse import Warehouse
from warehouse_sim.environment import WarehouseEnv


def test_warehouse_env_initialization():
    """Test WarehouseEnv initialization"""
    warehouse = Warehouse()
    env = WarehouseEnv(warehouse)
    
    assert env.warehouse is warehouse
    assert env.action_space.n == 4
    assert env.observation_space.shape == (400,)
    assert env.max_steps == 500
    assert env.current_step == 0


def test_warehouse_env_reset():
    """Test environment reset functionality"""
    warehouse = Warehouse()
    env = WarehouseEnv(warehouse)
    
    # Modify environment state
    env.current_step = 100
    
    # Reset environment
    obs, info = env.reset()
    
    assert env.current_step == 0
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (400,)
    assert isinstance(info, dict)


def test_warehouse_env_step():
    """Test environment step functionality"""
    warehouse = Warehouse()
    env = WarehouseEnv(warehouse)
    
    # Reset environment
    obs, info = env.reset()
    
    # Take a step
    action = 0  # Idle action
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    assert isinstance(next_obs, np.ndarray)
    assert next_obs.shape == (400,)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert env.current_step == 1


def test_warehouse_env_observation():
    """Test observation generation"""
    warehouse = Warehouse()
    env = WarehouseEnv(warehouse)
    
    obs = env._get_observation()
    
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (400,)
    assert obs.dtype == np.int32
    # Each shelf should have 6 or fewer items
    assert all(0 <= count <= 6 for count in obs)


def test_warehouse_env_reward_calculation():
    """Test reward calculation"""
    warehouse = Warehouse()
    env = WarehouseEnv(warehouse)
    
    # Initially no completed orders
    reward = env._calculate_reward()
    assert reward == 0