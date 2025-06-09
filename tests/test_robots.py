"""
Tests for the Robot class
"""
import pytest
from warehouse_sim.warehouse import Warehouse
from warehouse_sim.robot import Robot


def test_robot_initialization():
    """Test robot initialization"""
    warehouse = Warehouse()
    robot = Robot(warehouse, 1)

    assert robot.robot_id == 1
    assert robot.available == True
    assert robot.mode == 0  # Should start in available mode
    assert robot.current_location == (0, 0)
    assert robot.shelf is None


def test_robot_assignment():
    """Test robot assignment to shelf"""
    warehouse = Warehouse()
    robot = Robot(warehouse, 1)

    # Assign robot to shelf
    robot.assigne(5, 10, (2, 3))

    assert robot.mode == 1  # Should be in "going to shelf" mode
    assert robot.shelf == 5
    assert robot.shelf_location == (2, 3)
    assert robot.target_location == (2, 3)
    assert robot.time_left == 10
    assert robot.available == False


def test_robot_step():
    """Test robot step movement"""
    warehouse = Warehouse()
    robot = Robot(warehouse, 1)

    # Set target location
    robot.target_location = (1, 1)
    initial_location = robot.current_location

    # Take a step
    robot.step()

    # Robot should move towards target
    new_location = robot.current_location
    assert new_location != initial_location or new_location == robot.target_location


def test_test():
    """Keep the existing simple test"""
    assert 5 == 5
