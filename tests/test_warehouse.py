"""
Tests for the Warehouse class
"""
import pytest
import numpy as np
from warehouse_sim.warehouse import Warehouse


def test_warehouse_initialization():
    """Test that warehouse initializes correctly"""
    warehouse = Warehouse()
    
    assert warehouse.time == 0
    assert len(warehouse.shelfs) == 400
    assert len(warehouse.robots) == 6
    assert len(warehouse.picking_stations) == 1
    assert warehouse.stock.sum() == 50 * 48  # 50 item types * 48 items each


def test_warehouse_sample():
    """Test item sampling from probability distribution"""
    warehouse = Warehouse()
    
    # Sample should return an integer between 0 and 49
    sample = warehouse.sample()
    assert isinstance(sample, int)
    assert 0 <= sample < 50


def test_warehouse_available():
    """Test available items check"""
    warehouse = Warehouse()
    
    available = warehouse.available()
    assert len(available) == 50
    assert all(available)  # Initially all items should be available


def test_warehouse_item_in_shelfs():
    """Test counting items in shelfs"""
    warehouse = Warehouse()
    
    # Test for item type 0
    item_counts = warehouse.itemInShelfs(0)
    assert len(item_counts) == 400
    assert sum(item_counts) == 48  # Should have 48 items of type 0 total


def test_warehouse_reset():
    """Test warehouse reset functionality"""
    warehouse = Warehouse()
    
    # Modify some state
    warehouse.time = 100
    
    # Reset
    warehouse.reset()
    
    # Check that time is reset
    assert warehouse.time == 0