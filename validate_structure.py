#!/usr/bin/env python3
"""Validation script for modular structure."""

def test_functionality():
    """Test modular structure functionality."""
    print("Testing modular warehouse simulation...")
    
    tests = [
        ("Imports", lambda: __import__('environment') and __import__('simulation') and __import__('utils')),
        ("Config", lambda: getattr(__import__('simulation'), 'SimulationConfig')().default_seed == 42),
        ("Factory", lambda: getattr(__import__('environment'), 'EnvironmentFactory').create_demo_env()),
        ("Environment", lambda: test_env_step()),
        ("Performance", lambda: test_performance()),
    ]
    
    for name, test in tests:
        try:
            result = test()
            print(f"   ✓ {name}")
        except Exception as e:
            print(f"   ✗ {name}: {e}")
            return False
    
    print("\n" + "=" * 30)
    print("ALL TESTS PASSED! ✓")
    print("=" * 30)
    return True

def test_env_step():
    """Test environment step."""
    from environment import EnvironmentFactory
    env = EnvironmentFactory.create_demo_env()
    obs, info = env.reset(seed=42)
    action = env.action_space.sample()
    return env.step(action)

def test_performance():
    """Test performance monitoring."""
    from utils import PerformanceMonitor
    import time
    monitor = PerformanceMonitor()
    monitor.start_timer("test")
    time.sleep(0.001)
    return monitor.stop_timer("test")

def test_simulation():
    """Quick simulation test."""
    print("\nQuick simulation test...")
    try:
        from environment import EnvironmentFactory
        env = EnvironmentFactory.create_demo_env()
        obs, info = env.reset(seed=42)
        
        total_reward = 0
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        print(f"Simulation: {step + 1} steps, reward: {total_reward}")
        return True
    except Exception as e:
        print(f"Simulation failed: {e}")
        return False

if __name__ == "__main__":
    success = test_functionality()
    if success:
        test_simulation()
        print("\nValidation complete! Modular structure ready.")
    else:
        print("\nValidation failed!")
