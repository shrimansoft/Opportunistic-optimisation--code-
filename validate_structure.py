#!/usr/bin/env python3
"""
Validation script to test the modular structure.
"""

def test_basic_functionality():
    """Test basic functionality of the modular structure."""
    print("Testing modular warehouse simulation structure...")
    
    # Test 1: Import all modules
    print("1. Testing imports...")
    try:
        from environment import WarehouseEnv, EnvironmentFactory, MonitoredWarehouseEnv
        from simulation import DemoRunner, SimulationConfig, ConfigValidator
        from utils import set_seed, PerformanceMonitor
        print("   ✓ All imports successful")
    except ImportError as e:
        print(f"   ✗ Import failed: {e}")
        return False
    
    # Test 2: Configuration validation
    print("2. Testing configuration validation...")
    try:
        config = SimulationConfig.default()
        ConfigValidator.validate_and_raise(config)
        print("   ✓ Configuration validation successful")
    except Exception as e:
        print(f"   ✗ Configuration validation failed: {e}")
        return False
    
    # Test 3: Environment factory
    print("3. Testing environment factory...")
    try:
        env = EnvironmentFactory.create_demo_env(layout_size="medium")
        obs, info = env.reset(seed=42)
        print(f"   ✓ Environment created successfully, observation shape: {obs.shape}")
    except Exception as e:
        print(f"   ✗ Environment factory failed: {e}")
        return False
    
    # Test 4: Single environment step
    print("4. Testing environment step...")
    try:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   ✓ Environment step successful, reward: {reward}")
    except Exception as e:
        print(f"   ✗ Environment step failed: {e}")
        return False
    
    # Test 5: Performance monitoring
    print("5. Testing performance monitoring...")
    try:
        monitor = PerformanceMonitor(enabled=True)
        monitor.start_timer("test")
        import time
        time.sleep(0.001)  # Small delay
        elapsed = monitor.stop_timer("test")
        print(f"   ✓ Performance monitoring successful, elapsed: {elapsed:.4f}s")
    except Exception as e:
        print(f"   ✗ Performance monitoring failed: {e}")
        return False
    
    # Test 6: Demo runner
    print("6. Testing demo runner initialization...")
    try:
        quick_config = SimulationConfig.quick_demo()
        demo_runner = DemoRunner(quick_config)
        print("   ✓ Demo runner initialized successfully")
    except Exception as e:
        print(f"   ✗ Demo runner initialization failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED! ✓")
    print("The modular structure is working correctly.")
    print("=" * 50)
    return True


def test_quick_simulation():
    """Run a quick simulation to test end-to-end functionality."""
    print("\nRunning quick simulation test...")
    
    try:
        from environment import EnvironmentFactory
        from simulation import SimulationConfig
        
        # Create environment
        env = EnvironmentFactory.create_demo_env()
        obs, info = env.reset(seed=42)
        
        total_reward = 0
        for step in range(10):  # Very short test
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"Quick simulation completed: {step + 1} steps, total reward: {total_reward}")
        return True
        
    except Exception as e:
        print(f"Quick simulation failed: {e}")
        return False


if __name__ == "__main__":
    success = test_basic_functionality()
    
    if success:
        test_quick_simulation()
        print("\nValidation complete! The modular structure is ready to use.")
    else:
        print("\nValidation failed! Please check the errors above.")
