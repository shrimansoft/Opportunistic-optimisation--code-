"""
Advanced demo runner showcasing the enhanced modular features.
"""

from typing import List, Dict, Any, Optional
from simulation.config import SimulationConfig
from simulation.config_validator import ConfigValidator
from simulation.demo_runner import DemoRunner
from environment import EnvironmentFactory, MonitoredWarehouseEnv
from utils import PerformanceMonitor


class AdvancedDemoRunner(DemoRunner):
    """
    Advanced demo runner that showcases enhanced features including:
    - Performance monitoring
    - Multiple environment types
    - Configuration validation
    - Detailed analytics
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize advanced demo runner.
        
        Args:
            config: Simulation configuration
        """
        super().__init__(config)
        
        # Validate configuration
        ConfigValidator.validate_and_raise(self.config)
        
        self.performance_data: Dict[str, Any] = {}
    
    def run_comprehensive_demo(self) -> None:
        """
        Run a comprehensive demo showcasing all features.
        """
        print("Advanced Warehouse Simulation Demo")
        print("=" * 60)
        print("Features: Performance Monitoring, Multiple Environments, Analytics")
        print("=" * 60)
        
        # Demo 1: Basic environment comparison
        self._demo_environment_types()
        
        # Demo 2: Performance monitoring showcase
        self._demo_performance_monitoring()
        
        # Demo 3: Configuration validation demo
        self._demo_configuration_validation()
        
        # Demo 4: Factory pattern demo
        self._demo_factory_patterns()
        
        # Final summary
        self._print_comprehensive_summary()
        
        input("\nPress Enter to exit...")
    
    def _demo_environment_types(self) -> None:
        """Demo different environment types."""
        print("\n" + "=" * 40)
        print("DEMO 1: Environment Types Comparison")
        print("=" * 40)
        
        env_types = [
            ("Basic Environment", "basic"),
            ("Research Environment", "research"),
            ("Demo Environment", "demo")
        ]
        
        for env_name, env_type in env_types:
            print(f"\n--- {env_name} ---")
            
            if env_type == "basic":
                env = EnvironmentFactory.create_basic_env()
            elif env_type == "research":
                env = EnvironmentFactory.create_research_env()
            else:
                env = EnvironmentFactory.create_demo_env()
            
            # Run short demo
            obs, info = env.reset()
            total_reward = 0
            
            for step in range(50):  # Short demo
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            print(f"Completed in {step + 1} steps with reward {total_reward:.2f}")
            print(f"Environment config: {env.reward_calculator.reward_strategy}, "
                  f"{env.observation_handler.observation_type}")
    
    def _demo_performance_monitoring(self) -> None:
        """Demo performance monitoring capabilities."""
        print("\n" + "=" * 40)
        print("DEMO 2: Performance Monitoring")
        print("=" * 40)
        
        # Create monitored environment
        warehouse = EnvironmentFactory.create_basic_env().warehouse
        monitored_env = MonitoredWarehouseEnv(
            warehouse=warehouse,
            enable_monitoring=True,
            log_frequency=25
        )
        
        print("Running simulation with performance monitoring...")
        
        obs, info = monitored_env.reset()
        
        for step in range(100):
            action = monitored_env.action_space.sample()
            obs, reward, terminated, truncated, info = monitored_env.step(action)
            
            if terminated or truncated:
                break
        
        # Print performance statistics
        monitored_env.print_final_statistics()
    
    def _demo_configuration_validation(self) -> None:
        """Demo configuration validation."""
        print("\n" + "=" * 40)
        print("DEMO 3: Configuration Validation")
        print("=" * 40)
        
        # Test valid configuration
        print("Testing valid configuration...")
        valid_config = SimulationConfig.default()
        try:
            ConfigValidator.validate_and_raise(valid_config)
            print("✓ Valid configuration passed validation")
        except ValueError as e:
            print(f"✗ Unexpected validation error: {e}")
        
        # Test invalid configuration
        print("\nTesting invalid configuration...")
        invalid_config = SimulationConfig(
            reward_strategy="invalid_strategy",
            observation_type="invalid_observation",
            max_steps=-100
        )
        
        try:
            ConfigValidator.validate_and_raise(invalid_config)
            print("✗ Invalid configuration should have failed")
        except ValueError as e:
            print("✓ Invalid configuration correctly rejected:")
            for line in str(e).split('\n'):
                if line.strip().startswith('-'):
                    print(f"  {line.strip()}")
    
    def _demo_factory_patterns(self) -> None:
        """Demo factory pattern usage."""
        print("\n" + "=" * 40)
        print("DEMO 4: Factory Pattern Usage")
        print("=" * 40)
        
        # Create environments using different factory methods
        factories = [
            ("Basic Factory", lambda: EnvironmentFactory.create_basic_env("small")),
            ("Research Factory", lambda: EnvironmentFactory.create_research_env("medium")),
            ("Config Factory", lambda: EnvironmentFactory.create_from_config(
                self.config, "medium", 42)),
        ]
        
        for factory_name, factory_func in factories:
            print(f"\n--- {factory_name} ---")
            env = factory_func()
            print(f"Created environment with:")
            print(f"  - Reward strategy: {env.reward_calculator.reward_strategy}")
            print(f"  - Observation type: {env.observation_handler.observation_type}")
            print(f"  - Max steps: {env.max_steps}")
            print(f"  - Warehouse layout: {env.warehouse.grid_size}")
    
    def _print_comprehensive_summary(self) -> None:
        """Print comprehensive demo summary."""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE DEMO SUMMARY")
        print("=" * 60)
        print("✓ Environment Types: Basic, Research, Demo environments tested")
        print("✓ Performance Monitoring: Timing and statistics collection")
        print("✓ Configuration Validation: Parameter validation and error handling")
        print("✓ Factory Patterns: Multiple creation patterns demonstrated")
        print("\nModular Architecture Benefits:")
        print("  • Separation of concerns")
        print("  • Easy configuration and customization")
        print("  • Performance monitoring and analytics")
        print("  • Robust error handling and validation")
        print("  • Extensible factory patterns")
        print("=" * 60)


def run_advanced_demo():
    """Convenience function to run the advanced demo."""
    config = SimulationConfig.comprehensive_demo()
    runner = AdvancedDemoRunner(config)
    runner.run_comprehensive_demo()


if __name__ == "__main__":
    run_advanced_demo()
