"""Advanced demo showcasing modular features."""

from simulation import DemoRunner, SimulationConfig, ConfigValidator
from environment import EnvironmentFactory, MonitoredWarehouseEnv


class AdvancedDemoRunner(DemoRunner):
    """Advanced demo runner showcasing enhanced features."""
    
    def __init__(self, config=None):
        super().__init__(config)
        ConfigValidator.validate_and_raise(self.config)
    
    def run_comprehensive_demo(self) -> None:
        """Run comprehensive demo."""
        print("Advanced Warehouse Simulation Demo")
        print("=" * 50)
        
        self._demo_environment_types()
        self._demo_performance_monitoring()
        self._demo_configuration_validation()
        
        print("\n" + "=" * 50)
        print("Demo Features Demonstrated:")
        print("✓ Multiple environment types")
        print("✓ Performance monitoring")
        print("✓ Configuration validation")
        print("✓ Factory patterns")
        print("=" * 50)
        
        input("\nPress Enter to exit...")
    
    def _demo_environment_types(self) -> None:
        """Demo different environment types."""
        print("\n--- Environment Types ---")
        
        envs = [
            ("Basic", EnvironmentFactory.create_basic_env()),
            ("Research", EnvironmentFactory.create_research_env()),
            ("Demo", EnvironmentFactory.create_demo_env())
        ]
        
        for name, env in envs:
            obs, info = env.reset()
            total_reward = 0
            
            for step in range(20):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            
            print(f"{name}: {step + 1} steps, reward: {total_reward:.2f}")
    
    def _demo_performance_monitoring(self) -> None:
        """Demo performance monitoring."""
        print("\n--- Performance Monitoring ---")
        
        warehouse = EnvironmentFactory.create_basic_env().warehouse
        env = MonitoredWarehouseEnv(warehouse=warehouse, log_frequency=25)
        
        obs, info = env.reset()
        for step in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        env.print_final_statistics()
    
    def _demo_configuration_validation(self) -> None:
        """Demo configuration validation."""
        print("\n--- Configuration Validation ---")
        
        # Valid config
        try:
            ConfigValidator.validate_and_raise(SimulationConfig())
            print("✓ Valid config passed")
        except ValueError:
            print("✗ Valid config failed")
        
        # Invalid config
        try:
            invalid = SimulationConfig(reward_strategy="invalid", max_steps=-1)
            ConfigValidator.validate_and_raise(invalid)
            print("✗ Invalid config should have failed")
        except ValueError:
            print("✓ Invalid config correctly rejected")


def run_advanced_demo():
    """Run the advanced demo."""
    config = SimulationConfig.comprehensive_demo()
    runner = AdvancedDemoRunner(config)
    runner.run_comprehensive_demo()


if __name__ == "__main__":
    run_advanced_demo()
