# Advanced Modular Usage Guide

This guide demonstrates how to use the enhanced modular warehouse simulation system.

## Quick Start

### 1. Basic Usage (Original Functionality)
```python
# Run the original demo
python main.py
```

### 2. Advanced Features Demo
```python
# Run the advanced demo showcasing all new features
python advanced_demo.py
```

### 3. Validate Structure
```python
# Validate that everything is working
python validate_structure.py
```

## Advanced Usage Examples

### Environment Factory Pattern

```python
from environment import EnvironmentFactory

# Create different types of environments
basic_env = EnvironmentFactory.create_basic_env(layout_size="medium")
research_env = EnvironmentFactory.create_research_env(
    layout_size="large", 
    reward_strategy="combined",
    observation_type="grid_state"
)
demo_env = EnvironmentFactory.create_demo_env()

# Custom environment
custom_env = EnvironmentFactory.create_custom_env(
    warehouse_params={
        "layout_size": "small",
        "difficulty": "hard",
        "seed": 123
    },
    env_params={
        "reward_strategy": "efficiency",
        "observation_type": "robot_positions",
        "max_steps": 2000
    }
)
```

### Configuration Management

```python
from simulation import SimulationConfig, ConfigValidator

# Use predefined configurations
quick_config = SimulationConfig.quick_demo()
comprehensive_config = SimulationConfig.comprehensive_demo()
default_config = SimulationConfig.default()

# Create custom configuration
custom_config = SimulationConfig(
    demo_steps=200,
    render_frequency=10,
    reward_strategy="combined",
    observation_type="grid_state",
    available_layouts=["medium", "large"]
)

# Validate configuration
try:
    ConfigValidator.validate_and_raise(custom_config)
    print("Configuration is valid!")
except ValueError as e:
    print(f"Configuration error: {e}")
```

### Performance Monitoring

```python
from environment import MonitoredWarehouseEnv, EnvironmentFactory
from utils import PerformanceMonitor

# Create monitored environment
warehouse = EnvironmentFactory.create_basic_env().warehouse
monitored_env = MonitoredWarehouseEnv(
    warehouse=warehouse,
    enable_monitoring=True,
    log_frequency=50  # Log every 50 steps
)

# Run simulation with monitoring
obs, info = monitored_env.reset()
for step in range(100):
    action = monitored_env.action_space.sample()
    obs, reward, terminated, truncated, info = monitored_env.step(action)
    
    if terminated or truncated:
        break

# Get comprehensive statistics
monitored_env.print_final_statistics()
stats = monitored_env.get_episode_statistics()
print(f"Average reward: {stats['avg_reward']}")
```

### Custom Demo Runner

```python
from simulation import DemoRunner, SimulationConfig

class CustomDemoRunner(DemoRunner):
    def run_my_custom_demo(self):
        """Custom demo implementation."""
        config = SimulationConfig(
            demo_steps=50,
            available_layouts=["small"],
            reward_strategy="efficiency"
        )
        
        # Your custom demo logic here
        print("Running my custom demo...")
        
        # Use the factory to create environments
        from environment import EnvironmentFactory
        env = EnvironmentFactory.create_from_config(config, "small", 42)
        
        # Run simulation
        obs, info = env.reset()
        for step in range(config.demo_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step % 10 == 0:
                print(f"Step {step}, Reward: {reward}")
            
            if terminated or truncated:
                break

# Use custom demo runner
runner = CustomDemoRunner()
runner.run_my_custom_demo()
```

### Individual Component Usage

```python
# Use individual components
from environment import RewardCalculator, ObservationHandler, MetricsDisplay
from warehouse_sim.warehouse import Warehouse

# Create warehouse
warehouse = Warehouse(layout_size="medium", seed=42)

# Use modular components
reward_calc = RewardCalculator("combined")
obs_handler = ObservationHandler("grid_state")
metrics = MetricsDisplay()

# Calculate reward
reward = reward_calc.calculate_reward(warehouse)

# Get observation
observation = obs_handler.get_observation(warehouse)

# Display metrics
metrics.display_step_metrics(warehouse, 1)
metrics.display_robot_status(warehouse)
```

### Seed Management

```python
from utils import set_seed

# Set global seed for reproducibility
set_seed(42)

# This ensures consistent results across runs
```

### Constants and Enums

```python
from utils import RobotMode, SimulationDefaults

# Use robot mode constants
if robot.mode == RobotMode.IDLE:
    print("Robot is idle")

# Use simulation defaults
max_steps = SimulationDefaults.MAX_STEPS
default_pause = SimulationDefaults.DEFAULT_PAUSE_TIME
```

## Configuration Options

### SimulationConfig Parameters

```python
SimulationConfig(
    # Seed configuration
    default_seed=42,
    
    # Environment configuration
    max_steps=10000,
    reward_strategy="completed_orders",  # "completed_orders", "efficiency", "combined"
    observation_type="shelf_counts",     # "shelf_counts", "grid_state", "robot_positions"
    
    # Warehouse configuration
    buffer_enabled=True,
    column_height=8,
    difficulty="normal",                 # "easy", "normal", "hard"
    
    # Demo configuration
    demo_steps=500,
    render_frequency=1,
    pause_time=0.05,
    
    # Layout configurations
    available_layouts=["tiny", "small", "medium", "large"]
)
```

## Architecture Benefits

1. **Modularity**: Each component has a single responsibility
2. **Configurability**: Easy to customize without code changes
3. **Extensibility**: Easy to add new features
4. **Testability**: Components can be tested independently
5. **Performance**: Built-in monitoring and optimization
6. **Maintainability**: Clear separation of concerns

## Migration from Original Code

The new modular structure is 100% backward compatible. You can:

1. Continue using `python main.py` as before
2. Gradually adopt new features as needed
3. Extend functionality without modifying core code
4. Add custom configurations and demo scenarios

## Performance Considerations

- Use `MonitoredWarehouseEnv` for performance analysis
- Configure appropriate `render_frequency` for visual demos
- Use `SimulationConfig.quick_demo()` for fast testing
- Monitor memory usage with large numbers of episodes

## Troubleshooting

1. **Import Errors**: Ensure all modules are in the Python path
2. **Configuration Errors**: Use `ConfigValidator` to check parameters
3. **Performance Issues**: Enable monitoring to identify bottlenecks
4. **Memory Issues**: Monitor episode statistics and adjust accordingly
