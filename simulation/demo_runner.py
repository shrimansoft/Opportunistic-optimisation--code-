"""
Demo runner for warehouse simulation.
Orchestrates the demo workflow and handles execution.
"""

from typing import TYPE_CHECKING, Optional

from .config import SimulationConfig

if TYPE_CHECKING:
    from warehouse_sim.warehouse import Warehouse
    from environment.warehouse_env import WarehouseEnv


class DemoRunner:
    """
    Handles the execution of warehouse simulation demos.
    Provides a clean interface for running different types of demonstrations.
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize the demo runner.
        
        Args:
            config: Configuration for the simulation demo
        """
        self.config = config or SimulationConfig.default()
    
    def run_rware_demo(self) -> None:
        """
        Run the main RWare-style warehouse demonstration.
        This is the primary demo function that showcases different warehouse layouts.
        """
        from utils.random_utils import set_seed
        
        set_seed(self.config.default_seed)
        
        print("RWare-style Warehouse Simulation Demo")
        print("=" * 50)

        available_layouts = self.config.available_layouts or ["medium"]
        total_layouts = len(available_layouts)
        
        for i, layout_size in enumerate(available_layouts):
            self._run_single_layout_demo(layout_size, i + 1, total_layouts)
        
        self._display_final_summary()
        
        input("Press Enter to exit...")
    
    def _run_single_layout_demo(self, layout_size: str, demo_num: int, total_demos: int) -> None:
        """
        Run a demo for a single warehouse layout.
        
        Args:
            layout_size: Size of the warehouse layout to demo
            demo_num: Current demo number
            total_demos: Total number of demos
        """
        from warehouse_sim.warehouse import Warehouse
        from environment.warehouse_env import WarehouseEnv
        from environment.metrics_display import MetricsDisplay
        
        metrics_display = MetricsDisplay()
        metrics_display.display_demo_header(layout_size, demo_num, total_demos)
        
        # Create warehouse with specified layout
        warehouse = self._create_warehouse(layout_size, demo_num - 1)
        
        # Display warehouse information
        metrics_display.display_warehouse_info(warehouse, layout_size)
        
        # Create and reset environment
        env = self._create_environment(warehouse)
        obs, info = env.reset(seed=self.config.default_seed + demo_num - 1)
        
        print(f"Running simulation...")
        total_reward = self._run_simulation_loop(env)
        
        # Display completion summary
        metrics_display.display_demo_completion(total_reward, warehouse.average_delay())
        
        # Final visualization
        self._final_visualization(env, layout_size)
    
    def _create_warehouse(self, layout_size: str, demo_index: int) -> 'Warehouse':
        """
        Create a warehouse instance with the specified configuration.
        
        Args:
            layout_size: Size of the warehouse layout
            demo_index: Index of the current demo
            
        Returns:
            Warehouse: Configured warehouse instance
        """
        from warehouse_sim.warehouse import Warehouse
        
        return Warehouse(
            seed=self.config.default_seed + demo_index,
            buffer_enabled=self.config.buffer_enabled,
            layout_size=layout_size,
            column_height=self.config.column_height,
            difficulty=self.config.difficulty
        )
    
    def _create_environment(self, warehouse: 'Warehouse') -> 'WarehouseEnv':
        """
        Create a warehouse environment with the specified configuration.
        
        Args:
            warehouse: The warehouse instance
            
        Returns:
            WarehouseEnv: Configured environment instance
        """
        from environment.warehouse_env import WarehouseEnv
        
        return WarehouseEnv(
            warehouse=warehouse,
            reward_strategy=self.config.reward_strategy,
            observation_type=self.config.observation_type,
            max_steps=self.config.max_steps
        )
    
    def _run_simulation_loop(self, env: 'WarehouseEnv') -> float:
        """
        Run the main simulation loop for the environment.
        
        Args:
            env: The warehouse environment
            
        Returns:
            float: Total reward accumulated
        """
        total_reward = 0
        
        for step in range(self.config.demo_steps):
            # Environment step (this handles all warehouse operations internally)
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Render at specified frequency
            if step % self.config.render_frequency == 0:
                self._render_step(env, step)
            
            if terminated or truncated:
                break
        
        return total_reward
    
    def _render_step(self, env: 'WarehouseEnv', step: int) -> None:
        """
        Render a single step of the simulation.
        
        Args:
            env: The warehouse environment
            step: Current step number
        """
        try:
            print(f"\n--- Step {step} ---")
            
            # Use debug renderer for clear console output
            from warehouse_sim.renderer import debug_render_warehouse
            debug_render_warehouse(env.warehouse)
            
            # Try visual render (might hang in some environments)
            try:
                env.render(pause_time=self.config.pause_time)
            except Exception as e:
                print(f"Visual render skipped: {e}")
                
        except Exception as e:
            print(f"Rendering issue: {e}")
    
    def _final_visualization(self, env: 'WarehouseEnv', layout_size: str) -> None:
        """
        Perform final visualization for the demo.
        
        Args:
            env: The warehouse environment
            layout_size: Size of the layout
        """
        try:
            print(f"Final state visualization for {layout_size} layout...")
            env.render(pause_time=0.1)
        except Exception as e:
            print(f"Final rendering error: {e}")
    
    def _display_final_summary(self) -> None:
        """Display the final summary of all demos."""
        from environment.metrics_display import MetricsDisplay
        
        metrics_display = MetricsDisplay()
        metrics_display.display_final_summary()
