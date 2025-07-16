"""Demo runner for warehouse simulation."""

from typing import TYPE_CHECKING, Optional
from .config import SimulationConfig

if TYPE_CHECKING:
    from warehouse_sim.warehouse import Warehouse
    from environment.warehouse_env import WarehouseEnv


class DemoRunner:
    """Handles warehouse simulation demos."""
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
    
    def run_rware_demo(self) -> None:
        """Run RWare-style warehouse demonstration."""
        from utils.random_utils import set_seed
        
        set_seed(self.config.default_seed)
        print("RWare-style Warehouse Simulation Demo")
        print("=" * 50)

        available_layouts = self.config.available_layouts or ["medium"]
        
        for i, layout_size in enumerate(available_layouts):
            self._run_single_layout_demo(layout_size, i + 1, len(available_layouts))
        
        self._display_final_summary()
        input("Press Enter to exit...")
    
    def _run_single_layout_demo(self, layout_size: str, demo_num: int, total: int) -> None:
        """Run demo for single layout."""
        from warehouse_sim.warehouse import Warehouse
        from environment.warehouse_env import WarehouseEnv
        from environment.metrics_display import MetricsDisplay
        
        display = MetricsDisplay()
        display.display_demo_header(layout_size, demo_num, total)
        
        # Create warehouse and environment
        warehouse = Warehouse(seed=self.config.default_seed + demo_num - 1,
                            buffer_enabled=self.config.buffer_enabled,
                            layout_size=layout_size,
                            column_height=self.config.column_height,
                            difficulty=self.config.difficulty)
        
        display.display_warehouse_info(warehouse, layout_size)
        
        env = WarehouseEnv(warehouse, self.config.reward_strategy,
                          self.config.observation_type, self.config.max_steps)
        
        obs, info = env.reset(seed=self.config.default_seed + demo_num - 1)
        
        print("Running simulation...")
        total_reward = self._run_simulation_loop(env)
        
        display.display_completion(total_reward, warehouse.average_delay())
        self._final_visualization(env, layout_size)
    
    def _run_simulation_loop(self, env: 'WarehouseEnv') -> float:
        """Run simulation loop."""
        total_reward = 0
        
        for step in range(self.config.demo_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step % self.config.render_frequency == 0:
                self._render_step(env, step)
            
            if terminated or truncated:
                break
        
        return total_reward
    
    def _render_step(self, env: 'WarehouseEnv', step: int) -> None:
        """Render simulation step."""
        try:
            print(f"\n--- Step {step} ---")
            
            from warehouse_sim.renderer import debug_render_warehouse
            debug_render_warehouse(env.warehouse)
            
            try:
                env.render(pause_time=self.config.pause_time)
            except Exception as e:
                print(f"Visual render skipped: {e}")
                
        except Exception as e:
            print(f"Rendering issue: {e}")
    
    def _final_visualization(self, env: 'WarehouseEnv', layout_size: str) -> None:
        """Final visualization."""
        try:
            print(f"Final state visualization for {layout_size} layout...")
            env.render(pause_time=0.1)
        except Exception as e:
            print(f"Final rendering error: {e}")
    
    def _display_final_summary(self) -> None:
        """Display final summary."""
        from environment.metrics_display import MetricsDisplay
        MetricsDisplay().display_final_summary()
