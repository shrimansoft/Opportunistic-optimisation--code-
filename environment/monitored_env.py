"""
Enhanced warehouse environment with performance monitoring and logging.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from environment.warehouse_env import WarehouseEnv
from utils.performance_monitor import PerformanceMonitor, TimerContext


class MonitoredWarehouseEnv(WarehouseEnv):
    """
    Enhanced warehouse environment with built-in performance monitoring.
    Wraps the base WarehouseEnv with additional monitoring capabilities.
    """
    
    def __init__(self, 
                 *args,
                 enable_monitoring: bool = True,
                 log_frequency: int = 100,
                 **kwargs):
        """
        Initialize monitored warehouse environment.
        
        Args:
            *args: Arguments for base WarehouseEnv
            enable_monitoring: Whether to enable performance monitoring
            log_frequency: How often to log performance metrics (in steps)
            **kwargs: Keyword arguments for base WarehouseEnv
        """
        super().__init__(*args, **kwargs)
        
        self.monitor = PerformanceMonitor(enabled=enable_monitoring)
        self.log_frequency = log_frequency
        self._episode_rewards: list = []
        self._episode_lengths: list = []
        self._current_episode_reward = 0.0
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment with monitoring.
        
        Args:
            seed: Random seed
            options: Reset options
            
        Returns:
            Tuple of (observation, info)
        """
        # Record episode statistics
        if self.current_step > 0:
            self._episode_rewards.append(self._current_episode_reward)
            self._episode_lengths.append(self.current_step)
        
        self._current_episode_reward = 0.0
        
        with TimerContext(self.monitor, "reset"):
            return super().reset(seed, options)
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step environment with monitoring.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        with TimerContext(self.monitor, "step"):
            # Time warehouse step
            with TimerContext(self.monitor, "warehouse_step"):
                self._execute_warehouse_step()
            
            # Time reward calculation
            with TimerContext(self.monitor, "reward_calculation"):
                reward = self.reward_calculator.calculate_reward(self.warehouse)
            
            # Time observation generation
            with TimerContext(self.monitor, "observation"):
                obs = self.observation_handler.get_observation(self.warehouse)
            
            # Update step counter and episode reward
            self.current_step += 1
            self._current_episode_reward += reward
            
            # Determine episode termination
            terminated = self._is_terminated()
            truncated = self._is_truncated()
            
            # Enhanced info with monitoring data
            info = self._get_enhanced_step_info()
            
            # Log performance periodically
            if self.current_step % self.log_frequency == 0:
                self._log_performance()
            
            return obs, reward, terminated, truncated, info
    
    def render(self, mode="human", pause_time=0.05) -> None:
        """
        Render with timing.
        
        Args:
            mode: Render mode
            pause_time: Pause time
        """
        with TimerContext(self.monitor, "render"):
            super().render(mode, pause_time)
    
    def _get_enhanced_step_info(self) -> Dict[str, Any]:
        """
        Get enhanced step information including monitoring data.
        
        Returns:
            Dictionary with step and performance information
        """
        base_info = super()._get_step_info()
        
        # Add monitoring information
        base_info.update({
            'performance': self.monitor.get_summary(),
            'episode_reward': self._current_episode_reward,
            'avg_episode_reward': np.mean(self._episode_rewards) if self._episode_rewards else 0.0,
            'avg_episode_length': np.mean(self._episode_lengths) if self._episode_lengths else 0.0,
        })
        
        return base_info
    
    def _log_performance(self) -> None:
        """Log current performance metrics."""
        if self.monitor.enabled:
            summary = self.monitor.get_summary()
            print(f"\n[Performance] Step {self.current_step}: "
                  f"SPS={summary.get('steps_per_second', 'N/A')}, "
                  f"Avg Step={summary.get('avg_step_time', 'N/A')}, "
                  f"Episode Reward={self._current_episode_reward:.2f}")
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive episode statistics.
        
        Returns:
            Dictionary with episode statistics
        """
        return {
            'total_episodes': len(self._episode_rewards),
            'episode_rewards': self._episode_rewards.copy(),
            'episode_lengths': self._episode_lengths.copy(),
            'avg_reward': np.mean(self._episode_rewards) if self._episode_rewards else 0.0,
            'std_reward': np.std(self._episode_rewards) if self._episode_rewards else 0.0,
            'avg_length': np.mean(self._episode_lengths) if self._episode_lengths else 0.0,
            'std_length': np.std(self._episode_lengths) if self._episode_lengths else 0.0,
            'performance': self.monitor.get_summary(),
        }
    
    def print_final_statistics(self) -> None:
        """Print comprehensive final statistics."""
        stats = self.get_episode_statistics()
        
        print("\n" + "=" * 60)
        print("FINAL EPISODE STATISTICS")
        print("=" * 60)
        print(f"Total Episodes: {stats['total_episodes']}")
        print(f"Average Reward: {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"Average Length: {stats['avg_length']:.1f} ± {stats['std_length']:.1f}")
        
        if stats['episode_rewards']:
            print(f"Best Episode Reward: {max(stats['episode_rewards']):.2f}")
            print(f"Worst Episode Reward: {min(stats['episode_rewards']):.2f}")
        
        # Print performance summary
        self.monitor.print_summary()
