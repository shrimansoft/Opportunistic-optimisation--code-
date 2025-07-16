"""Enhanced warehouse environment with monitoring."""

import numpy as np
from typing import Tuple, Dict, Any
from environment.warehouse_env import WarehouseEnv
from utils.performance_monitor import PerformanceMonitor, TimerContext


class MonitoredWarehouseEnv(WarehouseEnv):
    """Warehouse environment with performance monitoring."""
    
    def __init__(self, *args, enable_monitoring: bool = True, log_frequency: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitor = PerformanceMonitor(enabled=enable_monitoring)
        self.log_frequency = log_frequency
        self._episode_rewards = []
        self._current_episode_reward = 0.0
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.current_step > 0:
            self._episode_rewards.append(self._current_episode_reward)
        self._current_episode_reward = 0.0
        
        with TimerContext(self.monitor, "reset"):
            return super().reset(seed, options)
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        with TimerContext(self.monitor, "step"):
            obs, reward, terminated, truncated, info = super().step(action)
            self._current_episode_reward += reward
            
            # Enhanced info with monitoring
            info.update({
                'performance': self.monitor.get_summary(),
                'episode_reward': self._current_episode_reward,
                'avg_episode_reward': np.mean(self._episode_rewards) if self._episode_rewards else 0.0,
            })
            
            if self.current_step % self.log_frequency == 0:
                print(f"[Performance] Step {self.current_step}: "
                      f"Episode Reward={self._current_episode_reward:.2f}")
            
            return obs, reward, terminated, truncated, info
    
    def render(self, mode="human", pause_time=0.05) -> None:
        with TimerContext(self.monitor, "render"):
            super().render(mode, pause_time)
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """Get episode statistics."""
        return {
            'total_episodes': len(self._episode_rewards),
            'avg_reward': np.mean(self._episode_rewards) if self._episode_rewards else 0.0,
            'performance': self.monitor.get_summary(),
        }
    
    def print_final_statistics(self) -> None:
        """Print final statistics."""
        stats = self.get_episode_statistics()
        print(f"\nEpisode Stats: {stats['total_episodes']} episodes, "
              f"avg reward: {stats['avg_reward']:.2f}")
        self.monitor.print_summary()
