"""
Performance monitoring and profiling utilities.
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    total_time: float = 0.0
    step_times: List[float] = field(default_factory=list)
    render_times: List[float] = field(default_factory=list)
    warehouse_step_times: List[float] = field(default_factory=list)
    reward_calculation_times: List[float] = field(default_factory=list)
    observation_times: List[float] = field(default_factory=list)
    
    @property
    def avg_step_time(self) -> float:
        """Average step time."""
        return sum(self.step_times) / len(self.step_times) if self.step_times else 0.0
    
    @property
    def avg_render_time(self) -> float:
        """Average render time."""
        return sum(self.render_times) / len(self.render_times) if self.render_times else 0.0
    
    @property
    def steps_per_second(self) -> float:
        """Steps per second."""
        return 1.0 / self.avg_step_time if self.avg_step_time > 0 else 0.0


class PerformanceMonitor:
    """
    Monitors and tracks performance metrics during simulation.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize performance monitor.
        
        Args:
            enabled: Whether monitoring is enabled
        """
        self.enabled = enabled
        self.metrics = PerformanceMetrics()
        self._timers: Dict[str, float] = {}
        self._start_time: Optional[float] = None
    
    def start_total_timer(self) -> None:
        """Start the total simulation timer."""
        if self.enabled:
            self._start_time = time.time()
    
    def stop_total_timer(self) -> None:
        """Stop the total simulation timer."""
        if self.enabled and self._start_time is not None:
            self.metrics.total_time = time.time() - self._start_time
    
    def start_timer(self, name: str) -> None:
        """
        Start a named timer.
        
        Args:
            name: Name of the timer
        """
        if self.enabled:
            self._timers[name] = time.time()
    
    def stop_timer(self, name: str) -> Optional[float]:
        """
        Stop a named timer and record the elapsed time.
        
        Args:
            name: Name of the timer
            
        Returns:
            Elapsed time in seconds, or None if monitoring disabled
        """
        if not self.enabled or name not in self._timers:
            return None
        
        elapsed = time.time() - self._timers[name]
        del self._timers[name]
        
        # Record time in appropriate metric list
        if name == "step":
            self.metrics.step_times.append(elapsed)
        elif name == "render":
            self.metrics.render_times.append(elapsed)
        elif name == "warehouse_step":
            self.metrics.warehouse_step_times.append(elapsed)
        elif name == "reward_calculation":
            self.metrics.reward_calculation_times.append(elapsed)
        elif name == "observation":
            self.metrics.observation_times.append(elapsed)
        
        return elapsed
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.enabled:
            return {"monitoring": "disabled"}
        
        return {
            "total_time": f"{self.metrics.total_time:.2f}s",
            "total_steps": len(self.metrics.step_times),
            "avg_step_time": f"{self.metrics.avg_step_time:.4f}s",
            "avg_render_time": f"{self.metrics.avg_render_time:.4f}s",
            "steps_per_second": f"{self.metrics.steps_per_second:.1f}",
            "total_renders": len(self.metrics.render_times),
        }
    
    def print_summary(self) -> None:
        """Print a formatted performance summary."""
        if not self.enabled:
            print("Performance monitoring is disabled.")
            return
        
        summary = self.get_summary()
        print("\n" + "=" * 50)
        print("PERFORMANCE SUMMARY")
        print("=" * 50)
        for key, value in summary.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print("=" * 50)


class TimerContext:
    """Context manager for timing code blocks."""
    
    def __init__(self, monitor: PerformanceMonitor, timer_name: str):
        """
        Initialize timer context.
        
        Args:
            monitor: Performance monitor instance
            timer_name: Name of the timer
        """
        self.monitor = monitor
        self.timer_name = timer_name
    
    def __enter__(self):
        """Start the timer."""
        self.monitor.start_timer(self.timer_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the timer."""
        self.monitor.stop_timer(self.timer_name)
