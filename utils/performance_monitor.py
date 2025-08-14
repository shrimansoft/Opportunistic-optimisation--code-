"""Performance monitoring utilities."""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    total_time: float = 0.0
    step_times: List[float] = field(default_factory=list)
    render_times: List[float] = field(default_factory=list)
    
    @property
    def avg_step_time(self) -> float:
        return sum(self.step_times) / len(self.step_times) if self.step_times else 0.0
    
    @property
    def steps_per_second(self) -> float:
        return 1.0 / self.avg_step_time if self.avg_step_time > 0 else 0.0


class PerformanceMonitor:
    """Monitors performance metrics during simulation."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.metrics = PerformanceMetrics()
        self._timers: Dict[str, float] = {}
        self._start_time: Optional[float] = None
    
    def start_total_timer(self) -> None:
        if self.enabled:
            self._start_time = time.time()
    
    def stop_total_timer(self) -> None:
        if self.enabled and self._start_time:
            self.metrics.total_time = time.time() - self._start_time
    
    def start_timer(self, name: str) -> None:
        if self.enabled:
            self._timers[name] = time.time()
    
    def stop_timer(self, name: str) -> Optional[float]:
        if not self.enabled or name not in self._timers:
            return None
        
        elapsed = time.time() - self._timers[name]
        del self._timers[name]
        
        if name == "step":
            self.metrics.step_times.append(elapsed)
        elif name == "render":
            self.metrics.render_times.append(elapsed)
        
        return elapsed
    
    def get_summary(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"monitoring": "disabled"}
        
        return {
            "total_time": f"{self.metrics.total_time:.2f}s",
            "total_steps": len(self.metrics.step_times),
            "avg_step_time": f"{self.metrics.avg_step_time:.4f}s",
            "steps_per_second": f"{self.metrics.steps_per_second:.1f}",
        }
    
    def print_summary(self) -> None:
        if not self.enabled:
            print("Performance monitoring disabled.")
            return
        
        summary = self.get_summary()
        print("\nPerformance Summary:")
        for key, value in summary.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")


class TimerContext:
    """Context manager for timing code blocks."""
    
    def __init__(self, monitor: PerformanceMonitor, timer_name: str):
        self.monitor = monitor
        self.timer_name = timer_name
    
    def __enter__(self):
        self.monitor.start_timer(self.timer_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.stop_timer(self.timer_name)
