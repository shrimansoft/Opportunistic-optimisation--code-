"""Configuration validation utilities."""

from typing import List
from .config import SimulationConfig


class ConfigValidator:
    """Validates simulation configuration parameters."""
    
    VALID_STRATEGIES = ["completed_orders", "efficiency", "combined"]
    VALID_OBSERVATIONS = ["shelf_counts", "grid_state", "robot_positions"]
    VALID_LAYOUTS = ["tiny", "small", "medium", "large"]
    VALID_DIFFICULTIES = ["easy", "normal", "hard"]
    
    @classmethod
    def validate(cls, config: SimulationConfig) -> List[str]:
        """Validate configuration and return error list."""
        errors = []
        
        if config.reward_strategy not in cls.VALID_STRATEGIES:
            errors.append(f"Invalid reward_strategy: {config.reward_strategy}")
        
        if config.observation_type not in cls.VALID_OBSERVATIONS:
            errors.append(f"Invalid observation_type: {config.observation_type}")
        
        if config.difficulty not in cls.VALID_DIFFICULTIES:
            errors.append(f"Invalid difficulty: {config.difficulty}")
        
        if config.available_layouts:
            invalid = [l for l in config.available_layouts if l not in cls.VALID_LAYOUTS]
            if invalid:
                errors.append(f"Invalid layouts: {invalid}")
        
        for attr, value in [("max_steps", config.max_steps), ("demo_steps", config.demo_steps), 
                           ("render_frequency", config.render_frequency), ("column_height", config.column_height)]:
            if value <= 0:
                errors.append(f"{attr} must be positive")
        
        if config.pause_time < 0:
            errors.append("pause_time cannot be negative")
        
        return errors
    
    @classmethod
    def validate_and_raise(cls, config: SimulationConfig) -> None:
        """Validate and raise ValueError if invalid."""
        errors = cls.validate(config)
        if errors:
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")
