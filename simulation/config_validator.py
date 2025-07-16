"""
Configuration validation and type safety utilities.
"""

from typing import List, Optional
from .config import SimulationConfig


class ConfigValidator:
    """
    Validates simulation configuration parameters.
    Ensures configuration integrity and provides helpful error messages.
    """
    
    VALID_REWARD_STRATEGIES = ["completed_orders", "efficiency", "combined"]
    VALID_OBSERVATION_TYPES = ["shelf_counts", "grid_state", "robot_positions"]
    VALID_LAYOUT_SIZES = ["tiny", "small", "medium", "large"]
    VALID_DIFFICULTIES = ["easy", "normal", "hard"]
    
    @classmethod
    def validate_config(cls, config: SimulationConfig) -> List[str]:
        """
        Validate a simulation configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate reward strategy
        if config.reward_strategy not in cls.VALID_REWARD_STRATEGIES:
            errors.append(f"Invalid reward_strategy '{config.reward_strategy}'. "
                         f"Must be one of: {cls.VALID_REWARD_STRATEGIES}")
        
        # Validate observation type
        if config.observation_type not in cls.VALID_OBSERVATION_TYPES:
            errors.append(f"Invalid observation_type '{config.observation_type}'. "
                         f"Must be one of: {cls.VALID_OBSERVATION_TYPES}")
        
        # Validate difficulty
        if config.difficulty not in cls.VALID_DIFFICULTIES:
            errors.append(f"Invalid difficulty '{config.difficulty}'. "
                         f"Must be one of: {cls.VALID_DIFFICULTIES}")
        
        # Validate layout sizes
        if config.available_layouts:
            invalid_layouts = [layout for layout in config.available_layouts 
                             if layout not in cls.VALID_LAYOUT_SIZES]
            if invalid_layouts:
                errors.append(f"Invalid layout sizes: {invalid_layouts}. "
                             f"Must be from: {cls.VALID_LAYOUT_SIZES}")
        
        # Validate numeric ranges
        if config.max_steps <= 0:
            errors.append("max_steps must be positive")
        
        if config.demo_steps <= 0:
            errors.append("demo_steps must be positive")
        
        if config.render_frequency <= 0:
            errors.append("render_frequency must be positive")
        
        if config.pause_time < 0:
            errors.append("pause_time cannot be negative")
        
        if config.column_height <= 0:
            errors.append("column_height must be positive")
        
        return errors
    
    @classmethod
    def validate_and_raise(cls, config: SimulationConfig) -> None:
        """
        Validate configuration and raise exception if invalid.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        errors = cls.validate_config(config)
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))
