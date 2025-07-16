"""
Random utilities for seed management and reproducibility.
"""

import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducible results.
    
    Args:
        seed: The random seed to set
    """
    random.seed(seed)
    np.random.seed(seed)
