from dataclasses import dataclass
import numpy as np

@dataclass
class GraphDataset:
    """Container for graph structured data."""
    X: np.ndarray
    Y: np.ndarray
    A: np.ndarray
    N: list

