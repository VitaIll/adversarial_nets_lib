from dataclasses import dataclass
import numpy as np

@dataclass
class GraphDataset:
    """
    Container for graph structured data. 
    Optionally stores an initial outcome state ``Y0`` used by structural
    models that map baseline outcomes to current outcomes.
    """
    X: np.ndarray
    Y: np.ndarray
    A: np.ndarray
    N: list
    Y0: np.ndarray | None = None