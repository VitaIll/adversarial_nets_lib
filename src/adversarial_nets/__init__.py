"""Structural GNN Library for Adversarial Estimation on Graphs."""

import warnings

warnings.filterwarnings(
    "ignore", message="An issue occurred while importing 'torch-sparse'"
)
warnings.filterwarnings(
    "ignore", message="An issue occurred while importing 'torch-cluster'"
)

from .generator.generator import (
    GeneratorBase,
    GroundTruthGenerator,
    SyntheticGenerator,
)
from .data import GraphDataset
from .estimator.estimator import AdversarialEstimator
from .utils.utils import (
    create_dataset,
    evaluate_discriminator,
    objective_function,
)

__version__ = "0.1.1"
__all__ = [
    "GeneratorBase",
    "GroundTruthGenerator",
    "SyntheticGenerator",
    "AdversarialEstimator",
    "create_dataset",
    "evaluate_discriminator",
    "objective_function",
    "GraphDataset",
]

