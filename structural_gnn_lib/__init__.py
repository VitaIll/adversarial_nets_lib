"""Structural GNN Library for Adversarial Estimation on Graphs"""

from .generator.generator import (
    GeneratorBase,
    GroundTruthGenerator,
    SyntheticGenerator,
    linear_in_means_model
)
from .discriminator.discriminator import GraphDiscriminator
from .estimator.estimator import AdversarialEstimator
from .utils.utils import (
    create_dataset,
    evaluate_discriminator,
    objective_function
)

__version__ = "0.1.0"
__all__ = [
    "GeneratorBase",
    "GroundTruthGenerator", 
    "SyntheticGenerator",
    "linear_in_means_model",
    "GraphDiscriminator",
    "AdversarialEstimator",
    "create_dataset",
    "evaluate_discriminator",
    "objective_function"
]