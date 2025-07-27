from skopt import gp_minimize

from ..generator.generator import GroundTruthGenerator, SyntheticGenerator
from ..utils.utils import objective_function

class AdversarialEstimator:
    def __init__(
        self,
        ground_truth_data,
        structural_model,
        initial_params,
        bounds,
        discriminator_factory,
        gp_params=None,
    ):
        """
        Initialize the adversarial estimator.
        
        Parameters
        ----------
        ground_truth_data : object
            Data object containing attributes X, Y, A, N
        structural_model : callable
            Function that generates synthetic outcomes
        initial_params : array-like
            Initial parameter values
        bounds : list
            Bounds for parameters used by the optimizer
        discriminator_factory : callable
            Callable returning a discriminator model given ``input_dim``
        gp_params : dict, optional
            Additional parameters passed to ``gp_minimize``
        """
        self.ground_truth_generator = GroundTruthGenerator(
            ground_truth_data.X,
            ground_truth_data.Y,
            ground_truth_data.A,
            ground_truth_data.N
        )
        
        self.synthetic_generator = SyntheticGenerator(
            self.ground_truth_generator, 
            structural_model
        )
        
        self.initial_params = initial_params
        self.bounds = bounds
        self.discriminator_factory = discriminator_factory
        self.gp_params = gp_params or {}

        
    def estimate(self, m, num_epochs=20, verbose=True):
        """Run the adversarial estimation."""

        def objective_with_generator(theta):
            return objective_function(
                theta,
                self.ground_truth_generator,
                self.synthetic_generator,
                self.discriminator_factory,
                m=m,
                num_epochs=num_epochs,
                verbose=verbose,
            )


        gp_options = {
            'n_calls': 50,
            'n_initial_points': 15,
            'noise': 0.1,
            'acq_func': 'EI',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': verbose,
        }
        gp_options.update(self.gp_params)

        result = gp_minimize(
            objective_with_generator,
            self.bounds,
            **gp_options,
        )
        
        return result
