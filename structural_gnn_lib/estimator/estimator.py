from skopt import gp_minimize
from ..generator.generator import GroundTruthGenerator, SyntheticGenerator
from ..utils.utils import objective_function

class AdversarialEstimator:
    def __init__(self, ground_truth_data, structural_model, initial_params, bounds, optimizer=None):
        """
        Initialize the adversarial estimator.
        
        Parameters:
        -----------
        ground_truth_data : dict
            Dictionary containing X, Y, A, N
        structural_model : callable
            Function that generates synthetic outcomes
        initial_params : array-like
            Initial parameter values
        optimizer : object, optional
            Optimizer to use (defaults to gp_minimize)
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
        self.optimizer = optimizer

        
    def estimate(self, m, num_epochs=20, n_calls=500, verbose=True):
        """
        Run the adversarial estimation.
        """

        def objective_with_generator(theta):
            return objective_function(
                theta,
                self.ground_truth_generator,
                self.synthetic_generator,  # Reuse the same instance
                m,
                num_epochs=num_epochs,
                verbose=verbose
            )


        result = gp_minimize(
            objective_with_generator,
            self.bounds,
            n_calls=n_calls,
            n_initial_points=int(0.3 * n_calls),
            noise=0.1,
            acq_func='EI',
            random_state=42,
            n_jobs=-1,
            verbose=verbose
        )
        
        return result