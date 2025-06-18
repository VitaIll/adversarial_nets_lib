# structural_gnn_lib/estimator/estimator.py

import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from ..generator.generator import GroundTruthGenerator
from ..utils.utils import objective_function
import time


class AdversarialEstimator:
    """
    Adversarial estimator for structural parameters using GNN discriminator.
    """
    
    def __init__(self, x, y, adjacency, initial_params=None, bounds=None, optimizer=None):
        """
        Initialize the adversarial estimator.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Node features matrix (n × k)
        y : numpy.ndarray
            Node outcomes matrix (n × l)
        adjacency : numpy.ndarray
            Adjacency matrix (n × n)
        initial_params : list or numpy.ndarray, optional
            Initial parameter values
        bounds : list of tuples, optional
            Parameter bounds for optimization
        optimizer : callable, optional
            Custom optimizer function
        """
        # Create ground truth generator
        node_indices = list(range(x.shape[0]))
        self.ground_truth_generator = GroundTruthGenerator(x, y, adjacency, node_indices)
        
        # Set initial parameters and bounds
        self.initial_params = initial_params if initial_params is not None else [0.0, 1.0]
        self.bounds = bounds if bounds is not None else [(-5.0, 5.0), (-5.0, 5.0)]
        
        # Store results
        self.result = None
        self.estimated_params = None
        self.optimization_history = []
        
        # Set optimizer
        self.optimizer = optimizer if optimizer is not None else self._default_optimizer
    
    def _default_optimizer(self, objective, space, verbose=True):
        """Default optimizer using Gaussian Process minimization."""
        
        # Callback to track optimization progress
        def gp_callback(res):
            self.optimization_history.append({
                'iteration': len(self.optimization_history),
                'params': res.x_iters[-1],
                'objective': res.func_vals[-1],
                'best_params': res.x,
                'best_objective': res.fun
            })
            if verbose and len(self.optimization_history) % 10 == 0:
                print(f"Iteration {len(self.optimization_history)}: "
                      f"Best objective = {res.fun:.4f}, "
                      f"Best params = {res.x}")
        
        # Safe objective wrapper to handle exceptions
        def safe_objective(params):
            try:
                return objective(params)
            except Exception as e:
                print(f"Error in objective evaluation: {e}")
                return 1.0  # Return worst possible accuracy
        
        result = gp_minimize(
            safe_objective,
            space,
            n_calls=500,                # Total evaluations
            n_initial_points=400,       # Initial random evaluations
            noise=0.2,                  # Explicitly model noise
            acq_func='EI',              # Expected Improvement acquisition function
            callback=gp_callback,
            random_state=42,
            n_jobs=-1,                  # Parallel execution
            verbose=verbose
        )
        
        return result
    
    def estimate(self, m=100, num_epochs=10, verbose=True):
        """
        Run adversarial estimation to find optimal parameters.
        
        Parameters:
        -----------
        m : int
            Number of nodes to sample for subgraphs
        num_epochs : int
            Number of epochs to train discriminator
        verbose : bool
            Whether to print progress information
        
        Returns:
        --------
        numpy.ndarray
            Estimated parameters
        """
        if verbose:
            print(f"Starting adversarial estimation with m={m} sampled nodes...")
            print(f"Initial parameters: {self.initial_params}")
            print(f"Parameter bounds: {self.bounds}")
        
        # Create search space
        space = [Real(low, high) for low, high in self.bounds]
        
        # Define objective function wrapper
        def wrapped_objective(params):
            return objective_function(
                params, 
                self.ground_truth_generator, 
                m, 
                num_epochs=num_epochs,
                verbose=False  # Suppress per-iteration output
            )
        
        # Run optimization
        start_time = time.time()
        self.result = self.optimizer(wrapped_objective, space, verbose)
        end_time = time.time()
        
        # Extract results
        self.estimated_params = self.result.x
        
        if verbose:
            print(f"\nOptimization completed in {end_time - start_time:.2f} seconds")
            print(f"Estimated parameters: {self.estimated_params}")
            print(f"Final objective value: {self.result.fun:.4f}")
        
        return self.estimated_params
    
    def get_optimization_history(self):
        """Get the optimization history."""
        return self.optimization_history