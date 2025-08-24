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
            metric="neg_logloss",
        ):
        """
        Initialize the adversarial estimator.
        
        Parameters
        ----------
            ground_truth_data : object
            Data object containing attributes ``X``, ``Y``, ``A``, ``N`` and
            optionally an initial outcome state ``Y0``.
        structural_model : callable
            Function implementing the structural mapping

            ``structural_model(X, P, Y0, theta) -> Y'``
        initial_params : array-like
            Initial parameter values
        bounds : list
            Bounds for parameters used by the optimizer
        discriminator_factory : callable
            Callable returning a discriminator model given ``input_dim``
        gp_params : dict, optional
            Additional parameters passed to ``gp_minimize``
        metric : str, optional
            Evaluation metric for the discriminator. Passed to
            :func:`objective_function`.
        """
        self.ground_truth_generator = GroundTruthGenerator(
            ground_truth_data.X,
            ground_truth_data.Y,
            ground_truth_data.A,
            ground_truth_data.N
        )
        
        self.synthetic_generator = SyntheticGenerator(
            self.ground_truth_generator,
            structural_model,
            initial_outcomes=getattr(ground_truth_data, "Y0", None),
        )
        
        self.initial_params = initial_params
        self.bounds = bounds
        self.discriminator_factory = discriminator_factory
        self.gp_params = gp_params or {}
        self.metric = metric
        
    def estimate(
            self,
            m,
            num_epochs=20,
            k_hops=1,
            verbose=True,
            discriminator_params=None,
            training_params=None,
        ):
        """Run the adversarial estimation.

        Parameters
        ----------
        m : int
            Number of nodes to sample for subgraphs.
        num_epochs : int, optional
            Number of epochs to train the discriminator.
        k_hops : int, optional
            Radius of the ego network sampled around each target node.
        verbose : bool, optional
            Whether to print progress information.
        discriminator_params : dict, optional
            Additional keyword arguments forwarded to ``discriminator_factory``.
        training_params : dict, optional
            Keyword arguments forwarded to :func:`objective_function` to
            control the training routine (e.g. ``batch_size`` or ``lr``).
        """

        training_params = training_params or {}

        def objective_with_generator(theta):
            return objective_function(
                theta,
                self.ground_truth_generator,
                self.synthetic_generator,
                m=m,
                num_epochs=num_epochs,
                k_hops=k_hops,
                discriminator_factory=self.discriminator_factory,
                discriminator_params=discriminator_params,
                verbose=verbose,
                metric=self.metric,
                **training_params,
            )
        
        gp_options = {
            'n_calls': 150,
            'n_initial_points': 70,
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
            **gp_options
        )
        
        return result
