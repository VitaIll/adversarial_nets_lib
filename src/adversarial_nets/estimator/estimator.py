from skopt import gp_minimize
import optuna
import random
from tqdm.auto import tqdm
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
        self.calibrated_params = None
        self.calibration_study = None

    def estimate(
            self,
            m=None,
            num_epochs=None,
            k_hops=None,
            verbose=True,
            discriminator_params=None,
            training_params=None,
        ):
        """Run the adversarial estimation.

        Parameters
        ----------
        m : int, optional
            Number of nodes to sample for subgraphs. If ``None`` and
            calibration has been performed, the calibrated value is used.
        num_epochs : int, optional
            Number of epochs to train the discriminator. Falls back to the
            calibrated value or ``20`` if unspecified.
        k_hops : int, optional
            Radius of the ego network sampled around each target node. Falls
            back to the calibrated value or ``1`` if unspecified.
        verbose : bool, optional
            Whether to print progress information during discriminator
            training.
        discriminator_params : dict, optional
            Additional keyword arguments forwarded to ``discriminator_factory``.
        training_params : dict, optional
            Keyword arguments forwarded to :func:`objective_function` to
            control the training routine (e.g. ``batch_size`` or ``lr``).
        """

        discriminator_params = discriminator_params or {}
        training_params = training_params or {}

        if self.calibrated_params:
            calib_disc = self.calibrated_params.get("discriminator_params", {})
            calib_train = self.calibrated_params.get("training_params", {}).copy()
        else:
            calib_disc, calib_train = {}, {}

        m = calib_train.pop("m", m)
        num_epochs = calib_train.pop("num_epochs", num_epochs)
        k_hops = calib_train.pop("k_hops", k_hops)

        if m is None:
            raise ValueError(
                "Parameter m must be specified either directly or via calibrated params."
            )
        if num_epochs is None:
            num_epochs = 20
        if k_hops is None:
            k_hops = 1

        discriminator_params = {**calib_disc, **discriminator_params}
        training_params = {**calib_train, **training_params}

        print(
            "Starting estimation with parameters:\n"
            f"m={m}, num_epochs={num_epochs}, k_hops={k_hops}\n"
            f"discriminator_params={discriminator_params}\n"
            f"training_params={training_params}"
        )

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
                **training_params
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

        total_calls = gp_options.get('n_calls', 0)
        pbar = tqdm(total=total_calls, desc="Estimating") if verbose else None

        def _callback(res):
            if pbar is not None:
                pbar.update(1)

        result = gp_minimize(
            objective_with_generator,
            self.bounds,
            callback=[_callback],
            **gp_options
        )

        if pbar is not None:
            pbar.close()

        return result

    def callibrate(
            self,
            search_space,
            optimizer_params,
            metric_name,
            k=10,
            m=1500,
            num_epochs=5,
            k_hops=1,
            discriminator_verbose=False,
            direction="minimize",
        ):
        """Calibrate discriminator hyperparameters using Optuna.

        Parameters
        ----------
        search_space : dict
            Dictionary defining Optuna search space. Expected keys
            ``"discriminator_params"`` and ``"training_params"`` each mapping
            parameter names to callables ``lambda trial: ...``.
        optimizer_params : dict
            Parameters for :func:`optuna.create_study`. May include
            ``n_trials`` to specify the number of optimization trials.
        metric_name : str
            Calibration metric to optimize (passed to
            :func:`evaluate_discriminator`).
        k : int, optional
            Number of randomly drawn ``theta`` values per trial.
        m, num_epochs, k_hops : int, optional
            Arguments controlling subgraph sampling and discriminator
            training. They can be overridden by sampled training parameters.
        discriminator_verbose : bool, optional
            Whether to print discriminator training progress.
        direction : {"minimize", "maximize"}, optional
            Direction of optimization passed to :func:`optuna.create_study`.
        """

        n_trials = optimizer_params.pop("n_trials", 50)
        study = optuna.create_study(direction=direction, **optimizer_params)
        pbar = tqdm(total=n_trials * k, desc="Calibrating")

        def objective(trial):
            disc_search = search_space.get("discriminator_params", {})
            train_search = search_space.get("training_params", {})

            disc_params = {name: sampler(trial) for name, sampler in disc_search.items()}
            train_params = {name: sampler(trial) for name, sampler in train_search.items()}

            m_trial = train_params.pop("m", m)
            num_epochs_trial = train_params.pop("num_epochs", num_epochs)
            k_hops_trial = train_params.pop("k_hops", k_hops)

            performances = []
            for _ in range(k):
                theta = [random.uniform(low, high) for (low, high) in self.bounds]
                perf = objective_function(
                    theta,
                    self.ground_truth_generator,
                    self.synthetic_generator,
                    discriminator_factory=self.discriminator_factory,
                    m=m_trial,
                    num_epochs=num_epochs_trial,
                    k_hops=k_hops_trial,
                    verbose=discriminator_verbose,
                    metric=metric_name,
                    discriminator_params=disc_params,
                    **train_params
                )
                performances.append(perf)
                pbar.update(1)
            return float(sum(performances) / len(performances))

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        pbar.close()

        best_params = study.best_params
        disc_keys = search_space.get("discriminator_params", {}).keys()
        train_keys = search_space.get("training_params", {}).keys()

        self.calibrated_params = {
            "discriminator_params": {k: best_params[k] for k in disc_keys if k in best_params},
            "training_params": {k: best_params[k] for k in train_keys if k in best_params},
        }
        self.calibration_study = study

        return None
