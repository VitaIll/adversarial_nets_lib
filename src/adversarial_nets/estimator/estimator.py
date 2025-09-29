import random

import numpy as np
import optuna
from scipy.optimize import minimize
from skopt import gp_minimize
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
            outer_optimizer="gp",
            outer_optimizer_params=None,
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
            Additional parameters passed to ``gp_minimize``. Retained for
            backward compatibility with earlier versions of the API.
        metric : str, optional
            Evaluation metric for the discriminator. Passed to
            :func:`objective_function`.
        outer_optimizer : {"gp", "nelder-mead"}, optional
            Outer optimization routine. Defaults to Gaussian process based
            Bayesian optimization (``"gp"``). Set to ``"nelder-mead"`` to use
            SciPy's derivative-free simplex solver.
        outer_optimizer_params : dict, optional
            Additional keyword arguments forwarded to the selected outer
            optimizer. When ``outer_optimizer="gp"`` these correspond to
            parameters of :func:`skopt.gp_minimize`. When
            ``outer_optimizer="nelder-mead"`` they are passed to
            :func:`scipy.optimize.minimize`.
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
        self.metric = metric
        self.outer_optimizer = (outer_optimizer or "gp").lower()

        valid_optimizers = {"gp", "nelder-mead"}
        if self.outer_optimizer not in valid_optimizers:
            raise ValueError(
                f"Unsupported outer optimizer '{outer_optimizer}'. "
                f"Expected one of {sorted(valid_optimizers)}."
            )

        params_candidates = []
        if gp_params is not None:
            params_candidates.append(gp_params)
        if outer_optimizer_params is not None:
            params_candidates.append(outer_optimizer_params)

        if len(params_candidates) > 1:
            raise ValueError(
                "Specify either 'gp_params' or 'outer_optimizer_params', not both."
            )

        base_params = params_candidates[0] if params_candidates else {}
        base_dict = dict(base_params) if base_params else {}
        if self.outer_optimizer == "gp":
            # Keep gp_params alias for backwards compatibility. Mutations on
            # either attribute affect the same underlying dictionary.
            self.gp_params = base_dict
            self.outer_optimizer_params = self.gp_params
        else:
            self.outer_optimizer_params = base_dict
            self.gp_params = dict(gp_params or {})
        self.calibrated_params = None
        self.calibration_study = None

    def estimate(
            self,
            m=None,
            num_epochs=None,
            k_hops=None,
            num_runs=1,
            verbose=True,
            show_progress=True,
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
        num_runs : int, optional
            Number of independent training/evaluation repetitions per
            objective evaluation. Results are averaged across runs.
        verbose : bool, optional
            Whether to print summary information for each optimization step.
        show_progress : bool, optional
            Whether to display a progress bar for the outer optimization.
        discriminator_params : dict, optional
            Additional keyword arguments forwarded to ``discriminator_factory``.
        training_params : dict, optional
            Keyword arguments forwarded to :func:`objective_function` to
            control the training routine (e.g. ``batch_size``, ``lr``,
            ``weight_decay`` or ``label_smoothing``). Use
            ``training_verbose`` to re-enable per-epoch discriminator logs.
        """

        discriminator_params = discriminator_params or {}
        training_params = training_params or {}

        if self.calibrated_params:
            calib_disc = self.calibrated_params.get("discriminator_params", {})
            calib_train = self.calibrated_params.get("training_params", {}).copy()
        else:
            calib_disc, calib_train = {}, {}

        discriminator_params = {**calib_disc, **discriminator_params}
        training_params = {**calib_train, **training_params}

        m = training_params.pop("m", m)
        num_epochs = training_params.pop("num_epochs", num_epochs)
        k_hops = training_params.pop("k_hops", k_hops)
        num_runs = training_params.pop("num_runs", num_runs)

        if m is None:
            raise ValueError(
                "Parameter m must be specified either directly or via calibrated params."
            )
        if num_epochs is None:
            num_epochs = 20
        if k_hops is None:
            k_hops = 1
        if num_runs is None:
            num_runs = 1

        if verbose:
            print(
                "Starting estimation with parameters:\n"
                f"m={m}, num_epochs={num_epochs}, k_hops={k_hops}, num_runs={num_runs}\n"
                f"discriminator_params={discriminator_params}\n"
                f"training_params={training_params}"
            )

        training_verbose = training_params.pop("training_verbose", None)
        if training_verbose is None:
            training_verbose = False

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
                num_runs=num_runs,
                training_verbose=training_verbose,
                **training_params
            )

        if self.outer_optimizer == "gp":
            gp_options = {
                "n_calls": 150,
                "n_initial_points": 70,
                "noise": 0.1,
                "acq_func": "EI",
                "random_state": 42,
                "n_jobs": -1,
                "verbose": verbose,
            }

            gp_options.update(dict(self.outer_optimizer_params))

            total_calls = gp_options.get("n_calls", 0)
            pbar = tqdm(
                total=total_calls if total_calls else None,
                desc="Estimating",
                disable=not show_progress,
            )

            def _callback(res):
                if pbar is not None:
                    pbar.update(1)

            result = gp_minimize(
                objective_with_generator,
                self.bounds,
                callback=[_callback],
                **gp_options,
            )

            if pbar is not None:
                pbar.close()

            return result

        if self.outer_optimizer == "nelder-mead":
            nm_params = dict(self.outer_optimizer_params)

            options_user = nm_params.pop("options", {}) or {}
            x0 = nm_params.pop("x0", self.initial_params)
            if x0 is None:
                raise ValueError(
                    "Nelder-Mead optimization requires an initial parameter vector."
                )

            lower_bounds = upper_bounds = None
            if self.bounds:
                lower_bounds = np.array(
                    [(-np.inf if low is None else low) for (low, _) in self.bounds],
                    dtype=float,
                )
                upper_bounds = np.array(
                    [(np.inf if high is None else high) for (_, high) in self.bounds],
                    dtype=float,
                )

            x0 = np.asarray(x0, dtype=float)
            if lower_bounds is not None:
                x0 = np.clip(x0, lower_bounds, upper_bounds)

            default_options = {
                "maxiter": 200,
                "xatol": 1e-4,
                "fatol": 1e-4,
                "adaptive": True,
            }
            default_options.update(options_user)
            default_options.setdefault("disp", verbose)
            default_options["disp"] = bool(default_options["disp"])

            estimated_total = default_options.get("maxiter") or default_options.get("maxfev")
            pbar = tqdm(
                total=estimated_total,
                desc="Estimating",
                disable=not show_progress,
            )

            user_callback = nm_params.pop("callback", None)
            user_tol = nm_params.pop("tol", None)
            user_args = nm_params.pop("args", ())
            if nm_params.pop("bounds", None) is not None:
                # Bounds are handled manually through clipping. Ignore any provided
                # bounds to avoid SciPy raising an error for unsupported arguments.
                pass

            minimize_kwargs = nm_params
            if user_args:
                minimize_kwargs["args"] = user_args
            if user_tol is not None:
                minimize_kwargs["tol"] = user_tol

            def _clip(theta):
                theta = np.asarray(theta, dtype=float)
                if lower_bounds is not None:
                    theta = np.clip(theta, lower_bounds, upper_bounds)
                return theta

            def nm_objective(theta):
                clipped_theta = _clip(theta)
                return objective_with_generator(clipped_theta.tolist())

            def wrapped_callback(xk, *cb_args):
                if pbar is not None:
                    pbar.update(1)
                if user_callback is not None:
                    user_callback(_clip(xk), *cb_args)

            result = minimize(
                nm_objective,
                x0,
                method="Nelder-Mead",
                callback=wrapped_callback,
                options=default_options,
                **minimize_kwargs,
            )

            if pbar is not None:
                pbar.close()

            if lower_bounds is not None:
                result.x = _clip(result.x)

            return result

        raise ValueError(
            f"Unsupported outer optimizer '{self.outer_optimizer}'. "
            "Expected 'gp' or 'nelder-mead'."
        )

    def callibrate(
            self,
            search_space,
            optimizer_params,
            metric_name,
            k=10,
            m=1500,
            num_epochs=5,
            k_hops=1,
            num_runs=1,
            discriminator_verbose=False,
            direction="minimize",
        ):
        """Calibrate discriminator hyperparameters using Optuna.

        Parameters
        ----------
        search_space : dict
            Dictionary defining Optuna search space. Expected keys
            ``"discriminator_params"`` and ``"training_params"`` each mapping
            parameter names to callables ``lambda trial: ...``. Training
            parameters can include values such as ``lr``, ``batch_size``,
            ``weight_decay`` or ``label_smoothing``.
        optimizer_params : dict
            Parameters for :func:`optuna.create_study`. May include
            ``n_trials`` to specify the number of optimization trials.
        metric_name : str
            Calibration metric to optimize (passed to
            :func:`evaluate_discriminator`).
        k : int, optional
            Number of randomly drawn ``theta`` values per trial.
        m, num_epochs, k_hops, num_runs : int, optional
            Arguments controlling subgraph sampling, discriminator training
            and the number of repeated trainings/evaluations per objective
            computation. They can be overridden by sampled training
            parameters.
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
            num_runs_trial = train_params.pop("num_runs", num_runs)

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
                    num_runs=num_runs_trial,
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
