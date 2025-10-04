import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import random
import numpy as np
from sklearn.model_selection import train_test_split


def create_dataset(real_subgraphs, synthetic_subgraphs):
    """
    Create a dataset combining real and synthetic subgraphs with class labels.
    
    Parameters:
    -----------
    real_subgraphs : list
        List of PyTorch Geometric Data objects from the ground truth
    synthetic_subgraphs : list
        List of PyTorch Geometric Data objects from the synthetic simulator
    
    Returns:
    --------
    list
        Combined dataset with class labels (0 for real, 1 for synthetic)
    """
    dataset = []
    for data in real_subgraphs:
        data.label = torch.tensor(0, dtype=torch.long)
        dataset.append(data)
    for data in synthetic_subgraphs:
        data.label = torch.tensor(1, dtype=torch.long)
        dataset.append(data)
    return dataset


def evaluate_discriminator(model, loader, device, metric="neg_logloss"):
    """Evaluate the discriminator model according to a specified metric.

    Parameters
    ----------
    model : torch.nn.Module
        The GNN discriminator model
    loader : torch_geometric.data.DataLoader
        DataLoader containing evaluation data
    device : torch.device
        Device to run computations on
    metric : str
        Metric to compute. Supported values are ``"neg_logloss"``,
        ``"accuracy"``, ``"neg_brier_score"``, ``"ace"`` (average calibration
        error) and ``"ece"`` (expected calibration error).

    Returns
    -------
    float
        Evaluation value according to ``metric``. For ``"accuracy"`` the value
        is the standard accuracy score. For the other options the value keeps
        the adversarial convention used by the estimator: the returned metric is
        negated so that *more negative* values correspond to a less accurate
        discriminator. This allows the outer optimization routine, which
        minimizes the reported quantity, to seek parameters that lead to the
        worst discriminator performance when using cross-entropy-based losses.
    """

    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
    from sklearn.calibration import calibration_curve

    model.eval()
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            labels = batch.label
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if metric == "neg_logloss":
        value = -log_loss(y_true, y_prob, labels=[0, 1])
    elif metric == "accuracy":
        value = accuracy_score(y_true, y_pred)
    elif metric == "neg_brier_score":
        value = -brier_score_loss(y_true, y_prob)
    elif metric in {"ace", "ece"}:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
        bins = np.linspace(0.0, 1.0, 11)
        binids = np.digitize(y_prob, bins) - 1
        bin_total = np.bincount(binids, minlength=10)
        nonzero = bin_total > 0
        diff = np.abs(prob_true - prob_pred)
        if metric == "ace":
            value = diff.mean()
        else:
            weights = bin_total[nonzero] / len(y_prob)
            value = np.sum(diff * weights)
    else:
        raise ValueError(
            "metric must be one of 'neg_logloss', 'accuracy', 'neg_brier_score', 'ace', 'ece'",
        )

    return value

def objective_function(
    theta,
    ground_truth_generator,
    synthetic_generator,
    discriminator_factory,
    m=1500,
    num_epochs=5,
    k_hops=1,
    verbose=True,
    metric="neg_logloss",
    batch_size=256,
    lr=0.01,
    weight_decay=0.0,
    label_smoothing=0.0,
    discriminator_params=None,
    seeds=None,
    num_runs=1,
    training_verbose=None,
):
    """
    Objective function for parameter estimation.

    For candidate parameters ``theta``, generates synthetic outcomes, trains a
    GNN discriminator to distinguish between real and synthetic data, and
    evaluates the discriminator on a held-out split using the specified
    ``metric``.

    Parameters:
    -----------
    theta : list or numpy.ndarray
        Candidate parameters theta
    ground_truth_generator : GroundTruthGenerator
        The ground truth generator
    synthetic_generator : SyntheticGenerator
        The synthetic generator (reused across calls)
    discriminator_factory : callable
        Callable that returns a discriminator model given ``input_dim``.
    m : int
        Number of nodes to sample for subgraphs
    num_epochs : int
        Number of epochs to train the discriminator
    k_hops : int, optional
        Radius of the ego network sampled around each target node.
    verbose : bool
        Whether to print summary information after each training run.
    training_verbose : bool, optional
        Whether to print discriminator training progress for each epoch. If
        ``None`` (default) this mirrors the value of ``verbose``.
    metric : str
        Evaluation metric passed to ``evaluate_discriminator``. Supported
        values are ``"neg_logloss"``, ``"accuracy"`` and
        ``"neg_brier_score"``.
    batch_size : int, optional
        Batch size used by the ``DataLoader``.
    lr : float, optional
        Learning rate for the optimizer.
    weight_decay : float, optional
        Weight decay to apply in the optimizer.
    label_smoothing : float, optional
        Label smoothing factor used in the loss function.
    discriminator_params : dict, optional
        Additional keyword arguments forwarded to ``discriminator_factory``.
    seeds : sequence of int, optional
        Optional seeds ``(sampling, reserved, training)``. If ``None``
        (default), fresh seeds are drawn for each call ensuring different
        subsamples and discriminator initialization. The second entry is kept
        for backwards compatibility and is currently unused.
    num_runs : int, optional
        Number of independent training/evaluation repetitions using fresh
        samples. The final objective is the average over these runs.

    Returns:
    --------
    float
        Average value of ``metric`` on the test split across ``num_runs``
        different training/evaluation cycles (objective to minimize).
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if training_verbose is None:
        training_verbose = verbose

    sys_rng = random.SystemRandom()
    if seeds is not None:
        seeds = list(seeds) + [sys_rng.randrange(2**32) for _ in range(3 - len(seeds))]

    def single_run(run_seeds):
        synthetic_generator.generate_outcomes(theta)

        n = ground_truth_generator.num_nodes
        if m > n:
            raise ValueError(
                f"Requested {m} subgraphs but only {n} nodes are available."
            )
        k = min(m, n)

        if run_seeds is None:
            run_seeds = [sys_rng.randrange(2**32) for _ in range(3)]

        node_rng = random.Random(run_seeds[0])

        sampled_nodes = node_rng.sample(range(n), k)

        real_subgraphs = ground_truth_generator.sample_subgraphs(sampled_nodes, k_hops=k_hops)
        synthetic_subgraphs = synthetic_generator.sample_subgraphs(sampled_nodes, k_hops=k_hops)

        train_seed = run_seeds[2]
        torch.manual_seed(train_seed)
        np.random.seed(train_seed)

        dataset = create_dataset(real_subgraphs, synthetic_subgraphs)

        train_data, test_data = train_test_split(
            dataset, test_size=0.3, random_state=train_seed
        )
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        input_dim = real_subgraphs[0].x.shape[1]
        disc_params = discriminator_params or {}
        model = discriminator_factory(input_dim, **disc_params).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                outputs = model(batch)
                loss = criterion(outputs, batch.label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if training_verbose:
                print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")

        return evaluate_discriminator(model, test_loader, device, metric)

    results = []
    for i in range(num_runs):
        if seeds is None:
            run_seeds = None
        else:
            run_seeds = [(s + i) % (2**32) for s in seeds]
        test_objective = single_run(run_seeds)
        results.append(test_objective)
        if verbose:
            print(
                f"Run {i + 1}/{num_runs} test objective ({metric}) for theta={theta}: {test_objective:.4f}"
            )

    return float(sum(results) / len(results))
