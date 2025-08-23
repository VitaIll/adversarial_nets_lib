import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader 
import random
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
    """
    Evaluate the discriminator model according to a specified metric.

    Parameters:
    -----------
    model : torch.nn.Module
        The GNN discriminator model
    loader : torch_geometric.data.DataLoader
        DataLoader containing evaluation data
    device : torch.device
        Device to run computations on
    metric : str
        Metric to compute. Supported values are ``"neg_logloss"``,
        ``"accuracy"`` and ``"neg_brier_score"``.

    Returns:
    --------
    float
        Evaluation value according to ``metric``. For ``"accuracy"`` the
        value is the standard accuracy score. For the other options the value
        is negated so that lower values correspond to better discriminator
        performance; hence they can be directly minimized by the outer
        optimization routine.
    """

    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

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

    if metric == "neg_logloss":
        value = -log_loss(y_true, y_prob, labels=[0, 1])
    elif metric == "accuracy":
        value = accuracy_score(y_true, y_pred)
    elif metric == "neg_brier_score":
        value = -brier_score_loss(y_true, y_prob)
    else:
        raise ValueError(
            "metric must be one of 'neg_logloss', 'accuracy', 'neg_brier_score'"
        )

    return value


def objective_function(
    theta,
    ground_truth_generator,
    synthetic_generator,
    discriminator_factory,
    m=500,
    num_epochs=5,
    verbose=True,
    metric="neg_logloss",
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
    verbose : bool
        Whether to print progress information
    metric : str
        Evaluation metric passed to ``evaluate_discriminator``. Supported
        values are ``"neg_logloss"``, ``"accuracy"`` and
        ``"neg_brier_score"``.

    Returns:
    --------
    float
        Value of ``metric`` on the test split (objective to minimize).
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    synthetic_generator.generate_outcomes(theta)

    n = ground_truth_generator.num_nodes
    sampled_nodes = random.sample(range(n), min(m, n))

    real_subgraphs = ground_truth_generator.sample_subgraphs(sampled_nodes)
    synthetic_subgraphs = synthetic_generator.sample_subgraphs(sampled_nodes)

    dataset = create_dataset(real_subgraphs, synthetic_subgraphs)

    train_data, test_data = train_test_split(dataset, test_size=0.3, random_state=42)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    input_dim = real_subgraphs[0].x.shape[1]
    model = discriminator_factory(input_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

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
        if verbose:
            print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")

    test_objective = evaluate_discriminator(model, test_loader, device, metric)

    if verbose:
        print(f"Test objective ({metric}) for theta={theta}: {test_objective:.4f}")

    return test_objective
