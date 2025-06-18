# structural_gnn_lib/utils/utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import numpy as np
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


def evaluate_discriminator(model, loader, device):
    """
    Evaluate the discriminator model.
    
    Parameters:
    -----------
    model : GraphDiscriminator
        The GNN discriminator model
    loader : torch_geometric.data.DataLoader
        DataLoader containing evaluation data
    device : torch.device
        Device to run computations on
    
    Returns:
    --------
    float
        Classification accuracy
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            _, predicted = torch.max(outputs, 1)
            total += batch.label.size(0)
            correct += (predicted == batch.label).sum().item()
    
    accuracy = correct / total
    return accuracy


def objective_function(theta, ground_truth_generator, m, num_epochs=10, verbose=False):
    """
    Objective function for parameter estimation.
    
    For candidate parameters theta, generates synthetic outcomes, trains a GNN 
    discriminator to distinguish between real and synthetic data, and returns
    the test accuracy (which we want to minimize).
    
    Parameters:
    -----------
    theta : list or numpy.ndarray
        Candidate parameters theta
    ground_truth_generator : GroundTruthGenerator
        The ground truth generator
    m : int
        Number of nodes to sample for subgraphs
    num_epochs : int
        Number of epochs to train the discriminator
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    float
        Test accuracy of the discriminator (objective to minimize)
    """
    from ..generator.generator import SyntheticGenerator, linear_in_means_model
    from ..discriminator.discriminator import GraphDiscriminator
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create synthetic generator with linear-in-means model
    synthetic_generator = SyntheticGenerator(ground_truth_generator, linear_in_means_model)
    
    # Generate synthetic outcomes
    synthetic_generator.generate_outcomes(theta)
    
    # Sample nodes
    n = ground_truth_generator.num_nodes
    sampled_nodes = random.sample(range(n), min(m, n))
    
    # Generate subgraphs
    real_subgraphs = ground_truth_generator.sample_subgraphs(sampled_nodes)
    synthetic_subgraphs = synthetic_generator.sample_subgraphs(sampled_nodes)
    
    # Create labeled dataset
    dataset = create_dataset(real_subgraphs, synthetic_subgraphs)
    
    # Train-test split
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    # Initialize discriminator
    input_dim = real_subgraphs[0].x.shape[1]  # x and y concatenated
    hidden_dim = 16  # Small hidden dimension to keep params < 100
    model = GraphDiscriminator(input_dim, hidden_dim).to(device)
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Higher LR for simpler model
    criterion = nn.CrossEntropyLoss()
    
    # Train discriminator
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            
            # Forward pass
            outputs = model(batch)
            loss = criterion(outputs, batch.label)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if verbose and epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")
    
    # Evaluate on test set
    test_accuracy = evaluate_discriminator(model, test_loader, device)
    
    if verbose:
        print(f"Test accuracy for theta={theta}: {test_accuracy:.4f}")
    
    return test_accuracy