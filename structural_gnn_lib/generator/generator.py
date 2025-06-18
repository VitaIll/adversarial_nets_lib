import torch
import numpy as np
import networkx as nx
from abc import ABC
from torch_geometric.data import Data


class GeneratorBase(ABC):
    """Abstract base class for data generators."""
    
    def __init__(self, x, y, adjacency, node_indices):
        """
        Initialize the generator with graph data.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Node features matrix (n × k)
        y : numpy.ndarray
            Node outcomes matrix (n × l)
        adjacency : numpy.ndarray
            Adjacency matrix (n × n)
        node_indices : list
            List of node indices
        """
        self.x = x
        self.y = y
        self.adjacency = adjacency
        self.node_indices = node_indices
        self.num_nodes = len(node_indices)
        
        self.G = nx.from_numpy_array(adjacency)
    
    def sample_subgraphs(self, node_ids):
        """
        Extract induced subgraphs centered on specified nodes.
        
        For each node in node_ids, creates a subgraph containing the node and all
        its neighbors, with features and outcomes preserved from the original graph.
        
        Parameters:
        -----------
        node_ids : list
            List of node indices to sample subgraphs from
        
        Returns:
        --------
        list
            List of PyTorch Geometric Data objects representing subgraphs
        """
        subgraphs = []
        for node in node_ids:
           
            nodes = [node] + list(self.G.neighbors(node))
            subgraph = self.G.subgraph(nodes).copy()
            
            mapping = {n: i for i, n in enumerate(nodes)}
            subgraph = nx.relabel_nodes(subgraph, mapping)
            
            x_sub = torch.tensor(self.x[nodes], dtype=torch.float)
            y_sub = torch.tensor(self.y[nodes], dtype=torch.float)
            
            features = torch.cat([x_sub, y_sub], dim=1)
            
            edge_index = torch.tensor(list(subgraph.edges), dtype=torch.long).t().contiguous()
            if edge_index.numel() > 0:
                edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                
            data = Data(x=features, edge_index=edge_index, 
                        original_nodes=nodes,  
                        original_graph=subgraph)  
            subgraphs.append(data)
        return subgraphs


class GroundTruthGenerator(GeneratorBase):
    """Generator for ground truth data."""
    
    def __init__(self, x, y, adjacency, node_indices):
        """
        Initialize ground truth generator with real data.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Node features matrix (n × k)
        y : numpy.ndarray
            Node outcomes matrix (n × l)
        adjacency : numpy.ndarray
            Adjacency matrix (n × n)
        node_indices : list
            List of node indices
        """
        super().__init__(x, y, adjacency, node_indices)


class SyntheticGenerator(GeneratorBase):
    """Generator for synthetic data using structural models."""
    
    def __init__(self, ground_truth_generator, structural_model):
        """
        Initialize synthetic generator inheriting structure from ground truth.
        
        Parameters:
        -----------
        ground_truth_generator : GroundTruthGenerator
            Ground truth generator instance to inherit X, A, N from
        structural_model : callable
            Function that takes (X, A, theta) and returns synthetic Y
        """
     
        super().__init__(
            ground_truth_generator.x,
            ground_truth_generator.y,  # Will be overwritten by generate_outcomes
            ground_truth_generator.adjacency,
            ground_truth_generator.node_indices
        )
        self.structural_model = structural_model
    
    def generate_outcomes(self, theta):
        """
        Generate synthetic outcomes using the structural model.
        
        Parameters:
        -----------
        theta : numpy.ndarray or list
            Parameter vector for the structural model
        
        Returns:
        --------
        numpy.ndarray
            Generated outcomes Y' matrix
        """
        self.y = self.structural_model(self.x, self.adjacency, theta)
        return self.y



def linear_in_means_model(x, adjacency, theta):
    """
    Linear-in-means structural model: y_i = a + b * mean(x_j) for j in neighbors(i)
    
    Parameters:
    -----------
    x : numpy.ndarray
        Node features (n × k)
    adjacency : numpy.ndarray
        Adjacency matrix (n × n)
    theta : list or numpy.ndarray
        Parameters [a, b]
    
    Returns:
    --------
    numpy.ndarray
        Generated outcomes (n × 1)
    """
    a, b = theta
    n = x.shape[0]
    y = np.zeros((n, 1))
    
    G = nx.from_numpy_array(adjacency)
    
    for i in range(n):
        neighbors = list(G.neighbors(i))
        if neighbors:
            mean_neighbor_x = np.mean(x[neighbors])
        else:
            mean_neighbor_x = 0.0
        y[i] = a + b * mean_neighbor_x
    
    return y