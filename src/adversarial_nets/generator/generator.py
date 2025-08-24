import warnings

warnings.filterwarnings("ignore", message="An issue occurred while importing 'torch-sparse'")
warnings.filterwarnings("ignore", message="An issue occurred while importing 'torch-cluster'")

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
    
    def sample_subgraphs(self, node_ids, k_hops=1):
        """Extract ego subgraphs centered on specified nodes.

        For each node in ``node_ids`` an ego network with radius ``k_hops``
        is extracted from the original graph. Node features and outcomes are
        preserved from the original graph and combined into a
        :class:`~torch_geometric.data.Data` object.

        Parameters
        ----------
        node_ids : list
            List of node indices to sample subgraphs from.
        k_hops : int, optional
            Radius of the ego network around each ``node``. ``k_hops=1``
            reduces to the previous behaviour of including only immediate
            neighbours. Defaults to ``1``.

        Returns
        -------
        list
            List of PyTorch Geometric ``Data`` objects representing subgraphs.
        """
        subgraphs = []
        for node in node_ids:
            # Use networkx's ego_graph to obtain nodes within ``k_hops``
            subgraph = nx.ego_graph(self.G, node, radius=k_hops).copy()

            nodes = list(subgraph.nodes)
            mapping = {n: i for i, n in enumerate(nodes)}
            subgraph = nx.relabel_nodes(subgraph, mapping)

            x_sub = torch.tensor(self.x[nodes], dtype=torch.float)
            y_sub = torch.tensor(self.y[nodes], dtype=torch.float)

            if x_sub.dim() == 1:
                x_sub = x_sub.unsqueeze(1)
            if y_sub.dim() == 1:
                y_sub = y_sub.unsqueeze(1)

            features = torch.cat([x_sub, y_sub], dim=1)

            edge_index = (
                torch.tensor(list(subgraph.edges), dtype=torch.long).t().contiguous()
            )
            if edge_index.numel() > 0:
                edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            data = Data(
                x=features,
                edge_index=edge_index,
                original_nodes=nodes,
                original_graph=subgraph,
            )
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

    def __init__(self, ground_truth_generator, structural_model, initial_outcomes=None):
        """Initialize synthetic generator inheriting structure from ground truth.

        Parameters
        ----------
        ground_truth_generator : GroundTruthGenerator
            Ground truth generator instance to inherit ``X``, ``A``, ``N`` from.
        structural_model : callable
            Function implementing the structural mapping

            ``structural_model(X, P, Y0, theta) -> Y'``
        initial_outcomes : numpy.ndarray, optional
            Initial outcome state ``Y0`` used by the structural model. If
            ``None`` a zero matrix with the appropriate shape is used.
        """

        super().__init__(
            ground_truth_generator.x,
            ground_truth_generator.y,
            ground_truth_generator.adjacency,
            ground_truth_generator.node_indices,
        )

        self.structural_model = structural_model
        self.initial_outcomes = (
            initial_outcomes
            if initial_outcomes is not None
            else np.zeros_like(ground_truth_generator.y)
        )

        # Peer operator with zero diagonal as specified in the README
        self.peer_operator = self.adjacency - np.diag(np.diag(self.adjacency))

    def generate_outcomes(self, theta):
        """Generate synthetic outcomes using the structural model."""

        # Support structural models that either require an initial outcome
        # state ``Y0`` or ignore it. We inspect the callable signature and
        # pass the appropriate number of arguments accordingly.
        try:
            from inspect import signature

            n_params = len(signature(self.structural_model).parameters)
        except Exception:
            # Fallback: assume the full four-argument signature.
            n_params = 4

        if n_params == 4:
            args = (self.x, self.peer_operator, self.initial_outcomes, theta)
        elif n_params == 3:
            args = (self.x, self.peer_operator, theta)
        else:
            raise TypeError(
                "structural_model must accept either 3 or 4 positional arguments"
            )

        self.y = self.structural_model(*args)
        return self.y


