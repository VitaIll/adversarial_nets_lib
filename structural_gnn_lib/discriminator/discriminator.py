# structural_gnn_lib/discriminator/discriminator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GraphDiscriminator(torch.nn.Module):
    """
    Lightweight Graph Neural Network discriminator for distinguishing real and synthetic graphs.
    Uses only a single convolutional layer to keep parameters under 100.
    """
    
    def __init__(self, input_dim, hidden_dim=16, num_classes=2):
        """
        Initialize the lightweight GNN discriminator.
        
        Parameters:
        -----------
        input_dim : int
            Dimension of input node features
        hidden_dim : int
            Dimension of hidden node representations (default=16 for <100 params)
        num_classes : int
            Number of output classes (2 for binary classification)
        """
        super(GraphDiscriminator, self).__init__()
        
        # Single GNN layer
        self.conv = GCNConv(input_dim, hidden_dim)
        
        # Simple linear classifier (no hidden layers to minimize parameters)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
   
    def forward(self, data):
        """
        Forward pass through the GNN.
        
        Parameters:
        -----------
        data : torch_geometric.data.Data
            Input graph data
        
        Returns:
        --------
        torch.Tensor
            Logits for each class
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Single graph convolution with ReLU activation
        x = F.relu(self.conv(x, edge_index))
        
        # Light dropout for regularization
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Global pooling to get graph-level representation
        x = global_mean_pool(x, batch)
        
        # Direct classification (no hidden layers)
        logits = self.classifier(x)
        
        return logits