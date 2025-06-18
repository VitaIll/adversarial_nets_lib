## Adversarial estimation on graphs 

Adversarial estimator for graph structural models 

For ground truth dataset graph dataset $G = (X,Y,N,A)$ where:
- X is a matrix of $n \times k$ exogenous characteristics of individual nodes, i.e. each node is asociated with $k$ dimensinal vector of features
- Y is a matrix of $n \times l$ endogenous outcomes of individual nodes, i.e. each node is asociated with $k$ dimensinal vector of outcomes
- N = \{0,...,n\} is set of node indicises
- A $n\times n$ is and adjacency matrix, symmetric and $A \in \{0,1\}^{n\times n}$

Structural model $m_{\theta}: R^{n \times k } \to R^{n \times l }$, $m$ is parametrized by uknown vector $\theta$.

Synthetic dataset $G(\theta)' = (X,Y',N,A)$ where $Y'=m_{\theta}(X,A, \theta)$

GNN discriminator $D: g_i \to [0,1]$, $g_i$ is a graph.

We search for $\theta*$ such that:
```math
  \theta* \in \arg \min_{\theta} \max_{D} L(G'(\theta),G)
```
where the loss $L$ is some classification quality metric we want to minimize (e.g. accuracy).


