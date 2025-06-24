# Adversarial estimation on graphs 

Adversarial estimator for graph structural models extends theoretical framework proposed by Kaji et al. (2023) to structural models defined on graphs (strategic communication games, peer effect models etc.). With graph data, we face unique challenges, in tabular datasets arbitrary row is usually regarded a single realization from joint distribution of exogenous and outcome variables, with graph data the graph itself is essentially a single random realization hence we need to create variability that would allow us to train the discriminative classifier, our current approach is to sample subgraphs from the ground truth and synthetic dataset and label them according to their origin to create necessary variability required for adversarial strategy, intuitive justification of this strategy would be something akin of ergodic theorem for signal transmission on networks, i.e. if individual generates a signal transmitting to his peers, given sufficient number of steps from the point of origin the effects will eventually dissipate. Currently implemented experiments thus rely on $k$-hop ego sampling from the ground truth and synthetic data. Further challenges are posed, by multiple equilibria of dynamic network models, lack of closed form asymptotics and the necessity to optimize discriminator architecture to suit different classes of structural models.

Below I briefly formalize the estimation problem:

For ground truth dataset graph dataset $G = (X,Y,N,A)$ where:
- X is a matrix of $n \times k$ exogenous characteristics of individual nodes, i.e. each node is associated with $k$ dimensional vector of features
- Y is a matrix of $n \times l$ endogenous outcomes of individual nodes, i.e. each node is associated with $l$ dimensional vector of outcomes
- N = \{0,...,n\} is set of node indices
- A $n\times n$ is an adjacency matrix, symmetric and $A \in \{0,1\} ^{n\times n}$

Structural model $m_{\theta}: R^{n \times k } \to R^{n \times l }$, $m$ is parametrized by unknown vector $\theta$.

Synthetic dataset $G(\theta)' = (X,Y',N,A)$ where $Y'=m_{\theta}(X,A, \theta)$

GNN discriminator $D: g_i \to [0,1]$, $g_i$ is a subgraph sampled from $G$ or $G'$. The discriminator is essentially a binary classifier which predicts if given sampled example belongs to ground truth or synthetic data.

We search for $\theta^*$ such that:
```math
  \theta^* \in \arg \min_{\theta}\max_{D}\mathop{{}\mathbb{E}}L_D(G'(\theta),G)
```
where the loss $L_D$ is some classification quality metric we want to minimize induced by the optimal classifier $D^*$, evaluated on the test set (e.g. accuracy or negative cross-entropy).

## Reference:
Kaji, T., Manresa, E., & Pouliot, G. (2023). An adversarial approach to structural estimation. Econometrica, 91(6), 2041-2063.

## Practical implementation

To build and train GNN discriminator I use components from PyTorch Geometric module for deep learning on graphs. Generator is implemented with base class unified interface for sampling, both, ground truth and synthetic data are handled by the generator. Generator for ground truth data is essentially a sampling manager, while generator for synthetic data requires in addition mapping function defining the structural model, and instance of ground truth generator to ensure that exogenous characteristic of synthetic data are an exact copy of those in ground truth dataset. The synthetic generator also implements a generate_outcomes method to produce counterfactuals outcome values based on supplied structural parameters. Util functions are mostly for encapsulating the discriminator training and testing into the outside minimization objective. Default minimization method is Bayesian optimization with surrogate models, since it attacks complex black-box objectives without the need for analytical derivative and combines benefits of global search with benefits of local refinement.

## linear_in_means_model.ipynb
Is a test notebook showcasing the estimation on 2-parameter case where objective and optimization progress can be visualized.

## Notes
- As of now utils are specific to the discriminator use in linear in means experiment, but should be generalized.
- Architecture of GNN for the experiment is chosen ad hoc since the identification is strong.
- Linear experiment uses accuracy as a minimization objective, for more complex models more sensitive metrics are necessary.