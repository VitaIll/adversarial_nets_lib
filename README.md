# Adversarial estimation on graphs

This project adapts the adversarial estimator of Kaji, Manresa & Pouliot (2023) to structural models defined on graphs (e.g., strategic communication, peer effects). In tabular settings, each row is treated as an i.i.d. draw. With graph data the entire graph is typically a single realization, so we induce variability for the discriminator by sampling subgraphs from both the ground-truth and the synthetic graph and training a classifier to distinguish their origins. Intuitively, local neighborhoods carry enough information about global structure to power discrimination—consistent with common GNN practice (e.g., $k$-hop ego sampling) and graph-GAN work that discriminates on local walks/subgraphs. ([arXiv][1], [pytorch-geometric.readthedocs.io][2])

## Formal setup

**Observed graph.** Let $G=(X,Y,N,A)$ with:

* $X\in\mathbb{R}^{n\times k}$: exogenous node covariates (row $i$ is $x_i^\top$).
* $Y\in\mathbb{R}^{n\times \ell}$: endogenous node outcomes (row $i$ is $y_i^\top$).
* $N=\{1,\dots,n\}$: node index set.
* $A\in\{0,1\}^{n\times n}$: undirected adjacency, symmetric. We work with a **peer operator** $\,P(A)\,$ that **excludes self-links**:

  $$
  P(A)\;=\;A-\operatorname{diag}(A)\qquad\text{(no self-to-self influence)}
  $$

  Optionally $P(A)$ may be row/degree normalized; the formulation below only requires $P(A)$ to have zero diagonal.

**Structural model (generator).** A structural mapping

$$
m_\theta:\ \big(X,\,P(A),\,Y^{(0)},\,\xi\big)\ \longmapsto\ Y' \in \mathbb{R}^{n\times \ell}
$$

takes covariates, the peer operator, and an **initial outcome state** $Y^{(0)}$ (e.g., pre-interaction signals, baseline choices, or an initialization such as zeros) and returns simulated outcomes $Y'$. The innovation $\xi$ captures simulation randomness if present.

* **Single-step (peer-to-peer, no self-loop):**

  $$
  Y' \;=\; m_\theta\!\Big(X,\;P(A)\,Y^{(0)}\,,\;\xi\Big),
  $$

  i.e., each $y_i'$ depends on $x_i$ and **peers’ initial outcomes** $\{y_j^{(0)}: j\in \mathcal{N}(i)\}$, but not on $y_i^{(0)}$ directly.
* **Multi-step propagation (optional):** for $t=0,\dots,T-1$,

  $$
  Y^{(t+1)} \;=\; m_\theta\!\Big(X,\;P(A)\,Y^{(t)}\,,\;\xi^{(t)}\Big),\qquad Y'\equiv Y^{(T)}.
  $$

  This allows $k$-hop effects to accrue while maintaining zero diagonal in $P(A)$.

**Synthetic graph.** $G'(\theta)=(X,\;Y',\;N,\;A)$ where $Y'=m_\theta(X,P(A),Y^{(0)},\xi)$. Exogenous features and the topology are held fixed so that identification comes from matching the **distribution of outcomes over (sampled) subgraphs**.

**Subgraph sampling.** Let $\mathsf{S}$ be a randomized sampler (e.g., $k$-hop ego nets, rooted random-walk subgraphs). Sampling from $G$ induces $p_{\text{data}}^{\mathsf{S}}$ over subgraphs $g$; sampling from $G'(\theta)$ induces $p_\theta^{\mathsf{S}}$. ([pytorch-geometric.readthedocs.io][2])

**Discriminator.** A GNN-based discriminator $D_\phi: g \mapsto [0,1]$ outputs the probability that subgraph $g$ is from the ground-truth distribution $p_{\text{data}}^{\mathsf{S}}$.

**Adversarial objective (Goodfellow-style).** We estimate $\theta$ by the GAN minimax program

```math
\min_{\theta}\ \max_{\phi}\ 
\mathbb{E}_{g\sim p_{\text{data}}^{\mathsf{S}}}\big[\log D_\phi(g)\big]
+ \mathbb{E}_{g\sim p_{\theta}^{\mathsf{S}}}\big[\log\!\big(1-D_\phi(g)\big)\big],
```

which reduces (for an optimal discriminator) to minimizing the Jensen–Shannon divergence between $p_{\text{data}}^{\mathsf{S}}$ and $p_{\theta}^{\mathsf{S}}$. This is the standard GAN formulation with generator $m_\theta$ and discriminator $D_\phi$. ([arXiv][3], [papers.neurips.cc][4])

Equivalently, your earlier statement “minimize a classification loss achieved by the optimal $D$” corresponds to minimizing the negative log-likelihood / cross-entropy implied by the expression above (accuracy is a coarser surrogate). The adversarial estimator connects to the Kaji–Manresa–Pouliot framework, which studies the statistical properties of such minimax estimators for structural models. ([arXiv][1], [Institute for Fiscal Studies][5])

## Practical implementation

* **Discriminator.** Implement $D_\phi$ with PyTorch Geometric. Use $k$-hop ego nets or rooted random-walk subgraphs as $\mathsf{S}$; both are available as transforms/utilities (e.g., `RootedEgoNets`). ([pytorch-geometric.readthedocs.io][2])
* **Generators.**

  * *Ground truth generator*: sampling manager over $G$ producing subgraphs $g\sim p_{\text{data}}^{\mathsf{S}}$.
  * *Synthetic generator*: wraps the structural mapping $m_\theta$. It reuses $X,A$ (and chosen $Y^{(0)}$) from the ground truth, constructs $P(A)$ with zero diagonal, and produces $Y'$. It exposes `generate_outcomes(θ)` to simulate counterfactual outcomes and align subgraph sampling with the ground-truth sampler.
* **Optimization.** The outer problem is black-box in $\theta$; Bayesian optimization is a reasonable default. Use the **binary cross-entropy** induced by the GAN objective (not raw accuracy) as the scalar objective passed to the optimizer; this aligns the estimator with the minimax program above. ([arXiv][3])

## `linear_in_means_model.ipynb`

A two-parameter testbed illustrating training curves and objective values. (For linear-in-means $Y=(I-\rho P)^{-1}(X\beta+\varepsilon)$, the no-self-loop restriction is satisfied by construction via $P(A)$ and invertibility requires $|\rho|<1/\lambda_{\max}(P)$.)

## Notes

* Current utils target the linear-in-means demo; they can be generalized to other structural classes by swapping $m_\theta$ and the subgraph sampler $\mathsf{S}$.
* GNN architecture for $D_\phi$ is chosen ad hoc in the demo; stronger identification may tolerate simple discriminators.
* The demo reports accuracy; for estimation we recommend minimizing the cross-entropy implied by the GAN objective.

## Reference

* Goodfellow, I., et al. (2014). *Generative Adversarial Networks.* NeurIPS. ([papers.neurips.cc][4])
* Kaji, T., Manresa, E., & Pouliot, G. (2023). *An adversarial approach to structural estimation.* *Econometrica.* (working-paper version: 2020 arXiv / IFS). ([arXiv][1], [Institute for Fiscal Studies][5])
* Hamilton, W., Ying, R., & Leskovec, J. (2017). *GraphSAGE: Inductive Representation Learning on Large Graphs.* NeurIPS. (for neighborhood sampling design). ([papers.neurips.cc][6])
* Bojchevski, A., et al. (2018). *NetGAN: Generating Graphs via Random Walks.* ICML. (graph-GAN perspective; WGAN objective). ([Proceedings of Machine Learning Research][7])

