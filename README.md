# Adversarial estimation on graphs

Adversarial estimator for graph structural models extends the theoretical framework proposed by Kaji et al. (2023) to structural models defined on graphs (strategic communication games, peer-effect models, etc.). With graph data, the entire network is typically a single realization; to create variability for the discriminator we sample subgraphs from both the ground truth and the synthetic graph and label them by origin. Intuitively, local neighborhoods contain enough information about global structure for discrimination; our current implementation uses \$k\$-hop ego sampling. Additional challenges include multiple equilibria in dynamic network models, lack of closed-form asymptotics, and the need to tailor the discriminator architecture to the structural class.

## Formal setup

**Observed data.** Let \$G=(X,Y,N,A)\$ where:

* \$X\in\mathbb{R}^{n\times k}\$ are exogenous node covariates (row \$i\$ is \$x\_i^\top\$).
* \$Y\in\mathbb{R}^{n\times \ell}\$ are endogenous node outcomes (row \$i\$ is \$y\_i^\top\$).
* \$N={1,\dots,n}\$ is the node index set.
* \$A\in{0,1}^{n\times n}\$ is a symmetric adjacency matrix.

We work with a **peer operator** \$P(A)\$ that **excludes self-links**:

```math
P(A) = A - \operatorname{diag}(A)
```

Optionally, \$P(A)\$ may be row/degree-normalized; we only require \$\operatorname{diag}(P(A))=0\$.

**Structural model (generator).** A structural mapping

```math
m_\theta:\ \big(X,\ P(A),\ Y^{(0)},\ \xi\big)\ \longmapsto\ Y' \in \mathbb{R}^{n\times \ell}
```

takes covariates, the peer operator, and an **initial outcome state** \$Y^{(0)}\$ (e.g., pre-interaction signals, baselines, or zeros) and returns simulated outcomes \$Y'\$. The innovation \$\xi\$ captures simulation randomness if present.

* **Single-step (peer-to-peer, no self-loop):**

```math
Y' = m_\theta\!\Big(X,\ P(A)\,Y^{(0)},\ \xi\Big),
```

i.e., each \$y\_i'\$ depends on \$x\_i\$ and **peers’ initial outcomes** \${y\_j^{(0)}: j\in\mathcal{N}(i)}\$, but not on \$y\_i^{(0)}\$ directly.

* **Multi-step propagation (optional):** for \$t=0,\dots,T-1\$,

```math
Y^{(t+1)} = m_\theta\!\Big(X,\ P(A)\,Y^{(t)},\ \xi^{(t)}\Big),
\qquad Y' \equiv Y^{(T)}.
```

**Synthetic data.** Define \$G'(\theta)=(X,\ Y',\ N,\ A)\$ with \$Y'=m\_\theta(X,P(A),Y^{(0)},\xi)\$. Exogenous features and topology are held fixed so identification comes from matching the **distribution of outcomes over sampled subgraphs**.

**Subgraph sampling.** Let \$\mathsf{S}\$ denote a randomized sampler (e.g., \$k\$-hop ego nets or rooted random-walk subgraphs). Sampling from \$G\$ induces \$p\_{\text{data}}^{\mathsf{S}}\$ over subgraphs \$g\$; sampling from \$G'(\theta)\$ induces \$p\_{\theta}^{\mathsf{S}}\$.

**Discriminator.** A GNN discriminator \$D\_\phi: g\mapsto\[0,1]\$ outputs the probability that \$g\$ came from \$p\_{\text{data}}^{\mathsf{S}}\$.

**Adversarial objective (Goodfellow-style).** Estimate \$\theta\$ by

```math
\min_{\theta}\ \max_{\phi}\ 
\mathbb{E}_{g\sim p_{\text{data}}^{\mathsf{S}}}\!\big[\log D_\phi(g)\big]
\;+\;
\mathbb{E}_{g\sim p_{\theta}^{\mathsf{S}}}\!\big[\log\big(1-D_\phi(g)\big)\big].
```

At the optimal discriminator this minimizes the Jensen–Shannon divergence between \$p\_{\text{data}}^{\mathsf{S}}\$ and \$p\_{\theta}^{\mathsf{S}}\$. Your earlier “minimize the classification loss under the optimal \$D\$” is equivalent to minimizing the implied cross-entropy; raw accuracy is a much coarser surrogate.

## Practical implementation

* **Discriminator.** Implement \$D\_\phi\$ with PyTorch Geometric. Use the same sampler \$\mathsf{S}\$ (e.g., \$k\$-hop ego nets or rooted random-walk subgraphs) for real and synthetic graphs to keep the target distribution fixed.
* **Generators.**

  * *Ground-truth generator*: sampling manager over \$G\$ to produce \$g\sim p\_{\text{data}}^{\mathsf{S}}\$.
  * *Synthetic generator*: wraps \$m\_\theta\$, reuses \$X\$, \$A\$ (and chosen \$Y^{(0)}\$), constructs \$P(A)\$ with zero diagonal, and exposes `generate_outcomes(θ)` for counterfactual simulation.
* **Optimization.** Treat the outer problem as black-box in \$\theta\$. Bayesian optimization is a reasonable default; use **binary cross-entropy** from the objective above (not accuracy) as the scalar loss.

## `linear_in_means_model.ipynb`

A two-parameter testbed illustrating training curves and objective values. For linear-in-means,

```math
Y = (I-\rho P)^{-1}(X\beta+\varepsilon),
```

the no-self-loop restriction holds via \$P(A)\$ and invertibility requires

```math
|\rho| < 1/\lambda_{\max}(P).
```

## Notes

* Utilities currently target the linear-in-means demo; they should generalize by swapping \$m\_\theta\$ and the sampler \$\mathsf{S}\$.
* The demo’s GNN for \$D\_\phi\$ is ad hoc; strong identification often tolerates simple discriminators.
* Prefer cross-entropy over accuracy for the outer objective.

## Reference

Kaji, T., Manresa, E., & Pouliot, G. (2023). *An adversarial approach to structural estimation*. **Econometrica**, 91(6), 2041–2063.

