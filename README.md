# Adversarial estimation on graphs

This project adapts the adversarial estimator of Kaji, Manresa & Pouliot (2023) to structural models defined on graphs (e.g., strategic communication, peer effects). In tabular settings, each row is treated as an i.i.d. draw. With graph data the entire graph is typically a single realization, so we induce variability for the discriminator by sampling subgraphs from both the ground-truth and the synthetic graph and training a classifier to distinguish their origins. Intuitively, local neighborhoods carry enough information about global structure to power discrimination—consistent with common GNN practice (e.g., \$k\$-hop ego sampling) and graph-GANs that discriminate on local walks/subgraphs.

## Formal setup

**Observed graph.** Let \$G=(X,Y,N,A)\$ with:

* \$X\in\mathbb{R}^{n\times k}\$: exogenous node covariates (row \$i\$ is \$x\_i^\top\$).
* \$Y\in\mathbb{R}^{n\times \ell}\$: endogenous node outcomes (row \$i\$ is \$y\_i^\top\$).
* \$N={0,\dots,n}\$: node index set.
* \$A\in{0,1}^{n\times n}\$: undirected adjacency, symmetric. We work with a **peer operator** \$P(A)\$ that **excludes self-links**:

  $$
  P(A)\;=\;A-\operatorname{diag}(A)\qquad\text{(no self\text{-}to\text{-}self influence)}.
  $$

  Optionally \$P(A)\$ may be row/degree normalized; the formulation only requires \$\operatorname{diag}(P(A))=0\$.

**Structural model (generator).** A structural mapping

$$
m_\theta:\ \big(X,\,P(A),\,Y^{(0)},\,\xi\big)\ \longmapsto\ Y' \in \mathbb{R}^{n\times \ell}
$$

takes covariates, the peer operator, and an **initial outcome state** \$Y^{(0)}\$ (e.g., pre-interaction signals, baseline choices, or an initialization such as zeros) and returns simulated outcomes \$Y'\$. The innovation \$\xi\$ captures simulation randomness if present.

* **Single-step (peer-to-peer, no self-loop):**

  $$
  Y' \;=\; m_\theta\!\Big(X,\;P(A)\,Y^{(0)}\,,\;\xi\Big),
  $$

  i.e., each \$y\_i'\$ depends on \$x\_i\$ and **peers’ initial outcomes** \${y\_j^{(0)}: j\in \mathcal{N}(i)}\$, but not on \$y\_i^{(0)}\$ directly.
* **Multi-step propagation (optional):** for \$t=0,\dots,T-1\$,

  $$
  Y^{(t+1)} \;=\; m_\theta\!\Big(X,\;P(A)\,Y^{(t)}\,,\;\xi^{(t)}\Big),
  \qquad Y'\equiv Y^{(T)}.
  $$

  This allows \$k\$-hop effects to accrue while maintaining a zero diagonal in \$P(A)\$.

**Synthetic graph.** \$G'(\theta)=(X,;Y',;N,;A)\$ where \$Y'=m\_\theta(X,P(A),Y^{(0)},\xi)\$. Exogenous features and the topology are held fixed so that identification comes from matching the **distribution of outcomes over (sampled) subgraphs**.

**Subgraph sampling.** Let \$\mathsf{S}\$ be a randomized sampler (e.g., \$k\$-hop ego nets, rooted random-walk subgraphs). Sampling from \$G\$ induces \$p\_{\text{data}}^{\mathsf{S}}\$ over subgraphs \$g\$; sampling from \$G'(\theta)\$ induces \$p\_\theta^{\mathsf{S}}\$.

**Discriminator.** A GNN-based discriminator \$D\_\phi: g \mapsto \[0,1]\$ outputs the probability that subgraph \$g\$ is from the ground-truth distribution \$p\_{\text{data}}^{\mathsf{S}}\$.

**Adversarial objective (Goodfellow-style).** Estimate \$\theta\$ by the GAN minimax program

$$
\min_{\theta}\ \max_{\phi}\ 
\mathbb{E}_{g\sim p_{\text{data}}^{\mathsf{S}}}\!\big[\log D_\phi(g)\big]
\;+\;
\mathbb{E}_{g\sim p_{\theta}^{\mathsf{S}}}\!\big[\log\!\big(1-D_\phi(g)\big)\big].
$$

At the optimal discriminator, this minimizes the Jensen–Shannon divergence between \$p\_{\text{data}}^{\mathsf{S}}\$ and \$p\_{\theta}^{\mathsf{S}}\$. Equivalently, your earlier statement “minimize a classification loss achieved by the optimal \$D\$” corresponds to minimizing the negative log-likelihood / cross-entropy implied by the expression above (accuracy is a coarser surrogate). The adversarial estimator connects to the Kaji–Manresa–Pouliot framework, which studies the statistical properties of such minimax estimators for structural models.

## Practical implementation

* **Discriminator.** Implement \$D\_\phi\$ with PyTorch Geometric. Use \$k\$-hop ego nets or rooted random-walk subgraphs as \$\mathsf{S}\$; keep the sampler identical across real/synthetic pipelines.
* **Generators.**

  * *Ground-truth generator*: sampling manager over \$G\$ producing subgraphs \$g\sim p\_{\text{data}}^{\mathsf{S}}\$.
  * *Synthetic generator*: wraps the structural mapping \$m\_\theta\$. It reuses \$X,A\$ (and chosen \$Y^{(0)}\$) from the ground truth, constructs \$P(A)\$ with zero diagonal, and produces \$Y'\$. It exposes `generate_outcomes(θ)` to simulate counterfactual outcomes and align subgraph sampling with the ground-truth sampler.
* **Optimization.** The outer problem is black-box in \$\theta\$; Bayesian optimization is a reasonable default. Use the **binary cross-entropy** induced by the GAN objective (not raw accuracy) as the scalar objective passed to the optimizer.

## `linear_in_means_model.ipynb`

A two-parameter testbed illustrating training curves and objective values. (For linear-in-means \$Y=(I-\rho P)^{-1}(X\beta+\varepsilon)\$, the no-self-loop restriction is satisfied by construction via \$P(A)\$ and invertibility requires \$|\rho|<1/\lambda\_{\max}(P)\$.)

## Notes

* Current utils target the linear-in-means demo; they can be generalized to other structural classes by swapping \$m\_\theta\$ and the subgraph sampler \$\mathsf{S}\$.
* GNN architecture for \$D\_\phi\$ is chosen ad hoc in the demo; stronger identification may tolerate simple discriminators.
* The demo reports accuracy; for estimation we recommend minimizing the cross-entropy implied by the GAN objective.

## References

* Goodfellow, I., et al. (2014). *Generative Adversarial Networks.* NeurIPS.
* Kaji, T., Manresa, E., & Pouliot, G. (2023). *An adversarial approach to structural estimation.* *Econometrica*, 91(6), 2041–2063.
* Hamilton, W., Ying, R., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs (GraphSAGE).* NeurIPS.
* Bojchevski, A., et al. (2018). *NetGAN: Generating Graphs via Random Walks.* ICML.

---

### Optional, major (non-blocking) recommendations

1. **Use the explicit GAN loss during outer optimization.** Replace accuracy with the expected log loss above to keep the generator’s target aligned with the minimax program.
2. **Consider a Wasserstein variant for stability.** If discriminator overpowering or vanishing gradients arise, switch to a WGAN-style critic with gradient penalty.
3. **Sampler auditing.** Document \$\mathsf{S}\$ (ego-hop vs. random-walk) and keep it identical across real/synthetic pipelines to avoid an “estimator with changing target.”
4. **Multiple equilibria handling.** If \$m\_\theta\$ admits multiple equilibria, encode an equilibrium selection rule \$s\_\theta(X,P(A),\upsilon)\$ (with tie-breaking randomness \$\upsilon\$) so the generator’s stochastic output distribution is explicit.
5. **Diagnostics beyond the adversarial loss.** Supplement with MMD or moment checks on subgraph statistics (degree mix, assortativity, motif counts) to catch mode collapse that a single \$D\_\phi\$ might miss.
