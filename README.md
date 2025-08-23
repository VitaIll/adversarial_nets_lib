# Adversarial Estimation on Graphs

This repository implements **adversarial structural estimation** for graph-based models (peer effects, strategic communication games, diffusion, etc.), extending the framework of Kaji–Manresa–Pouliot (Econometrica, 2023) to settings where data are observed on a **single network**. Because a graph is typically one realization (not i.i.d. rows), we build variability by sampling **subgraphs** from the observed (“real”) network and from a **structural simulator** (“generator”), then train a **GNN discriminator** to distinguish real vs simulated subgraphs. Structural parameters are chosen to make the discriminator’s job as hard as possible—mirroring the original GAN min–max game.

---

## Notation 

**Observed graph.**

$$
G_{\text{data}} = (X, Y, A),
\quad X\in\mathbb{R}^{n\times k},\; Y\in\mathbb{R}^{n\times \ell},\;
A\in\{0,1\}^{n\times n}\ \text{(symmetric; zero diagonal by default)}.
$$

**Structural simulator (generator).**
We model outcomes via a structural **operator** on outcomes:

$$
F_\theta:\ \mathbb{R}^{n\times \ell}\to \mathbb{R}^{n\times \ell},\qquad 
Y^{(t+1)} = F_\theta\!\big(X, A, Y^{(t)}, \varepsilon^{(t)}\big),
$$

with shocks $\varepsilon$ (and an equilibrium-selection rule if multiple equilibria exist). This unifies:

* **Static mapping (special case):** $m_\theta(X,A,\varepsilon) = F_\theta(X,A,\mathbf{0},\varepsilon)$ (no dependence on $Y$).
* **Fixed-point/equilibrium:** $Y_\theta = F_\theta(X,A,Y_\theta,\varepsilon)$.
* **K-step dynamics:** start from $Y^{(0)} = Y_{\text{init}}$ and iterate $K$ times to get $Y^{(K)}$.

The model induces a **synthetic graph** $G_\theta=(X, Y_\theta, A)$ and a distribution $p_\theta$ over outcomes (via $\varepsilon$ and equilibrium selection).

**Identity generalization (requested).**
We allow $F_\theta$ to include the **identity operator** as a limiting/edge case:

$$
F_{\theta_{\text{id}}}(X,A,Y,\varepsilon) \equiv Y,
$$

or via a convex mixture

$$
F_{\theta,\alpha}(X,A,Y,\varepsilon)
= \alpha\,\tilde F_\theta(X,A,Y,\varepsilon) + (1-\alpha)\,Y,\qquad \alpha\in[0,1].
$$

We also allow $Y_{\text{init}}$ to be:

* zeros or noise (fresh simulation),
* a baseline predictor,
* the **observed** $Y_{\text{data}}$ (warm start).

> ⚠️ **Degeneracy warning.**
> If $Y_{\text{init}}=Y_{\text{data}}$ and either $K=0$ **or** $\alpha=0$, then $G_\theta=G_{\text{data}}$ and the adversarial objective becomes uninformative. Treat identity/warm-start as **diagnostic baselines** or enforce constraints (e.g., $K\!\ge\!1$, $\alpha\!\ge\!\alpha_{\min}\!>\!0$) or penalties keeping the identity neighborhood out of the feasible set during estimation.

**Subgraph sampling.**
Let $\mathcal{S}$ denote a subgraph sampler (e.g., $k$-hop ego-nets, random-walk-induced subgraphs). Sampling induces two distributions over subgraphs $g\in\mathcal{G}$:

$$
p_{\text{data}}^{\mathcal{S}}\quad\text{from }G_{\text{data}},\qquad
p_{\theta}^{\mathcal{S}}\quad\text{from }G_\theta.
$$

**Discriminator.**
A GNN $D_\phi:\mathcal{G}\to[0,1]$ outputs the probability a subgraph is **real**.

---

## Adversarial Objective

Using the standard GAN (Goodfellow et al.) cross-entropy on **subgraphs**:

$$
V(\phi,\theta)
=\mathbb{E}_{g\sim p_{\text{data}}^{\mathcal{S}}}\!\big[\log D_\phi(g)\big]
\;+\;
\mathbb{E}_{g\sim p_\theta^{\mathcal{S}}}\!\big[\log (1-D_\phi(g))\big].
$$

Estimate $\theta$ via the min–max game:

$$
\theta^\star \in \arg\min_\theta\ \max_\phi\ V(\phi,\theta).
$$

With the optimal discriminator
$D_\phi^\star(g) = \frac{p_{\text{data}}^{\mathcal{S}}(g)}{p_{\text{data}}^{\mathcal{S}}(g) + p_\theta^{\mathcal{S}}(g)}$,
the generator minimizes the **Jensen–Shannon divergence** between
$p_{\text{data}}^{\mathcal{S}}$ and $p_\theta^{\mathcal{S}}$ (up to a constant).
*Training note:* many implementations use the **non-saturating** generator loss
$-\,\mathbb{E}_{g\sim p_\theta^{\mathcal{S}}}\log D_\phi(g)$ for better gradients.

---

## Practical Implementation

### Components

* **Discriminator (`D_φ`).**
  Implemented with PyTorch Geometric as a GNN binary classifier on subgraphs (node features, optional edge features, outcomes).

* **Generator (structural simulator).**
  Unified interface handles *both* real and synthetic sampling. The **real** branch is a sampling manager over $G_{\text{data}}$. The **synthetic** branch wraps your structural operator $F_\theta$, reusing $X,A$ from the real graph to align covariates and topology, and exposes:

  * `generate_outcomes(θ, y_init=None, n_steps=1, mix_alpha=None, rng=None)` → $Y^{(K)}$.

    * `y_init=None` → default init (zeros/noise).
    * `mix_alpha∈[0,1]` → applies $F_{\theta,\alpha} = \alpha \tilde F_\theta + (1-\alpha) I$.
    * Guards recommended to forbid identity during **training** but allow it in **ablations**.

* **Subgraph sampler (`𝒮`).**
  First-class component (e.g., $k$-hop, degree-stratified, random-walk with restart). The choice of $\mathcal{S}$ defines the target subgraph distribution and matters for identifiability.

* **Outer optimizer over $\theta$.**
  Default: **Bayesian optimization** (black-box friendly, handles non-differentiable $F_\theta$). If $F_\theta$ is differentiable end-to-end, you may alternate gradient steps in $(\phi,\theta)$ like standard GANs; otherwise keep BO.

### Training protocol (suggested)

1. **Split** sampled subgraphs into train/val/test at the *subgraph* level (avoid leakage by node overlap if relevant).
2. **Train** $D_\phi$ with cross-entropy to distinguish $p_{\text{data}}^{\mathcal{S}}$ vs $p_\theta^{\mathcal{S}}$.
3. **Update** $\theta$ (BO step or gradient step) based on validation $V(\phi,\theta)$.
4. **Repeat** until convergence/stopping.
5. **Report** test $V(\phi,\theta^\star)$ and (optionally) calibration metrics of $D_\phi$ to ensure a meaningful game.

### Multiple equilibria & shocks

Specify the **equilibrium-selection rule** and the distribution of shocks $\varepsilon$; both are part of $p_\theta$. For dynamic models, document $K$, $\alpha$, and the initialization $Y_{\text{init}}$.

---

## Example: `linear_in_means_model.ipynb`

A two-parameter peer-effects testbed demonstrating:

* $k$-hop ego-net sampling,
* training a simple GNN discriminator,
* the min–max optimization trace for $\theta$,
* ablations with identity/warm-start (diagnostics only).

---

## Notes & Current Limitations

* Utilities are currently tailored to the linear-in-means demo; generalization is planned.
* For quick visualization the notebook may show **accuracy**, but training and selection use **log-loss** aligned with the GAN objective.
* Richer discriminators and smarter samplers (e.g., degree/centrality stratification) often improve identification.
* For very large graphs, add assumptions (stationarity/mixing or large-$n$ regimes) justifying that $\mathcal{S}$ yields a stable subgraph distribution.

---

## Optional Recommendations

* **Guard against identity degeneracy.** Enforce $K\!\ge\!1$ and/or $\alpha\!\ge\!\alpha_{\min}\!>\!0$ during training; allow identity only in ablations.
* **Make `𝒮` pluggable.** Expose clear interfaces so experiments vary only $F_\theta$ or $\mathcal{S}$.
* **Use proper scoring rules.** Prefer cross-entropy/Brier over accuracy for model selection and BO targets.
* **Log equilibrium details.** Record equilibrium selection, shock seeds, and sampler config for reproducibility.

---

## References

* **Kaji, T., Manresa, E., & Pouliot, G. (2023).** *An Adversarial Approach to Structural Estimation.* **Econometrica**, 91(6), 2041–2063.
* **Goodfellow, I., Pouget-Abadie, J., et al. (2014).** *Generative Adversarial Nets.* NeurIPS.
* **Goodfellow, I. (2016).** *NIPS 2016 Tutorial: Generative Adversarial Networks.* (non-saturating loss; practical tips).

---

## Citation

If you use this code, please cite both this repository.

