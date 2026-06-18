# Ontometric Dual‑Stream Architecture (ODSA): A Physics‑Guided Design Pattern for Frugal World Models

**Author:** [Massimiliano Concas]  
**Date:** [June 18 2026]  
**Affiliation:** [Ciber Fabbrica]  

## Abstract
We present the **Ontometric Dual‑Stream Architecture (ODSA)**, a design pattern that transforms high‑dimensional sensor streams into compact, generalizable predictive models. ODSA feeds two parallel input pathways into a lightweight neural network: the **territory** (raw, unlabeled sensor readings) and the **map** (a causally computed stream of dimensionless, self‑referenced ontometric features derived from the system’s own healthy state). The network learns to correlate the two streams, using the map as a relational scaffold that anchors the interpretation of raw data. We validate ODSA on the NASA C‑MAPSS turbofan degradation benchmark under extreme Green AI constraints (≤32 k parameters, ≤50 epochs, CPU only). A single ODSA model trained on all four engine fleets simultaneously achieves NASA scores of **429 (FD001), 1645 (FD002), 607 (FD003), and 2074 (FD004)**, meeting every challenge target and outperforming a raw‑sensor baseline by a factor of 250. We further show that ODSA’s efficiency stems from a geometric property: ontometric features are approximately *proportional* to the target variable, providing consistent gradient directions that collapse the condition number of the optimization landscape. This allows the model to converge with less data, fewer steps, and larger learning rates. Finally, we demonstrate that ODSA naturally scales horizontally by incorporating multiple ontometric maps for different physical subsystems or domains, enabling a single network to function as a generalist world model. The architecture is domain‑agnostic, scale‑invariant, regime‑invariant, and model‑invariant—offering a principled route to frugal, interpretable AI for the physical world.

---

## 1. Introduction
Modern machine learning for physical systems often follows a brute‑force paradigm: ingest raw sensor values into a large model and let it jointly discover a useful internal representation and a predictive mapping. This approach is computationally expensive, data‑hungry, and frequently fails under tight resource constraints. The **Ontometry** framework [1] provided a mathematical explanation for this inefficiency: absolute measurement scales impose a hidden computational burden—the $O(D^2)$ overhead—that can be eliminated by expressing the system in its own dimensionless, intrinsic coordinates. Relational Calculus (RC) operationalizes this insight by prescribing a method to build dimensionless ratios anchored to the system’s own limits (North Stars).

However, RC alone does not dictate *which* ratios to build; it is a domain‑agnostic grammar. In this paper, we translate the ontometric principle into a concrete neural architecture: **ODSA**. Rather than replacing raw data with hand‑crafted features, ODSA retains both: the raw, semantically void sensor stream (the *territory*) and a causally computed stream of dimensionless, physically meaningful state variables (the *map*). The network then learns to align the two, using the map as a relational scaffold that linearizes the prediction problem. We demonstrate ODSA on the most widely used benchmark for prognostics—NASA’s C‑MAPSS turbofan dataset—and show that it meets stringent accuracy and efficiency targets while providing a generalizable blueprint for frugal world models.

---

## 2. Related Work
**Physics‑Informed Neural Networks (PINNs)** embed physical laws as soft penalty terms in the loss function. ODSA differs fundamentally: it injects physics into the input structure, not the optimization objective. This architectural separation allows the network to learn the residual dynamics without being constrained by approximate equations.

**Feature engineering** traditionally replaces raw data with derived statistics. ODSA does not discard raw data; it augments it with an invariant map, enabling the network to learn the correspondence between the two streams—a form of implicit representation learning guided by physics.

**Multi‑stream networks** have been used for multimodal data (e.g., RGB+depth). ODSA’s dual streams are of a different nature: one is the raw physical signal, the other is a mathematically derived relational scaffold. The interaction between them is not a simple fusion but a *correlation* that teaches the network what to attend to.

**World models** (Dreamer, JEPA) learn latent state representations. ODSA provides an explicit, interpretable state scaffold, eliminating the need for the network to discover the state from scratch and drastically reducing the required model capacity.

---

## 3. Ontometric Dual‑Stream Architecture (ODSA)

### 3.1 Design Principle
ODSA is founded on a single premise:  
*When building a predictive model for a sensorized physical system, do not force the neural network to invent a representation from unlabeled, absolute‑scale numbers. Instead, provide both the raw sensory stream (territory) and a causally computed stream of dimensionless, invariant state variables (the map). The network learns to correlate the two, achieving efficient, generalizable prediction.*

The map is constructed following the Relational Calculus recipe: identify healthy baselines per operating regime, compute dimensionless deltas, and derive composite invariants that capture the system’s degradation state. Critically, the map is **proportional** to the hidden target—it moves in lockstep with the quantity to be predicted, while the raw territory is confounded by operating‑point variations.

### 3.2 Architecture
The ODSA consists of two parallel input pathways that are concatenated and fed into a lightweight recurrent or feedforward network.

- **Raw stream ($X_{\text{raw}}$):** the original sensor readings at each time step, possibly scaled.
- **Ontometric stream ($X_{\text{onto}}$):** a set of features computed online, causally, using only past data. These features are dimensionless ratios relative to a per‑engine, per‑regime healthy baseline. Typical components include a composite degradation index, its rate of change, and an urgency signal that amplifies near critical thresholds.

The network processes the concatenated vector $[X_{\text{raw}}, X_{\text{onto}}]$ and produces a prediction $\hat{y}$. No auxiliary loss forces the network to reconstruct the ontometric features—the scaffold works purely by providing a correlated information channel that shapes the gradients.

### 3.3 Online Ontometric Feature Construction
The ontometric stream is built via a deterministic, causal pipeline:

1. **Regime identification:** discretize the operating point (e.g., flight condition) from control settings.
2. **Healthy baseline capture:** for each physical asset, buffer the first $B$ cycles (e.g., $B=10$) and record sensor values per regime. Compute a robust baseline (median or EMA) for each sensor in each regime.
3. **Dimensionless delta:** for every subsequent time step, compute  
   $\delta_s(t) = \frac{s(t) - \text{baseline}_\text{regime}(s)}{\text{baseline}_\text{regime}(s)}$  
4. **Composite invariants:** aggregate deltas from groups of functionally related sensors into scalar health indices (e.g., a “pain index,” a “compensatory effort index” $\Theta$). Apply exponential moving averages to smooth.
5. **Kinematic derivatives:** compute velocity (lagged difference) and acceleration of the smoothed indices to capture trend dynamics.
6. **Criticality signals:** define a threshold that marks the entry into a terminal phase. Construct urgency as the rate of approach divided by distance to threshold, and alarm as the product of the health index and urgency.

Every step is causal (no future information), uses a history buffer ≤30 steps, and requires no global dataset scan.

### 3.4 Geometric Rationale: Why the Map Accelerates Learning
The efficiency of ODSA can be understood entirely through the geometry of the optimization landscape. Let $y$ be the target (RUL). The network’s loss gradient with respect to a weight $w_i$ connected to feature $x_i$ is proportional to $x_i \cdot \frac{\partial L}{\partial \hat{y}}$.

- For a **raw sensor** $x_i$, the relationship $x_i \to y$ is confounded by regime changes. The error signal $\frac{\partial L}{\partial \hat{y}}$ is noisy and often contradictory, yielding gradients that point in different directions for different samples. The optimizer sees a rugged, ill‑conditioned landscape.
- For an **ontometric feature** $x_j$, the construction removes the operating‑point offset, making $x_j$ approximately **proportional** to $y$. The error signal is consistent across samples, and the gradient vector points steadily downhill. The landscape becomes smooth and nearly isotropic.

In effect, proportionality provides **direction**. The optimizer, blind to physical meaning, simply follows the most consistent gradient signal. Because the ontometric features offer a high gradient signal‑to‑noise ratio (gSNR), the network rapidly aligns its weights to exploit them, achieving convergence with:
- **Less data**: each sample conveys cleaner information.
- **Fewer steps**: the loss surface is well‑conditioned, allowing large, stable updates.
- **Higher learning rate**: consistent gradients prevent divergence even with aggressive step sizes.

This geometric interpretation demystifies the “magic” of ODSA: the human sees semantics (degradation indices, urgency), but the network sees only a smoothly sloping vector field that leads directly to the solution.

---

## 4. Case Study: C‑MAPSS Turbofan RUL Prediction

### 4.1 Dataset and Challenge Constraints
The NASA C‑MAPSS dataset [2] contains run‑to‑failure trajectories of turbofan engines under 4 different operating/fault regimes (FD001–FD004). The task is to predict Remaining Useful Life (RUL) at each time step from 21 noisy sensor measurements and 3 control signals. We adopt the **Green AI Challenge** constraints:
- ≤11 sensors out of 21.
- All features must be computable online, causally, with a fixed history buffer ≤30 steps.
- Model size ≤32 000 trainable parameters, flash memory footprint <130 KB, inference <0.7 MFLOPs.
- Training <5 min per seed on a modern CPU, ≤50 epochs.
- A single model must serve all four fleets (Protocol B – unified fleet‑wide training).
- Evaluation uses the NASA asymmetric scoring function ($a_1=13$, $a_2=10$, RUL clipped at 125) over ≥5 random seeds.

### 4.2 ODSA Implementation
**Sensor selection:** 11 sensors were chosen via systematic correlation analysis with RUL and domain knowledge, covering fuel‑flow potential, compressor/turbine effort, and fan health.

**Ontometric features (the map):**  
- Per‑engine, per‑regime healthy baseline captured from the first 10 cycles (EMA).  
- Dimensionless deltas computed as above.  
- **Pain index $T_\text{ema}$:** mean absolute delta of four wear‑sensitive sensors, smoothed (span≈10).  
- **Compensatory effort $\Theta_\text{ema}$:** mean absolute delta of six effort‑related sensors, halved and smoothed.  
- **Velocity $v_\Theta$:** lag‑5 difference of $\Theta_\text{ema}$.  
- **Urgency:** $v_\Theta$ divided by distance to threshold $\Theta_\text{thresh}=0.003$, gated for $\Theta_\text{ema} > 0.0015$.  
- **Alarm:** $\Theta_\text{ema} \times$ urgency.  
- Kinematic derivatives of the pain index ($v_T$, $a_T$) are also included.  
- Operating settings ($op1, op2, op3$) are passed through as context.

**Model:** 2‑layer LSTM (48→32 units) + Dense(16) bottleneck + linear output, total 24 437 parameters. Batch normalization at input. Dropout 0.1 after the dense layer.  
**Training:** Adam optimizer with warm‑up cosine decay (lr max $10^{-3}$, min $10^{-5}$), weighted asymmetric NASA loss (late penalties 3×, early 1×, extra 2× for RUL<30). Batch size 256, early stopping on validation loss (patience 10). Engine‑wise 80/20 split fixed across seeds.  
**Baseline:** Identical architecture and training, but fed only the 11 raw sensors and operating settings (no ontometric map).

### 4.3 Results
| Dataset | Target NASA Score | ODSA (Unified, 5 seeds) | Raw‑Sensor Baseline (Unified, 5 seeds) | Improvement Factor |
|---------|------------------|--------------------------|----------------------------------------|-------------------|
| FD001   | <470             | **429 ± 58**             | 105 850 ± 11 888                       | 247×              |
| FD002   | <3200            | **1645 ± 175**           | 354 121 ± 39 838                       | 215×              |
| FD003   | <1200            | **607 ± 112**            | 114 883 ± 12 905                       | 189×              |
| FD004   | <4600            | **2074 ± 166**           | 394 868 ± 44 425                       | 190×              |

ODSA meets every challenge target with comfortable margins and low seed variance. The raw‑sensor baseline, despite identical capacity and training, fails catastrophically—its NASA scores are 2–3 orders of magnitude worse, consistent with the $O(D^2)$ overhead predicted by Ontometry.

### 4.4 Analysis
**Scale distortion collapse:** The raw sensor inputs span hundreds of physical units, creating a condition number $D^2 \approx 10^8$ that prevents convergence within 50 epochs. The ontometric features compress the dynamic range to $O(1)$ while maintaining strong, monotonic correlation with RUL, collapsing the condition number to near 1.

**Gradient signal clarity:** We computed the gradient signal‑to‑noise ratio (gSNR) for each input feature on the training set. Ontometric features exhibit gSNR values 50–200× higher than raw sensors, directly explaining the optimizer’s rapid alignment with the scaffold.

**Convergence speed:** ODSA reaches a validation loss in 15–20 epochs that the baseline fails to approach after 50 epochs, even though the baseline is allowed to train to the full epoch limit.

**Model invariance:** We replaced the LSTM with a simple feedforward network (3 layers, 24 k params) and retrained; the performance degraded by less than 5%, confirming that the ontometric features have already linearized the problem sufficiently that the model architecture becomes secondary.

---

## 5. Scaling ODSA Horizontally: Multi‑Map Generalization
The ODSA pattern scales naturally to multiple physical domains or subsystems within a single model. Suppose we have $K$ distinct assets (e.g., turbofan, battery, hydraulic pump) each with its own sensor suite and degradation dynamics. The R‑framework (or any suitable domain theory) can produce, for each asset, a separate ontometric map $\mathcal{M}_k$ that is proportional to the asset’s hidden state. By feeding all $K$ maps alongside the combined raw sensor streams, a single network can learn to:

- Detect which asset is currently active from the raw territory.
- Route attention to the corresponding map $\mathcal{M}_k$.
- Suppress maps that are irrelevant for the current sample, since they provide no consistent gradient signal.

This **implicit modularization** arises automatically from the gradient dynamics: the map $\mathcal{M}_k$ that is proportional to the target will have high gSNR for samples from asset $k$, while other maps will have low gSNR. The optimizer naturally amplifies the useful map and ignores the rest, without any explicit gating mechanism. The result is a **generalist world model** that can handle diverse domains in a single forward pass, with a parameter count far smaller than an ensemble of specialist models.

Because the maps share a common structure (dimensionless ratios, bounded ranges, kinematic derivatives), the network can also transfer knowledge across domains—e.g., the concept of “urgency near a critical threshold” learned from engines can help interpret a similar signal in batteries.

---

## 6. Discussion

### 6.1 Model Invariance and Deployment Flexibility
ODSA decouples representation from prediction. The ontometric map is a fixed, deterministic computation; any function approximator can consume it. This means the same feature pipeline can serve a microcontroller‑sized polynomial as easily as a cloud‑hosted Transformer. As hardware improves, the model can be upgraded without re‑engineering the sensor interface.

### 6.2 Interpretability and Safety
Every feature in the ontometric stream has a clear physical meaning. One can trace a high urgency prediction back to a rising $\Theta_\text{ema}$ and further to specific sensor deviations. This audit trail is essential for safety‑critical applications (e.g., aviation, medical devices) where black‑box models are unacceptable.

### 6.3 Green AI and Ecological Impact
By eliminating the $O(D^2)$ waste, ODSA reduces training FLOPs by orders of magnitude. The C‑MAPSS model trains in under 3 minutes on a laptop CPU. Scaling the approach to larger industrial fleets would yield proportionate energy savings, directly addressing the ecological footprint of AI.

### 6.4 Limitations and Future Work
The current validation is on a single benchmark. Future work will deploy ODSA on other asset types (batteries, wind turbines) and on multi‑domain datasets to quantify the horizontal scaling properties. The approach relies on the availability of a reliable healthy baseline; in systems without a clear “as‑new” period, the baseline may need to be continuously adapted, which introduces additional research challenges.

---

## 7. Conclusion
We have introduced ODSA, a dual‑stream architecture that augments raw sensor data with a causally computed ontometric map. This map, built from dimensionless invariants, provides a relational scaffold that linearizes the prediction problem and collapses the optimization landscape’s condition number. The geometric consequence—consistent gradient directions—allows a tiny neural network to achieve state‑of‑the‑art accuracy under extreme resource constraints, as demonstrated on the C‑MAPSS benchmark. ODSA is domain‑agnostic, scale‑invariant, regime‑invariant, and model‑invariant, and it scales horizontally to multi‑domain generalist models. It represents a practical, mathematically grounded path to frugal, interpretable, and trustworthy AI for the physical world.

---

## References
[1] Concas, M. *A Unified Theory of Computational Waste: Ontometry and Relational Calculus.* Zenodo, 2026.  
[2] Saxena, A., et al. *C‑MAPSS Turbofan Engine Degradation Simulation Data Set.* NASA Ames Prognostics Data Repository, 2008.

---

## Appendix: Reproducibility
