# The Green AI Challenge: C-MAPSS Computational Efficiency Target
## Performance/Compute as the True North for Real‑World Prognostics

### 1. Core Philosophy
The dominant metric in machine health monitoring has been raw predictive accuracy. This has led to models with millions of parameters, 100+ engineered features, and multi‑GPU inference pipelines—solutions that are physically impossible to deploy on the low‑cost microcontrollers that actually sit next to industrial assets.

This challenge flips the metric. We do not ask “Can you get the lowest possible error?”
We ask: “Can you reach a clearly defined, operationally useful accuracy floor while consuming three orders of magnitude less compute and energy than the typical brute‑force approach?”

The measure of success is performance per unit computational cost, evaluated under a set of non‑negotiable physical constraints that mirror real‑world edge deployment. A solution that passes all gates is immediately transferable to wind turbines, diesel generators, robotic actuators, and any other rotating machinery where only a tiny sensor subset, a coin‑sized chip, and a single invariant model are available.

### 2. Benchmark Data & Operational Challenge
We use the official NASA C-MAPSS Turbofan Engine Degradation Simulation (FD001–FD004), provided by the NASA Ames Prognostics Center of Excellence.

* **FD001:** single operating condition, one fault mode.
* **FD002:** six operating conditions (sea level → 40 000 ft, varying Mach), one fault mode.
* **FD003:** single operating condition, two fault modes.
* **FD004:** six operating conditions, two fault modes.

The data reflect a true multi‑regime, multi‑fault environment with altitude‑dependent sensor shifts, throttle transients, and overlapping degradation trajectories. The primary challenge is to produce one single model that works reliably across all these regimes simultaneously.

### 3. Mandatory Constraints (No Exceptions, No Workarounds)

#### 3.1 Fleet‑Wide Unification
The ultimate binding constraint for challenge success is fleet-wide unification. Edge devices cannot swap models per operating point. However, for empirical validation against the official baseline (see Section 7), the evaluation script will run two distinct training protocols side-by-side[cite: 1]. To qualify, the submitted architecture must pass the challenge gates strictly under the unified protocol.

#### 3.2 Sensor Budget
You may read at most 11 of the 21 sensor measurement columns (s1–s21).
The three operational settings (op1, op2, op3) are always available and do not count toward this budget.
Any feature derived from the allowed sensors must obey the online‑computation rule below.

#### 3.3 Online Feature Computation Only
Every input feature to the model must be computable sample‑by‑sample in a streaming, causal manner using only:

* the current raw sensor values and operating settings,
* a fixed‑size history buffer of past raw readings (buffer length ≤ 30 time steps),
* simple arithmetic, exponential moving averages, or finite differences,
* constants that are either known physical constants fixed a priori or trainable parameters that count against the total budgets.

Strictly forbidden:

* Any statistic that requires a full dataset scan (e.g., global median, per‑regime baselines computed offline), principal components, or any non‑causal transformation.
* Any use of future time steps or the RUL ground truth at inference time.

#### 3.4 Parameter Budget
≤ 32 000 trainable parameters (weights + biases + any learnable scaling factors, embedding vectors, or constants used during feature computation). This includes every scalar that receives a gradient update during training.

#### 3.5 Deployment Storage
The total size of everything that must reside in flash memory for inference must be < 130 KB in full float32 precision. No quantization, no weight sharing, no compressed formats—every stored value counts as a float32. The 130 KB limit is absolute.

#### 3.6 Inference Compute
< 0.7 MegaFLOPs per prediction (window‑to‑RUL).
A floating‑point multiply‑add counts as 2 operations. The budget must cover all arithmetic inside the model forward pass plus any arithmetic performed in the online feature computation step.

#### 3.7 Training Budget
The entire end‑to‑end procedure—from raw `.txt` files to the final model checkpoint—must complete in under 5 minutes wall‑clock time per individual training run (seed)[cite: 1]. The reference machine for this limit is an Apple M1 (or equivalent modern x86 laptop) with 8 GB RAM, no GPU acceleration, and Python ≥3.9. 

The training must not exceed 50 total passes (epochs) over the dataset[cite: 1]. 

### 4. Target Accuracy (The Operational Floor)
The model must achieve all of the following scores on the official NASA scoring function (asymmetric, late‑prediction‑penalizing) on the respective test sets. A single model that fails even one dataset does not qualify.

| Dataset | Target NASA Score | Context |
| :--- | :--- | :--- |
| **FD001** | < 470 | Single condition, single fault. Trivial for a large model; a test of whether the tiny model retains minimal regression ability. |
| **FD002** | < 3 200 | Six operating conditions, single fault. The large flight‑envelope span introduces strong regime shifts that typically inflate error. |
| **FD003** | < 1 200 | Single condition, two faults. Dual degradation modes demand that the model disentangle competing failure signatures. |
| **FD004** | < 4 600 | Six conditions, two faults. The binding constraint—equivalent to an RMSE of roughly 30–35 cycles on test lifetimes of 50–300 cycles. |

### 5. Official Baseline & Evaluation Protocol
To guarantee a defensible, independent control, the script must include a scaled version of the industry-standard Zheng LSTM architecture, executed under identical physical constraints[cite: 1]. 

**The Baseline Architecture:**
The control model is a 2-layer LSTM with a hidden size of 36 in both layers, followed by a single dense output unit[cite: 1]. Utilizing 11 input features and a 30-time-step window, this baseline inherently utilizes roughly 17,554 parameters and 0.66 MFLOPS per prediction, securely under the required challenge ceilings[cite: 1]. 

*Baseline Literature Reference:* S. Zheng, K. Ristovski, A. Farahat, and C. Gupta, "Long Short-Term Memory Network for Remaining Useful Life prediction," 2017 IEEE International Conference on Prognostics and Health Management (ICPHM). DOI: 10.1109/ICPHM.2017.7998311[cite: 1].

**The A/B Evaluation Protocol:**
The final script must strictly enforce the following benchmark testing conditions[cite: 1]:
* **Dual Protocols:** The script must run both Protocol A (per-subset training) and Protocol B (fleet-wide unified training) side-by-side using the same architectures and budgets[cite: 1].
* **Statistical Rigor:** Each model must be executed across at least 5 random seeds[cite: 1]. The script must output the mean and standard deviation of the separate NASA scores for FD001–FD004[cite: 1].
* **Literature Context:** The script must hardcode and print the published full-Zheng model scores for each subset as a reference before initiating the empirical tests[cite: 1]. 
* **Evaluation Parameters:** All scoring must utilize the standard NASA constants $a_1=13$ and $a_2=10$, with the Remaining Useful Life (RUL) strictly clipped at 125[cite: 1].

### 6. What Constitutes a Valid Solution
A submission is only valid if it satisfies all of the following:

* A single model object (no post‑hoc weighting of sub‑models).
* Total parameter count ≤32 000, disk size <130 KB, and inference FLOPs <0.7 M.
* Training script finishes in <5 minutes per seed on the reference CPU and uses ≤50 epochs[cite: 1].
* Achieves FD001 NASA score <470 and FD004 NASA score <4 600 strictly under the fleet-wide unified protocol.
* Executes the side-by-side dual protocol baseline evaluation utilizing the established scoring metrics ($a_1=13$, $a_2=10$, RUL clipped at 125) and 5 random seeds[cite: 1].

#### Submission Format
The solution must be submitted as a single, self‑contained Python script. When executed, it must build, train, and evaluate both the proposed model and the Zheng baseline. It must print a summary that includes the number of parameters, serialized model size, inference FLOP count, and the complete matrix of NASA subset scores (mean and standard deviation) for both protocols[cite: 1].
