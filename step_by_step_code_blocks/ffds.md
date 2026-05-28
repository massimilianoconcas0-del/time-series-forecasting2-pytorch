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

The data reflect a true multi‑regime, multi‑fault environment with altitude‑dependent sensor shifts, throttle transients, and overlapping degradation trajectories. The challenge is to produce one single model that works reliably across all these regimes simultaneously.

### 3. Mandatory Constraints (No Exceptions, No Workarounds)

#### 3.1 Fleet‑Wide Unification
One single model architecture and one set of trained weights for all four sub‑datasets. No per‑dataset fine‑tuning, no ensembles, no mixture‑of‑experts. Edge devices cannot swap models per operating point.

#### 3.2 Sensor Budget
You may read at most 11 of the 21 sensor measurement columns (s1–s21).
The three operational settings (op1, op2, op3) are always available and do not count toward this budget.
Any feature derived from the allowed sensors must obey the online‑computation rule below.

#### 3.3 Online Feature Computation Only
Every input feature to the model must be computable sample‑by‑sample in a streaming, causal manner using only:

* the current raw sensor values and operating settings,
* a fixed‑size history buffer of past raw readings (buffer length ≤ 30 time steps),
* simple arithmetic, exponential moving averages, or finite differences,
* constants that are either:
    * known physical constants fixed a priori, or
    * trainable parameters that count against the total parameter and storage budgets.

Strictly forbidden:

* Any statistic that requires a full dataset scan (e.g., global median, per‑regime baselines computed offline), principal components, or any non‑causal transformation.
* Any use of future time steps or the RUL ground truth at inference time.

#### 3.4 Parameter Budget
≤ 32 000 trainable parameters (weights + biases + any learnable scaling factors, embedding vectors, or constants used during feature computation). This includes every scalar that receives a gradient update during training.

#### 3.5 Deployment Storage
The total size of everything that must reside in flash memory for inference must be < 130 KB in full float32 precision.

This includes:

* all model weights,
* any learnable or fixed constants used in online feature computation,
* any lookup tables, normalization coefficients, or auxiliary arrays.

No quantization, no weight sharing, no compressed formats—every stored value counts as a float32 (or equivalent bit‑width without effective precision reduction). The 130 KB limit is absolute.

#### 3.6 Inference Compute
< 0.7 MegaFLOPs per prediction (window‑to‑RUL).
A floating‑point multiply‑add counts as 2 operations. The budget must cover all arithmetic inside the model forward pass plus any arithmetic performed in the online feature computation step (e.g., EMA updates, differences). Simple indexing or assignment operations are free.
Measure using a standard FLOP counter (e.g., TensorFlow Profiler, PyTorch FLOPs counter) on the final exported model, fed with a single input window.

#### 3.7 Training Budget
The entire end‑to‑end procedure—from raw `.txt` files to the final model checkpoint—must complete in under 5 minutes wall‑clock time on a reference CPU machine with the following characteristics:

* Apple M1 (or equivalent modern x86 laptop, e.g., Intel i7‑1165G7, AMD Ryzen 7 5800U),
* 8 GB RAM,
* no GPU acceleration (only CPU compute),
* Python ≥3.9, using only allowed libraries (see §6).

The training must not exceed 50 total passes (epochs) over the unified mixed dataset of FD001–FD004. Pre‑computing features on disk is allowed only if the computation itself obeys the online rule and its cost is included in the 5‑minute limit.

### 4. Target Accuracy (The Operational Floor)
The model must achieve all of the following scores on the official NASA scoring function (asymmetric, late‑prediction‑penalizing) on the respective test sets. There is no averaging, no ranking by a single metric—the model must pass each gate individually. A single model that fails even one dataset does not qualify.

| Dataset | Target NASA Score | Context |
| :--- | :--- | :--- |
| **FD001** | < 470 | Single condition, single fault. Trivial for a large model; a test of whether the tiny model retains minimal regression ability. |
| **FD002** | < 3 200 | Six operating conditions, single fault. The large flight‑envelope span introduces strong regime shifts that typically inflate error. This bound is well below typical heavy‑model benchmarks. |
| **FD003** | < 1 200 | Single condition, two faults. Dual degradation modes demand that the model disentangle competing failure signatures without dedicated fault labels. |
| **FD004** | < 4 600 | Six conditions, two faults. The binding constraint—equivalent to an RMSE of roughly 30–35 cycles on test lifetimes of 50–300 cycles. Operationally actionable for scheduling maintenance, without demanding research‑grade SOTA. |

These targets are not state‑of‑the‑art. Modern large ensembles routinely score FD001 ~200, FD002 ~2 500, FD003 ~700, and FD004 ~3 000. They are deliberately set at the threshold where a prediction becomes industrially useful—and they are all attainable by a single, tiny, genuinely green model. Surpassing them yields no extra credit; the objective is to hit them under the full computational and memory budget—not to maximise accuracy at any cost. The multi‑gate design ensures that the model cannot collapse on difficult regimes while exploiting easy ones, making the challenge a true test of unified, edge‑capable prognostics.

### 5. The True Objective: Performance / Compute
We define efficiency implicitly by the hard gates above. A solution that passes all constraints achieves:

* >95% reduction in trainable parameters,
* >97% reduction in memory footprint,
* >98% reduction in inference compute,

relative to the typical 500k‑param+ CNN‑LSTM benchmark—while still delivering actionable RUL estimates across the entire flight envelope. This outcome proves that standard brute‑force deep learning is structurally dependent on computational waste. The challenge demonstrates that a unified, physics‑aware, exquisitely efficient model can replace them, unlocking truly green, accessible, and deployable AI.

### 6. What Constitutes a Valid Solution
A submission is only valid if it satisfies all of the following:

* A single model object (no post‑hoc weighting of sub‑models).
* Reads ≤11 raw sensor channels (as per §3.2) and complies with the online‑feature rule (§3.3).
* Total parameter count ≤32 000 (including all trainable constants).
* Size on disk in float32 <130 KB (model weights + all constants needed for inference).
* Inference FLOPs per prediction <0.7 M (including feature computation arithmetic).
* Training script finishes in <5 minutes on the reference CPU and uses ≤50 epochs of the unified dataset.
* Achieves FD001 NASA score <470 and FD004 NASA score <4 600.
* No escape hatches—the solution must be fully self‑contained and reproducible.

#### Submission Format
The solution must be submitted as a single, self‑contained Python script (Python ≥3.9). When executed in a clean environment with only the allowed packages (`numpy`, `pandas`, `tensorflow>=2.12` or `pytorch>=2.0`, `scikit‑learn` for scalers only, and their standard dependencies), it must:

* Load the raw C-MAPSS `.txt` files,
* Build, train, and save the model within the time limit,
* Print a summary that includes: number of parameters, serialized model size, inference FLOP count, and the NASA scores on all four test sets.

No compiled extensions, no external data beyond the official C-MAPSS files, and no network access are permitted. The script serves as both submission and proof.
