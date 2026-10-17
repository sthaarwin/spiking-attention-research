# Research Roadmap: Multiplication-Free Linear Spiking Attention

**Working Title:** Multiplication-Free Spiking Attention: Replacing Softmax with Dynamic Threshold Gating in Linear Neuromorphic Transformers  
**Domain:** Neuromorphic Computing / Spiking Neural Networks (SNNs) / Efficient Transformers  
**Primary Deliverable:** Research manuscript + reproducible implementation

---

## 1. Problem & Core Hypothesis

### 1.1 The Bottleneck

Standard Vision Transformers rely on scaled dot-product attention:

\[
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

This creates several challenges for neuromorphic and resource-constrained systems:

1. **Arithmetic cost:** Conventional attention relies heavily on dense multiply-accumulate operations.
2. **Quadratic token interaction:** Explicit \(QK^T\) creates \(O(N^2)\) token-pair interactions.
3. **Softmax dependence:** Softmax produces dense floating-point attention weights that do not naturally map to sparse spike-based computation.
4. **Representation stability:** Simply removing softmax can produce poorly normalized activations, weak token selectivity, or neuron saturation.

### 1.2 Proposed Mechanism

The project investigates whether neuronal dynamics can replace explicit softmax normalization while retaining useful attention selectivity.

The proposed attention block combines:

- **Linear associative attention:** Reorder the computation as

\[
Q(K^TV)
\]

to avoid explicitly constructing the \(N\times N\) attention matrix.

- **Adaptive threshold gating:** Use a dynamic LIF threshold based on local activation statistics:

\[
\theta_d=\alpha\cdot Mean(U)+\beta
\]

- **Spike-based interaction:** Represent intermediate activations as sparse/binary spikes and use additive accumulation in the attention interaction.

### 1.3 Precise Research Claim

The primary claim is intentionally narrower than "the entire network is multiplication-free":

> **The proposed method aims to eliminate multiplicative attention interaction while preserving attention selectivity and competitive recognition accuracy.**

Learned input/projection layers may still contain multiplications unless an explicitly multiplication-free projection mechanism is separately implemented.

### 1.4 Central Research Question

> Can adaptive neuronal threshold dynamics provide the normalization and selectivity of softmax attention while allowing the attention interaction itself to be implemented using additive spike accumulation?

### 1.5 Hypotheses

**H1 — Accuracy:** The proposed method can approach the accuracy of a Spikformer-style attention baseline.

**H2 — Efficiency:** Linear associative attention reduces the asymptotic token-interaction complexity from \(O(N^2)\) to \(O(N)\), subject to the dimensionality and implementation of the reordered computation.

**H3 — Selectivity:** Adaptive thresholding produces more discriminative activation/attention behavior than an unnormalized linear attention control.

**H4 — Sparsity:** The proposed mechanism can maintain a healthy firing rate without widespread dead or saturated layers.

**H5 — Computational cost:** The attention interaction requires substantially fewer multiply operations than conventional attention, with energy savings reported as theoretical estimates unless hardware measurements are available.

---

## 2. Scope and Terminology

### 2.1 What "Multiplication-Free" Means

The project should distinguish between:

- **Multiplication-free attention interaction:** No multiplication is used in the proposed attention aggregation itself.
- **Multiplication-free network:** No multiplication anywhere in projections, normalization, embedding, classifier, etc.

The primary project targets the first claim.

### 2.2 What "Linear Attention" Means

The project uses the associative rearrangement:

\[
Q(K^TV)
\]

instead of explicitly calculating:

\[
(QK^T)V
\]

The implementation and complexity analysis must explicitly account for the dimensions of \(Q\), \(K\), and \(V\), rather than claiming \(O(N)\) solely from the algebraic rearrangement.

### 2.3 Energy Accounting

Energy numbers derived from assumed pJ/op values are **theoretical computational estimates**, not direct measurements of silicon power consumption.

Actual hardware energy should only be claimed if measured on a target accelerator, neuromorphic processor, FPGA, or other physical platform.

---

## 3. Directory Structure

```text
spiking-attention-research/
├── configs/
│   ├── cifar10_baseline.yaml
│   ├── cifar10_proposed.yaml
│   ├── cifar100.yaml
│   └── event.yaml
│
├── models/
│   ├── __init__.py
│   ├── backbones.py
│   ├── attention_vanilla.py
│   ├── attention_ssa.py
│   ├── attention_linear.py
│   ├── attention_proposed.py
│   └── neurons.py
│
├── utils/
│   ├── __init__.py
│   ├── energy_meter.py
│   ├── entropy_tracker.py
│   ├── firing_rate.py
│   ├── complexity.py
│   ├── reproducibility.py
│   └── datasets.py
│
├── experiments/
│   ├── ablation_threshold/
│   ├── ablation_attention/
│   ├── complexity/
│   └── event_vision/
│
├── notebooks/
│   └── kaggle_runner.ipynb
│
├── checkpoints/
├── results/
├── train.py
├── evaluate.py
├── profile.py
├── README.md
└── ROADMAP.md
```

---

# 4. Experimental Design

## 4.1 Core Controlled Comparison

Train the following four models under matched conditions:

### Run 1 — Floating-Point Reference

Small Vision Transformer with conventional softmax attention.

**Purpose:** Establish a conventional attention reference.

### Run 2 — SOTA/Primary SNN Baseline

Standard Spikformer-style Spiking Self-Attention.

**Purpose:** Establish the primary SNN comparison.

### Run 3 — Negative Control

Linear/unnormalized spike attention without adaptive threshold normalization.

**Purpose:** Test whether simply removing softmax is sufficient.

### Run 4 — Proposed Method

Linear Spike Attention + Adaptive Threshold LIF.

**Purpose:** Test the complete hypothesis.

---

## 4.2 Critical Threshold Ablation

Add a dedicated factorial ablation to isolate the source of improvement:

| Variant | Linear Attention | Adaptive Threshold |
|---|---:|---:|
| A | No | No |
| B | Yes | No |
| C | No | Yes |
| D | Yes | Yes |

The goal is to determine whether observed behavior comes from:

1. linear associative attention,
2. adaptive thresholding,
3. or the interaction between both mechanisms.

---

## 4.3 Threshold Sensitivity

Evaluate:

- fixed threshold,
- adaptive threshold,
- different values of \(\alpha\),
- different values of \(\beta\),
- different target firing-rate regimes.

Track:

- accuracy,
- firing rate,
- activation distribution,
- entropy/selectivity,
- gradient stability.

---

# 5. Phased 12-Week Implementation Plan

## Phase 1 — Baseline Verification (Weeks 1–2)

- [ ] Initialize Git repository.
- [ ] Set up Python + PyTorch + SpikingJelly.
- [ ] Verify CPU execution.
- [ ] Set up Kaggle GPU workflow.
- [ ] Implement deterministic/reproducible experiment configuration.
- [ ] Run a 20-epoch sanity run of a small Spikformer baseline on CIFAR-10.
- [ ] Record training curves and final validation accuracy.

### Milestone 1

Baseline pipeline trains successfully and produces reproducible results.

---

## Phase 2 — Proposed Attention Implementation (Weeks 3–4)

- [ ] Implement associative linear attention.
- [ ] Implement ThresholdGatedSpikingAttention.
- [ ] Implement:

\[
\theta_d=\alpha Mean(U)+\beta
\]

- [ ] Verify tensor shape:

```text
[T, B, N, D] → [T, B, N, D]
```

- [ ] Verify non-zero gradients.
- [ ] Verify gradients for \(\alpha\) and \(\beta\).
- [ ] Verify neuron state reset between batches.
- [ ] Add firing-rate instrumentation.
- [ ] Add operation-count instrumentation.

### Milestone 2

The proposed attention block trains without numerical instability and maintains measurable spike activity.

---

## Phase 3 — Controlled Ablations (Weeks 5–6)

Train the four core variants under identical experimental conditions:

- [ ] Conventional softmax reference.
- [ ] Spikformer SSA.
- [ ] Unnormalized/negative-control attention.
- [ ] Proposed linear + adaptive-threshold attention.

Suggested initial configuration:

- CIFAR-10
- \(T=4\) timesteps
- AdamW
- cosine learning-rate schedule
- matched model size
- matched training epochs
- fixed random seeds

### Milestone 3

Obtain a complete comparison of:

- Top-1 accuracy
- training stability
- firing rate
- operation counts
- parameter count
- inference/runtime measurements where available.

---

## Phase 4 — Mechanism Ablation (Weeks 6–7)

Run the 2×2 factorial experiment:

- [ ] Standard attention + fixed threshold.
- [ ] Linear attention + fixed threshold.
- [ ] Standard attention + adaptive threshold.
- [ ] Linear attention + adaptive threshold.

Also test:

- [ ] threshold sensitivity,
- [ ] firing-rate sensitivity,
- [ ] entropy/selectivity behavior.

### Milestone 4

Determine whether adaptive thresholding provides measurable benefits independently of linear attention.

---

## Phase 5 — Generalization & Event-Based Vision (Weeks 8–9)

Evaluate the strongest controlled variants on:

- [ ] CIFAR-100.
- [ ] CIFAR10-DVS or DVS128-Gesture.

Measure:

- Top-1 accuracy,
- firing rate,
- temporal behavior,
- operation count,
- robustness across timesteps.

### Milestone 5

Establish whether the proposed mechanism remains useful beyond CIFAR-10 frame-based classification.

---

## Phase 6 — Efficiency & Analytical Profiling (Weeks 9–10)

Implement operation accounting:

\[
SOPs =
\sum_l\sum_t
\sum_i
1_{S_l[t,i]=1}
\cdot FanOut_l
\]

Track separately:

- additions,
- multiplications,
- spike operations,
- projection operations,
- attention interaction operations,
- total FLOPs,
- estimated energy.

### Theoretical Energy Model

If using assumed operation-energy values:

\[
E_{estimated}
=
SOPs\cdot E_{AC}
+
FLOPs\cdot E_{MAC}
\]

Document all assumptions and technology sources.

### Important

Do not report theoretical pJ calculations as measured hardware energy.

### Milestone 6

Produce publication-quality efficiency and complexity plots.

---

## Phase 7 — Manuscript & Reproducibility (Weeks 11–12)

- [ ] Write Introduction.
- [ ] Write Related Work.
- [ ] Write Mathematical Formulation.
- [ ] Explain the adaptive threshold mechanism.
- [ ] Document complexity analysis.
- [ ] Document experimental setup.
- [ ] Write ablation results.
- [ ] Write limitations.
- [ ] Prepare figures and tables.
- [ ] Add reproducibility instructions.
- [ ] Release configuration files.
- [ ] Release checkpoints where licensing permits.
- [ ] Prepare final manuscript.

### Milestone 7

Complete reproducible research package and manuscript draft.

---

# 6. Target Metrics

| Metric | Target / Question | Verification |
|---|---|---|
| Top-1 Accuracy | Approach primary SNN baseline | `evaluate.py` |
| Accuracy Gap | Report absolute difference, not only pass/fail | `evaluate.py` |
| Firing Rate | Avoid dead/saturated layers | `firing_rate.py` |
| Attention Interaction | Minimize/remove multiplications | `energy_meter.py` |
| Complexity | Verify scaling experimentally and analytically | `complexity.py` |
| Parameter Count | Report matched model size | model summary |
| FLOPs | Compare all variants | profiler |
| SOPs | Count active spike operations | `energy_meter.py` |
| Estimated Energy | Report theoretical estimate with assumptions | energy analysis |
| Entropy | Analyze selectivity/distribution | `entropy_tracker.py` |
| Gradient Stability | Verify trainability | training logs |
| Temporal Robustness | Compare across timesteps | event experiments |

---

# 7. Complexity Analysis

Let:

- \(N\) = number of tokens,
- \(D\) = embedding dimension.

Conventional attention explicitly forms:

\[
QK^T
\]

which involves approximately:

\[
O(N^2D)
\]

token-interaction computation.

The associative formulation:

\[
Q(K^TV)
\]

changes the order of multiplication/accumulation and avoids constructing the \(N\times N\) attention matrix.

However, the final complexity should be reported using the actual dimensions:

\[
O(ND^2 + ND^2)
\]

or the exact cost derived from the implemented tensor shapes.

**Do not claim unconditional \(O(N)\) complexity without specifying which dimensions are treated as constants.**

---

# 8. Attention Selectivity Analysis

One of the central questions is whether adaptive thresholding can replace some of the functional role normally provided by softmax.

Track:

### Activation entropy

\[
H(P)=-\sum_i P_i\log P_i
\]

where \(P\) is a normalized representation used solely for analysis.

### Firing-rate distribution

Measure:

\[
R_l =
\frac{\text{number of active spikes}}
{\text{number of possible spikes}}
\]

per layer and timestep.

### Selectivity

Compare:

- token activation variance,
- entropy,
- top-k activation concentration,
- firing-rate distributions.

The objective is not necessarily to minimize entropy. Instead, determine whether the model preserves **useful discriminative structure** without collapsing into either uniform activation or saturation.

---

# 9. Required Figures

The final research package should aim to contain:

### Figure 1 — Architecture

```text
Input
  ↓
Spiking Patch / Token Encoding
  ↓
Q / K / V
  ↓
Linear Associative Attention
  ↓
Adaptive Threshold LIF
  ↓
Spiking Transformer Block
  ↓
Classifier
```

### Figure 2 — Conventional vs Proposed Attention

Show:

```text
Softmax:
Q → QKᵀ → Softmax → V

Proposed:
Q → KᵀV → Q(·) → Adaptive Spike Gate
```

### Figure 3 — Accuracy Comparison

Compare the four primary variants.

### Figure 4 — Firing Rate

Layer-by-layer firing rate distribution.

### Figure 5 — Complexity

Runtime/operation scaling against token count \(N\).

### Figure 6 — Operation Breakdown

Show:

- MACs,
- ACs,
- SOPs,
- attention operations.

### Figure 7 — Entropy / Selectivity

Compare activation selectivity between:

- softmax,
- Spikformer SSA,
- negative control,
- proposed method.

### Figure 8 — Threshold Ablation

Show the effect of \(\alpha\), \(\beta\), and adaptive vs fixed thresholds.

---

# 10. Expected Failure Modes

A successful research project does not require every hypothesis to be confirmed.

Document failures such as:

### Failure A — Accuracy collapse

Possible interpretation:

The adaptive threshold does not preserve enough normalization/selectivity.

### Failure B — Firing-rate explosion

Possible interpretation:

Threshold adaptation is insufficient to stabilize membrane dynamics.

### Failure C — Dead neurons

Possible interpretation:

Threshold becomes too restrictive.

### Failure D — No efficiency improvement

Possible interpretation:

Projection layers, memory movement, or implementation overhead dominate the attention savings.

### Failure E — Linear attention does not scale as expected

Possible interpretation:

The theoretical associative formulation does not translate directly into practical runtime advantages for the tested dimensions.

These outcomes are still scientifically useful if properly controlled and analyzed.

---

# 11. Reproducibility Requirements

Every reported experiment should record:

- random seed,
- dataset version,
- model configuration,
- number of parameters,
- timestep count,
- optimizer,
- learning rate,
- scheduler,
- batch size,
- number of epochs,
- hardware,
- software versions,
- training duration,
- checkpoint,
- evaluation script.

Use identical settings for controlled comparisons wherever possible.

---

# 12. Publication Positioning

Potential venues should be treated as targets rather than guarantees.

### Primary categories

- ML conferences focused on efficient/deep learning.
- Neuromorphic computing venues.
- IEEE neural-network and signal-processing venues.

The manuscript should emphasize the **scientific question and empirical evidence**, not the prestige of the venue.

---

# 13. Final Research Narrative

The project should ultimately tell this story:

### Problem

Softmax attention is difficult to map efficiently onto sparse neuromorphic computation and explicit attention has unfavorable token-interaction scaling.

### Observation

Simply removing softmax can destroy normalization and attention selectivity.

### Hypothesis

Adaptive neuronal thresholds may provide a biologically compatible gating mechanism that replaces part of softmax's functional role.

### Method

Combine:

1. associative linear attention,
2. spike-based representations,
3. adaptive LIF thresholding.

### Evidence

Demonstrate:

- competitive classification accuracy,
- stable firing behavior,
- reduced attention multiplication,
- favorable operation scaling,
- preserved activation selectivity,
- generalization to event-based data.

### Limitation

The method does not automatically make the entire neural network multiplication-free. Projection and embedding operations must be accounted for separately.

### Contribution

A carefully controlled empirical study of whether **adaptive spiking threshold dynamics can serve as a practical normalization/gating mechanism for efficient linear attention.**

---

# 14. 24-Hour Prototype Plan

Before committing to the full 12-week roadmap, build the smallest possible proof of concept.

### Hours 1–4

Implement:

```text
Q, K, V
  ↓
KᵀV
  ↓
Q(KᵀV)
```

with small synthetic tensors.

### Hours 4–8

Implement fixed-threshold and adaptive-threshold LIF gating.

### Hours 8–12

Create the four-way threshold/attention ablation.

### Hours 12–18

Train on a small CIFAR-10 subset.

Record:

- loss,
- accuracy,
- firing rate,
- gradient magnitude.

### Hours 18–24

Generate:

- activation plots,
- firing-rate plots,
- operation counts,
- initial comparison table.

If the proposed mechanism cannot learn a simple classification problem, stop and debug the mechanism before scaling to a full Spikformer.

---

# 15. Creativity Exercise — SCAMPER

Use SCAMPER to generate possible extensions:

**Substitute:**  
Replace adaptive mean thresholding with percentile-based or membrane-statistic thresholding.

**Combine:**  
Combine threshold adaptation with temporal spike accumulation.

**Adapt:**  
Adapt the threshold independently per head or per token group.

**Modify:**  
Use a learnable nonlinear threshold function.

**Put to another use:**  
Apply the mechanism to event-based gesture recognition rather than image classification.

**Eliminate:**  
Remove explicit normalization entirely and test whether neuronal dynamics alone stabilize attention.

**Reverse:**  
Instead of changing the threshold according to activation magnitude, investigate whether activation scaling can be controlled by the recent firing history.

---

# 16. Definition of Success

The project should **not** depend on achieving a predetermined accuracy improvement.

A scientifically successful outcome is:

> A reproducible experiment that determines whether adaptive threshold gating can preserve useful selectivity in multiplication-reduced linear spiking attention, together with a rigorous accounting of accuracy, sparsity, computational complexity, and estimated energy.

A positive result strengthens the proposed mechanism.

A negative result identifies where the hypothesis fails and provides evidence for future research.

---

## Current Priority

**Do not start with the full 12-week implementation.**

First establish this minimal chain:

```text
Linear Attention
       ↓
Adaptive Threshold
       ↓
Spike Output
       ↓
Learning
       ↓
Measurable Selectivity
```

Once that works, scale to the complete Spikformer architecture.
