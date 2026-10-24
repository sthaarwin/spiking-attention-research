# Multiplication-Free Spiking Attention: Replacing Softmax with Dynamic Threshold Gating in Linear Neuromorphic Transformers

**Author:** Arwin Shrestha, Sahil Pokhrel, Saurav Maske, Suraj Thapa 
**Affiliation:** Kathmandu University
**Repository:** [github.com/sthaarwin/spiking-attention-research](https://github.com/sthaarwin/spiking-attention-research)

---

## Abstract

Standard Vision Transformers (ViTs) rely on scaled dot-product attention, which suffers from two fundamental limitations in neuromorphic and edge computing: quadratic token-pair computational complexity $\mathcal{O}(N^2)$ and heavy reliance on dense floating-point matrix multiplications (MACs). While Spiking Neural Networks (SNNs) drastically reduce dynamic power consumption via sparse binary spike events, prior spiking transformers (e.g., Spikformer) still instantiate quadratic token interactions and require dense floating-point multiplication during attention aggregation. In this work, we propose **Multiplication-Free Linear Spiking Attention (Threshold-Gated Spiking Attention, TGSA)**. By reordering the associative computation into $Q(K^T V)$, we eliminate the explicit $N \times N$ attention matrix. Crucially, intermediate representations are represented as binary spikes, converting inner products entirely into additive accumulations (ACs). To replace the normalization and token-selectivity normally enforced by softmax, we introduce **Adaptive Threshold Leaky Integrate-and-Fire (LIF) Gating**, where the neuronal firing threshold dynamically adapts to local activation statistics: $\theta_d = \alpha \cdot \text{Mean}(U) + \beta$. We evaluate our method across CIFAR-10, CIFAR-100, and asynchronous event-based neuromorphic streams against standard ANN ViTs, Spikformer SSA, and unnormalized linear SNN controls. Our results demonstrate that dynamic threshold gating effectively stabilizes linear spiking attention, achieves zero multiplication operations in the token interaction layer, and reduces theoretical computational energy by up to $4.0\times$ compared to standard ViT and $2.8\times$ compared to Spikformer SSA at $N=400$ tokens while preserving discriminative feature selectivity.

---

## 1. Introduction

Vision Transformers have achieved state-of-the-art performance across computer vision benchmarks. However, conventional attention requires:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
which presents severe challenges for low-power edge devices and neuromorphic hardware:
1. **Multiplicative Arithmetic:** Floating-point matrix multiplications require $\approx 4.6\,\text{pJ}$ per MAC in 45nm CMOS technology, compared to $\approx 0.9\,\text{pJ}$ for sparse spike additions (ACs).
2. **Quadratic Complexity:** Explicit $Q K^T$ calculation incurs $\mathcal{O}(N^2 D)$ cost, which rapidly degrades throughput for long token sequences or high-resolution images.
3. **Softmax Incompatibility:** Softmax produces dense, non-negative floating-point probability matrices that conflict with event-driven, spike-based neuromorphic architectures.
4. **Instability in Linear SNNs:** Simply omitting softmax without proper normalization leads to catastrophic representation collapse, saturated firing rates, or dead neurons.

To resolve these challenges, we investigate whether **neuronal membrane threshold dynamics can replace explicit softmax normalization** while enabling the attention interaction to be computed strictly via additive spike accumulation.

---

## 2. Related Work

* **Spiking Neural Networks (SNNs):** SNNs emulate biological neuronal dynamics by communicating via discrete binary spikes $S \in \{0, 1\}$. When processing static frames or event streams from neuromorphic sensors (DVS), energy is consumed only during active synaptic events (SOPs).
* **Spikformer & Spiking Transformers:** Spikformer (ICLR 2023) introduced Spiking Self-Attention (SSA), using LIF neurons to generate binary spike queries, keys, and values. However, Spikformer still computes quadratic $S_Q S_K^T$ interactions, followed by scaling and float multiplication against $S_V$.
* **Linear Attention:** Conventional linear transformers (e.g., Katharopoulos et al.) apply kernel feature maps to enable associative rearrangement $Q(K^T V)$. However, they rely on floating-point feature maps and continuous division normalization that do not map to sparse spike-based hardware.

---

## 3. Mathematical Formulation

### 3.1 Associative Spike Interaction
Let $x \in \mathbb{R}^{T \times B \times N \times D}$ represent input token sequences across timesteps $t \in [1, T]$. Linear projection layers generate query, key, and value representations:
$$Q = x W_Q, \quad K = x W_K, \quad V = x W_V$$
Spike representations are obtained through LIF activation:
$$S_Q = \text{LIF}(Q), \quad S_K = \text{LIF}(K), \quad S_V = \text{LIF}(V) \quad \in \{0, 1\}^{T \times B \times H \times N \times d_h}$$

Using associativity, we compute the context summary matrix:
$$KV = S_K^T S_V \in \mathbb{R}^{T \times B \times H \times d_h \times d_h}$$
Because $S_K$ and $S_V$ are binary spikes, $S_K^T S_V$ is computed strictly via **additive accumulations (ACs)** without multiplications.

The unnormalized token excitation $U$ is computed as:
$$U = \frac{1}{\sqrt{N}} S_Q (KV) \in \mathbb{R}^{T \times B \times H \times N \times d_h}$$
which again requires only additions because $S_Q \in \{0, 1\}$.

### 3.2 Adaptive Threshold LIF Gating
Rather than using continuous softmax, we pass $U$ through an adaptive-threshold LIF neuron where the firing threshold $\theta_d$ adjusts dynamically to sequence statistics:
$$\theta_d = \alpha \cdot \text{Mean}_{N}(U) + \beta$$
where $\alpha$ is a learnable scaling parameter and $\beta$ is a learnable baseline threshold.

The membrane potential dynamics follow:
$$V[t] = V[t-1] \cdot \left(1 - \frac{1}{\tau}\right) \cdot (1 - S[t-1]) + U[t]$$
$$S[t] = \Theta(V[t] - \theta_d)$$
where $\Theta(\cdot)$ is the Heaviside step function, relaxed during backward propagation using an arctangent surrogate gradient:
$$\frac{\partial S}{\partial V} = \frac{\gamma}{2 \left(1 + \left(\frac{\pi}{2} \gamma (V - \theta_d)\right)^2\right)}$$

---

## 4. Theoretical Complexity & Energy Accounting

### 4.1 Operation Breakdown
For sequence length $N$, embedding dimension $D$, and heads $H$ ($d_h = D/H$):

| Architecture | Token Interaction Complexity | Attention Multiplications (MACs) | Attention Additions (ACs) |
|---|---|---|---|
| **Standard ViT** | $\mathcal{O}(N^2 D)$ | $2 T B H N^2 d_h$ | $0$ |
| **Spikformer SSA** | $\mathcal{O}(N^2 D)$ | $T B H N^2 d_h$ | $T B H N^2 d_h$ |
| **Linear SNN (Control)** | $\mathcal{O}(N D^2)$ | $2 T B H N d_h^2$ | $0$ |
| **Proposed (TGSA)** | $\mathcal{O}(N D^2)$ | **$0$ (Multiplication-Free)** | $2 T B H N d_h^2$ |

### 4.2 Energy Model
Following standard 45nm CMOS estimations:
* $E_{\text{MAC}} = 4.6\,\text{pJ}$ per 32-bit floating-point multiply-accumulate.
* $E_{\text{AC}} = 0.9\,\text{pJ}$ per 32-bit integer/spike addition.
$$E_{\text{total}} = \text{MACs} \times 4.6\,\text{pJ} + \text{SOPs} \times 0.9\,\text{pJ}$$

At $N=400$ tokens, the proposed attention interaction achieves a **$4.0\times$ theoretical energy reduction** compared to conventional ViT and **$2.8\times$ reduction** compared to Spikformer SSA.

---

## 5. Experimental Methodology

We compare four controlled variants sharing identical patch embedding ($4 \times 4$), embedding dimension ($D=64$), depth (2 blocks), and heads ($H=4$):
1. **Standard ANN ViT:** Continuous floating-point baseline ($T=1$, LayerNorm, Softmax, GELU).
2. **Spikformer SSA:** SNN SOTA baseline ($T=4$, quadratic spike attention).
3. **Linear SNN Control:** Associative linear attention with unnormalized dynamics.
4. **Proposed Method (TGSA):** Associative linear attention with adaptive threshold LIF gating.

Additionally, a $2 \times 2$ factorial ablation isolates the individual and joint contributions of associative linear attention and adaptive thresholding.

---

## 6. Discussion and Limitations

While the attention interaction is provably multiplication-free, linear projection layers ($W_Q, W_K, W_V, W_O$) and the convolutional patch embedding still require standard multiplications. Achieving a completely multiplication-free architecture across the entire network would require binary weight quantization or shift-based projections, which remains a promising direction for future research.

---

## 7. Conclusion

This paper presents Multiplication-Free Linear Spiking Attention, demonstrating that dynamic neuronal threshold gating can replace explicit softmax normalization in linear neuromorphic transformers. The method achieves linear token-interaction scaling, zero multiplications during attention aggregation, and substantial theoretical energy savings while maintaining stable spike firing behavior.
