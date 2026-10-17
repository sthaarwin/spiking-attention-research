# Multiplication-Free Linear Spiking Attention

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![Neuromorphic: SpikingJelly](https://img.shields.io/badge/Neuromorphic-SpikingJelly-brightgreen.svg)](https://spikingjelly.readthedocs.io/)

> **Replacing Softmax with Dynamic Threshold Gating in Linear Neuromorphic Transformers**  
> Empirical research codebase and reproducible benchmark suite for multiplication-free linear spiking attention.

---

## 🔬 Core Research Hypothesis

Standard Vision Transformers rely on scaled dot-product attention requiring quadratic token interaction $\mathcal{O}(N^2)$ and dense multiply-accumulate operations. This project investigates whether neuronal dynamics can replace explicit softmax normalization while retaining attention selectivity:

1. **Linear Associative Attention**: Reorder computation as $Q(K^T V)$ to eliminate quadratic token-pair matrix construction.
2. **Adaptive Threshold Gating**: Dynamic LIF threshold based on local activation statistics:
   $$\theta_d = \alpha \cdot \text{Mean}(U) + \beta$$
3. **Multiplication-Free Interaction**: Represent intermediate activations as binary spikes and implement attention interaction using additive spike accumulation.

---

## 📁 Repository Structure

```text
spiking-attention-research/
├── configs/                     # Experiment configurations (CIFAR-10, CIFAR-100, Events)
├── models/                      # Attention mechanisms and spiking backbones
│   ├── attention_vanilla.py     # Conventional softmax attention baseline
│   ├── attention_ssa.py         # Spikformer Spiking Self-Attention baseline
│   ├── attention_linear.py      # Unnormalized associative linear attention control
│   ├── attention_proposed.py   # Proposed linear attention + adaptive threshold LIF
│   ├── neurons.py               # Standard LIF and Adaptive Threshold LIF neurons
│   └── backbones.py             # Spiking Vision Transformer architecture
├── utils/                       # Metrics, profiling, and instrumentation
│   ├── energy_meter.py          # SOPs and theoretical energy accounting
│   ├── entropy_tracker.py       # Activation entropy and selectivity metrics
│   ├── firing_rate.py           # Layer-wise spike rate tracking
│   ├── complexity.py            # FLOPs and computational scaling analysis
│   └── reproducibility.py       # Deterministic seeds and environment logging
├── experiments/                 # Factorial ablations and benchmark runners
├── notebooks/                   # Kaggle / Colab GPU execution notebooks
├── runner.py                    # Autonomous multi-agent harness loop engine
├── ROADMAP.md                   # Complete 12-week research plan and milestone targets
└── README.md
```

---

## 🚀 Quickstart

### 1. Harness Loop Setup
This repository is configured with the `harness-loop` autonomous agent engineering engine:

```bash
# Setup multi-agent configuration
python runner.py --setup

# Run autonomous loop
python runner.py
```

### 2. Manual Testing
```bash
python3 -m unittest discover tests
```

---

## 📜 Research Roadmap & Milestones

See [ROADMAP.md](ROADMAP.md) for detailed experimental designs, 2×2 factorial ablations, theoretical energy models, and expected failure modes.
