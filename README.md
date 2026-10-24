# Multiplication-Free Linear Spiking Attention

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![Neuromorphic: SpikingJelly](https://img.shields.io/badge/Neuromorphic-SpikingJelly-brightgreen.svg)](https://spikingjelly.readthedocs.io/)
[![Tests: Pytest](https://img.shields.io/badge/Tests-20%20Passed-brightgreen.svg)](tests/)

> **Replacing Softmax with Dynamic Threshold Gating in Linear Neuromorphic Transformers**  
> Empirical research codebase, full experiment suite, and reproducible benchmarks for multiplication-free linear spiking attention compared directly against conventional Vision Transformers and Spikformer.

---

## 🔬 Core Research Hypothesis

Standard Vision Transformers rely on scaled dot-product attention requiring quadratic token interaction $\mathcal{O}(N^2)$ and dense multiply-accumulate operations. This project investigates whether neuronal dynamics can replace explicit softmax normalization while retaining attention selectivity:

1. **Linear Associative Attention**: Reorder computation as $Q(K^T V)$ to eliminate quadratic token-pair matrix construction ($\mathcal{O}(N D^2)$ vs $\mathcal{O}(N^2 D)$).
2. **Adaptive Threshold Gating**: Dynamic LIF threshold based on local activation statistics:
   $$\theta_d = \alpha \cdot \text{Mean}(U) + \beta$$
3. **Multiplication-Free Interaction**: Represent intermediate activations as binary spikes and implement attention aggregation using strictly additive spike accumulation (**$0$ MACs in attention interaction**).

---

## 📊 Core Controlled Comparison

The codebase trains and compares five matched architectures:

| Model | Paradigm | Attention Mechanism | Timesteps | Attention MACs | Energy Model |
|---|---|---|---|---|---|
| **Standard ViT** | ANN (Dense) | Scaled Dot-Product Softmax | $T=1$ | Quadratic ($2 N^2 d_h$) | Dense FP32 MACs ($4.6\,\text{pJ}$) |
| **Spikformer SSA** | SNN (Spiking) | Spiking Self-Attention | $T=4$ | Quadratic ($N^2 d_h$) | Mixed MACs & ACs |
| **Linear Control** | SNN (Spiking) | Unnormalized Associative Linear | $T=4$ | Linear ($2 N d_h^2$) | Mixed MACs & ACs |
| **Proposed (TGSA)** | SNN (Spiking) | Threshold-Gated Linear Spiking | $T=4$ | **$0$ (Multiplication-Free)** | **Pure ACs ($0.9\,\text{pJ}$)** |

---

## 📁 Repository Structure

```text
spiking-attention-research/
├── configs/                     # Experiment configurations (YAML)
│   ├── vit_standard.yaml        # Conventional ANN Vision Transformer baseline
│   ├── cifar10_baseline.yaml    # Spikformer SSA baseline
│   ├── cifar10_proposed.yaml    # Proposed threshold-gated spiking attention
│   ├── cifar100.yaml            # CIFAR-100 generalization benchmark
│   ├── event.yaml               # Neuromorphic 2-channel event vision benchmark
│   ├── ablation_linear_fixed.yaml # Factorial Variant B (Linear + Fixed Threshold)
│   └── ablation_ssa_adaptive.yaml # Factorial Variant C (SSA + Adaptive Threshold)
├── models/                      # Neural network modules and backbones
│   ├── attention_vanilla.py     # Conventional softmax attention baseline
│   ├── attention_ssa.py         # Spikformer Spiking Self-Attention baseline
│   ├── attention_linear.py      # Unnormalized associative linear attention control
│   ├── attention_proposed.py   # Proposed linear attention + adaptive threshold LIF
│   ├── neurons.py               # Standard LIF and Adaptive Threshold LIF neurons
│   └── backbones.py             # SpikingVisionTransformer & StandardVisionTransformer
├── utils/                       # Metrics, profiling, datasets, and visualization
│   ├── datasets.py              # CIFAR-10, CIFAR-100, DVS event streams, synthetic
│   ├── energy_meter.py          # SOPs and theoretical energy accounting (AC vs MAC)
│   ├── entropy_tracker.py       # Activation entropy and selectivity metrics
│   ├── firing_rate.py           # Layer-wise spike rate tracking and health checks
│   ├── complexity.py            # FLOPs, MACs, ACs, and scaling analysis
│   ├── reproducibility.py       # Deterministic seeds and environment logging
│   └── visualize.py             # Publication-quality plotting engine (Figures 3-8)
├── experiments/                 # Automated experimental ablation suites
│   ├── ablation_attention/      # 4-model comparison (Standard ViT vs SNNs)
│   ├── ablation_threshold/      # 2x2 factorial ablation & threshold sweeps
│   ├── complexity/              # Sequence length scaling analysis (N=16 to 400)
│   └── event_vision/            # Asynchronous neuromorphic event benchmarks
├── notebooks/                   # GPU execution notebooks
│   └── kaggle_runner.ipynb      # Kaggle / Colab ready-to-run GPU benchmark notebook
├── paper/                       # Research manuscript draft
│   └── manuscript.md            # Complete academic paper draft
├── checkpoints/                 # Saved best.pt and latest.pt model weights
├── results/                     # Metric JSONs, CSV summaries, and figures
│   └── figures/                 # Generated publication plots
├── train.py                     # Unified training CLI
├── evaluate.py                  # Standalone evaluation & profiling CLI
├── profile.py                   # Complexity profiling CLI
├── ROADMAP.md                   # 12-week research plan and milestone definitions
└── README.md
```

---

## 🚀 Quickstart

### 1. Installation
```bash
# Clone repository
git clone https://github.com/sthaarwin/spiking-attention-research.git
cd spiking-attention-research

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest tests/
```

### 3. Fast Local Smoke Tests
You can verify the entire pipeline locally on CPU in seconds using `--synthetic`:
```bash
# Smoke test standard ViT
python train.py --attn_type standard_vit --synthetic --epochs 1

# Smoke test proposed multiplication-free attention
python train.py --attn_type proposed --synthetic --epochs 1

# Evaluate saved checkpoint
python evaluate.py --checkpoint checkpoints/proposed_dim64_seed42/best.pt --synthetic
```

---

## 🧪 Running Experiments

### 1. Primary Attention Ablation (Standard ViT vs SNNs)
```bash
python experiments/ablation_attention/run_attention_ablation.py --epochs 20 --batch_size 64
```
Generates:
* `results/ablation_attention/comparison_table.csv`
* `results/figures/figure3_accuracy_comparison.png`
* `results/figures/pareto_accuracy_energy.png` (Accuracy vs. Theoretical Energy Cost)

### 2. 2×2 Factorial Threshold Ablation
```bash
python experiments/ablation_threshold/run_threshold_ablation.py --epochs 15 --batch_size 64
```
Isolates whether improvements originate from linear associative attention, adaptive threshold gating, or their interaction.

### 3. Complexity & Token Scaling Profiling
```bash
python experiments/complexity/run_complexity_analysis.py
```
Evaluates operation scaling across $N \in [16, 64, 144, 256, 400]$ tokens and generates:
* `results/figures/figure5_complexity_scaling.png`
* `results/figures/figure6_operation_breakdown.png`

### 4. Neuromorphic Event-Based Vision
```bash
python experiments/event_vision/run_event_experiment.py --timesteps 4 8 16 --epochs 10
```

---

## 💾 Saved Model Checkpoints

Every training run automatically archives checkpoints to both:
1. `checkpoints/<model_name>_dim<D>_seed<S>/` (`best.pt`, `latest.pt`)
2. `results/<experiment_name>/<model_name>/checkpoints/` (`best.pt`, `latest.pt`)

Each checkpoint preserves:
* Model parameter weights (`model_state_dict`)
* Optimizer state (`optimizer_state_dict`)
* Best validation accuracy (`val_acc`)
* Epoch number and complete YAML configuration dictionary

---

## 📄 Research Manuscript Draft

See [`paper/manuscript.md`](paper/manuscript.md) for the complete academic research paper draft, including formal mathematical derivations, energy models, related work, and empirical discussion.
