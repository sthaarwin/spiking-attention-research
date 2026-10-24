"""
Publication-Quality Visualization Engine for Neuromorphic Spiking Attention Research.
Generates Figures 3-8 and Pareto Tradeoff plots matching ROADMAP.md specifications.
"""

import os
from typing import Dict, List, Optional, Any
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np


# Academic paper styling
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "grid.alpha": 0.3,
})

COLORS = {
    "standard_vit": "#1f77b4",  # Blue (Dense FP32)
    "vanilla": "#1f77b4",
    "ssa": "#ff7f0e",           # Orange (Spikformer SNN)
    "linear": "#2ca02c",        # Green (Linear SNN control)
    "proposed": "#d62728",      # Red (Proposed Multiplication-Free)
}


def plot_accuracy_comparison(
    histories: Dict[str, List[Dict[str, Any]]],
    output_path: str = "results/figures/figure3_accuracy_comparison.png",
) -> str:
    """
    Figure 3: Training and Validation Accuracy trajectories over epochs
    across all primary model variants (including Standard ViT).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for model_name, records in histories.items():
        epochs = [r["epoch"] for r in records]
        train_acc = [r["train_acc"] * 100 for r in records]
        val_acc = [r["val_acc"] * 100 for r in records]
        color = COLORS.get(model_name.lower(), "#7f7f7f")

        ax1.plot(epochs, train_acc, label=model_name, color=color, linestyle="--")
        ax2.plot(epochs, val_acc, label=model_name, color=color)

    ax1.set_title("Training Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.grid(True)
    ax1.legend()

    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_firing_rate_distribution(
    layer_rates: Dict[str, float],
    output_path: str = "results/figures/figure4_firing_rates.png",
) -> str:
    """Figure 4: Layer-wise firing rate distribution bar chart."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    names = list(layer_rates.keys())
    # Shorten names for clean display
    short_names = [n.replace("blocks.", "B").replace(".attn.", ".A.") for n in names]
    rates = [r * 100 for r in layer_rates.values()]

    bars = ax.bar(range(len(rates)), rates, color="#d62728", alpha=0.85, edgecolor="black")
    ax.axhline(y=1.0, color="gray", linestyle=":", label="1% Dead Threshold")
    ax.axhline(y=95.0, color="orange", linestyle=":", label="95% Saturation Threshold")

    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels(short_names, rotation=45, ha="right")
    ax.set_ylabel("Active Firing Rate (%)")
    ax.set_title("Layer-Wise Firing Rate Distribution (Proposed Model)")
    ax.grid(axis="y")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_complexity_scaling(
    scaling_data: Dict[str, List[Dict[str, Any]]],
    output_path: str = "results/figures/figure5_complexity_scaling.png",
) -> str:
    """Figure 5: Operation scaling against token count N."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for method, rows in scaling_data.items():
        tokens = [r["tokens"] for r in rows]
        macs = [r["macs"] / 1e6 for r in rows]
        energy = [r["estimated_energy_pj"] / 1e6 for r in rows]
        color = COLORS.get(method.lower(), "#7f7f7f")

        ax1.plot(tokens, macs, marker="o", label=method.capitalize(), color=color)
        ax2.plot(tokens, energy, marker="s", label=method.capitalize(), color=color)

    ax1.set_title("Multiply-Accumulate Operations (MACs)")
    ax1.set_xlabel("Sequence Length / Tokens (N)")
    ax1.set_ylabel("MMACs ($10^6$)")
    ax1.grid(True)
    ax1.legend()

    ax2.set_title("Estimated Theoretical Energy")
    ax2.set_xlabel("Sequence Length / Tokens (N)")
    ax2.set_ylabel(r"Theoretical Energy ($\mu$J / forward)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_operation_breakdown(
    token_comparison: Dict[str, Dict[str, float]],
    output_path: str = "results/figures/figure6_operation_breakdown.png",
) -> str:
    """Figure 6: Bar chart breakdown of MACs vs ACs and Energy savings."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))

    methods = list(token_comparison.keys())
    x = np.arange(len(methods))
    width = 0.35

    macs = [token_comparison[m]["macs"] / 1e6 for m in methods]
    acs = [token_comparison[m]["acs"] / 1e6 for m in methods]

    ax.bar(x - width/2, macs, width, label="MACs ($10^6$)", color="#1f77b4", edgecolor="black")
    ax.bar(x + width/2, acs, width, label="ACs ($10^6$, Spike Additions)", color="#2ca02c", edgecolor="black")

    ax.set_ylabel("Million Operations ($10^6$)")
    ax.set_title("Computational Operation Split at $N=400$ Tokens")
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in methods])
    ax.legend()
    ax.grid(axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_accuracy_energy_pareto(
    model_results: Dict[str, Dict[str, float]],
    output_path: str = "results/figures/pareto_accuracy_energy.png",
) -> str:
    """
    Accuracy vs. Theoretical Energy Cost Pareto Frontier Tradeoff plot:
    Compares Standard ANN ViT against SNN baselines and proposed method.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))

    for name, metrics in model_results.items():
        acc = metrics["top1_acc"] * 100
        energy_uj = metrics["energy_uj"]
        color = COLORS.get(name.lower(), "#7f7f7f")
        marker = "^" if "standard" in name.lower() or "vit" in name.lower() else "o"
        size = 140 if "proposed" in name.lower() else 100

        ax.scatter(energy_uj, acc, color=color, s=size, label=name, marker=marker, edgecolors="black", zorder=4)
        ax.annotate(
            rf"  {name}" + "\n" + rf"  ({energy_uj:.1f}$\mu$J, {acc:.1f}%)",
            (energy_uj, acc),
            fontsize=9,
            va="center",
        )

    ax.set_xlabel(r"Theoretical Energy Consumption per Sample ($\mu$J) [Lower is Better]")
    ax.set_ylabel("Top-1 Classification Accuracy (%) [Higher is Better]")
    ax.set_title("Accuracy vs. Computational Energy Tradeoff (CIFAR-10)")
    ax.grid(True, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path
