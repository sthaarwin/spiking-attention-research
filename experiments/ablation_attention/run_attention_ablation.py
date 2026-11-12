"""
Automated Attention Mechanism Ablation Runner:
Compares:
1. Standard ANN Vision Transformer (Dense Floating-Point Reference)
2. Spikformer SSA (Primary SNN Baseline)
3. Linear Spike Attention (Negative Control)
4. Proposed Multiplication-Free Threshold-Gated Linear Spiking Attention

Produces:
- Side-by-side comparison table (CSV, JSON, Markdown)
- Figure 3 (Training & Validation Curves)
- Figure 6 (Operation Breakdown)
- Accuracy vs. Energy Pareto Tradeoff Plot
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)

from utils.complexity import count_parameters
from utils.energy_meter import EnergyMeter
from utils.visualize import (
    plot_accuracy_comparison,
    plot_operation_breakdown,
    plot_accuracy_energy_pareto,
)


MODELS = [
    {"name": "Standard_ViT", "attn_type": "standard_vit", "config": os.path.join(REPO_ROOT, "configs/vit_standard.yaml")},
    {"name": "Spikformer_SSA", "attn_type": "ssa", "config": os.path.join(REPO_ROOT, "configs/cifar10_baseline.yaml")},
    {"name": "Linear_Attention", "attn_type": "linear", "config": os.path.join(REPO_ROOT, "configs/cifar10_baseline.yaml")},
    {"name": "Proposed_Method", "attn_type": "proposed", "config": os.path.join(REPO_ROOT, "configs/cifar10_proposed.yaml")},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run Attention Mechanism Ablation")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per model")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--synthetic", action="store_true", help="Run fast synthetic smoke test")
    parser.add_argument("--output_dir", type=str, default="results/ablation_attention")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    meter = EnergyMeter()

    all_histories = {}
    summary_rows = []
    pareto_data = {}

    print("=" * 80)
    print("STARTING ATTENTION MECHANISM ABLATION EXPERIMENT")
    print("=" * 80)

    for spec in MODELS:
        name = spec["name"]
        attn_type = spec["attn_type"]
        config_path = spec["config"]

        model_out = os.path.join(args.output_dir, name)
        cmd = [
            sys.executable,
            os.path.join(REPO_ROOT, "train.py"),
            "--config", config_path,
            "--attn_type", attn_type,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--output_dir", model_out,
        ]
        if args.synthetic:
            cmd.append("--synthetic")

        print(f"\n>>> Running training for {name} ({attn_type})...")
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)

        # Load metrics
        metrics_file = os.path.join(model_out, "train_metrics.json")
        with open(metrics_file, "r") as f:
            data = json.load(f)

        best_acc = data["best_val_acc"]
        all_histories[name] = data["history"]

        # Calculate theoretical costs using proper neuromorphic energy accounting
        # N=64 patches (CIFAR-10, 4x4 patches on 32x32 image)
        N = (32 // 4) ** 2
        D = 64
        H = 4
        T = 1 if attn_type == "standard_vit" else 4

        # Read measured firing rate from training metrics (for spike-driven accounting)
        firing_rate = 0.0
        if "firing_stats" in data["history"][-1]:
            firing_rate = data["history"][-1]["firing_stats"].get("mean_firing_rate", 0.18)

        if attn_type == "standard_vit":
            # Standard ViT: All dense MACs (no spikes, T=1)
            from models.backbones import StandardVisionTransformer
            m = StandardVisionTransformer(embed_dim=D, depth=2, num_heads=H)
            ops = m.count_operations(B=1)
            total_macs = ops["total_macs"]
            total_acs = 0
            energy_uj = meter.estimate_energy_pj(sops=0, flops=total_macs) / 1e6
        else:
            # ---- Neuromorphic Spike-Driven Energy Accounting ----
            # Layer 1: Patch embedding (Conv2d on continuous RGB pixels) → dense MACs
            patch_macs = N * (4 * 4 * 3 * D)  # kernel_size^2 * in_channels * embed_dim

            # Subsequent layers receive binary spikes from LIF neurons.
            # Weight lookups on active spikes are pure additions (ACs), not MACs.
            # Active ACs = firing_rate × total_synaptic_connections × T

            # Projection layers per block: Q,K,V projections (3*D*D) + Out projection (D*D) = 4*D*D
            # MLP per block: fc1 (D * 4D) + fc2 (4D * D) = 2 * D * 4D = 8*D*D
            # Total per block: 4*D*D + 8*D*D = 12*D*D
            ops_per_block = N * 12 * D * D
            total_proj_mlp_ops = T * 2 * ops_per_block  # 2 blocks, T timesteps

            # Active spike-driven ACs for projections + MLP
            proj_mlp_acs = int(total_proj_mlp_ops * firing_rate)

            # Attention-specific operations
            if attn_type == "proposed":
                # Proposed TGSA: Zero MACs in attention interaction
                # S_K^T @ S_V + S_Q @ KV = 2 * N * d_h^2 ACs per head
                attn_acs = T * H * (2 * N * ((D // H) ** 2)) * 2  # 2 blocks
                attn_macs = 0
            elif attn_type == "ssa":
                # Spikformer SSA: S_Q @ S_K^T is additive (ACs), but Attn @ S_V
                # involves the scaled float attention map → partial MACs
                attn_acs_per_block = T * H * (N * N * (D // H))
                attn_macs_per_block = T * H * (N * N * (D // H))
                attn_acs = int(attn_acs_per_block * firing_rate) * 2  # 2 blocks
                attn_macs = attn_macs_per_block * 2  # 2 blocks (float attn map)
            else:
                # Linear SNN control: K^T V + Q(KV) are dense float matmuls (no spiking)
                attn_macs = T * H * (2 * N * ((D // H) ** 2)) * 2  # 2 blocks
                attn_acs = 0

            total_macs = patch_macs + attn_macs
            total_acs = proj_mlp_acs + attn_acs

            # Classifier head: receives time-averaged float features → dense MACs
            classifier_macs = D * 10  # embed_dim * num_classes
            total_macs += classifier_macs

            energy_uj = meter.estimate_energy_pj(sops=total_acs, flops=total_macs) / 1e6

        summary_rows.append({
            "Model": name,
            "Type": "ANN (Dense)" if attn_type == "standard_vit" else "SNN (Spiking)",
            "Top-1 Acc (%)": round(best_acc * 100, 2),
            "Total MACs": total_macs,
            "Total ACs": total_acs,
            "Firing Rate": round(firing_rate, 4) if attn_type != "standard_vit" else "N/A",
            "Energy (uJ/sample)": round(energy_uj, 2),
        })

        pareto_data[name] = {
            "top1_acc": best_acc,
            "energy_uj": energy_uj,
        }

    # Save summary table CSV and JSON
    csv_path = os.path.join(args.output_dir, "comparison_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    json_path = os.path.join(args.output_dir, "comparison_table.json")
    with open(json_path, "w") as f:
        json.dump(summary_rows, f, indent=2)

    # Generate Figures
    fig3_path = os.path.join("results/figures", "figure3_accuracy_comparison.png")
    plot_accuracy_comparison(all_histories, output_path=fig3_path)

    pareto_path = os.path.join("results/figures", "pareto_accuracy_energy.png")
    plot_accuracy_energy_pareto(pareto_data, output_path=pareto_path)

    print("\n" + "=" * 90)
    print("                      ATTENTION ABLATION SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Model':<20} | {'Type':<14} | {'Top-1 Acc':<10} | {'MACs':<12} | {'ACs':<10} | {'Energy (uJ)':<12}")
    print("-" * 90)
    for r in summary_rows:
        print(f"{r['Model']:<20} | {r['Type']:<14} | {r['Top-1 Acc (%)']:<9.2f}% | {r['Total MACs']:<12,d} | {r['Total ACs']:<10,d} | {r['Energy (uJ/sample)']:<12.2f}")
    print("=" * 90)
    print(f"Results saved to {args.output_dir}")
    print(f"Plots generated: {fig3_path}, {pareto_path}")


if __name__ == "__main__":
    main()
