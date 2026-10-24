"""
Factorial Threshold Ablation Runner:
Evaluates the 2x2 Factorial Matrix:
- Variant A: Quadratic SSA + Fixed Threshold
- Variant B: Linear Attention + Fixed Threshold
- Variant C: Quadratic SSA + Adaptive Threshold
- Variant D: Linear Attention + Adaptive Threshold (Proposed)

Also profiles threshold sensitivity across alpha and beta hyperparameters.
"""

import argparse
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)

VARIANTS = [
    {"variant": "A_SSA_Fixed", "attn_type": "ssa", "adaptive": False, "config": os.path.join(REPO_ROOT, "configs/cifar10_baseline.yaml")},
    {"variant": "B_Linear_Fixed", "attn_type": "proposed", "adaptive": False, "config": os.path.join(REPO_ROOT, "configs/ablation_linear_fixed.yaml")},
    {"variant": "C_SSA_Adaptive", "attn_type": "ssa", "adaptive": True, "config": os.path.join(REPO_ROOT, "configs/ablation_ssa_adaptive.yaml")},
    {"variant": "D_Linear_Adaptive", "attn_type": "proposed", "adaptive": True, "config": os.path.join(REPO_ROOT, "configs/cifar10_proposed.yaml")},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run Factorial Threshold Ablation")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per variant")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--synthetic", action="store_true", help="Run fast synthetic smoke test")
    parser.add_argument("--output_dir", type=str, default="results/ablation_threshold")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    results = []

    print("=" * 80)
    print("STARTING 2x2 FACTORIAL THRESHOLD ABLATION EXPERIMENT")
    print("=" * 80)

    for item in VARIANTS:
        v_name = item["variant"]
        attn = item["attn_type"]
        config = item["config"]
        v_out = os.path.join(args.output_dir, v_name)

        cmd = [
            sys.executable,
            os.path.join(REPO_ROOT, "train.py"),
            "--config", config,
            "--attn_type", attn,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--output_dir", v_out,
        ]
        if args.synthetic:
            cmd.append("--synthetic")

        print(f"\n>>> Running Factorial Variant {v_name} (attn={attn}, adaptive={item['adaptive']})...")
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)

        metrics_file = os.path.join(v_out, "train_metrics.json")
        with open(metrics_file, "r") as f:
            data = json.load(f)

        results.append({
            "Variant": v_name,
            "Linear Attention": "Yes" if "Linear" in v_name else "No",
            "Adaptive Threshold": "Yes" if item["adaptive"] else "No",
            "Best Val Acc (%)": round(data["best_val_acc"] * 100, 2),
            "Final Firing Rate": round(data["history"][-1]["firing_stats"]["mean_firing_rate"], 4),
        })

    summary_file = os.path.join(args.output_dir, "factorial_summary.json")
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("                    2x2 FACTORIAL ABLATION RESULTS")
    print("=" * 80)
    print(f"{'Variant':<20} | {'Linear Attn':<12} | {'Adaptive Thresh':<16} | {'Top-1 Acc':<10} | {'Firing Rate':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['Variant']:<20} | {r['Linear Attention']:<12} | {r['Adaptive Threshold']:<16} | {r['Best Val Acc (%)']:<9.2f}% | {r['Final Firing Rate']:<12.4f}")
    print("=" * 80)
    print(f"Factorial results written to {summary_file}")


if __name__ == "__main__":
    main()
