"""
Comprehensive Computational Complexity and Token Scaling Analysis:
Generates publication plots Figure 5 and Figure 6.
Evaluates asymptotic behavior as sequence length N varies: N in [16, 64, 144, 256, 400].
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)

from utils.complexity import profile_attention_scaling
from utils.visualize import plot_complexity_scaling, plot_operation_breakdown


def parse_args():
    parser = argparse.ArgumentParser(description="Run Complexity & Token Scaling Analysis")
    parser.add_argument("--output_dir", type=str, default="results/complexity")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    token_counts = (16, 64, 144, 256, 400)
    print("Profiling token scaling across N =", token_counts)

    scaling_results = profile_attention_scaling(
        dim=64, num_heads=4, token_counts=token_counts, T=4, B=1
    )

    # Save scaling json
    out_json = os.path.join(args.output_dir, "scaling_metrics.json")
    with open(out_json, "w") as f:
        json.dump(scaling_results, f, indent=2)

    # Plot Figure 5
    fig5_path = os.path.join("results/figures", "figure5_complexity_scaling.png")
    plot_complexity_scaling(scaling_results, output_path=fig5_path)

    # Extract N=400 comparison for Figure 6
    idx = token_counts.index(400)
    comparison_400 = {
        "vanilla": scaling_results["vanilla"][idx],
        "ssa": scaling_results["ssa"][idx],
        "linear": scaling_results["linear"][idx],
        "proposed": scaling_results["proposed"][idx],
    }

    # Plot Figure 6
    fig6_path = os.path.join("results/figures", "figure6_operation_breakdown.png")
    plot_operation_breakdown(comparison_400, output_path=fig6_path)

    print("\n" + "=" * 80)
    print("             COMPLEXITY ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Generated Figure 5 (Complexity Scaling):    {fig5_path}")
    print(f"Generated Figure 6 (Operation Breakdown):  {fig6_path}")
    print(f"Metrics saved to {out_json}")


if __name__ == "__main__":
    main()
