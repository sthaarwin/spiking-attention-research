"""
Neuromorphic Event-Based Vision Experiment Runner:
Tests model robustness on 2-channel asynchronous event streams across timesteps T in [4, 8, 16].
Compares Spikformer SSA baseline against Proposed Linear Spiking Attention.
"""

import argparse
import json
import os
import sys
import time
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)

from models.backbones import SpikingVisionTransformer
from utils.datasets import get_event_loaders
from utils.reproducibility import set_seed, get_device
from utils.firing_rate import FiringRateTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Run Event Vision Experiment")
    parser.add_argument("--timesteps", type=int, nargs="+", default=[4, 8], help="List of timesteps T to test")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per run")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--synthetic", action="store_true", help="Synthetic smoke test")
    parser.add_argument("--output_dir", type=str, default="results/event_vision")
    return parser.parse_args()


def train_eval_event_model(attn_type: str, T: int, epochs: int, batch_size: int, synthetic: bool, device: torch.device):
    set_seed(42)
    train_loader, test_loader = get_event_loaders(batch_size=batch_size, T=T, synthetic=synthetic)

    model = SpikingVisionTransformer(
        in_channels=2,
        embed_dim=32,
        depth=1,
        num_heads=2,
        T=T,
        attn_type=attn_type,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    tracker = FiringRateTracker(model)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            model.reset()
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        # Eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(device), targets.to(device)
                model.reset()
                logits = model(images)
                preds = logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        val_acc = correct / max(1, total)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    firing_stats = tracker.summarize()
    return best_val_acc, firing_stats["mean_firing_rate"]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()

    results = []
    print("=" * 80)
    print("STARTING NEUROMORPHIC EVENT VISION EXPERIMENT")
    print("=" * 80)

    for T in args.timesteps:
        for attn in ["ssa", "proposed"]:
            print(f"\n>>> Running Event Benchmark: Model={attn.upper()}, Timesteps T={T}...")
            acc, firing_rate = train_eval_event_model(
                attn_type=attn,
                T=T,
                epochs=args.epochs,
                batch_size=args.batch_size,
                synthetic=args.synthetic,
                device=device,
            )
            results.append({
                "Model": attn.upper(),
                "Timesteps T": T,
                "Best Accuracy (%)": round(acc * 100, 2),
                "Mean Firing Rate": round(firing_rate, 4),
            })

    out_file = os.path.join(args.output_dir, "temporal_scaling.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("               EVENT VISION RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<15} | {'Timesteps T':<12} | {'Top-1 Accuracy':<16} | {'Firing Rate':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['Model']:<15} | {r['Timesteps T']:<12} | {r['Best Accuracy (%)']:<15.2f}% | {r['Mean Firing Rate']:<12.4f}")
    print("=" * 80)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
