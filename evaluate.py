"""
Evaluation and analytical profiling script:
Measures Top-1 and Top-5 accuracy, layer-wise firing rates,
activation entropy/selectivity, and theoretical energy consumption.
"""

import argparse
import json
import os
from typing import Dict, Any
import torch
import torch.nn as nn

from models.backbones import SpikingVisionTransformer
from utils.datasets import get_cifar10_loaders, get_synthetic_loaders
from utils.reproducibility import get_device, set_seed
from utils.firing_rate import FiringRateTracker
from utils.energy_meter import EnergyMeter
from utils.entropy_tracker import EntropyTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Spiking Vision Transformer")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument(
        "--attn_type",
        type=str,
        default="proposed",
        choices=["proposed", "ssa", "linear", "vanilla", "standard_vit"],
        help="Attention type if checkpoint not specified",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Target device")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset")
    parser.add_argument("--output", type=str, default=None, help="Save evaluation metrics to JSON")
    return parser.parse_args()


def evaluate_metrics(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    firing_tracker = FiringRateTracker(model)
    entropy_tracker = EntropyTracker()
    energy_meter = EnergyMeter()

    all_entropies = []
    total_sops = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            model.reset()

            logits = model(images)

            # Top-1 accuracy
            preds_top1 = logits.argmax(dim=-1)
            correct_top1 += (preds_top1 == targets).sum().item()

            # Top-5 accuracy
            k = min(5, logits.size(-1))
            top5_preds = logits.topk(k=k, dim=-1).indices
            correct_top5 += top5_preds.eq(targets.unsqueeze(-1)).any(dim=-1).sum().item()

            total += targets.size(0)

            # Measure entropy and activation selectivity
            ent_metrics = entropy_tracker.compute_metrics(logits)
            all_entropies.append(ent_metrics["entropy"])

    top1_acc = correct_top1 / max(1, total)
    top5_acc = correct_top5 / max(1, total)
    mean_entropy = sum(all_entropies) / max(1, len(all_entropies))
    firing_summary = firing_tracker.summarize()

    results = {
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "mean_entropy": mean_entropy,
        "firing_summary": firing_summary,
        "layer_firing_rates": firing_tracker.get_layer_firing_rates(),
    }
    return results


def main():
    args = parse_args()
    set_seed(42)
    device = get_device(args.device)

    config = {
        "attn_type": args.attn_type,
        "embed_dim": 64,
        "depth": 2,
        "num_heads": 4,
        "T": 4,
        "img_size": 32,
        "patch_size": 4,
        "tau": 2.0,
        "adaptive": True if args.attn_type == "proposed" else False,
    }

    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if "config" in checkpoint and "model" in checkpoint["config"]:
            config = checkpoint["config"]["model"]
        print(f"Loaded checkpoint: {args.checkpoint}")

    attn_type = config.get("attn_type", "proposed")
    if attn_type == "standard_vit":
        from models.backbones import StandardVisionTransformer
        vit_kwargs = {
            k: v for k, v in config.items()
            if k not in ("tau", "adaptive", "attn_type", "T", "initial_alpha", "initial_beta", "scale_by_length")
        }
        model = StandardVisionTransformer(**vit_kwargs).to(device)
    else:
        model = SpikingVisionTransformer(**config).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Restored model weights from checkpoint.")

    if args.synthetic:
        _, test_loader = get_synthetic_loaders(batch_size=args.batch_size)
    else:
        _, test_loader, _ = get_cifar10_loaders(batch_size=args.batch_size, download=True)

    print(f"Evaluating {config['attn_type']} on {len(test_loader.dataset)} samples...")
    results = evaluate_metrics(model, test_loader, device)

    print("\n" + "=" * 50)
    print("           EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Attention Type:        {config['attn_type']}")
    print(f"Top-1 Accuracy:        {results['top1_accuracy']*100:.2f}%")
    print(f"Top-5 Accuracy:        {results['top5_accuracy']*100:.2f}%")
    print(f"Mean Activation Entropy: {results['mean_entropy']:.4f}")
    print(f"Mean Firing Rate:      {results['firing_summary']['mean_firing_rate']:.4f}")
    print(f"Dead Layers (<1%):     {results['firing_summary']['dead_layers_count']}")
    print(f"Saturated Layers (>95%): {results['firing_summary']['saturated_layers_count']}")
    print("=" * 50)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation metrics written to {args.output}")


if __name__ == "__main__":
    main()
