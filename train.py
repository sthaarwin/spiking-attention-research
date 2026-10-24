"""
Training script for Spiking Vision Transformer attention baselines and proposed method.
Supports CIFAR-10, synthetic smoke tests, checkpointing, and metric logging.
"""

import argparse
import json
import os
import time
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
import yaml

from models.backbones import SpikingVisionTransformer
from utils.datasets import get_cifar10_loaders, get_synthetic_loaders
from utils.reproducibility import set_seed, get_device, save_run_metadata
from utils.firing_rate import FiringRateTracker
from utils.complexity import count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Spiking Vision Transformer")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument(
        "--attn_type",
        type=str,
        default=None,
        choices=["proposed", "ssa", "linear", "vanilla", "standard_vit"],
        help="Attention or model variant",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--embed_dim", type=int, default=None, help="Embedding dimension")
    parser.add_argument("--depth", type=int, default=None, help="Number of transformer blocks")
    parser.add_argument("--num_heads", type=int, default=None, help="Number of attention heads")
    parser.add_argument("--T", type=int, default=None, help="Number of timesteps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset for smoke tests")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Load config from YAML if provided and overlay CLI arguments."""
    config: Dict[str, Any] = {
        "model": {
            "attn_type": "proposed",
            "embed_dim": 64,
            "depth": 2,
            "num_heads": 4,
            "T": 4,
            "img_size": 32,
            "patch_size": 4,
            "tau": 2.0,
            "adaptive": True,
            "mlp_ratio": 4.0,
        },
        "dataset": {
            "name": "cifar10",
            "batch_size": 64,
            "num_workers": 2,
        },
        "training": {
            "epochs": 20,
            "lr": 1e-3,
            "weight_decay": 1e-2,
            "seed": 42,
        },
    }

    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)
            for section in ["model", "dataset", "training"]:
                if section in yaml_config:
                    config[section].update(yaml_config[section])

    # Command-line explicit overrides
    if args.attn_type is not None:
        config["model"]["attn_type"] = args.attn_type
        if args.attn_type != "proposed" and "adaptive" not in config["model"]:
            config["model"]["adaptive"] = False
    if args.embed_dim is not None:
        config["model"]["embed_dim"] = args.embed_dim
    if args.depth is not None:
        config["model"]["depth"] = args.depth
    if args.num_heads is not None:
        config["model"]["num_heads"] = args.num_heads
    if args.T is not None:
        config["model"]["T"] = args.T
    if args.synthetic:
        config["dataset"]["name"] = "synthetic"
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["dataset"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["lr"] = args.lr
    if args.weight_decay is not None:
        config["training"]["weight_decay"] = args.weight_decay
    if args.seed is not None:
        config["training"]["seed"] = args.seed

    return config


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)

        # Crucial for SNNs: reset membrane potentials between independent batches
        model.reset()

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = total_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            model.reset()
            logits = model(images)
            loss = criterion(logits, targets)

            total_loss += loss.item() * targets.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    test_loss = total_loss / max(1, total)
    test_acc = correct / max(1, total)
    return test_loss, test_acc


def main():
    args = parse_args()
    config = load_config(args)

    set_seed(config["training"]["seed"])
    device = get_device(args.device)

    run_name = f"{config['model']['attn_type']}_dim{config['model']['embed_dim']}_seed{config['training']['seed']}"
    output_dir = args.output_dir or os.path.join("results", run_name)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    save_run_metadata(output_dir, config)

    # Dataset setup
    if config["dataset"]["name"] == "synthetic":
        train_loader, test_loader = get_synthetic_loaders(
            batch_size=config["dataset"]["batch_size"]
        )
    else:
        train_loader, test_loader, _ = get_cifar10_loaders(
            batch_size=config["dataset"]["batch_size"],
            num_workers=config["dataset"].get("num_workers", 2),
            download=True,
        )

    # Model initialization
    attn_type = config["model"]["attn_type"]
    if attn_type == "standard_vit":
        from models.backbones import StandardVisionTransformer
        vit_kwargs = {
            k: v for k, v in config["model"].items()
            if k not in ("tau", "adaptive", "attn_type", "T", "initial_alpha", "initial_beta", "scale_by_length")
        }
        model = StandardVisionTransformer(**vit_kwargs).to(device)
    else:
        model = SpikingVisionTransformer(**config["model"]).to(device)

    param_info = count_parameters(model)
    print(f"Initialized {config['model']['attn_type']} model with {param_info['trainable_parameters']:,} trainable parameters.")
    print(f"Training on device: {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    epochs = config["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    tracker = FiringRateTracker(model)

    history = []
    best_val_acc = 0.0

    # Ensure central checkpoints directory exists for long-term persistence
    central_ckpt_dir = os.path.join("checkpoints", run_name)
    os.makedirs(central_ckpt_dir, exist_ok=True)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        epoch_duration = time.time() - t0
        firing_stats = tracker.summarize()

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
            "firing_stats": firing_stats,
            "duration_sec": epoch_duration,
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_duration:.1f}s] - "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}% | "
            f"Mean Firing Rate: {firing_stats['mean_firing_rate']:.3f}"
        )

        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "config": config,
        }

        # Save latest checkpoint
        latest_path = os.path.join(checkpoints_dir, "latest.pt")
        torch.save(checkpoint_state, latest_path)
        torch.save(checkpoint_state, os.path.join(central_ckpt_dir, "latest.pt"))

        # Save best checkpoint
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(checkpoints_dir, "best.pt")
            torch.save(checkpoint_state, best_path)
            torch.save(checkpoint_state, os.path.join(central_ckpt_dir, "best.pt"))
            print(f"--> Saved new best checkpoint (Val Acc: {val_acc*100:.2f}%) to {best_path} and {central_ckpt_dir}/best.pt")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f}s. Best Val Acc: {best_val_acc*100:.2f}%.")

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "train_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"best_val_acc": best_val_acc, "history": history}, f, indent=2)
    print(f"Results and checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
