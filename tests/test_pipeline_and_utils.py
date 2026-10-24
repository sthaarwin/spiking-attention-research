"""
Unit tests for end-to-end training pipeline, datasets, reproducibility,
firing rate tracking, and model complexity utilities.
"""

import os
import tempfile
import unittest
import torch
import torch.nn as nn

from models.backbones import SpikingVisionTransformer
from utils.datasets import get_synthetic_loaders, get_cifar10_transforms
from utils.reproducibility import set_seed, get_device, save_run_metadata, get_environment_info
from utils.firing_rate import FiringRateTracker
from utils.complexity import count_parameters, profile_attention_scaling


class TestPipelineAndUtils(unittest.TestCase):
    def setUp(self):
        set_seed(42)
        self.device = torch.device("cpu")

    def test_reproducibility_seeding(self):
        set_seed(1234)
        a = torch.randn(5, 5)
        set_seed(1234)
        b = torch.randn(5, 5)
        self.assertTrue(torch.equal(a, b))

    def test_save_run_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"model": {"attn_type": "proposed"}}
            meta_path = save_run_metadata(tmpdir, config)
            self.assertTrue(os.path.exists(meta_path))

    def test_synthetic_loaders(self):
        train_loader, test_loader = get_synthetic_loaders(
            batch_size=8, num_train_samples=16, num_test_samples=8, num_classes=10
        )
        self.assertEqual(len(train_loader.dataset), 16)
        self.assertEqual(len(test_loader.dataset), 8)

        images, targets = next(iter(train_loader))
        self.assertEqual(images.shape, (8, 3, 32, 32))
        self.assertEqual(targets.shape, (8,))

    def test_cifar10_transforms(self):
        train_tf, test_tf = get_cifar10_transforms()
        self.assertIsNotNone(train_tf)
        self.assertIsNotNone(test_tf)

    def test_firing_rate_tracker(self):
        model = SpikingVisionTransformer(
            img_size=32, patch_size=4, embed_dim=32, depth=1, num_heads=2, T=2, attn_type="proposed"
        )
        tracker = FiringRateTracker(model)
        x = torch.randn(2, 3, 32, 32)
        _ = model(x)

        summary = tracker.summarize()
        self.assertIn("mean_firing_rate", summary)
        self.assertGreaterEqual(summary["mean_firing_rate"], 0.0)
        self.assertLessEqual(summary["mean_firing_rate"], 1.0)
        self.assertGreater(summary["num_spiking_layers"], 0)

        is_healthy, _ = tracker.check_health()
        self.assertIsInstance(is_healthy, bool)

    def test_complexity_and_scaling(self):
        model = SpikingVisionTransformer(
            img_size=32, patch_size=4, embed_dim=32, depth=1, num_heads=2, T=2
        )
        params = count_parameters(model)
        self.assertGreater(params["trainable_parameters"], 0)
        self.assertEqual(params["total_parameters"], params["trainable_parameters"])

        scaling = profile_attention_scaling(
            dim=32, num_heads=2, token_counts=(16, 32), T=2, B=1
        )
        self.assertIn("proposed", scaling)
        self.assertIn("ssa", scaling)
        self.assertIn("linear", scaling)
        self.assertIn("vanilla", scaling)

        # Proposed attention interaction must have 0 MACs
        for entry in scaling["proposed"]:
            self.assertEqual(entry["attention_macs"], 0)
            self.assertGreater(entry["attention_acs"], 0)

    def test_train_step_all_four_variants(self):
        """Verify that all 4 model variants train and backpropagate without error."""
        variants = ["proposed", "ssa", "linear", "vanilla"]
        x = torch.randn(2, 3, 32, 32)
        y = torch.tensor([1, 2], dtype=torch.long)
        criterion = nn.CrossEntropyLoss()

        for variant in variants:
            model = SpikingVisionTransformer(
                img_size=32,
                patch_size=4,
                embed_dim=32,
                depth=1,
                num_heads=2,
                T=2,
                attn_type=variant,
            )
            model.reset()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.assertFalse(torch.isnan(loss))
            self.assertGreater(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
