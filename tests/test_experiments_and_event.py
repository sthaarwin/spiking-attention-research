"""
Unit tests for StandardVisionTransformer, event-based vision datasets,
visualization generators, and experiment runners.
"""

import os
import tempfile
import unittest
import torch

from models.backbones import StandardVisionTransformer
from utils.datasets import get_event_loaders
from utils.visualize import (
    plot_accuracy_comparison,
    plot_firing_rate_distribution,
    plot_complexity_scaling,
    plot_operation_breakdown,
    plot_accuracy_energy_pareto,
)


class TestExperimentsAndEvent(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_standard_vision_transformer_forward_and_ops(self):
        model = StandardVisionTransformer(
            img_size=32,
            patch_size=4,
            in_channels=3,
            num_classes=10,
            embed_dim=32,
            depth=1,
            num_heads=2,
        )
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (2, 10))

        # Backward pass
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(model.head.weight.grad)

        # Operation counts
        ops = model.count_operations(B=1)
        self.assertIn("total_macs", ops)
        self.assertGreater(ops["total_macs"], 0)
        self.assertEqual(ops["total_acs"], 0)

    def test_event_vision_loaders(self):
        train_loader, test_loader = get_event_loaders(
            batch_size=4, T=4, img_size=32, synthetic=True
        )
        images, targets = next(iter(train_loader))
        self.assertEqual(images.shape, (4, 2, 32, 32))
        self.assertEqual(targets.shape, (4,))

    def test_visualization_generators(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Accuracy comparison
            history_mock = {
                "Standard_ViT": [{"epoch": 1, "train_acc": 0.5, "val_acc": 0.4}],
                "Proposed": [{"epoch": 1, "train_acc": 0.45, "val_acc": 0.42}],
            }
            p1 = os.path.join(tmpdir, "fig3.png")
            plot_accuracy_comparison(history_mock, output_path=p1)
            self.assertTrue(os.path.exists(p1) and os.path.getsize(p1) > 0)

            # 2. Firing rates
            rates_mock = {"block1": 0.25, "block2": 0.35}
            p2 = os.path.join(tmpdir, "fig4.png")
            plot_firing_rate_distribution(rates_mock, output_path=p2)
            self.assertTrue(os.path.exists(p2) and os.path.getsize(p2) > 0)

            # 3. Complexity scaling
            scaling_mock = {
                "vanilla": [{"tokens": 16, "macs": 1000, "estimated_energy_pj": 4600}],
                "proposed": [{"tokens": 16, "macs": 500, "estimated_energy_pj": 2300}],
            }
            p3 = os.path.join(tmpdir, "fig5.png")
            plot_complexity_scaling(scaling_mock, output_path=p3)
            self.assertTrue(os.path.exists(p3) and os.path.getsize(p3) > 0)

            # 4. Operation breakdown
            ops_mock = {
                "vanilla": {"macs": 1000, "acs": 0},
                "proposed": {"macs": 500, "acs": 500},
            }
            p4 = os.path.join(tmpdir, "fig6.png")
            plot_operation_breakdown(ops_mock, output_path=p4)
            self.assertTrue(os.path.exists(p4) and os.path.getsize(p4) > 0)

            # 5. Pareto tradeoff
            pareto_mock = {
                "Standard_ViT": {"top1_acc": 0.75, "energy_uj": 5.4},
                "Proposed": {"top1_acc": 0.72, "energy_uj": 1.2},
            }
            p5 = os.path.join(tmpdir, "pareto.png")
            plot_accuracy_energy_pareto(pareto_mock, output_path=p5)
            self.assertTrue(os.path.exists(p5) and os.path.getsize(p5) > 0)


if __name__ == "__main__":
    unittest.main()
