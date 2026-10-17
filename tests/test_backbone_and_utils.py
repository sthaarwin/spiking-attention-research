"""
Unit tests for Spiking Vision Transformer backbone, EnergyMeter, and EntropyTracker.
"""

import unittest
import torch
import torch.nn as nn

from models.backbones import SpikingVisionTransformer, SpikingTransformerBlock
from models.attention_proposed import ThresholdGatedSpikingAttention
from utils.energy_meter import EnergyMeter
from utils.entropy_tracker import EntropyTracker


class TestBackboneAndUtils(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.B = 2
        self.C = 3
        self.H = 32
        self.W = 32
        self.T = 4
        self.num_classes = 10
        self.embed_dim = 64

    def test_spiking_vision_transformer_forward(self):
        model = SpikingVisionTransformer(
            img_size=self.H,
            patch_size=4,
            in_channels=self.C,
            num_classes=self.num_classes,
            embed_dim=self.embed_dim,
            depth=2,
            num_heads=4,
            T=self.T,
        )
        x = torch.randn(self.B, self.C, self.H, self.W)
        logits = model(x)

        # Output shape must be [B, num_classes]
        self.assertEqual(logits.shape, (self.B, self.num_classes))

        # Backward gradient flow
        loss = logits.sum()
        loss.backward()
        self.assertIsNotNone(model.head.weight.grad)

        # Reset state
        model.reset()

    def test_energy_meter_accounting(self):
        meter = EnergyMeter(e_mac_pj=4.6, e_ac_pj=0.9)

        # Record dummy spike tensor with 25% firing rate
        spikes = torch.zeros(self.T, self.B, 16, self.embed_dim)
        spikes[:, :, :, :16] = 1.0  # 25% active

        sops = meter.calculate_sops(spikes, fan_out=128)
        self.assertGreater(sops, 0)

        energy_pj = meter.estimate_energy_pj(sops=sops, flops=10000)
        self.assertGreater(energy_pj, 0.0)
        self.assertEqual(energy_pj, sops * 0.9 + 10000 * 4.6)

    def test_entropy_tracker_selectivity(self):
        tracker = EntropyTracker()

        # Dummy attention activations [T, B, H, N, d_h]
        activations = torch.rand(self.T, self.B, 4, 16, 16)
        metrics = tracker.compute_metrics(activations)

        self.assertIn("entropy", metrics)
        self.assertIn("firing_rate", metrics)
        self.assertIn("top_k_concentration", metrics)
        self.assertGreaterEqual(metrics["entropy"], 0.0)
        self.assertGreaterEqual(metrics["firing_rate"], 0.0)
        self.assertLessEqual(metrics["firing_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
