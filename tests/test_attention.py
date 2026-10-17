"""
Comprehensive unit tests for Linear Attention and Adaptive Threshold Gating.
Verifies:
1. Output tensor shape preservation [T, B, N, D] -> [T, B, N, D]
2. Differentiability and non-zero gradients for alpha, beta, and projection weights
3. State reset functionality between batches
4. Multiplication-free attention operation counts
5. Firing rate bounds and numerical stability
"""

import unittest
import torch
import torch.nn as nn

from models.neurons import AdaptiveThresholdLIFNode, SurrogateHeaviside
from models.attention_linear import AssociativeLinearAttention
from models.attention_proposed import ThresholdGatedSpikingAttention


class TestSpikingAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.T = 4    # Timesteps
        self.B = 2    # Batch size
        self.N = 16   # Tokens / Sequence length
        self.D = 64   # Embedding dimension
        self.heads = 4

    def test_adaptive_threshold_lif_shape_and_dynamics(self):
        node = AdaptiveThresholdLIFNode(
            tau=2.0,
            initial_alpha=0.5,
            initial_beta=1.0,
            adaptive=True,
        )
        x = torch.randn(self.T, self.B, self.N, self.D, requires_grad=True)
        spikes = node(x)

        # Shape preservation
        self.assertEqual(spikes.shape, (self.T, self.B, self.N, self.D))

        # Binary spike verification (elements must be 0.0 or 1.0)
        unique_vals = torch.unique(spikes).detach().cpu().numpy()
        for val in unique_vals:
            self.assertIn(val, [0.0, 1.0])

        # Backward gradients verification
        loss = spikes.sum()
        loss.backward()

        self.assertIsNotNone(node.alpha.grad)
        self.assertIsNotNone(node.beta.grad)
        self.assertNotEqual(node.alpha.grad.item(), 0.0)
        self.assertNotEqual(node.beta.grad.item(), 0.0)

    def test_lif_neuron_reset(self):
        node = AdaptiveThresholdLIFNode(tau=2.0, adaptive=True)
        x = torch.randn(self.T, self.B, self.N, self.D)

        out1 = node(x)
        self.assertIsNotNone(node.v_membrane)

        node.reset()
        self.assertIsNone(node.v_membrane)
        self.assertIsNone(node.last_spike)
        self.assertEqual(node.total_spikes, 0)
        self.assertEqual(node.total_elements, 0)

        # Running again after reset should produce identical output
        out2 = node(x)
        self.assertTrue(torch.allclose(out1, out2))

    def test_associative_linear_attention_shape_and_ops(self):
        attn = AssociativeLinearAttention(dim=self.D, num_heads=self.heads)
        x = torch.randn(self.T, self.B, self.N, self.D, requires_grad=True)
        out = attn(x)

        self.assertEqual(out.shape, (self.T, self.B, self.N, self.D))

        # Backward pass
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(attn.q_proj.weight.grad)
        self.assertIsNotNone(attn.k_proj.weight.grad)
        self.assertIsNotNone(attn.v_proj.weight.grad)

        # Operation counts
        ops = attn.count_attention_ops(self.T, self.B, self.N)
        self.assertIn("attention_macs", ops)
        self.assertGreater(ops["attention_macs"], 0)

    def test_threshold_gated_spiking_attention_full_chain(self):
        model = ThresholdGatedSpikingAttention(
            dim=self.D,
            num_heads=self.heads,
            initial_alpha=1.0,
            initial_beta=1.0,
            adaptive=True,
        )

        x = torch.randn(self.T, self.B, self.N, self.D, requires_grad=True)
        out = model(x)

        # Shape check
        self.assertEqual(out.shape, (self.T, self.B, self.N, self.D))

        # Backward gradient flow for alpha, beta, and projection weights
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(model.attention_lif.alpha.grad)
        self.assertIsNotNone(model.attention_lif.beta.grad)
        self.assertNotEqual(model.attention_lif.alpha.grad.item(), 0.0)
        self.assertNotEqual(model.attention_lif.beta.grad.item(), 0.0)
        self.assertIsNotNone(model.q_proj.weight.grad)
        self.assertIsNotNone(model.k_proj.weight.grad)
        self.assertIsNotNone(model.v_proj.weight.grad)
        self.assertIsNotNone(model.out_proj.weight.grad)

        # Multiplication-free verification
        ops = model.count_operations(self.T, self.B, self.N)
        self.assertEqual(ops["attention_macs"], 0, "Attention interaction must contain 0 MACs!")
        self.assertGreater(ops["attention_acs"], 0, "Attention interaction must use ACs (additions)!")

        # Firing rates check
        rates = model.get_firing_rates()
        for name, rate in rates.items():
            self.assertGreaterEqual(rate, 0.0)
            self.assertLessEqual(rate, 1.0)

    def test_factorial_ablation_modes(self):
        # Variant D: Linear Attention + Adaptive Threshold
        model_d = ThresholdGatedSpikingAttention(dim=self.D, num_heads=self.heads, adaptive=True)
        # Variant B: Linear Attention + Fixed Threshold (non-adaptive)
        model_b = ThresholdGatedSpikingAttention(dim=self.D, num_heads=self.heads, adaptive=False)

        x = torch.randn(self.T, self.B, self.N, self.D)
        out_d = model_d(x)
        out_b = model_b(x)

        self.assertEqual(out_d.shape, out_b.shape)
        # Model D has learnable alpha, Model B has buffer alpha == 0
        self.assertTrue(isinstance(model_d.attention_lif.alpha, nn.Parameter))
        self.assertFalse(isinstance(model_b.attention_lif.alpha, nn.Parameter))


if __name__ == "__main__":
    unittest.main()
