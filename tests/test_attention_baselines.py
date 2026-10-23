"""
Unit tests for Baseline Attention Mechanisms:
1. Vanilla Softmax Vision Transformer Attention (Floating-point Reference)
2. Spikformer-style Spiking Self-Attention (SSA, Primary SNN Baseline)
"""

import unittest
import torch

from models.attention_vanilla import VanillaSoftmaxAttention
from models.attention_ssa import SpikingSelfAttention


class TestAttentionBaselines(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.T = 4
        self.B = 2
        self.N = 16
        self.D = 64
        self.num_heads = 4

    def test_vanilla_softmax_attention(self):
        attn = VanillaSoftmaxAttention(dim=self.D, num_heads=self.num_heads)
        x = torch.randn(self.T, self.B, self.N, self.D, requires_grad=True)
        out = attn(x)

        # Output shape contract
        self.assertEqual(out.shape, (self.T, self.B, self.N, self.D))

        # Backward gradient flow
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(attn.q_proj.weight.grad)
        self.assertIsNotNone(attn.k_proj.weight.grad)
        self.assertIsNotNone(attn.v_proj.weight.grad)

        # Reset state check
        attn.reset()

    def test_spikformer_spiking_self_attention(self):
        attn = SpikingSelfAttention(dim=self.D, num_heads=self.num_heads)
        x = torch.randn(self.T, self.B, self.N, self.D, requires_grad=True)
        out = attn(x)

        # Output shape contract
        self.assertEqual(out.shape, (self.T, self.B, self.N, self.D))

        # Backward gradient flow
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(attn.q_proj.weight.grad)

        # Reset state check
        attn.reset()


if __name__ == "__main__":
    unittest.main()
