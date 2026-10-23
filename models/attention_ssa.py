"""
Spikformer-style Spiking Self-Attention (SSA).
Serves as the primary SOTA Spiking Neural Network baseline (Run 2).
Based on: Spikformer: When Spiking Neural Networks Meet Vision Transformer (ICLR 2023).
"""

from typing import Dict, Optional
import torch
import torch.nn as nn

from .neurons import AdaptiveThresholdLIFNode


class SpikingSelfAttention(nn.Module):
    """
    Standard Spiking Self-Attention (SSA) as introduced in Spikformer.
    Computes quadratic spike-based self-attention:
        S_Q, S_K, S_V = LIF(Q), LIF(K), LIF(V)
        Attn = (S_Q @ S_K^T) * scale
        Out = LIF(Attn @ S_V)
    
    Input shape:  [T, B, N, D]
    Output shape: [T, B, N, D]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_drop: float = 0.0,
        tau: float = 2.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 0.125  # Standard Spikformer scaling constant

        # Linear projections
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        # Spiking neurons for intermediate representations
        self.q_lif = AdaptiveThresholdLIFNode(tau=tau, adaptive=False, initial_beta=1.0)
        self.k_lif = AdaptiveThresholdLIFNode(tau=tau, adaptive=False, initial_beta=1.0)
        self.v_lif = AdaptiveThresholdLIFNode(tau=tau, adaptive=False, initial_beta=1.0)
        self.attn_lif = AdaptiveThresholdLIFNode(tau=tau, adaptive=False, initial_beta=1.0)

        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def reset(self) -> None:
        """Reset internal membrane states of all spiking neurons."""
        self.q_lif.reset()
        self.k_lif.reset()
        self.v_lif.reset()
        self.attn_lif.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        x: [T, B, N, D]
        Returns: [T, B, N, D]
        """
        T, B, N, D = x.shape

        x_flat = x.reshape(T * B, N, D)
        q = self.q_proj(x_flat).reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = self.k_proj(x_flat).reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = self.v_proj(x_flat).reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        # Intermediate binary spikes
        s_q = self.q_lif(q)
        s_k = self.k_lif(k)
        s_v = self.v_lif(v)

        # Quadratic spike attention: S_Q @ S_K^T -> [T, B, H, N, N]
        attn = torch.matmul(s_q, s_k.transpose(-2, -1)) * self.scale

        # Aggregation: Attn @ S_V -> [T, B, H, N, d_h]
        u = torch.matmul(attn, s_v)

        # Spiking post-activation
        s_out = self.attn_lif(u)

        # Reshape and linear projection
        out = s_out.permute(0, 1, 3, 2, 4).reshape(T * B, N, D)
        out = self.out_proj(out)
        out = self.proj_drop(out)

        return out.reshape(T, B, N, D)
