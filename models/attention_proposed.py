"""
Threshold-Gated Spiking Attention:
Multiplication-Free Linear Spiking Attention using Adaptive Threshold LIF Gating.
Replaces Softmax with Dynamic Threshold Gating: theta_d = alpha * Mean(U) + beta.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .neurons import AdaptiveThresholdLIFNode


class ThresholdGatedSpikingAttention(nn.Module):
    """
    Proposed Multiplication-Free Linear Spiking Attention.
    
    Architecture:
        1. Linear Q, K, V projections followed by spiking activation.
        2. Spike-based associative linear aggregation:
              KV = S_K^T @ S_V       (Additive Accumulation / ACs)
              U  = S_Q   @ KV        (Additive Accumulation / ACs)
        3. Adaptive Threshold LIF Gating:
              theta_d = alpha * Mean(U) + beta
              S_out   = Spike(U - theta_d)
        4. Linear output projection W_o.
    
    Zero matrix multiplications in the attention interaction:
    Binary spike inputs convert all inner products into additions.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_drop: float = 0.0,
        initial_alpha: float = 1.0,
        initial_beta: float = 1.0,
        tau: float = 2.0,
        adaptive: bool = True,
        scale_by_length: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale_by_length = scale_by_length
        self.adaptive = adaptive

        # Q, K, V linear projections
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        # Spiking neurons for intermediate Q, K, V representations
        self.q_lif = AdaptiveThresholdLIFNode(tau=tau, adaptive=False, initial_beta=1.0)
        self.k_lif = AdaptiveThresholdLIFNode(tau=tau, adaptive=False, initial_beta=1.0)
        self.v_lif = AdaptiveThresholdLIFNode(tau=tau, adaptive=False, initial_beta=1.0)

        # Core Attention Normalizer: Dynamic Threshold LIF Gating
        self.attention_lif = AdaptiveThresholdLIFNode(
            tau=tau,
            initial_alpha=initial_alpha,
            initial_beta=initial_beta,
            adaptive=adaptive,
            reduction_dims=(-2,),  # Normalizes across token sequence N
        )

        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def reset(self) -> None:
        """Reset all constituent spiking neurons."""
        self.q_lif.reset()
        self.k_lif.reset()
        self.v_lif.reset()
        self.attention_lif.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        x: [T, B, N, D]
        Returns: [T, B, N, D]
        """
        T, B, N, D = x.shape

        # Linear projections
        x_flat = x.reshape(T * B, N, D)
        q = self.q_proj(x_flat).reshape(T, B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x_flat).reshape(T, B, N, self.num_heads, self.head_dim)
        v = self.v_proj(x_flat).reshape(T, B, N, self.num_heads, self.head_dim)

        # Transpose to [T, B, H, N, d_h]
        q = q.permute(0, 1, 3, 2, 4)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)

        # Convert to intermediate binary spikes
        s_q = self.q_lif(q)
        s_k = self.k_lif(k)
        s_v = self.v_lif(v)

        # Associative linear attention:
        # Step 1: KV = S_K^T @ S_V
        # S_K^T: [T, B, H, d_h, N]
        # S_V:   [T, B, H, N, d_h]
        # KV:    [T, B, H, d_h, d_h]
        kv = torch.matmul(s_k.transpose(-2, -1), s_v)

        # Step 2: Excitation U = S_Q @ KV
        # S_Q:   [T, B, H, N, d_h]
        # KV:    [T, B, H, d_h, d_h]
        # U:     [T, B, H, N, d_h]
        u = torch.matmul(s_q, kv)

        # Sequence length scaling
        if self.scale_by_length and N > 0:
            u = u / (N ** 0.5)

        # Step 3: Adaptive Threshold Gating (replaces Softmax)
        # Dynamically computes theta_d = alpha * Mean(U) + beta
        # Generates output spikes: [T, B, H, N, d_h]
        s_out = self.attention_lif(u)

        # Reshape to [T, B, N, D] and project out
        out = s_out.permute(0, 1, 3, 2, 4).reshape(T * B, N, D)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out.reshape(T, B, N, D)

    def get_firing_rates(self) -> Dict[str, float]:
        """Return firing rates for intermediate and output spike layers."""
        return {
            "q_firing_rate": self.q_lif.get_firing_rate(),
            "k_firing_rate": self.k_lif.get_firing_rate(),
            "v_firing_rate": self.v_lif.get_firing_rate(),
            "attention_firing_rate": self.attention_lif.get_firing_rate(),
        }

    def count_operations(self, T: int, B: int, N: int) -> Dict[str, int]:
        """
        Operation breakdown:
        The attention aggregation is strictly additive (ACs) because inputs are spikes.
        Projection layers are standard MACs.
        """
        # Linear projection MACs: 3 * D * D for Q,K,V + D * D for Out
        projection_macs = T * B * N * (4 * self.dim * self.dim)

        # Attention interaction ACs: S_K^T @ S_V + S_Q @ KV
        attention_acs = T * B * self.num_heads * (2 * N * (self.head_dim ** 2))

        return {
            "projection_macs": projection_macs,
            "attention_macs": 0,  # Zero multiplications in attention interaction!
            "attention_acs": attention_acs,
        }
