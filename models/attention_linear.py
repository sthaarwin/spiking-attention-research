"""
Associative Linear Attention Mechanism: Q(K^T V).
Eliminates explicit O(N^2) token-pair attention matrix instantiation.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class AssociativeLinearAttention(nn.Module):
    """
    Multi-Head Associative Linear Attention.
    
    Reorders computation from:
        (Q K^T) V  -->  O(N^2 * d_h)
    to:
        Q (K^T V)  -->  O(N * d_h^2)
    
    Input shape:  [T, B, N, D]
    Output shape: [T, B, N, D]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_drop: float = 0.0,
        scale_by_length: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.scale_by_length = scale_by_length

        # Linear projections
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def reset(self) -> None:
        """Stateless compatibility method."""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        x: [T, B, N, D]
        Returns: [T, B, N, D]
        """
        T, B, N, D = x.shape

        # Linear projections across all timesteps: [T * B, N, D]
        x_flat = x.reshape(T * B, N, D)
        q = self.q_proj(x_flat).reshape(T, B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x_flat).reshape(T, B, N, self.num_heads, self.head_dim)
        v = self.v_proj(x_flat).reshape(T, B, N, self.num_heads, self.head_dim)

        # Transpose to [T, B, H, N, d_h]
        q = q.permute(0, 1, 3, 2, 4)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)

        # Associative linear attention:
        # Step 1: Compute context KV = K^T V
        # K:  [T, B, H, N, d_h] -> K^T: [T, B, H, d_h, N]
        # V:  [T, B, H, N, d_h]
        # KV: [T, B, H, d_h, d_h]
        kv = torch.matmul(k.transpose(-2, -1), v)

        # Step 2: Compute attended representation Q(KV)
        # Q:   [T, B, H, N, d_h]
        # KV:  [T, B, H, d_h, d_h]
        # Out: [T, B, H, N, d_h]
        out = torch.matmul(q, kv)

        # Optional length scaling: 1 / sqrt(N) or 1 / N to normalize energy
        if self.scale_by_length and N > 0:
            out = out / (N ** 0.5)

        # Reshape back to [T, B, N, D]
        out = out.permute(0, 1, 3, 2, 4).reshape(T * B, N, D)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out.reshape(T, B, N, D)

    def count_attention_ops(self, T: int, B: int, N: int) -> dict:
        """Calculate theoretical MACs and ACs for the attention aggregation."""
        # K^T V: [d_h, N] x [N, d_h] -> d_h * N * d_h MACs per head
        # Q (KV): [N, d_h] x [d_h, d_h] -> N * d_h * d_h MACs per head
        macs_per_head = 2 * N * (self.head_dim ** 2)
        total_macs = T * B * self.num_heads * macs_per_head
        return {
            "attention_macs": total_macs,
            "complexity_order": f"O(T * B * H * N * d_h^2) = O({T * B * self.dim * self.head_dim} * N)",
        }

    def count_operations(self, T: int, B: int, N: int) -> dict:
        """
        Operation breakdown for associative linear attention:
        Linear attention reorders (Q K^T) V to Q (K^T V):
        K^T @ V: N * d_h * d_h MACs
        Q @ (KV): N * d_h * d_h MACs
        """
        projection_macs = T * B * N * (4 * self.dim * self.dim)
        attention_macs = T * B * self.num_heads * (2 * N * (self.head_dim ** 2))
        return {
            "projection_macs": projection_macs,
            "attention_macs": attention_macs,
            "attention_acs": 0,
        }

