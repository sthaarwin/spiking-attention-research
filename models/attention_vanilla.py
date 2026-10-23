"""
Conventional Scaled Dot-Product Attention (Floating-Point ViT Reference Baseline).
Implements: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaSoftmaxAttention(nn.Module):
    """
    Standard Vision Transformer Multi-Head Self-Attention.
    Serves as the conventional floating-point reference baseline (Run 1).
    
    Input shape:  [T, B, N, D]
    Output shape: [T, B, N, D]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
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
        x_flat = x.reshape(T * B, N, D)

        q = self.q_proj(x_flat).reshape(T * B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x_flat).reshape(T * B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x_flat).reshape(T * B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product: Q K^T / sqrt(d) -> [T * B, H, N, N]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Attended representation: [T * B, H, N, d_h]
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(T * B, N, D)
        out = self.out_proj(out)
        out = self.proj_drop(out)

        return out.reshape(T, B, N, D)
