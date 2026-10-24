"""
Spiking Vision Transformer backbones built from adaptive-threshold spiking
neurons and pluggable attention mechanisms (Vanilla Softmax, SSA, Linear, and Proposed).
"""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn

from .neurons import AdaptiveThresholdLIFNode
from .attention_proposed import ThresholdGatedSpikingAttention
from .attention_ssa import SpikingSelfAttention
from .attention_linear import AssociativeLinearAttention
from .attention_vanilla import VanillaSoftmaxAttention


def create_attention_layer(
    attn_type: str,
    dim: int,
    num_heads: int = 4,
    tau: float = 2.0,
    adaptive: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """Factory creating an attention module by name."""
    attn_type = attn_type.lower()
    if attn_type in ("proposed", "tgsa"):
        return ThresholdGatedSpikingAttention(
            dim=dim, num_heads=num_heads, tau=tau, adaptive=adaptive, **kwargs
        )
    elif attn_type in ("ssa", "spikformer"):
        return SpikingSelfAttention(
            dim=dim, num_heads=num_heads, tau=tau, adaptive=adaptive, **kwargs
        )
    elif attn_type in ("linear", "associative"):
        return AssociativeLinearAttention(
            dim=dim, num_heads=num_heads, **kwargs
        )
    elif attn_type in ("vanilla", "softmax"):
        return VanillaSoftmaxAttention(
            dim=dim, num_heads=num_heads, **kwargs
        )
    else:
        raise ValueError(
            f"Unknown attention type '{attn_type}'. Supported: 'proposed', 'ssa', 'linear', 'vanilla'."
        )


class SpikingTransformerBlock(nn.Module):
    """Pre-normalization-free transformer block with configurable attention and spiking MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        tau: float = 2.0,
        attn_type: str = "proposed",
        adaptive: bool = True,
        **attn_kwargs: Any,
    ):
        super().__init__()
        self.attn = create_attention_layer(
            attn_type=attn_type,
            dim=dim,
            num_heads=num_heads,
            tau=tau,
            adaptive=adaptive,
            **attn_kwargs,
        )

        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.lif1 = AdaptiveThresholdLIFNode(tau=tau)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.lif2 = AdaptiveThresholdLIFNode(tau=tau)

    def reset(self) -> None:
        if hasattr(self.attn, "reset"):
            self.attn.reset()
        self.lif1.reset()
        self.lif2.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, N, D]
        x = x + self.attn(x)
        x = x + self.lif2(self.fc2(self.lif1(self.fc1(x))))
        return x


class SpikingVisionTransformer(nn.Module):
    """
    Spiking Vision Transformer for static image classification.
    Supports controlled comparison across 4 attention mechanisms.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        T: int = 4,
        attn_type: str = "proposed",
        tau: float = 2.0,
        adaptive: bool = True,
        mlp_ratio: float = 4.0,
        **attn_kwargs: Any,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.T = T
        self.attn_type = attn_type

        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.patch_lif = AdaptiveThresholdLIFNode(tau=tau)

        self.blocks = nn.ModuleList(
            [
                SpikingTransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    tau=tau,
                    attn_type=attn_type,
                    adaptive=adaptive,
                    **attn_kwargs,
                )
                for _ in range(depth)
            ]
        )
        self.head = nn.Linear(embed_dim, num_classes)

    def reset(self) -> None:
        self.patch_lif.reset()
        for block in self.blocks:
            block.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [T, B, C, H, W]
        t, b = x.shape[0], x.shape[1]
        x = x.flatten(0, 1)                            # [T*B, C, H, W]
        x = self.patch_embed(x)                        # [T*B, D, h, w]
        x = x.flatten(2).transpose(1, 2)               # [T*B, N, D]
        x = x.reshape(t, b, x.shape[1], x.shape[2])    # [T, B, N, D]
        x = self.patch_lif(x)

        for block in self.blocks:
            x = block(x)

        # Global temporal and token average pooling
        x = x.mean(dim=(0, 2))                          # [B, D]
        return self.head(x)


class StandardTransformerBlock(nn.Module):
    """
    Conventional Pre-LN Vision Transformer block:
    LayerNorm -> Multi-Head Softmax Attention -> LayerNorm -> GELU MLP
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = VanillaSoftmaxAttention(dim=dim, num_heads=num_heads, proj_drop=proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(proj_drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        x_norm = self.norm1(x).unsqueeze(0)  # [1, B, N, D]
        attn_out = self.attn(x_norm).squeeze(0)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class StandardVisionTransformer(nn.Module):
    """
    Conventional floating-point Vision Transformer (ANN ViT).
    Serves as the dense non-spiking reference for accuracy, FLOPs, and energy benchmarks.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        proj_drop: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_patches = (img_size // patch_size) ** 2
        self.T = 1

        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.blocks = nn.ModuleList(
            [
                StandardTransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    proj_drop=proj_drop,
                )
                for _ in range(depth)
            ]
        )
        self.head = nn.Linear(embed_dim, num_classes)

    def reset(self) -> None:
        """Compatibility method for standard ViT."""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.patch_embed(x)               # [B, D, h, w]
        x = x.flatten(2).transpose(1, 2)      # [B, N, D]
        x = self.norm(x)

        for block in self.blocks:
            x = block(x)

        x = x.mean(dim=1)                     # [B, D]
        return self.head(x)

    def count_operations(self, B: int = 1) -> Dict[str, int]:
        """Calculate total theoretical MACs and ACs for standard ViT."""
        N = self.num_patches
        D = self.embed_dim
        patch_macs = B * N * (self.patch_size * self.patch_size * self.in_channels * D)
        total_attn_macs = 0
        total_mlp_macs = 0
        for block in self.blocks:
            attn_ops = block.attn.count_operations(T=1, B=B, N=N)
            total_attn_macs += attn_ops["projection_macs"] + attn_ops["attention_macs"]
            total_mlp_macs += B * N * (2 * D * int(D * 4.0))

        classifier_macs = B * D * self.num_classes
        total_macs = patch_macs + total_attn_macs + total_mlp_macs + classifier_macs
        return {
            "total_macs": total_macs,
            "total_acs": 0,
            "attention_macs": total_attn_macs,
            "attention_acs": 0,
        }

