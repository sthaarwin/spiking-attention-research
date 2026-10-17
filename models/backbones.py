"""
Spiking Vision Transformer backbones built from adaptive-threshold spiking
neurons and multiplication-free threshold-gated spiking attention.
"""

import torch
import torch.nn as nn

from .neurons import AdaptiveThresholdLIFNode
from .attention_proposed import ThresholdGatedSpikingAttention


class SpikingTransformerBlock(nn.Module):
    """Pre-normalization-free transformer block with spiking attention and MLP."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0, tau: float = 2.0):
        super().__init__()
        self.attn = ThresholdGatedSpikingAttention(dim=dim, num_heads=num_heads, tau=tau)

        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.lif1 = AdaptiveThresholdLIFNode(tau=tau)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.lif2 = AdaptiveThresholdLIFNode(tau=tau)

    def reset(self) -> None:
        self.attn.reset()
        self.lif1.reset()
        self.lif2.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, N, D]
        x = x + self.attn(x)
        x = x + self.lif2(self.fc2(self.lif1(self.fc1(x))))
        return x


class SpikingVisionTransformer(nn.Module):
    """Spiking Vision Transformer for static image classification."""

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
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.T = T

        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.patch_lif = AdaptiveThresholdLIFNode(tau=2.0)

        self.blocks = nn.ModuleList(
            [SpikingTransformerBlock(embed_dim, num_heads=num_heads) for _ in range(depth)]
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
