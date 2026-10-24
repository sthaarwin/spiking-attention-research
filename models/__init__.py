"""
Neuromorphic Spiking Attention Models package.
"""

from .neurons import AdaptiveThresholdLIFNode, SurrogateHeaviside
from .attention_linear import AssociativeLinearAttention
from .attention_proposed import ThresholdGatedSpikingAttention

try:
    from .attention_vanilla import VanillaSoftmaxAttention
except ImportError:
    VanillaSoftmaxAttention = None

try:
    from .attention_ssa import SpikingSelfAttention
except ImportError:
    SpikingSelfAttention = None

try:
    from .backbones import (
        SpikingVisionTransformer,
        SpikingTransformerBlock,
        StandardVisionTransformer,
        StandardTransformerBlock,
        create_attention_layer,
    )
except ImportError:
    SpikingVisionTransformer = None
    SpikingTransformerBlock = None
    StandardVisionTransformer = None
    StandardTransformerBlock = None
    create_attention_layer = None

__all__ = [
    "AdaptiveThresholdLIFNode",
    "SurrogateHeaviside",
    "AssociativeLinearAttention",
    "ThresholdGatedSpikingAttention",
    "VanillaSoftmaxAttention",
    "SpikingSelfAttention",
    "SpikingVisionTransformer",
    "SpikingTransformerBlock",
    "StandardVisionTransformer",
    "StandardTransformerBlock",
    "create_attention_layer",
]
