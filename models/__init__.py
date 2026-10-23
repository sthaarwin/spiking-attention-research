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
    from .backbones import SpikingVisionTransformer, SpikingTransformerBlock
except ImportError:
    SpikingVisionTransformer = None
    SpikingTransformerBlock = None

__all__ = [
    "AdaptiveThresholdLIFNode",
    "SurrogateHeaviside",
    "AssociativeLinearAttention",
    "ThresholdGatedSpikingAttention",
    "VanillaSoftmaxAttention",
    "SpikingSelfAttention",
    "SpikingVisionTransformer",
    "SpikingTransformerBlock",
]
