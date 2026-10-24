"""
Layer-wise firing rate instrumentation, tracking, and pathological condition detection.
Detects dead neurons (<1% firing) and saturated neurons (>95% firing).
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from models.neurons import AdaptiveThresholdLIFNode


class FiringRateTracker:
    """
    Monitors firing rates across all spiking neuron modules in a neural network.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def get_layer_firing_rates(self) -> Dict[str, float]:
        """Collect firing rates from all named AdaptiveThresholdLIFNode modules."""
        rates = {}
        for name, module in self.model.named_modules():
            if isinstance(module, AdaptiveThresholdLIFNode):
                rates[name] = module.get_firing_rate()
        return rates

    def summarize(self) -> Dict[str, float]:
        """Compute summary statistics across all spiking layers in the model."""
        rates = self.get_layer_firing_rates()
        if not rates:
            return {
                "mean_firing_rate": 0.0,
                "min_firing_rate": 0.0,
                "max_firing_rate": 0.0,
                "num_spiking_layers": 0,
                "dead_layers_count": 0,
                "saturated_layers_count": 0,
            }

        values = list(rates.values())
        mean_rate = sum(values) / len(values)
        min_rate = min(values)
        max_rate = max(values)
        dead_count = sum(1 for v in values if v < 0.01)
        saturated_count = sum(1 for v in values if v > 0.95)

        return {
            "mean_firing_rate": mean_rate,
            "min_firing_rate": min_rate,
            "max_firing_rate": max_rate,
            "num_spiking_layers": len(values),
            "dead_layers_count": dead_count,
            "saturated_layers_count": saturated_count,
        }

    def check_health(self) -> Tuple[bool, List[str]]:
        """
        Check if any layer exhibits dead or saturated firing behavior.
        Returns (is_healthy, list_of_warnings).
        """
        rates = self.get_layer_firing_rates()
        warnings = []
        for name, rate in rates.items():
            if rate < 0.01:
                warnings.append(f"Layer '{name}' has near-zero firing rate: {rate:.4f} (potential dead neurons)")
            elif rate > 0.95:
                warnings.append(f"Layer '{name}' has saturated firing rate: {rate:.4f} (potential saturation)")

        is_healthy = len(warnings) == 0
        return is_healthy, warnings
