"""
Analytical profiling of spiking activation distributions: entropy, firing rate,
and top-k concentration.
"""

import torch
import torch.nn.functional as F


class EntropyTracker:
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def compute_metrics(self, activations: torch.Tensor, k: int = 4) -> dict:
        p = F.softmax(activations.float(), dim=-1)

        entropy = -(p * torch.log(p + self.epsilon)).sum(dim=-1).mean().item()
        firing_rate = (activations > 0).float().mean().item()

        k = min(k, p.shape[-1])
        top_k = torch.topk(p, k=k, dim=-1).values.sum(dim=-1)
        total = p.sum(dim=-1)
        top_k_concentration = (top_k / (total + self.epsilon)).mean().item()

        return {
            "entropy": entropy,
            "firing_rate": firing_rate,
            "top_k_concentration": top_k_concentration,
        }
