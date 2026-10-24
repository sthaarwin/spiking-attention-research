"""
Complexity and efficiency profiling: parameter counts, operation accounting (MACs, ACs, SOPs),
and asymptotic scaling analysis across sequence lengths.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from .energy_meter import EnergyMeter


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters in a PyTorch module."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": total - trainable,
    }


def profile_attention_scaling(
    dim: int = 64,
    num_heads: int = 4,
    token_counts: Tuple[int, ...] = (16, 64, 144, 256, 400),
    T: int = 4,
    B: int = 1,
) -> Dict[str, List[Dict[str, float]]]:
    """
    Computes theoretical operation counts across different sequence lengths N
    comparing Quadratic Softmax, Quadratic SSA, Associative Linear Attention,
    and Proposed Multiplication-Free Attention.
    """
    from models.attention_vanilla import VanillaSoftmaxAttention
    from models.attention_ssa import SpikingSelfAttention
    from models.attention_linear import AssociativeLinearAttention
    from models.attention_proposed import ThresholdGatedSpikingAttention

    vanilla = VanillaSoftmaxAttention(dim=dim, num_heads=num_heads)
    ssa = SpikingSelfAttention(dim=dim, num_heads=num_heads)
    linear = AssociativeLinearAttention(dim=dim, num_heads=num_heads)
    proposed = ThresholdGatedSpikingAttention(dim=dim, num_heads=num_heads)

    results = {
        "vanilla": [],
        "ssa": [],
        "linear": [],
        "proposed": [],
    }

    meter = EnergyMeter()

    for n in token_counts:
        # Vanilla
        ops_v = vanilla.count_operations(T=T, B=B, N=n)
        e_v = meter.estimate_energy_pj(sops=0, flops=ops_v["projection_macs"] + ops_v["attention_macs"])
        results["vanilla"].append({
            "tokens": n,
            "macs": ops_v["projection_macs"] + ops_v["attention_macs"],
            "acs": 0,
            "attention_macs": ops_v["attention_macs"],
            "attention_acs": 0,
            "estimated_energy_pj": e_v,
        })

        # SSA
        ops_ssa = ssa.count_operations(T=T, B=B, N=n)
        e_ssa = meter.estimate_energy_pj(
            sops=ops_ssa["attention_acs"],
            flops=ops_ssa["projection_macs"] + ops_ssa["attention_macs"]
        )
        results["ssa"].append({
            "tokens": n,
            "macs": ops_ssa["projection_macs"] + ops_ssa["attention_macs"],
            "acs": ops_ssa["attention_acs"],
            "attention_macs": ops_ssa["attention_macs"],
            "attention_acs": ops_ssa["attention_acs"],
            "estimated_energy_pj": e_ssa,
        })

        # Linear
        ops_l = linear.count_operations(T=T, B=B, N=n)
        e_l = meter.estimate_energy_pj(sops=0, flops=ops_l["projection_macs"] + ops_l["attention_macs"])
        results["linear"].append({
            "tokens": n,
            "macs": ops_l["projection_macs"] + ops_l["attention_macs"],
            "acs": 0,
            "attention_macs": ops_l["attention_macs"],
            "attention_acs": 0,
            "estimated_energy_pj": e_l,
        })

        # Proposed (Multiplication-free in attention interaction)
        ops_p = proposed.count_operations(T=T, B=B, N=n)
        e_p = meter.estimate_energy_pj(
            sops=ops_p["attention_acs"],
            flops=ops_p["projection_macs"]
        )
        results["proposed"].append({
            "tokens": n,
            "macs": ops_p["projection_macs"],
            "acs": ops_p["attention_acs"],
            "attention_macs": 0,
            "attention_acs": ops_p["attention_acs"],
            "estimated_energy_pj": e_p,
        })

    return results
