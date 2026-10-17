"""
Neuromorphic Neurons: Adaptive Threshold and Standard LIF Nodes.
Implements surrogate gradient backpropagation and dynamic threshold gating:
    theta_d = alpha * Mean(U) + beta
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ATanSurrogate(torch.autograd.Function):
    """
    Arctangent surrogate gradient function for Heaviside step.
    Forward: Heaviside step: 1.0 if x >= 0 else 0.0.
    Backward: Derivative of arctan(alpha * x): alpha / (2 * (1 + (pi/2 * alpha * x)^2)).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0.0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad = alpha / (2.0 * (1.0 + (math.pi / 2.0 * alpha * x).pow(2)))
        return grad_output * grad, None


class SurrogateHeaviside(nn.Module):
    """Surrogate spike generation module."""

    def __init__(self, alpha: float = 2.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ATanSurrogate.apply(x, self.alpha)


class AdaptiveThresholdLIFNode(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) neuron with dynamic threshold gating.
    
    Threshold equation:
        theta_d = alpha * Mean(U) + beta
    
    When adaptive=False, operates as a standard fixed-threshold LIF node
    with threshold = beta.
    """

    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = False,
        v_reset: float = 0.0,
        initial_alpha: float = 1.0,
        initial_beta: float = 1.0,
        surrogate_alpha: float = 2.0,
        adaptive: bool = True,
        reduction_dims: Tuple[int, ...] = (-2,),  # Average across token dimension N by default
    ):
        super().__init__()
        self.tau = tau
        self.decay = 1.0 / tau
        self.decay_input = decay_input
        self.v_reset = v_reset
        self.adaptive = adaptive
        self.reduction_dims = reduction_dims

        # Learnable gating parameters
        if self.adaptive:
            self.alpha = nn.Parameter(torch.tensor(float(initial_alpha)))
        else:
            self.register_buffer("alpha", torch.tensor(0.0))

        self.beta = nn.Parameter(torch.tensor(float(initial_beta)))

        # Spike generator
        self.spike_fn = SurrogateHeaviside(alpha=surrogate_alpha)

        # Internal state
        self.v_membrane: Optional[torch.Tensor] = None
        self.last_spike: Optional[torch.Tensor] = None

        # Instrumentation
        self.total_spikes: int = 0
        self.total_elements: int = 0

    def reset(self) -> None:
        """Reset internal membrane state and spike buffers between batches."""
        self.v_membrane = None
        self.last_spike = None
        self.total_spikes = 0
        self.total_elements = 0

    def compute_dynamic_threshold(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute dynamic threshold:
            theta_d = alpha * Mean(U) + beta
        Keeps dimension matched via keepdim=True over reduction_dims.
        """
        if not self.adaptive:
            return self.beta

        # Local activation statistics
        mean_u = u.mean(dim=self.reduction_dims, keepdim=True)
        # Ensure dynamic threshold remains strictly positive with softplus or smooth clamping
        theta_d = self.alpha * mean_u + self.beta
        return F.relu(theta_d) + 1e-4

    def single_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single time-step integration:
        x: [B, N, D] or [B, H, N, d_h]
        Returns (spike, v_membrane).
        """
        if self.v_membrane is None:
            self.v_membrane = torch.zeros_like(x)
            self.last_spike = torch.zeros_like(x)

        # Membrane decay with hard reset
        v_prev = self.v_membrane * (1.0 - self.last_spike) + self.v_reset * self.last_spike
        v_leak = v_prev * (1.0 - self.decay)
        u = v_leak + x

        # Dynamic threshold computation
        theta_d = self.compute_dynamic_threshold(u)

        # Spike generation via surrogate function
        spike = self.spike_fn(u - theta_d)

        # Update state
        self.v_membrane = u
        self.last_spike = spike

        # Track firing statistics
        with torch.no_grad():
            self.total_spikes += int(spike.sum().item())
            self.total_elements += spike.numel()

        return spike, u

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multi-step forward:
        x: [T, B, N, D] or [T, B, H, N, d_h]
        Returns: spikes of same shape as x.
        """
        time_steps = x.shape[0]
        spikes = []
        for t in range(time_steps):
            s_t, _ = self.single_step(x[t])
            spikes.append(s_t)
        return torch.stack(spikes, dim=0)

    def get_firing_rate(self) -> float:
        """Calculate active firing rate in range [0.0, 1.0]."""
        if self.total_elements == 0:
            return 0.0
        return self.total_spikes / self.total_elements
