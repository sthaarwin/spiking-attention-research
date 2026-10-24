"""
Utilities package for spiking-attention-research.
"""

from .energy_meter import EnergyMeter
from .entropy_tracker import EntropyTracker
from .datasets import (
    get_cifar10_loaders,
    get_cifar100_loaders,
    get_synthetic_loaders,
    get_cifar10_transforms,
    get_event_loaders,
)
from .reproducibility import set_seed, get_device, save_run_metadata, get_environment_info
from .firing_rate import FiringRateTracker
from .complexity import count_parameters, profile_attention_scaling

__all__ = [
    "EnergyMeter",
    "EntropyTracker",
    "get_cifar10_loaders",
    "get_cifar100_loaders",
    "get_synthetic_loaders",
    "get_cifar10_transforms",
    "get_event_loaders",
    "set_seed",
    "get_device",
    "save_run_metadata",
    "get_environment_info",
    "FiringRateTracker",
    "count_parameters",
    "profile_attention_scaling",
]
