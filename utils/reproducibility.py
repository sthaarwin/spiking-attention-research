"""
Reproducibility utilities: deterministic seed control, environment logging,
and execution metadata serialization.
"""

import json
import os
import platform
import random
import subprocess
from typing import Any, Dict, Optional
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Sets deterministic random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Resolve target execution device."""
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_git_commit() -> str:
    """Retrieve current git commit hash if in a git repository."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        return commit
    except Exception:
        return "unknown"


def get_environment_info() -> Dict[str, Any]:
    """Capture software and hardware runtime environment for strict reproducibility."""
    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": get_git_commit(),
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_device_count"] = torch.cuda.device_count()
    return info


def save_run_metadata(
    output_dir: str, config: Dict[str, Any], extra: Optional[Dict[str, Any]] = None
) -> str:
    """Serializes execution configuration and environment metadata to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    metadata = {
        "config": config,
        "environment": get_environment_info(),
    }
    if extra:
        metadata.update(extra)

    filepath = os.path.join(output_dir, "run_metadata.json")
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2)
    return filepath
