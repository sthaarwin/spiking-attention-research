"""
Computational complexity and analytical profiling script:
Evaluates MACs, ACs, and theoretical energy scaling across sequence lengths N.
"""

import sys
import os

# If imported as a module (e.g. by cProfile or torchvision internals),
# delegate to standard library profile to prevent namespace shadowing.
if __name__ != "__main__":
    _cwd = os.path.abspath(os.path.dirname(__file__))
    _clean_path = [p for p in sys.path if os.path.abspath(p) != _cwd]
    import importlib.util
    for _p in _clean_path:
        _cand = os.path.join(_p, "profile.py")
        if os.path.exists(_cand) and os.path.abspath(_cand) != os.path.abspath(__file__):
            _spec = importlib.util.spec_from_file_location("_stdlib_profile", _cand)
            if _spec and _spec.loader:
                _mod = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                globals().update(_mod.__dict__)
            break

import argparse
import json
from utils.complexity import profile_attention_scaling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Attention Scaling and Complexity")
    parser.add_argument("--dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--T", type=int, default=4, help="Number of timesteps")
    parser.add_argument("--output", type=str, default=None, help="Save profile results to JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    token_counts = (16, 64, 144, 256, 400)
    print(f"Profiling attention scaling for dim={args.dim}, heads={args.num_heads}, T={args.T}...")

    results = profile_attention_scaling(
        dim=args.dim,
        num_heads=args.num_heads,
        token_counts=token_counts,
        T=args.T,
        B=1,
    )

    print("\n" + "=" * 90)
    print(f"{'Mechanism':<12} | {'Tokens (N)':<10} | {'Total MACs':<12} | {'Total ACs':<12} | {'Attn MACs':<10} | {'Energy (pJ)':<12}")
    print("=" * 90)

    for method in ["vanilla", "ssa", "linear", "proposed"]:
        for row in results[method]:
            print(
                f"{method.capitalize():<12} | "
                f"{row['tokens']:<10} | "
                f"{row['macs']:<12,d} | "
                f"{row['acs']:<12,d} | "
                f"{row['attention_macs']:<10,d} | "
                f"{row['estimated_energy_pj']:<12,.1f}"
            )
        print("-" * 90)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Profiling results saved to {args.output}")


if __name__ == "__main__":
    main()
