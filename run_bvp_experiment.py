"""
run_bvp_experiment.py
=====================
Driver script for the Green-kernel boundary-value-problem experiment of
Section 7 (BVP formulation) of the companion paper.

Solves a one-dimensional second-order BVP whose Green-kernel reformulation
falls into the Kannan-stable threshold class

    u(t) = (A u)(t) + g(t) + q(t) H( int_0^1 u - c ),

with the closed-form parameters of `bvp_parameters()`. Reports continuous
and discrete Kannan diagnostics, the Picard-Nystroem iteration history and
the sup-norm error against the analytical fixed point.

Outputs:
  - tab:bvp-diagnostics, tab:bvp-history (printed to stdout),
  - figure fig_bvp.pdf (converged solution + residual decay).

Usage:
    python3 run_bvp_experiment.py
"""
from __future__ import annotations

import json
import os
import sys

from picard_nystrom import run_bvp_experiment


def main() -> int:
    print("=" * 92)
    print("  Green-kernel BVP experiment (Section 7, BVP formulation)")
    print("=" * 92)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    result = run_bvp_experiment(N=64, output_dir=out_dir)

    p = result["params"]
    diag_c = result["continuous_diagnostics"]
    diag_N = result["discrete_diagnostics"]

    print("\n  --- Parameters (Green formulation of -u''=a0 u + b0 + d0 H(int u - c)) ---")
    print(f"  a0 = {p['a0']:.4f}, b0 = {p['b0']:.4f}, d0 = {p['d0']:.4f}, "
          f"c = {p['c']:.4f}, R = {p['R']:.4f}")

    print("\n  --- Continuous diagnostics ---")
    for k, v in sorted(diag_c.items()):
        if isinstance(v, (int, float)):
            print(f"  {k:>10s} = {v:.6f}")
        else:
            print(f"  {k:>10s} = {v}")

    print("\n  --- Discrete diagnostics (N = 64) ---")
    for k, v in sorted(diag_N.items()):
        if isinstance(v, (int, float)):
            print(f"  {k:>10s} = {v:.6f}")
        else:
            print(f"  {k:>10s} = {v}")

    print("\n  --- Iteration history ---")
    print(f"  iterations:        {result['iterations']}")
    print(f"  final residual:    {result['final_residual']:.3e}")
    print(f"  sup-error vs exact:{result['sup_error_vs_exact']:.3e}")

    print(f"\n  Figure written: {result['figure']}")

    summary = {
        "iterations": result["iterations"],
        "final_residual": result["final_residual"],
        "sup_error_vs_exact": result["sup_error_vs_exact"],
        "lambda_N": diag_N.get("lambda_N", None),
    }
    print("\n  JSON summary: " + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
