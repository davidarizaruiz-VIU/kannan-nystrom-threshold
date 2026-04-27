"""
run_experiment6.py
==================
Driver script for Experiment 6 of the manuscript:
  "Single-step transient and the role of the Kannan envelope".

Reproduces the data reported in Section 7.8 (Table 7.7 + parametric search).

Usage from a terminal in the project folder:
    python3 run_experiment6.py
"""
from __future__ import annotations

import json
import sys

from picard_nystrom import run_singlestep_experiment


def _fmt(x, fmt=".4e"):
    return format(x, fmt) if isinstance(x, (int, float)) else str(x)


def main() -> int:
    print("=" * 72)
    print("  Experiment 6 — single-step transient and Kannan envelope sharpness")
    print("=" * 72)

    result = run_singlestep_experiment(
        N=128, sigma=0.058,
        n_random_per_family=1000,
        seed=42,
    )

    p = result["params"]
    d = result["diagnostics"]
    print(f"\nBenchmark: non-separable kernel  K(t,s) = alpha exp(-|t-s|)")
    print(f"  N = {p['N']},  sigma = {p['sigma']},  c = {p['c']},  R = {p['R']}")
    print(f"  Discrete diagnostics:")
    print(f"    lambda_N      = {d['lambda_N']:.6f}   (Kannan stability constant)")
    print(f"    kappa_N       = {d['kappa_N']:.6f}   (Banach rate ||A_N||_inf)")
    print(f"    delta_N       = {d['delta_N']:.6f}   (lower-branch separation)")
    print(f"    k_N envelope  = {d['k_N_envelope']:.6f}   (= lambda_N / (1 - lambda_N))")

    rs = result["random_sweep"]
    print(f"\n--- (A) Random initial-data sweep ({rs['n_per_family']} trials per family) ---")
    print(f"    Family                                 max switches over {rs['n_per_family']} trials")
    for family, ms in rs["max_switches_per_family"].items():
        print(f"    {family:<38s}   {ms}")
    print(f"    Global maximum across all families:                {rs['global_max_switches']}")

    ps = result["parametric_search"]
    print(f"\n--- (B) Parametric search for max r_1/r_0 in lower branch ---")
    print(f"    Family: U^0(t) = alpha + beta*t + gamma*sin(pi*t),"
          f"  ell_N(U^0) <= c")
    print(f"    Admissible grid points : {ps['n_admissible_initial_data']}")
    print(f"    Best alpha             : {ps['best_alpha']:+.4f}")
    print(f"    Best beta              : {ps['best_beta']:+.4f}")
    print(f"    Best gamma             : {ps['best_gamma']:+.4f}")
    print(f"    Best ell_N(U^0)        : {ps['best_ell0']:+.6f}   (must be <= c = {p['c']})")
    print(f"    Best r_0               : {ps['best_r0']:.6e}")
    print(f"    Best r_1               : {ps['best_r1']:.6e}")
    print(f"    Switch flip at step 1  : theta_1 = {ps['best_theta1']}")
    print(f"    --------------------------------------------------------------")
    print(f"    Best empirical r_1/r_0 : {ps['best_one_step_ratio']:.4f}")
    print(f"    Kannan envelope k_N    : {d['k_N_envelope']:.4f}")
    print(f"    Gap (k_N / max ratio)  : {d['k_N_envelope'] / ps['best_one_step_ratio']:.2f}x")

    print("\n" + "=" * 72)
    print("  JSON dump (for archival / programmatic use)")
    print("=" * 72)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
