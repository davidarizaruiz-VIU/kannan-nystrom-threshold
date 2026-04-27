"""
run_experiment5_highres.py
==========================
High-resolution extension of Experiment 5 (Section 7.7):
  Mesh-refinement self-consistency on the non-separable benchmark
  for N in {32, 64, 128, 256, 512, 1024}.

Reproduces the (extended) Table 7.5 of the manuscript and confirms the
asymptotic O(N^-2) decay predicted by Corollary 6.4 (kernel regularity-
limited Nystroem analysis on a kernel with cusp at {t=s}).

Usage from a terminal in the project folder:
    python3 run_experiment5_highres.py

Memory note: at N=1024 the dense Nystroem matrix occupies 8 MB; total
peak memory is well below 100 MB even with the four resolutions stored
simultaneously. Runs comfortably on a 14-inch MacBook Pro with 8 GB RAM.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from picard_nystrom import run_nonseparable_experiment


def main() -> int:
    print("=" * 80)
    print("  Experiment 5 (high-resolution) — Nystroem mesh refinement on the")
    print("                                   non-separable exponential benchmark")
    print("=" * 80)
    print()

    Ns = (32, 64, 128, 256, 512, 1024)
    print(f"  Resolutions tested: {Ns}")
    print(f"  Reference grid for the Nystroem extension: 401 equispaced points on [0,1]")
    print()

    t0 = time.perf_counter()
    out = run_nonseparable_experiment(Ns=Ns, output_dir=".")
    elapsed = time.perf_counter() - t0
    print(f"  Total wall-clock time: {elapsed:.2f} s")
    print()

    print("--- Discrete diagnostics convergence to continuous values ---")
    cont = out["continuous_diagnostics"]
    print(f"  Continuous: kappa={cont['kappa']:.6f}, mu_K={cont['mu_K']:.6f}, "
          f"int g={cont['g_int']:.6f}, delta_R={cont['delta_R']:.6f}, "
          f"lambda_G={cont['lambda_cont']:.6f}")
    print()
    print(f"  {'N':>5s}  {'kappa_N':>10s}  {'mu_N':>10s}  {'ell(G)':>10s}  "
          f"{'delta_NR':>10s}  {'lambda_N':>10s}  {'iters':>5s}")
    for r in out["resolution_results"]:
        print(f"  {r['N']:>5d}  {r['kappa_N']:>10.6f}  {r['mu_N']:>10.6f}  "
              f"{r['ell_G']:>10.6f}  {r['delta']:>10.6f}  "
              f"{r['lambda_N']:>10.6f}  {r['iters']:>5d}")

    print()
    print("--- Nystroem mesh-refinement self-consistency ||u_{2N}-u_N||_inf ---")
    print(f"  {'(N coarse, N fine)':>22s}  {'sup diff':>14s}  {'reduction':>10s}")
    prev = None
    for ref in out["mesh_refinement"]:
        red = (prev / ref["sup_diff"]) if prev is not None else None
        red_str = f"{red:>10.3f}" if red is not None else "       ---"
        print(f"  {('(' + str(ref['N_coarse']) + ', ' + str(ref['N_fine']) + ')'):>22s}  "
              f"{ref['sup_diff']:>14.3e}  {red_str}")
        prev = ref["sup_diff"]

    # Compute observed asymptotic order p from the two finest pairs
    diffs = [ref["sup_diff"] for ref in out["mesh_refinement"]]
    Nfines = [ref["N_fine"] for ref in out["mesh_refinement"]]
    if len(diffs) >= 2:
        p_last = -np.log(diffs[-1] / diffs[-2]) / np.log(Nfines[-1] / Nfines[-2])
        print(f"\n  Observed asymptotic order (last two pairs): p = {p_last:.4f}")
        print(f"  Theoretical order from Corollary 6.4         : p = 2 (kernel C^0 with cusp)")

    print()
    print("=" * 80)
    print("  JSON dump (for archival / programmatic use)")
    print("=" * 80)
    # Strip large arrays from JSON dump
    out_compact = {k: v for k, v in out.items() if k not in ("t_ref", "u_extensions")}
    print(json.dumps(out_compact, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
