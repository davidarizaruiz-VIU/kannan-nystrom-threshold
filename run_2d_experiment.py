"""
run_2d_experiment.py
====================
Driver script for Experiments 10 and 11 of the manuscript (Section 8.4):
  * Experiment 10: validation of the 2D Picard--Nyström on the separable
    bilinear benchmark with closed-form fixed point
        u^+(x,y) = (beta+sigma) + b (x+y),  b = 6 alpha (beta+sigma)/(24-7 alpha).
  * Experiment 11: mesh-refinement self-consistency on the non-separable
    exponential benchmark
        K(x,y; xi, eta) = alpha * exp(-|x-xi|) * exp(-|y-eta|),
    expected to display the predicted O(N^{-2}) Nyström rate.

File outputs:
  - fig_2d.pdf  (3-panel publication figure)

Usage from a terminal in the project folder:
    python3 run_2d_experiment.py

Memory note: the largest dense Nyström matrix occurs at N=64 (Experiment
11), namely 4096 x 4096 = 16M entries (128 MB in float64). The total
peak memory footprint stays below 1 GB on a 14-inch MacBook Pro M3 (8 GB
RAM); larger N (e.g. N=128, matrix 16384 x 16384 = 256M entries, 2 GB)
would require a sparse-matrix or low-rank approach beyond the scope of
this paper.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from picard_nystrom_2d import (
    benchmark_separable_2d_parameters,
    benchmark_nonseparable_2d_parameters,
    run_separable_2d_experiment,
    run_nonseparable_2d_experiment,
    plot_2d_overview,
)


def main() -> int:
    print("=" * 90)
    print("  Experiments 10-11 (Section 8.4) -- Two-dimensional extension")
    print("=" * 90)

    # ---------- Experiment 10: separable benchmark ---------------------
    print("\n--- Experiment 10: Separable bilinear benchmark (analytical) ---")
    p_sep = benchmark_separable_2d_parameters()
    print(f"  Parameters: alpha={p_sep['alpha']}, beta={p_sep['beta']}, "
          f"sigma={p_sep['sigma']}, c={p_sep['c']}, R={p_sep['R']}")
    print(f"  Closed-form fixed point: u^+(x,y) = (beta+sigma) + b (x+y)")
    print(f"  with b = 6*alpha*(beta+sigma)/(24 - 7*alpha) "
          f"= {6*p_sep['alpha']*(p_sep['beta']+p_sep['sigma'])/(24-7*p_sep['alpha']):.10f}")

    sep_results = []
    for N in (8, 16, 32, 64):
        t0 = time.perf_counter()
        sep = run_separable_2d_experiment(N=N)
        elapsed = time.perf_counter() - t0
        sep_results.append({
            "N":            N,
            "iters":        sep["iterations"],
            "residual":     sep["final_residual"],
            "sup_error":    sep["sup_error"],
            "ell_disc":     sep["ell_disc"],
            "ell_anal":     sep["ell_analytical"],
            "kappa_N":      sep["diag_disc"]["kappa_N"],
            "mu_N":         sep["diag_disc"]["mu_N"],
            "delta":        sep["diag_disc"]["delta"],
            "lambda_N":     sep["diag_disc"]["lambda_N"],
            "elapsed_s":    elapsed,
        })

    diag_cont_sep = sep["diag_cont"]
    print(f"\n  Continuous diagnostics (analytical):")
    print(f"    kappa_2  = {diag_cont_sep['kappa_2']:.6f}")
    print(f"    mu_K_2   = {diag_cont_sep['mu_K_2']:.6f}")
    print(f"    delta_R  = {diag_cont_sep['delta_R_2']:+.6f}")
    print(f"    gamma_R  = {diag_cont_sep['gamma_R_2']:.6f}")
    print(f"    lambda_R = {diag_cont_sep['lambda_R_2']:.6f}")
    print(f"    smallness lhs (3 kappa + 2 gamma) = "
          f"{diag_cont_sep['smallness_lhs']:.6f} < 1: "
          f"{diag_cont_sep['smallness_lhs'] < 1}")

    print(f"\n  Discrete results (Picard from x_0 = 0):")
    print(f"  {'N':>4s} {'iters':>5s} {'residual':>11s}  "
          f"{'sup-err vs exact':>18s}  {'ell_disc':>11s}  {'ell_anal':>11s}  "
          f"{'lambda_N':>10s}  {'time (s)':>9s}")
    for r in sep_results:
        print(f"  {r['N']:>4d} {r['iters']:>5d} "
              f"{r['residual']:>11.3e}  {r['sup_error']:>18.3e}  "
              f"{r['ell_disc']:>11.7f}  {r['ell_anal']:>11.7f}  "
              f"{r['lambda_N']:>10.6f}  {r['elapsed_s']:>9.3f}")

    # ---------- Experiment 11: non-separable benchmark -----------------
    print("\n--- Experiment 11: Non-separable exponential benchmark ---")
    p_ns = benchmark_nonseparable_2d_parameters()
    print(f"  Parameters: alpha={p_ns['alpha']}, beta={p_ns['beta']}, "
          f"sigma={p_ns['sigma']}, c={p_ns['c']}, R={p_ns['R']}")
    print(f"  Kernel: K(x,y; xi, eta) = alpha * exp(-|x-xi|) * exp(-|y-eta|)")
    print(f"  Reference grid for cross-resolution comparison: 41 x 41")

    t0 = time.perf_counter()
    nonsep = run_nonseparable_2d_experiment(Ns=(8, 16, 32, 64), n_ref=41)
    elapsed = time.perf_counter() - t0
    print(f"  Total wall-clock time: {elapsed:.2f} s")

    print(f"\n  Discrete diagnostics:")
    print(f"  {'N':>4s} {'kappa_N':>10s} {'mu_N':>10s} "
          f"{'ell_G':>10s} {'delta':>10s} {'lambda_N':>10s} "
          f"{'iters':>5s} {'residual':>11s} {'ell_U':>10s}")
    for r in nonsep["resolutions"]:
        print(f"  {r['N']:>4d} {r['kappa_N']:>10.6f} {r['mu_N']:>10.6f} "
              f"{r['ell_G']:>10.6f} {r['delta']:>10.6f} {r['lambda_N']:>10.6f} "
              f"{r['iters']:>5d} {r['final_residual']:>11.3e} {r['ell_U']:>10.6f}")

    print(f"\n  Mesh refinement (sup-norm differences on reference grid):")
    print(f"  {'(N coarse, N fine)':>22s}  {'sup diff':>14s}  "
          f"{'reduction':>10s}  {'order p':>8s}")
    prev_diff = None
    for ref in nonsep["mesh_refinement"]:
        red_str = f"{prev_diff/ref['sup_diff']:>10.3f}" if prev_diff is not None else "       ---"
        if prev_diff is not None and ref['sup_diff'] > 0:
            order = -np.log(ref['sup_diff'] / prev_diff) / np.log(2.0)
            ord_str = f"{order:>8.3f}"
        else:
            ord_str = "     ---"
        print(f"  {('(' + str(ref['N_coarse']) + ', ' + str(ref['N_fine']) + ')'):>22s}  "
              f"{ref['sup_diff']:>14.3e}  {red_str}  {ord_str}")
        prev_diff = ref["sup_diff"]
    print(f"\n  Theoretical Nyström rate (cusp kernel, 2D analogue of Sec 7.7): "
          f"O(N^{{-2}}) (i.e. p = 2)")

    # ---------- Figure ---------------------------------------------------
    plot_2d_overview(sep, nonsep, "./fig_2d.pdf")
    print(f"\nFigure written to ./fig_2d.pdf")

    # ---------- JSON dump -----------------------------------------------
    print("\n" + "=" * 90)
    print("  JSON dump (compact, for archival / programmatic use)")
    print("=" * 90)
    out = {
        "params_separable":      p_sep,
        "params_nonseparable":   p_ns,
        "continuous_diagnostics_separable": diag_cont_sep,
        "separable_results":     sep_results,
        "nonseparable_results":  nonsep["resolutions"],
        "mesh_refinement":       nonsep["mesh_refinement"],
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
