"""
run_two_sided_iteration.py
==========================
Empirical comparison of the standard Picard--Nystroem iteration with the
two-sided sweep (sub-solution / super-solution) introduced in Section 7.X
of the manuscript.

For two reference benchmarks --- the contractive Kannan setting of
Section 7.2 (separable kernel) and the bistable setting of Section 7.5
(separable kernel, large jump) --- the script

  (a) runs the plain Picard iteration from the canonical seed U^0 = c/2,
  (b) runs the two-sided Picard sweep from U_lower0 = 0 and U_upper0 = R,
  (c) reports iterations to tolerance and the terminal separation
      sigma_K = ||U^K_+ - U^K_-||_inf.

In the contractive regime the two sweeps must coalesce (sigma_K -> 0),
empirically certifying uniqueness of the discrete fixed point. In the
bistable regime they must split (sigma_K -> sigma_* > 0) and recover the
two distinct branch fixed points u_+, u_-.

Usage from a terminal in the project folder:
    python3 run_two_sided_iteration.py
"""
from __future__ import annotations

import json
import sys

import numpy as np

from picard_nystrom import (
    benchmark_parameters,
    bistability_parameters,
    bistability_branch_diagnostics,
    build_benchmark_functions,
    assemble,
    picard_nystrom,
    two_sided_picard,
    kannan_diagnostics,
)


def _run_one(label: str, p: dict, N: int = 128, tol: float = 1e-13):
    """Run plain Picard and two-sided sweep on a single benchmark."""
    K, g, q = build_benchmark_functions(p)
    sys_ = assemble(N, K, g, q, rule="gauss")
    diag = kannan_diagnostics(sys_, p["c"], p["R"])

    # (a) Plain Picard from canonical seed U^0 = c/2 * 1.
    pic = picard_nystrom(sys_, p["c"],
                         U0=np.full(N, p["c"] / 2.0),
                         tol=tol, nmax=500)

    # (b) Two-sided sweep from U_lo0 = 0, U_hi0 = R*1.
    twos = two_sided_picard(sys_, p["c"],
                            U_lower0=np.zeros(N),
                            U_upper0=np.full(N, p["R"]),
                            tol=tol, nmax=500)

    # Branch reference (always available; relevant in bistable regime).
    branch = bistability_branch_diagnostics(sys_, p["c"])

    # Terminal observables.
    sigma_terminal = float(twos["sigmas"][-1])
    pic_residual   = float(pic["residuals"][-1])
    twos_res_lo    = float(twos["residuals_lower"][-1])
    twos_res_hi    = float(twos["residuals_upper"][-1])

    # Distance of the terminal sweeps to the analytic branches.
    d_lo_to_uminus = float(np.max(np.abs(twos["U_lower"] - branch["u_minus"])))
    d_lo_to_uplus  = float(np.max(np.abs(twos["U_lower"] - branch["u_plus"])))
    d_hi_to_uminus = float(np.max(np.abs(twos["U_upper"] - branch["u_minus"])))
    d_hi_to_uplus  = float(np.max(np.abs(twos["U_upper"] - branch["u_plus"])))

    print(f"\n--- {label} ---")
    print(f"  Parameters: {p}")
    print(f"  N = {N}, lambda_N = {diag['lambda_N']:.6f}, "
          f"Kannan-admissible: {'YES' if diag['lambda_N'] < 0.5 else 'NO'}")
    print(f"  Branch separation L u_+ - L u_- = {branch['width_bistable']:.6f}")
    print(f"  Branch margins m^- = {branch['m_minus']:+.6f}, "
          f"m^+ = {branch['m_plus']:+.6f}")
    print(f"\n  Plain Picard (U^0 = c/2 * 1):")
    print(f"    iterations to tol = {pic['iters']:3d}    "
          f"residual = {pic_residual:.3e}")
    print(f"\n  Two-sided sweep (U_lo0 = 0, U_hi0 = R*1):")
    print(f"    iterations to tol           = {twos['iters']:3d}")
    print(f"    residual lower / upper      = "
          f"{twos_res_lo:.3e} / {twos_res_hi:.3e}")
    print(f"    terminal separation sigma_K = {sigma_terminal:.6e}")
    print(f"    distance(U_lo, u_-) = {d_lo_to_uminus:.3e}    "
          f"distance(U_lo, u_+) = {d_lo_to_uplus:.3e}")
    print(f"    distance(U_hi, u_-) = {d_hi_to_uminus:.3e}    "
          f"distance(U_hi, u_+) = {d_hi_to_uplus:.3e}")
    if sigma_terminal < 1e-6:
        verdict = "UNIQUENESS CERTIFIED (sigma_K ~ 0)"
    else:
        verdict = "MULTIPLICITY DETECTED (sigma_K > 0)"
    print(f"    Verdict: {verdict}")

    return {
        "label":             label,
        "params":            p,
        "N":                 N,
        "lambda_N":          float(diag["lambda_N"]),
        "branch_width":      float(branch["width_bistable"]),
        "m_minus":           float(branch["m_minus"]),
        "m_plus":            float(branch["m_plus"]),
        "picard_iters":      int(pic["iters"]),
        "picard_residual":   pic_residual,
        "twos_iters":        int(twos["iters"]),
        "twos_res_lower":    twos_res_lo,
        "twos_res_upper":    twos_res_hi,
        "sigma_terminal":    sigma_terminal,
        "d_lo_to_uminus":    d_lo_to_uminus,
        "d_lo_to_uplus":     d_lo_to_uplus,
        "d_hi_to_uminus":    d_hi_to_uminus,
        "d_hi_to_uplus":     d_hi_to_uplus,
        "uniqueness_certified": sigma_terminal < 1e-6,
    }


def main() -> int:
    print("=" * 90)
    print("  Empirical comparison: plain Picard vs two-sided monotone sweep")
    print("=" * 90)

    # Benchmark A: contractive Kannan regime (Section 7.2 reference parameters).
    p_contr = benchmark_parameters()
    res_contr = _run_one("Benchmark A (contractive Kannan regime, sec.~7.2)",
                         p_contr)

    # Benchmark B: bistable regime (Section 7.5 reference parameters).
    p_bist = bistability_parameters()
    res_bist = _run_one("Benchmark B (bistable regime, sec.~7.5)",
                        p_bist)

    # Print compact summary table.
    print("\n" + "=" * 90)
    print("  Compact summary")
    print("=" * 90)
    print(f"  {'benchmark':>34s}  {'lambda_N':>9s}  {'Pic iters':>10s}  "
          f"{'2-sweep iters':>14s}  {'sigma_K':>11s}  {'verdict':>20s}")
    for r in (res_contr, res_bist):
        verdict = "uniqueness" if r["uniqueness_certified"] else "multiplicity"
        print(f"  {r['label'][:34]:>34s}  {r['lambda_N']:>9.4f}  "
              f"{r['picard_iters']:>10d}  {r['twos_iters']:>14d}  "
              f"{r['sigma_terminal']:>11.3e}  {verdict:>20s}")

    print("\n" + "=" * 90)
    print("  JSON dump (compact)")
    print("=" * 90)
    print(json.dumps({"contractive": res_contr,
                      "bistable":    res_bist},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
