"""
run_bistability_experiment.py
=============================
Driver script for Experiment 8 of the manuscript (Section 7.11):
  "Bistable regime: coexistence of branch fixed points".

Empirically validates Theorem 3.10 (bistable regime) and Corollary 3.11
(quantitative basin radii) on the separable benchmark of Section 7.2,
with parameters deliberately chosen outside the contractive Kannan
regime of Theorem 4.1 (i.e., delta < 0).

Outputs printed to stdout:
  - Branch fixed points u^- and u^+ and their threshold images L u^pm.
  - Branch margins m^- = L u^- - c and m^+ = L u^+ - c.
  - Bistability check (m^- <= 0 < m^+) and width of the bistable interval.
  - Basin radii r^- = -m^-/||L|| and r^+ = m^+/||L|| (Cor. 3.11).
  - Two reference Picard runs (from x_0 = 0 and x_0 = 2 * 1).
  - Sweep table over the 1-parameter family x_0 = alpha * 1.
  - Confirmation that the global Kannan diagnostic of Theorem 4.1
    is INADMISSIBLE on these parameters (delta < 0).

File outputs:
  - fig_bistability.pdf  (4-panel publication-quality figure).

Usage from a terminal in the project folder:
    python3 run_bistability_experiment.py
"""
from __future__ import annotations

import json
import sys

import numpy as np

from picard_nystrom import (
    bistability_parameters,
    run_bistability_experiment,
    plot_bistability,
)


def main() -> int:
    print("=" * 86)
    print("  Experiment 8 (Section 7.11) -- Bistable regime")
    print("                                  Coexistence of branch fixed points")
    print("=" * 86)

    res = run_bistability_experiment(N=128, n_alpha=41, alpha_max=2.0)
    p   = res["params"]
    br  = res["branch"]
    kn  = res["global_kannan"]

    print(f"\nParameters (separable benchmark):")
    print(f"  alpha = {p['alpha']}, beta = {p['beta']}, rho = {p['rho']}, "
          f"sigma = {p['sigma']}")
    print(f"  c = {p['c']}, R = {p['R']},  Nystrom nodes N = {res['N']}")

    print(f"\nBranch fixed points and margins:")
    print(f"  ell(u^-) = {br['Lu_minus']:.10f}")
    print(f"  ell(u^+) = {br['Lu_plus']:.10f}")
    print(f"  m^- = ell(u^-) - c = {br['m_minus']:+.10f}    "
          f"(must be <= 0)")
    print(f"  m^+ = ell(u^+) - c = {br['m_plus']:+.10f}    "
          f"(must be  > 0)")
    print(f"  width of bistable interval (in c): "
          f"L (I-A)^-1 q = {br['width_bistable']:.10f}")
    print(f"  Bistability satisfied: "
          f"{'YES' if br['bistable'] else 'NO'}")
    print(f"  ||ell_N|| = sum |w_i|  = {br['norm_L']:.10f}")
    print(f"  Basin radii (Corollary 3.11):")
    print(f"    r^- = -m^- / ||ell|| = {br['r_minus_basin']:.10f}")
    print(f"    r^+ =  m^+ / ||ell|| = {br['r_plus_basin']:.10f}")

    print(f"\nGlobal Kannan diagnostic (Theorem 4.1) on these parameters:")
    print(f"  kappa_N  = {kn['kappa_N']:.6f}")
    print(f"  mu_N     = {kn['mu_N']:.6f}")
    print(f"  ell(G)   = {kn['ell_G']:.6f}")
    print(f"  delta    = {kn['delta']:+.6f}    "
          f"(negative => Theorem 4.1 INADMISSIBLE; "
          f"the bistable analysis of Theorem 3.10 applies)")
    print(f"  lambda_N = {kn['lambda_N']:.4f} "
          f"(reported as +inf when delta <= 0 by convention)")
    print(f"  admissible: {kn['admissible']}")

    rl = res["run_low"]
    rh = res["run_high"]
    print(f"\nReference Picard runs:")
    print(f"  x_0 = 0 (sup-norm dist to u^-: {rl['diff_lower']:.3e}, "
          f"to u^+: {rl['diff_upper']:.3e})")
    print(f"     iters = {rl['iters']:>3d}  final residual = "
          f"{rl['final_residual']:.3e}  attractor = {rl['attractor']}")
    print(f"  x_0 = 2 * 1 (sup-norm dist to u^-: {rh['diff_lower']:.3e}, "
          f"to u^+: {rh['diff_upper']:.3e})")
    print(f"     iters = {rh['iters']:>3d}  final residual = "
          f"{rh['final_residual']:.3e}  attractor = {rh['attractor']}")

    print(f"\nSweep over constant initial conditions x_0 = alpha * 1:")
    print(f"  {'alpha':>6s}  {'attractor':>10s}  {'iters':>6s}  "
          f"{'#cross':>7s}  {'final residual':>14s}  "
          f"{'sup-dist u^-':>13s}  {'sup-dist u^+':>13s}")
    for r in res["sweep"]:
        print(f"  {r['alpha']:>6.3f}  {r['attractor']:>10s}  "
              f"{r['iters']:>6d}  {r['n_crossings']:>7d}  "
              f"{r['final_residual']:>14.3e}  "
              f"{r['final_diff_lower']:>13.3e}  "
              f"{r['final_diff_upper']:>13.3e}")

    # Identify the empirical basin boundary (crossover alpha)
    attrs = [r["attractor"] for r in res["sweep"]]
    alphas = [r["alpha"]    for r in res["sweep"]]
    boundary = None
    for i in range(1, len(attrs)):
        if attrs[i] != attrs[i-1]:
            boundary = (alphas[i-1], alphas[i])
            break

    print(f"\nEmpirical basin boundary (crossover alpha):")
    if boundary is None:
        print(f"  No transition detected in the sweep.")
    else:
        print(f"  Lower-basin upper boundary in the sweep grid: "
              f"alpha in [{boundary[0]:.4f}, {boundary[1]:.4f}]")
        print(f"  Threshold value c = {p['c']}; ell(x_0) = alpha for these "
              f"initial data, so the predicted basin boundary in this family "
              f"is simply alpha = c.")

    # Maximum number of branch crossings observed
    max_cross = max(r["n_crossings"] for r in res["sweep"])
    print(f"\nMaximum number of branch crossings observed in sweep: "
          f"{max_cross}")
    print(f"  (Empirical evidence consistent with Remark 3.13: outside the "
          f"basins, multi-crossings can occur but are typically bounded by "
          f"a small constant for this benchmark.)")

    # Generate figure
    plot_bistability(res, "./fig_bistability.pdf")
    print(f"\nFigure written to ./fig_bistability.pdf")

    print("\n" + "=" * 86)
    print("  JSON dump (compact, for archival / programmatic use)")
    print("=" * 86)
    out = {
        "params":         p,
        "N":              res["N"],
        "branch_summary": {
            "Lu_minus":       br["Lu_minus"],
            "Lu_plus":        br["Lu_plus"],
            "m_minus":        br["m_minus"],
            "m_plus":         br["m_plus"],
            "width_bistable": br["width_bistable"],
            "norm_L":         br["norm_L"],
            "r_minus_basin":  br["r_minus_basin"],
            "r_plus_basin":   br["r_plus_basin"],
            "bistable":       br["bistable"],
        },
        "global_kannan_diag": kn,
        "run_low_summary":  {
            "iters": rl["iters"], "attractor": rl["attractor"],
            "final_residual": rl["final_residual"],
        },
        "run_high_summary": {
            "iters": rh["iters"], "attractor": rh["attractor"],
            "final_residual": rh["final_residual"],
        },
        "sweep_summary": [
            {"alpha": r["alpha"], "iters": r["iters"],
             "attractor": r["attractor"],
             "n_crossings": r["n_crossings"]}
            for r in res["sweep"]
        ],
        "max_crossings": max_cross,
        "boundary_grid": list(boundary) if boundary is not None else None,
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
