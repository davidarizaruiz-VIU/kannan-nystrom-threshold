"""
run_application.py
==================
Driver script for the applied case study of Section 7.10:
  Spatial population model with threshold-activated harvesting.

Solves the steady-state equation
    u(x) = int_0^1 K(x,y) u(y) dy + g(x) - h(x) H(int u - c),
with K(x,y) = alpha exp(-|x-y|), g(x) = (beta/2)(1+cos(pi x)),
h(x) = eta x, c the carrying-capacity threshold and R the
invariance ball radius. Reference: Cantrell & Cosner (2003).

Outputs:
  - Diagnostics certifying the Kannan stability conditions of Theorem 4.1.
  - Equilibrium density profile u_*(x), with comparison to the
    no-harvesting baseline (I-A)^{-1} g.
  - Sensitivity sweep: total equilibrium population vs harvesting effort eta.
  - Figure file fig_application.pdf.

Usage from a terminal in the project folder:
    python3 run_application.py
"""
from __future__ import annotations

import json
import sys

import numpy as np

from picard_nystrom import (
    population_parameters,
    run_population_application,
    run_population_sweep,
    plot_population_application,
)


def main() -> int:
    print("=" * 80)
    print("  Applied case study (Section 7.10) — spatial population model")
    print("                                       with threshold harvesting")
    print("=" * 80)

    # ----- Single-run diagnostics + equilibrium -----
    res = run_population_application(N=128)
    p = res["params"]
    dc = res["diagnostics_cont"]
    dd = res["diagnostics_disc"]

    print(f"\nNominal parameters:")
    print(f"  alpha={p['alpha']}, beta={p['beta']}, eta={p['eta']}, "
          f"c={p['c']}, R={p['R']}")
    print(f"  N = {res['N']} Gauss-Legendre nodes")

    print(f"\nContinuous Kannan diagnostics:")
    print(f"  kappa = {dc['kappa']:.5f}")
    print(f"  mu_K  = {dc['mu_K']:.5f}")
    print(f"  Lg    = {dc['g_int']:.5f}    Lq = {dc['Lq']:+.5f}")
    print(f"  ||g|| = {dc['norm_g']:.5f}    ||q|| = {dc['norm_q']:.5f}")
    print(f"  m_R   = delta_R = {dc['delta_R']:.5f}")
    print(f"  Invariance: {dc['invariance_lhs']:.5f} <= R = {dc['invariance_rhs']:.5f} "
          f"({'OK' if dc['invariance_lhs'] <= dc['invariance_rhs'] else 'FAIL'})")
    print(f"  Kannan LHS: 3 kappa + 2||q||/m_R = {dc['kannan_lhs']:.5f} < 1 "
          f"({'OK' if dc['kannan_lhs'] < 1 else 'FAIL'})")
    print(f"  Kannan constant: lambda_R = {dc['lambda_R']:.5f} < 1/2 "
          f"({'OK' if dc['lambda_R'] < 0.5 else 'FAIL'})")
    print(f"  Single-step transient (Lq + m_R): {dc['Lq']+dc['delta_R']:+.5f} > 0 "
          f"({'OK' if dc['Lq']+dc['delta_R'] > 0 else 'FAIL'})")

    print(f"\nDiscrete Kannan diagnostics (N={res['N']}):")
    print(f"  kappa_N = {dd['kappa_N']:.5f}    mu_N = {dd['mu_N']:.5f}")
    print(f"  delta_N = {dd['delta']:.5f}    lambda_N = {dd['lambda_N']:.5f}")

    print(f"\nIteration outcome:")
    print(f"  Iterations to tolerance 1e-13 : {res['iterations']}")
    print(f"  Final residual                : {res['final_residual']:.3e}")
    print(f"  Total population WITHOUT harvest: ell((I-A)^-1 g) = {res['ell_no_harvest']:.6f}")
    print(f"  Total population WITH harvest   : ell(u_*)         = {res['ell_star']:.6f}")
    print(f"  Reduction by regulator          : {res['harvest_reduction']:+.6f} "
          f"({100*res['harvest_reduction']/res['ell_no_harvest']:.2f}% of source)")
    print(f"  Margin above threshold c={p['c']}: {res['regulator_active_margin']:+.6f}")

    # ----- Sensitivity sweep -----
    print(f"\nSensitivity sweep over harvesting effort eta:")
    eta_max = 0.122    # just below the Kannan admissibility limit (~0.124)
    etas = np.linspace(0.0, eta_max, 13)
    sweep = run_population_sweep(etas, N=res["N"])

    print(f"  {'eta':>8s}  {'lambda_R':>10s}  {'admis':>6s}  {'ell(u*)':>10s}  {'iters':>6s}")
    for r in sweep["rows"]:
        ell_str = f"{r['ell_star']:.6f}" if r["ell_star"] is not None else "    --"
        it_str  = f"{r['iters']:6d}" if r["iters"] is not None else "    --"
        print(f"  {r['eta']:>8.4f}  {r['lambda_R']:>10.4f}  "
              f"{('YES' if r['admissible'] else 'NO'):>6s}  {ell_str:>10s}  {it_str}")

    # ----- Figure -----
    plot_population_application(res, sweep, "./fig_application.pdf")
    print(f"\nFigure written to ./fig_application.pdf")

    print("\n" + "=" * 80)
    print("  JSON dump (compact)")
    print("=" * 80)
    out = {
        "params": p,
        "diagnostics_cont": dc,
        "diagnostics_disc": dd,
        "iterations": res["iterations"],
        "final_residual": res["final_residual"],
        "ell_no_harvest": res["ell_no_harvest"],
        "ell_star": res["ell_star"],
        "harvest_reduction": res["harvest_reduction"],
        "sweep": sweep,
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
