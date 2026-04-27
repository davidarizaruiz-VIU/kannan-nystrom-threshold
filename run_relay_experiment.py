"""
run_relay_experiment.py
=======================
Driver script for the applied case study II of Section 7.13:
  Relay control with integral observation.

Solves the steady-state equation
    y(t) = alpha int_0^1 K(t,s) y(s) ds + g(t) + q(t) H(<y> - c),
with K(t,s) = exp(-2|t-s|), g(t) = beta cos(pi t / 2), q(t) = -sigma t,
the standard Tsypkin / Goebel-Sanfelice-Teel formulation of relay
control with integral observation.

Outputs:
  - Continuous and discrete Kannan diagnostics certifying admissibility.
  - Equilibrium output y_*(t), with comparison to the open-loop reference
    y_open = (I-A)^{-1} g.
  - Sensitivity sweep: equilibrium ell(y_*) vs. setpoint c, over the
    Kannan-admissible range of c.
  - Figure file fig_relay.pdf.

Usage from a terminal in the project folder:
    python3 run_relay_experiment.py
"""
from __future__ import annotations

import json
import sys

import numpy as np

from picard_nystrom import (
    relay_control_parameters,
    relay_control_continuous_diagnostics,
    run_relay_control_application,
    run_relay_setpoint_sweep,
    plot_relay_control,
)


def main() -> int:
    print("=" * 86)
    print("  Applied case study II (Section 7.13) — Relay control with integral")
    print("                                          observation")
    print("=" * 86)

    # ----- Single-run diagnostics + equilibrium -----
    res = run_relay_control_application(N=128)
    p   = res["params"]
    dc  = res["diagnostics_cont"]
    dd  = res["diagnostics_disc"]

    print(f"\nNominal parameters (relay control with integral observation):")
    print(f"  alpha={p['alpha']}, beta={p['beta']}, sigma={p['sigma']}, "
          f"c={p['c']}, R={p['R']}")
    print(f"  N = {res['N']} Gauss-Legendre nodes")

    print(f"\nContinuous Kannan diagnostics (relay control):")
    print(f"  kappa = alpha (1-exp(-1))             = {dc['kappa']:.6f}")
    print(f"  mu_K  = alpha (1 - 0.5(1-exp(-2)))    = {dc['mu_K']:.6f}")
    print(f"  Lg    = 2 beta / pi                   = {dc['Lg']:.6f}")
    print(f"  Lq    = -sigma / 2                    = {dc['Lq']:+.6f}")
    print(f"  ||q||_inf = sigma                     = {dc['norm_q']:.6f}")
    print(f"  m_R   = Lg - c - mu_K R               = {dc['delta_R']:+.6f}")
    print(f"  Invariance: {dc['invariance_lhs']:.4f} <= R = {dc['invariance_rhs']} "
          f"({'OK' if dc['invariance_lhs'] <= dc['invariance_rhs'] else 'FAIL'})")
    print(f"  Smallness 3 kappa + 2 ||q||/m_R       = {dc['kannan_lhs']:.6f} < 1 "
          f"({'OK' if dc['kannan_lhs'] < 1 else 'FAIL'})")
    print(f"  Kannan constant lambda_R              = {dc['lambda_R']:.6f} < 1/2 "
          f"({'OK' if dc['lambda_R'] < 0.5 else 'FAIL'})")
    print(f"  Single-step transient margin (Lq+m_R) = {dc['single_step_margin']:+.6f} "
          f"({'OK' if dc['single_step_margin'] > 0 else 'FAIL'})")

    print(f"\nDiscrete Kannan diagnostics (N={res['N']}):")
    print(f"  kappa_N = {dd['kappa_N']:.6f}    mu_N = {dd['mu_N']:.6f}")
    print(f"  delta_NR = {dd['delta']:+.6f}    lambda_N = {dd['lambda_N']:.6f}")

    print(f"\nIteration outcome:")
    print(f"  Iterations to tolerance 1e-13       : {res['iterations']}")
    print(f"  Final residual                      : {res['final_residual']:.3e}")
    print(f"  Open-loop equilibrium ell(y_open)   : {res['ell_open_loop']:.6f}")
    print(f"  Closed-loop equilibrium ell(y_*)    : {res['ell_star']:.6f}")
    print(f"  Regulation drop ell_open - ell_*    : {res['regulation_drop']:+.6f}  "
          f"({100*res['regulation_drop']/res['ell_open_loop']:+.2f}% of open-loop)")
    print(f"  Active margin ell(y_*) - c          : {res['active_margin']:+.6f}  "
          f"(positive => relay active at equilibrium)")

    # ----- Setpoint sweep -----
    print(f"\nSetpoint sweep over c (Kannan-admissible range):")
    # Maximum admissible c is c < Lg - mu_K R
    c_max = float(dc['Lg'] - dc['mu_K'] * p['R'])
    cs = np.linspace(0.10, c_max - 0.01, 21)
    sweep = run_relay_setpoint_sweep(cs, N=res["N"])

    print(f"  Open-loop ell(y_open)               = {sweep['ell_open']:.6f}")
    print(f"  Closed-loop ell(y_closed) (always on) = {sweep['ell_closed']:.6f}")
    print()
    print(f"  {'c':>7s}  {'lambda_N':>10s}  {'admissible':>11s}  "
          f"{'ell(y_*)':>10s}  {'theta_eq':>9s}  {'iters':>5s}")
    for r in sweep["rows"]:
        ell_str = f"{r['ell_star']:.6f}" if r['ell_star'] is not None else "      --"
        adm_str = "YES" if r["admissible"] else "NO"
        th_str  = f"{r['theta_eq']:.0f}" if r["theta_eq"] is not None else "  --"
        it_str  = f"{r['iters']:5d}" if r["iters"] is not None else "  -- "
        print(f"  {r['c']:>7.4f}  {r['lambda_N']:>10.4f}  {adm_str:>11s}  "
              f"{ell_str:>10s}  {th_str:>9s}  {it_str:>5s}")

    # Generate figure
    plot_relay_control(res, sweep, "./fig_relay.pdf")
    print(f"\nFigure written to ./fig_relay.pdf")

    print("\n" + "=" * 86)
    print("  JSON dump (compact)")
    print("=" * 86)
    out = {
        "params":           p,
        "diagnostics_cont": dc,
        "diagnostics_disc": dd,
        "iterations":       res["iterations"],
        "final_residual":   res["final_residual"],
        "ell_open_loop":    res["ell_open_loop"],
        "ell_closed_loop":  sweep["ell_closed"],
        "ell_star":         res["ell_star"],
        "regulation_drop":  res["regulation_drop"],
        "active_margin":    res["active_margin"],
        "sweep_summary": [
            {"c": r["c"], "lambda_N": r["lambda_N"],
             "admissible": r["admissible"],
             "ell_star": r["ell_star"], "theta_eq": r["theta_eq"],
             "iters": r["iters"]} for r in sweep["rows"]
        ],
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
