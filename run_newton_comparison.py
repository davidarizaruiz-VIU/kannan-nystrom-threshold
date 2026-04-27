"""
run_newton_comparison.py
========================
Empirical comparison of the Picard--Nystroem iteration with the
Chen--Mangasarian smoothed semismooth Newton method on the discrete
threshold equation

    F(U) = U - A_N U - G - H( w . U - c ) Q = 0.

Heaviside is not Lipschitz, so genuine semismooth Newton (Qi-Sun, 1993)
does not apply directly to F. The standard practical surrogate
(Chen & Mangasarian 1995; Facchinei & Pang 2003, Ch.~9) replaces H by
the smoothed sigmoid

    phi_eps(s) = 1 / (1 + exp(-s/eps)),

and applies plain Newton to the resulting C^infinity equation, with a
homotopy continuation eps -> 0. The Jacobian is

    J(U) = (I - A_N) - phi_eps'(w.U - c) Q w^T,

a rank-1 update of M = I - A_N inverted via the Sherman-Morrison
formula from a single LU factorisation.

For each benchmark and seed, we report iterations to reach the true
residual ||U - A_N U - G - H(w.U - c) Q||_inf below 1e-13, and the
attained accuracy. A scaling sweep over N in {64, 128, 256, 512, 1024}
quantifies wall-clock cost for the contractive benchmark.

Usage:
    python3 run_newton_comparison.py
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from picard_nystrom import (
    benchmark_parameters,
    bistability_parameters,
    bistability_branch_diagnostics,
    build_benchmark_functions,
    assemble,
    picard_nystrom,
    smoothed_newton,
    kannan_diagnostics,
)


# ---------- helper -----------------------------------------------------------
def _attractor_distance(U, branch):
    d_minus = float(np.max(np.abs(U - branch["u_minus"])))
    d_plus  = float(np.max(np.abs(U - branch["u_plus"])))
    return d_minus, d_plus


# ---------- benchmark blocks -------------------------------------------------
def block_contractive(N: int = 128) -> dict:
    p = benchmark_parameters()
    K, g, q = build_benchmark_functions(p)
    sys_ = assemble(N, K, g, q, rule="gauss")
    diag = kannan_diagnostics(sys_, p["c"], p["R"])

    U0 = np.full(N, p["c"] / 2.0)
    pic = picard_nystrom(sys_, p["c"], U0=U0, tol=1e-13, nmax=200)
    nw  = smoothed_newton(sys_, p["c"], U0=U0,
                          eps_init=1e-1, eps_min=1e-14, eps_factor=0.1,
                          tol=1e-13, nmax_inner=20, nmax_outer=20)

    err_methods = float(np.max(np.abs(pic["U"] - nw["U"])))
    return {
        "label":              "contractive (Sec. 7.2)",
        "N":                  N,
        "lambda_N":           float(diag["lambda_N"]),
        "picard_iters":       int(pic["iters"]),
        "picard_residual":    float(pic["residuals"][-1]),
        "newton_iters":       int(nw["total_iterations"]),
        "newton_residual_t":  float(nw["residuals_true"][-1]),
        "newton_residual_s":  float(nw["residuals_smooth"][-1]),
        "newton_converged":   bool(nw["converged"]),
        "newton_final_eps":   float(nw["final_eps"]),
        "agreement":          err_methods,
    }


def block_bistable(N: int = 128) -> dict:
    p = bistability_parameters()
    K, g, q = build_benchmark_functions(p)
    sys_ = assemble(N, K, g, q, rule="gauss")
    branch = bistability_branch_diagnostics(sys_, p["c"])

    rows = []
    for seed_name, U0 in (("lower seed (U0 = 0)",      np.zeros(N)),
                          ("upper seed (U0 = R . 1)", np.full(N, p["R"]))):
        pic = picard_nystrom(sys_, p["c"], U0=U0, tol=1e-13, nmax=200)
        nw  = smoothed_newton(sys_, p["c"], U0=U0,
                              eps_init=1e-1, eps_min=1e-14, eps_factor=0.1,
                              tol=1e-13, nmax_inner=20, nmax_outer=20)
        d_pic = _attractor_distance(pic["U"], branch)
        d_nw  = _attractor_distance(nw["U"],  branch)
        attractor_pic = "u^-" if d_pic[0] < d_pic[1] else "u^+"
        attractor_nw  = "u^-" if d_nw[0]  < d_nw[1]  else "u^+"
        rows.append({
            "seed":              seed_name,
            "picard_iters":      int(pic["iters"]),
            "picard_residual":   float(pic["residuals"][-1]),
            "picard_attractor":  attractor_pic,
            "picard_d_branch":   min(d_pic),
            "newton_iters":      int(nw["total_iterations"]),
            "newton_residual_t": float(nw["residuals_true"][-1]),
            "newton_attractor":  attractor_nw,
            "newton_d_branch":   min(d_nw),
            "consistent":        attractor_pic == attractor_nw,
        })

    return {
        "label":          "bistable (Sec. 7.5)",
        "N":              N,
        "branch_width":   float(branch["width_bistable"]),
        "rows":           rows,
    }


def block_scaling() -> dict:
    """Wall-clock comparison on the contractive benchmark, varying N."""
    p = benchmark_parameters()
    K, g, q = build_benchmark_functions(p)

    rows = []
    for N in (64, 128, 256, 512, 1024):
        sys_ = assemble(N, K, g, q, rule="gauss")
        U0   = np.full(N, p["c"] / 2.0)

        # Warm-up (JIT-free here, but caches and BLAS workspaces are touched).
        picard_nystrom(sys_, p["c"], U0=U0, tol=1e-13, nmax=200)
        smoothed_newton(sys_, p["c"], U0=U0,
                        eps_init=1e-1, eps_min=1e-14, eps_factor=0.1,
                        tol=1e-13, nmax_inner=20, nmax_outer=20)

        nrep = 200 if N <= 256 else 50
        t0 = time.perf_counter()
        for _ in range(nrep):
            pic = picard_nystrom(sys_, p["c"], U0=U0, tol=1e-13, nmax=200)
        t_pic = (time.perf_counter() - t0) / nrep * 1e3   # ms

        t0 = time.perf_counter()
        for _ in range(nrep):
            nw = smoothed_newton(sys_, p["c"], U0=U0,
                                 eps_init=1e-1, eps_min=1e-14,
                                 eps_factor=0.1, tol=1e-13,
                                 nmax_inner=20, nmax_outer=20)
        t_nw = (time.perf_counter() - t0) / nrep * 1e3    # ms

        rows.append({
            "N":            N,
            "picard_iters": int(pic["iters"]),
            "newton_iters": int(nw["total_iterations"]),
            "picard_ms":    t_pic,
            "newton_ms":    t_nw,
            "ratio_nw_pic": t_nw / t_pic,
        })
    return {"label": "scaling on contractive benchmark", "rows": rows}


# ---------- main -------------------------------------------------------------
def main() -> int:
    print("=" * 90)
    print("  Picard-Kannan vs smoothed semismooth Newton (Chen-Mangasarian)")
    print("=" * 90)

    contr = block_contractive(N=128)
    print("\n--- Block A : contractive benchmark, N = 128 ---")
    print(f"  lambda_N = {contr['lambda_N']:.4f}")
    print(f"  Picard:  iters = {contr['picard_iters']:3d}    "
          f"residual = {contr['picard_residual']:.2e}")
    print(f"  Newton:  iters = {contr['newton_iters']:3d}    "
          f"residual_true = {contr['newton_residual_t']:.2e}    "
          f"final eps = {contr['newton_final_eps']:.1e}")
    print(f"  ||U_pic - U_newton||_inf = {contr['agreement']:.2e}")
    print(f"  Newton/Picard iters ratio = "
          f"{contr['newton_iters']/contr['picard_iters']:.2f}")

    bist = block_bistable(N=128)
    print("\n--- Block B : bistable benchmark, N = 128 ---")
    print(f"  Branch separation ||u^+ - u^-||_inf = {bist['branch_width']:.4f}")
    for r in bist["rows"]:
        print(f"\n  {r['seed']}:")
        print(f"    Picard:  iters = {r['picard_iters']:3d}    "
              f"attractor = {r['picard_attractor']}    "
              f"d(branch) = {r['picard_d_branch']:.2e}")
        print(f"    Newton:  iters = {r['newton_iters']:3d}    "
              f"attractor = {r['newton_attractor']}    "
              f"d(branch) = {r['newton_d_branch']:.2e}")
        print(f"    Same attractor: {'YES' if r['consistent'] else 'NO'}")

    scal = block_scaling()
    print("\n--- Block C : wall-clock scaling on contractive benchmark ---")
    print(f"  {'N':>5s}  {'Pic iters':>9s}  {'Newton iters':>12s}  "
          f"{'Pic (ms)':>10s}  {'Newton (ms)':>12s}  {'ratio':>7s}")
    for r in scal["rows"]:
        print(f"  {r['N']:>5d}  {r['picard_iters']:>9d}  "
              f"{r['newton_iters']:>12d}  "
              f"{r['picard_ms']:>10.3f}  {r['newton_ms']:>12.3f}  "
              f"{r['ratio_nw_pic']:>7.2f}x")

    print("\n" + "=" * 90)
    print("  JSON dump (compact)")
    print("=" * 90)
    out = {"contractive": contr, "bistable": bist, "scaling": scal}
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
