"""
run_stability_experiment.py
===========================
Empirical validation of Theorem 3.X (Stability under data perturbations)
and Corollary 3.X (A priori Nystroem bound) of Section 3.

For the separable benchmark of Section 7.2 with reference parameters
    alpha=0.10, beta=1.00, rho=0.00, sigma=0.05, c=0.50, R=2.00,
we perturb each of (alpha, beta, sigma) independently by a sequence of
epsilon values, compute the perturbed fixed point u_*(eps) and verify
the predicted Lipschitz bound

    || u_*(0) - u_*(eps) ||_inf
       <= (1 + lambda_R) * ( R ||A - A_eps|| + ||g - g_eps|| + ||q - q_eps|| ).

The script reports, for each (parameter, eps) pair:
  - Actual sup-norm of the FP perturbation
  - Predicted bound (RHS of the stability estimate)
  - Tightness ratio (actual / bound), expected in (0, 1]

Usage from a terminal in the project folder:
    python3 run_stability_experiment.py
"""
from __future__ import annotations

import json
import sys

import numpy as np

from picard_nystrom import (
    benchmark_parameters,
    build_benchmark_functions,
    assemble,
    picard_nystrom,
    kannan_diagnostics,
    gauss_legendre_01,
)


def compute_fp(p, N=128, tol=1e-13):
    K, g, q = build_benchmark_functions(p)
    sys_ = assemble(N, K, g, q, rule="gauss")
    out = picard_nystrom(sys_, p["c"],
                        U0=np.full(N, p["c"] / 2.0),
                        tol=tol, nmax=200)
    return out["U"], sys_


def operator_diff_norm(p_ref, p_pert, N=128):
    """Compute ||A - A_pert|| in the discrete sup-norm operator topology."""
    t, w = gauss_legendre_01(N)
    K_ref,  _, _ = build_benchmark_functions(p_ref)
    K_pert, _, _ = build_benchmark_functions(p_pert)
    T_grid, S_grid = np.meshgrid(t, t, indexing="ij")
    K_diff = K_pert(T_grid, S_grid) - K_ref(T_grid, S_grid)
    A_diff = K_diff * w[np.newaxis, :]
    # ||A||_inf = max over rows of sum |A[i,j]|
    return float(np.max(np.sum(np.abs(A_diff), axis=1)))


def forcing_diff_norms(p_ref, p_pert, N=128):
    """Compute ||g - g_pert||_inf and ||q - q_pert||_inf."""
    t, _ = gauss_legendre_01(N)
    _, g_ref,  q_ref  = build_benchmark_functions(p_ref)
    _, g_pert, q_pert = build_benchmark_functions(p_pert)
    g_diff = float(np.max(np.abs(g_pert(t) - g_ref(t))))
    q_diff = float(np.max(np.abs(q_pert(t) - q_ref(t))))
    return g_diff, q_diff


def main() -> int:
    print("=" * 90)
    print("  Empirical validation of Theorem 3.X (Stability under data perturbations)")
    print("=" * 90)

    p_ref = benchmark_parameters()
    print(f"\nReference parameters: {p_ref}")

    # Reference fixed point and Kannan constant
    U_ref, sys_ref = compute_fp(p_ref)
    diag_ref = kannan_diagnostics(sys_ref, p_ref["c"], p_ref["R"])
    lambda_R = diag_ref["lambda_N"]
    R = p_ref["R"]

    ell_ref = float(sys_ref.w @ U_ref)
    print(f"Reference fixed point: ell_N(U_*) = {ell_ref:.10f}  (above c={p_ref['c']})")
    print(f"Discrete Kannan constant lambda_N = {lambda_R:.6f}")
    print(f"Predicted Lipschitz factor: (1 + lambda_R) = {1 + lambda_R:.6f}")
    print(f"R = {R}")

    print(f"\n{'parameter':>12s}  {'epsilon':>9s}  {'actual diff':>12s}  "
          f"{'predicted bound':>16s}  {'tightness':>10s}  {'OK':>3s}")
    print("-" * 90)

    rows = []
    for param in ("alpha", "beta", "sigma"):
        for eps in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
            p_pert = dict(p_ref)
            p_pert[param] = p_ref[param] + eps

            U_pert, sys_pert = compute_fp(p_pert)
            actual_diff = float(np.max(np.abs(U_ref - U_pert)))

            # Data perturbation norms
            opnorm_diff = operator_diff_norm(p_ref, p_pert)
            g_diff, q_diff = forcing_diff_norms(p_ref, p_pert)

            # Predicted bound from Theorem 3.X
            bound = (1 + lambda_R) * (R * opnorm_diff + g_diff + q_diff)

            tightness = actual_diff / bound if bound > 0 else float("nan")
            ok = "YES" if actual_diff <= bound + 1e-15 else "NO"

            print(f"  {param:>12s}  {eps:>9.1e}  {actual_diff:>12.4e}  "
                  f"{bound:>16.4e}  {tightness:>10.4f}  {ok:>3s}")
            rows.append({
                "parameter":  param,
                "epsilon":    eps,
                "actual_sup_diff": actual_diff,
                "opnorm_A_diff":   opnorm_diff,
                "g_diff_norm":     g_diff,
                "q_diff_norm":     q_diff,
                "predicted_bound": bound,
                "tightness":       tightness,
                "satisfied":       ok == "YES",
            })

    # Summary
    all_ok = all(r["satisfied"] for r in rows)
    max_tightness = max(r["tightness"] for r in rows if not np.isnan(r["tightness"]))
    min_tightness = min(r["tightness"] for r in rows if not np.isnan(r["tightness"]))

    print("-" * 90)
    print(f"Stability bound satisfied in all {len(rows)} perturbations: "
          f"{'YES' if all_ok else 'NO -- VIOLATED'}")
    print(f"Tightness ratio range: [{min_tightness:.4f}, {max_tightness:.4f}]")
    print(f"  (Ratio < 1 means the theoretical bound is conservative,")
    print(f"   ratio close to 1 means the bound is tight in this regime.)")

    # JSON dump
    print()
    print("=" * 90)
    print("  JSON dump (compact)")
    print("=" * 90)
    out = {
        "params":   p_ref,
        "lambda_N": lambda_R,
        "R":        R,
        "factor_1_plus_lambda": 1 + lambda_R,
        "rows":     rows,
        "all_satisfied": all_ok,
        "tightness_range": [min_tightness, max_tightness],
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
