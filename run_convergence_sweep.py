"""
run_convergence_sweep.py
========================
Empirical validation of Theorem 6.X (continuous-discrete Nystroem
convergence) across three kernel regularity classes:

  (S) Smooth:     K(t,s) = alpha * sin(pi t s)        in C^infinity,
  (L) Lipschitz:  K(t,s) = alpha * exp(-|t-s|)        Lipschitz on the diagonal,
  (B) BV:         K(t,s) = alpha * 1_{t <= s}         jump on the diagonal.

For each kernel we fix the same forcing g, jump q, threshold c and ball R,
verify that the discrete Kannan smallness 3 kappa_N + 2 ||q||/m_R < 1 holds
at every resolution N in {16,32,64,128,256,512}, run the Picard-Nystroem
fixed point, and tabulate the cross-resolution self-consistency error
||I_{2N} U_{2N} - I_N U_N||_infty on a common reference grid.

The expected rates are:
  smooth     :  spectral (faster than any algebraic rate),
  Lipschitz  :  O(N^{-2})    (Gauss-Legendre with C^1 kernel after splitting),
  BV         :  O(N^{-1})    (single jump in the integrand).

Outputs:
  - tables of self-consistency errors and empirical rates per kernel,
  - figure fig_convergence.pdf (3-panel log-log convergence curves),
  - JSON dump.

Usage:
    python3 run_convergence_sweep.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from picard_nystrom import (
    gauss_legendre_01,
    NystromSystem,
    picard_nystrom,
    kannan_diagnostics,
)


# ---------- kernel families --------------------------------------------------
def kernel_smooth(alpha):
    return lambda t, s: alpha * np.sin(np.pi * t * s)

def kernel_lipschitz(alpha):
    return lambda t, s: alpha * np.exp(-np.abs(t - s))

def kernel_bv(alpha):
    return lambda t, s: alpha * (t <= s).astype(float)


# Common forcing/jump on [0,1]: contractive Kannan, all kernels admissible.
def g_func(t):  return 1.0 + 0.0 * t           # uniform baseline
def q_func(t):  return 0.05 * (1.0 + t)         # mild positive jump
C_THR = 0.50
R_BALL = 2.00


# ---------- assembly ---------------------------------------------------------
def assemble_system(K_callable, N, c=C_THR):
    t, w = gauss_legendre_01(N)
    Tg, Sg = np.meshgrid(t, t, indexing='ij')
    A = K_callable(Tg, Sg) * w[np.newaxis, :]
    G = g_func(t)
    Q = q_func(t)
    return NystromSystem(t=t, w=w, A=A, G=G, Q=Q)


def nystrom_extension(t_coarse, U_coarse, K_callable, t_eval, c, G_eval, Q_eval, w_coarse):
    """Continuous Nystroem extension at points t_eval, given the discrete
    fixed point U_coarse on nodes t_coarse with weights w_coarse."""
    Tg, Sg = np.meshgrid(t_eval, t_coarse, indexing='ij')
    KU = (K_callable(Tg, Sg) * w_coarse[np.newaxis, :]) @ U_coarse  # (n_eval,)
    ell_disc = float(w_coarse @ U_coarse)
    theta = 1.0 if ell_disc > c else 0.0
    return KU + G_eval + theta * Q_eval


# ---------- per-kernel sweep -------------------------------------------------
def sweep_kernel(name, K_factory, alphas, Ns, c=C_THR, R=R_BALL):
    """For each alpha, run the resolution sweep; return diagnostics."""
    rows = []
    for alpha in alphas:
        K = K_factory(alpha)
        # Reference grid (same for all N) for cross-resolution comparison
        t_ref = np.linspace(0.0, 1.0, 1001)
        G_ref = g_func(t_ref)
        Q_ref = q_func(t_ref)

        # Run all Ns
        per_n = {}
        for N in Ns:
            sys_ = assemble_system(K, N, c=c)
            diag = kannan_diagnostics(sys_, c, R)
            res = picard_nystrom(sys_, c, U0=np.full(N, c / 2.0), tol=1e-13, nmax=300)
            U_ext = nystrom_extension(sys_.t, res['U'], K, t_ref, c,
                                      G_ref, Q_ref, sys_.w)
            per_n[N] = {
                'lambda_N': float(diag['lambda_N']),
                'kappa_N':  float(diag['kappa_N']),
                'iters':    int(res['iters']),
                'residual': float(res['residuals'][-1]),
                'U_ext':    U_ext,
            }

        # Cross-resolution self-consistency: ||U_ext_{2N} - U_ext_N||_inf
        sorted_N = sorted(Ns)
        consistency = []
        for i in range(len(sorted_N) - 1):
            Nc, Nf = sorted_N[i], sorted_N[i + 1]
            err = float(np.max(np.abs(per_n[Nf]['U_ext'] - per_n[Nc]['U_ext'])))
            consistency.append((Nc, Nf, err))

        # Empirical rate p between consecutive pairs
        rates = []
        for i in range(1, len(consistency)):
            Nc_prev, Nf_prev, e_prev = consistency[i - 1]
            Nc_curr, Nf_curr, e_curr = consistency[i]
            ratio_N = Nf_curr / Nf_prev
            ratio_e = e_prev / max(e_curr, 1e-300)
            p = float(np.log(ratio_e) / np.log(ratio_N)) if ratio_e > 0 else float('nan')
            rates.append(p)

        rows.append({
            'alpha':       float(alpha),
            'lambda_N_max': float(max(per_n[N]['lambda_N'] for N in Ns)),
            'iters':       per_n[max(Ns)]['iters'],
            'consistency': consistency,
            'rates':       rates,
            'U_ref_finest': per_n[max(Ns)]['U_ext'],
        })
    return {'name': name, 'rows': rows, 'Ns': sorted_N, 't_ref': t_ref}


# ---------- main -------------------------------------------------------------
def main() -> int:
    print("=" * 92)
    print("  Convergence sweep across kernel regularity classes")
    print("  Validating Theorem 6.X (Nystroem continuous-discrete convergence)")
    print("=" * 92)

    Ns = [16, 32, 64, 128, 256, 512]

    # alpha kept moderate so that Kannan smallness holds for all three kernels
    sweeps = []
    for name, factory, alpha in [
        ("smooth     (C^infty)", kernel_smooth,   0.20),
        ("Lipschitz  (cusp)",    kernel_lipschitz,0.10),
        ("BV         (jump)",    kernel_bv,       0.10),
    ]:
        s = sweep_kernel(name, factory, [alpha], Ns)
        sweeps.append(s)
        r = s['rows'][0]
        print(f"\n  --- {name} :  alpha = {alpha} ---")
        print(f"  lambda_N_max = {r['lambda_N_max']:.5f}   iters at N_max = {r['iters']}")
        print(f"  Consistency:")
        print(f"    {'(N_c, N_f)':>16s} {'||U_f - U_c||_inf':>22s} {'rate p':>10s}")
        for j, (Nc, Nf, err) in enumerate(r['consistency']):
            rate_s = ("--" if j == 0 else f"{r['rates'][j-1]:>10.3f}")
            print(f"    {f'({Nc:4d},{Nf:5d})':>16s} {err:>22.4e} {rate_s:>10s}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    titles = [
        r"(a) Smooth $K = \alpha\sin\pi t s$: spectral",
        r"(b) Lipschitz $K = \alpha e^{-|t-s|}$: $\mathcal{O}(N^{-2})$",
        r"(c) BV $K = \alpha\mathbf{1}_{\{t \leq s\}}$: $\mathcal{O}(N^{-1})$",
    ]
    expected_p = [None, 2.0, 1.0]
    for ax, sweep, title, p_exp in zip(axes, sweeps, titles, expected_p):
        r = sweep['rows'][0]
        Nf_vals = np.array([Nf for Nc, Nf, _ in r['consistency']], dtype=float)
        err_vals = np.array([e for Nc, Nf, e in r['consistency']])
        ax.loglog(Nf_vals, err_vals, "o-", color="C0", lw=1.6, ms=6,
                  label=r"$\|I_{2N}U_{2N}-I_N U_N\|_\infty$")
        if p_exp is not None:
            Nf_ref = np.array([Nf_vals[0], Nf_vals[-1]])
            err_ref = err_vals[0] * (Nf_ref / Nf_vals[0]) ** (-p_exp)
            ax.loglog(Nf_ref, err_ref, "k--", lw=1.0, alpha=0.7,
                      label=fr"$\mathcal{{O}}(N^{{-{p_exp:.0f}}})$ reference")
        ax.set_xlabel(r"$N$ (Gauss--Legendre nodes)")
        ax.set_ylabel("self-consistency error")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower left", framealpha=0.92)

    out_fig = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "fig_convergence.pdf")
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"\n  Figure written: {out_fig}")

    print("\n" + "=" * 92)
    print("  JSON dump (compact)")
    print("=" * 92)
    out = {
        sweep['name']: {
            'alpha':         sweep['rows'][0]['alpha'],
            'Ns':            sweep['Ns'],
            'lambda_N_max':  sweep['rows'][0]['lambda_N_max'],
            'iters_at_Nmax': sweep['rows'][0]['iters'],
            'consistency':   [(int(Nc), int(Nf), float(e))
                              for Nc, Nf, e in sweep['rows'][0]['consistency']],
            'rates':         [float(p) for p in sweep['rows'][0]['rates']],
        }
        for sweep in sweeps
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
