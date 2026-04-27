"""
run_real_data_application.py
============================
Real-data validation of the Kannan-stable Picard--Nystroem framework on the
Royama et al. (2017) spruce-budworm field dataset
(DOI 10.5061/dryad.t175g, CC0 1.0 Universal license).

The dataset records postdiapause larval densities at three Plot--Year cells in
New Brunswick (1981--1994), spanning the peak-to-decline phase of the budworm
outbreak that started in the early 1960s. Plot 1 in particular contains the
1985 outbreak peak (max density 993.38 larvae per m^2 of branch surface,
40 sampling dates) and the 1988 collapse (max 8.89, 11 dates), separated by a
dynamic transition through 1986 (max 379) and 1987 (max 148).

Within the present framework, the alternation between an outbreak and an
endemic seasonal equilibrium is the bistable regime of Theorem 3.X
(thm:bistable): a single operator T u = A u + g + q H(L u - c) admits both a
high-density fixed point u^+ (outbreak) and a low-density fixed point u^-
(endemic), with u^- < c < L u^+ realised by the Kannan margins of
\\eqref{eq:branch-margins}. The within-season profiles observed in 1985 and 1988
should therefore both be reproducible as fixed points of one and the same T,
selected by the Picard iteration via the basin partition of
Corollary~\\ref{cor:basin-radius}.

This script fits one parameter vector (alpha, beta, gamma, delta, mu, sigma, c)
of the bistable model

    K(t,s) = alpha * exp(-beta |t - s|),
    g(t)   = gamma * phi(t),                  (baseline forcing)
    q(t)   = delta * phi(t),                  (positive outbreak release)
    phi(t) = exp(-(t - mu)^2 / (2 sigma^2)),  (shared phenological shape)
    c      = scalar threshold,

to the joint requirement that the upper-branch fixed point u^+ matches the
1985 data and the lower-branch fixed point u^- matches the 1988 data, with the
hard constraint that the operator be in the bistable regime of Theorem 3.X
(L u^- < c < L u^+, i.e. m^- < 0 < m^+).

Reproduces:
  - Bistable diagnostics on the fitted operator,
  - Picard runs from low/high initial data certifying basin selection,
  - figure fig_realdata.pdf (4 panels: 1985 fit, 1988 fit, branch fixed
    points and basins, Picard convergence),
  - JSON summary.

Usage:
    python3 run_real_data_application.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

from picard_nystrom import gauss_legendre_01, picard_nystrom


# ---------- data -------------------------------------------------------------
def load_year(year: int):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'data', f'royama_plot1_{year}_larva.csv')
    days, dens = [], []
    with open(path, newline='') as f:
        r = csv.reader(f); next(r)
        for row in r:
            days.append(int(row[0]))
            dens.append(float(row[1]))
    return np.asarray(days), np.asarray(dens)


# ---------- model ------------------------------------------------------------
def assemble(theta, n_nodes: int = 64):
    alpha, beta, gamma, delta, mu, sigma, c = theta
    t, w = gauss_legendre_01(n_nodes)
    Tg, Sg = np.meshgrid(t, t, indexing='ij')
    K = alpha * np.exp(-beta * np.abs(Tg - Sg))
    A = K * w[np.newaxis, :]
    phi = np.exp(-(t - mu)**2 / (2.0 * sigma**2))
    G = gamma * phi
    Q = delta * phi
    return t, w, A, G, Q, c


def branch_fixed_points(theta, n_nodes: int = 64):
    """Compute u^- and u^+ via direct solve."""
    t, w, A, G, Q, c = assemble(theta, n_nodes)
    M = np.eye(n_nodes) - A
    u_minus = np.linalg.solve(M, G)
    u_plus = np.linalg.solve(M, G + Q)
    Lu_minus = float(w @ u_minus)
    Lu_plus = float(w @ u_plus)
    return t, w, u_minus, u_plus, Lu_minus, Lu_plus, c


# ---------- fit --------------------------------------------------------------
def loss_bistable(theta, t_85, u_85, t_88, u_88, weight_88: float = 1.0):
    """Joint loss: u^+ matches 1985, u^- matches 1988, bistability enforced."""
    try:
        t_grid, w, u_m, u_p, Lu_m, Lu_p, c = branch_fixed_points(theta)
    except np.linalg.LinAlgError:
        return 1.0e6

    # Data fit
    u_p_at_85 = np.interp(t_85, t_grid, u_p)
    u_m_at_88 = np.interp(t_88, t_grid, u_m)
    err_85 = float(np.sum((u_p_at_85 - u_85)**2))
    err_88 = float(np.sum((u_m_at_88 - u_88)**2))

    # Bistability penalty: m^- = Lu_m - c < 0, m^+ = Lu_p - c > 0
    m_minus = Lu_m - c
    m_plus = Lu_p - c
    pen = 0.0
    if m_minus > 0:                   # lower branch invalid
        pen += 1.0e3 * m_minus**2
    if m_plus < 0:                    # upper branch invalid
        pen += 1.0e3 * m_plus**2

    # Negative-density penalty (avoid u^- < 0)
    if u_m.min() < 0:
        pen += 1.0e3 * u_m.min()**2

    return err_85 + weight_88 * err_88 + pen


# ---------- diagnostics ------------------------------------------------------
def r_squared(u_obs, u_pred):
    ss_res = float(np.sum((u_obs - u_pred)**2))
    ss_tot = float(np.sum((u_obs - u_obs.mean())**2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')


# ---------- main -------------------------------------------------------------
def main() -> int:
    print("=" * 90)
    print("  Real-data bistable validation: Royama et al. 2017 (Plot 1, 1985 & 1988)")
    print("=" * 90)

    days_85, dens_85 = load_year(1985)
    days_88, dens_88 = load_year(1988)

    # Common time axis spanning both years' sampling windows.
    t_min = float(min(days_85.min(), days_88.min()))
    t_max = float(max(days_85.max(), days_88.max()))
    print(f"\n  Plot 1, 1985: {len(days_85)} obs, range {days_85.min()}--{days_85.max()},"
          f" max density = {dens_85.max():.2f}")
    print(f"  Plot 1, 1988: {len(days_88)} obs, range {days_88.min()}--{days_88.max()},"
          f" max density = {dens_88.max():.2f}")
    print(f"  Common time window: ordinal dates {int(t_min)}--{int(t_max)}")

    # Normalise time on the common window; normalise density by the global max.
    u_max = float(max(dens_85.max(), dens_88.max()))
    t_85 = (days_85 - t_min) / (t_max - t_min)
    t_88 = (days_88 - t_min) / (t_max - t_min)
    u_85 = dens_85 / u_max
    u_88 = dens_88 / u_max

    # The two years differ by two orders of magnitude in amplitude.
    # Scale 1988 to comparable contribution by SSE ratio.
    weight_88 = float(len(t_85) / max(len(t_88), 1)) * (u_85.var() / max(u_88.var(), 1e-12))
    print(f"\n  Loss weighting (1988 / 1985): {weight_88:.2e}  (balancing variance & sample size)")

    # Bounds: alpha kept moderate so kappa < 1/2 stays achievable
    bounds = [
        (0.01, 0.45),    # alpha
        (0.10, 50.0),    # beta
        (0.001, 1.0),    # gamma  (small endemic baseline)
        (0.10, 5.0),     # delta  (large outbreak release)
        (0.05, 0.95),    # mu
        (0.05, 0.50),    # sigma
        (0.01, 0.80),    # c (threshold; must lie between Lu^- and Lu^+)
    ]

    print("\n  Running differential evolution (joint bistable fit) ...")
    res = differential_evolution(
        loss_bistable, bounds,
        args=(t_85, u_85, t_88, u_88, weight_88),
        seed=42, maxiter=200, popsize=25, tol=1e-9,
        polish=True, workers=1, updating='deferred',
    )
    theta = res.x
    alpha, beta, gamma, delta, mu, sigma, c = theta
    print(f"  alpha = {alpha:.4f}   beta = {beta:.3f}   gamma = {gamma:.4f}   delta = {delta:.4f}")
    print(f"  mu    = {mu:.4f}     sigma = {sigma:.4f}    c    = {c:.4f}")
    print(f"  Final loss            = {res.fun:.4e}")

    # Diagnostics on the fitted operator
    t_grid, w, u_minus, u_plus, Lu_minus, Lu_plus, _ = branch_fixed_points(theta, n_nodes=128)
    m_minus = Lu_minus - c
    m_plus = Lu_plus - c
    print(f"\n  Branch fixed points  : Lu^- = {Lu_minus:.5f},  Lu^+ = {Lu_plus:.5f}")
    print(f"  Branch margins       : m^-  = {m_minus:+.5f}, m^+  = {m_plus:+.5f}")
    if m_minus < 0 and m_plus > 0:
        print(f"  Regime: BISTABLE (Theorem 3.X)")
    else:
        print(f"  Regime: NOT bistable (fit failed)")

    # Picard from low/high initial data, certify branch selection
    n = 128
    A_disc = (alpha * np.exp(-beta * np.abs(t_grid[:, None] - t_grid[None, :]))) * w[None, :]
    phi = np.exp(-(t_grid - mu)**2 / (2 * sigma**2))
    G = gamma * phi
    Q = delta * phi
    from picard_nystrom import NystromSystem
    sys_real = NystromSystem(t=t_grid, w=w, A=A_disc, G=G, Q=Q)
    pic_low = picard_nystrom(sys_real, c, U0=np.zeros(n), tol=1e-13, nmax=300)
    pic_high = picard_nystrom(sys_real, c, U0=np.full(n, 1.0), tol=1e-13, nmax=300)
    d_low_to_minus = float(np.max(np.abs(pic_low['U'] - u_minus)))
    d_high_to_plus = float(np.max(np.abs(pic_high['U'] - u_plus)))
    print(f"\n  Picard from U0 = 0           : iters = {pic_low['iters']:3d},"
          f" ||U - u^-||_inf = {d_low_to_minus:.2e}")
    print(f"  Picard from U0 = 1 . 1       : iters = {pic_high['iters']:3d},"
          f" ||U - u^+||_inf = {d_high_to_plus:.2e}")

    # Fit quality on each year
    u_p_at_85 = np.interp(t_85, t_grid, u_plus)
    u_m_at_88 = np.interp(t_88, t_grid, u_minus)
    R2_85 = r_squared(u_85, u_p_at_85)
    R2_88 = r_squared(u_88, u_m_at_88)
    rmse_85 = float(np.sqrt(np.mean((u_85 - u_p_at_85)**2)))
    rmse_88 = float(np.sqrt(np.mean((u_88 - u_m_at_88)**2)))
    print(f"\n  Fit quality (normalised):")
    print(f"    1985 (outbreak, vs u^+) : R^2 = {R2_85:.4f},  RMSE = {rmse_85:.4e}")
    print(f"    1988 (endemic,  vs u^-) : R^2 = {R2_88:.4f},  RMSE = {rmse_88:.4e}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    days_dense = np.linspace(t_min, t_max, 256)
    t_dense = (days_dense - t_min) / (t_max - t_min)
    u_minus_dense = np.interp(t_dense, t_grid, u_minus)
    u_plus_dense = np.interp(t_dense, t_grid, u_plus)

    ax = axes[0, 0]
    ax.scatter(days_85, dens_85, s=22, c='C3', label='1985 (outbreak)', zorder=3)
    ax.plot(days_dense, u_max * u_plus_dense, '-', color='C0', lw=1.8,
            label=fr'$u^+$: $R^2={R2_85:.3f}$')
    ax.set_xlabel('Ordinal date (1985)')
    ax.set_ylabel(r'Larval density (no.\ per m$^2$ branch)')
    ax.set_title('(a) Outbreak year vs upper-branch fixed point')
    ax.legend(framealpha=0.92)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(days_88, dens_88, s=26, c='C2', marker='s', label='1988 (endemic)', zorder=3)
    ax.plot(days_dense, u_max * u_minus_dense, '-', color='C0', lw=1.8,
            label=fr'$u^-$: $R^2={R2_88:.3f}$')
    ax.set_xlabel('Ordinal date (1988)')
    ax.set_ylabel(r'Larval density (no.\ per m$^2$ branch)')
    ax.set_title('(b) Endemic year vs lower-branch fixed point')
    ax.legend(framealpha=0.92)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t_grid, u_max * u_minus, '-', color='C2', lw=1.8, label=r'$u^-$ (endemic)')
    ax.plot(t_grid, u_max * u_plus, '-', color='C3', lw=1.8, label=r'$u^+$ (outbreak)')
    ax.axhline(u_max * c, color='k', ls='--', lw=0.9,
               label=fr'threshold $c$: $\ell(u^-)={Lu_minus:.3f},\ \ell(u^+)={Lu_plus:.3f}$')
    ax.set_xlabel(r'Normalised time $t \in [0,1]$')
    ax.set_ylabel(r'Density (no.\ per m$^2$)')
    ax.set_title(fr'(c) Bistable fixed points: $m^-={m_minus:+.3f},\ m^+={m_plus:+.3f}$')
    ax.legend(framealpha=0.92)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.semilogy(np.arange(len(pic_low['residuals'])) + 1, pic_low['residuals'], 'o-',
                color='C2', label=fr'from $U_0 = 0\ \to u^-$ ($\Delta={d_low_to_minus:.1e}$)')
    ax.semilogy(np.arange(len(pic_high['residuals'])) + 1, pic_high['residuals'], 's-',
                color='C3', label=fr'from $U_0 = \mathbf{{1}} \to u^+$ ($\Delta={d_high_to_plus:.1e}$)')
    ax.set_xlabel('Picard iteration $n$')
    ax.set_ylabel(r'Residual $\|U^{(n+1)} - U^{(n)}\|_\infty$')
    ax.set_title('(d) Picard convergence to each branch')
    ax.legend(framealpha=0.92)
    ax.grid(True, which='both', alpha=0.3)

    out_fig = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_realdata.pdf')
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"\n  Figure written: {out_fig}")

    # JSON
    print("\n" + "=" * 90)
    print("  JSON dump (compact)")
    print("=" * 90)
    out = {
        "data": {
            "source": "Royama et al. 2017, Dryad doi:10.5061/dryad.t175g",
            "n_obs_1985": int(len(days_85)),
            "n_obs_1988": int(len(days_88)),
            "max_density": u_max,
        },
        "fit": {
            "alpha": float(alpha), "beta": float(beta),
            "gamma": float(gamma), "delta": float(delta),
            "mu": float(mu), "sigma": float(sigma), "c": float(c),
            "loss": float(res.fun),
        },
        "bistable_diagnostics": {
            "Lu_minus": float(Lu_minus), "Lu_plus": float(Lu_plus),
            "m_minus": float(m_minus), "m_plus": float(m_plus),
            "regime": "BISTABLE" if (m_minus < 0 and m_plus > 0) else "NOT_BISTABLE",
        },
        "picard_certification": {
            "low_iters": int(pic_low['iters']),
            "high_iters": int(pic_high['iters']),
            "d_low_to_uminus": d_low_to_minus,
            "d_high_to_uplus": d_high_to_plus,
        },
        "fit_quality": {
            "R2_1985": R2_85, "RMSE_1985": rmse_85,
            "R2_1988": R2_88, "RMSE_1988": rmse_88,
        },
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
