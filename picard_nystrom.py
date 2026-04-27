"""
picard_nystrom.py
=================
Production-grade implementation of the Picard--Nystroem iteration for
Kannan-stable discontinuous threshold Fredholm integral equations of the
form

    u(t) = int_0^1 K(t,s) u(s) ds + g(t) + q(t) * H( int_0^1 u(s) ds - c ),

where H is the Heaviside step function. The discretization uses
Gauss--Legendre quadrature on [0,1]. All numerical kernels are vectorised;
no Python-level loops are used inside the iteration.

Author : David Ariza-Ruiz
Target : Journal of Computational and Applied Mathematics (Elsevier)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss


# ---------------------------------------------------------------------------
# Publication-quality plotting defaults
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "text.usetex": False,          # switch to True if a LaTeX toolchain is present
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "STIXGeneral", "Computer Modern Roman"],
    "mathtext.fontset": "cm",      # Computer Modern for math (Elsevier-friendly)
    "axes.labelsize": 12,
    "font.size": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.format": "pdf",       # vector format preferred by Elsevier
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,            # TrueType fonts (editable, no Type 3 issues)
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Quadrature utilities
# ---------------------------------------------------------------------------
def gauss_legendre_01(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (nodes, weights) of the N-point Gauss--Legendre rule on [0,1]."""
    x, w = leggauss(N)
    t = 0.5 * (x + 1.0)
    W = 0.5 * w
    return t, W


def composite_trapezoidal_01(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (nodes, weights) of the composite trapezoidal rule on [0,1]."""
    t = np.linspace(0.0, 1.0, N)
    h = 1.0 / (N - 1)
    w = np.full(N, h)
    w[0] *= 0.5
    w[-1] *= 0.5
    return t, w


def composite_simpson_01(N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Composite Simpson rule on [0,1] with N equispaced nodes.
    Requires N odd (i.e. even number of subintervals).
    """
    if N % 2 == 0:
        raise ValueError("Composite Simpson requires odd N (even number of subintervals)")
    t = np.linspace(0.0, 1.0, N)
    h = 1.0 / (N - 1)
    w = np.full(N, 2.0 * h / 3.0)
    w[1::2] = 4.0 * h / 3.0   # odd-indexed nodes (interior of each Simpson pair)
    w[0] = h / 3.0
    w[-1] = h / 3.0
    return t, w


# ---------------------------------------------------------------------------
# Assembly of the Nystroem system
# ---------------------------------------------------------------------------
@dataclass
class NystromSystem:
    t: np.ndarray   # nodes
    w: np.ndarray   # quadrature weights
    A: np.ndarray   # Nystroem matrix: A[i,j] = w_j K(t_i, t_j)
    G: np.ndarray   # G[i] = g(t_i)
    Q: np.ndarray   # Q[i] = q(t_i)


def assemble(N: int,
             K: Callable[[np.ndarray, np.ndarray], np.ndarray],
             g: Callable[[np.ndarray], np.ndarray],
             q: Callable[[np.ndarray], np.ndarray],
             rule: str = "gauss") -> NystromSystem:
    if rule == "gauss":
        t, w = gauss_legendre_01(N)
    elif rule == "trapezoidal":
        t, w = composite_trapezoidal_01(N)
    elif rule == "simpson":
        t, w = composite_simpson_01(N)
    else:
        raise ValueError(f"Unknown quadrature rule: {rule!r}")

    T_grid, S_grid = np.meshgrid(t, t, indexing="ij")
    A = w[np.newaxis, :] * K(T_grid, S_grid)
    return NystromSystem(t=t, w=w, A=A, G=g(t), Q=q(t))


# ---------------------------------------------------------------------------
# Discrete Kannan diagnostics
# ---------------------------------------------------------------------------
def kannan_diagnostics(sys_: NystromSystem, c: float, R: float) -> Dict[str, float]:
    """
    Compute the finite-dimensional Kannan stability indicators:

      kappa_N  = ||A_N||_inf
      mu_N     = ||ell_N o A_N||        (dual of sup-norm)
      delta    = ell_N(G) - c - mu_N R  (separation from threshold)
      lambda_N = (kappa_N + ||ell_N|| ||Q||_inf / delta) / (1 - kappa_N)

    The scheme is Kannan-stable on B_{N,R} when lambda_N < 1/2.
    """
    A, G, Q, w = sys_.A, sys_.G, sys_.Q, sys_.w
    kappa_N = float(np.max(np.sum(np.abs(A), axis=1)))
    mu_N    = float(np.sum(np.abs(w @ A)))
    ell_G   = float(w @ G)
    delta   = ell_G - c - mu_N * R
    normQ   = float(np.max(np.abs(Q)))
    ell_nrm = float(np.sum(np.abs(w)))

    if kappa_N >= 1.0 or delta <= 0.0:
        lam = float("inf")
    else:
        lam = (kappa_N + ell_nrm * normQ / delta) / (1.0 - kappa_N)

    return {
        "kappa_N": kappa_N,
        "mu_N": mu_N,
        "ell_G": ell_G,
        "delta": delta,
        "lambda_N": lam,
        "admissible": lam < 0.5,
    }


# ---------------------------------------------------------------------------
# Core iteration
# ---------------------------------------------------------------------------
def picard_nystrom(sys_: NystromSystem, c: float,
                   U0: np.ndarray | None = None,
                   tol: float = 1e-13,
                   nmax: int = 500) -> Dict[str, np.ndarray]:
    """
    Run the Picard--Nystroem iteration

        U^{n+1} = A_N U^{n} + G + theta_n Q,
        theta_n = H( w . U^{n} - c ).

    The function returns the last iterate plus the full residual and
    threshold histories, which allow full a posteriori analysis.
    """
    A, G, Q, w = sys_.A, sys_.G, sys_.Q, sys_.w
    N = G.size
    U = np.zeros(N) if U0 is None else np.asarray(U0, dtype=float).copy()

    residuals, thresholds, thetas = [], [], []

    for _ in range(nmax):
        ell = float(w @ U)
        theta = 1.0 if ell > c else 0.0

        U_new = A @ U + G + theta * Q
        res = float(np.max(np.abs(U_new - U)))

        residuals.append(res)
        thresholds.append(ell)
        thetas.append(theta)

        U = U_new
        if res < tol:
            break

    return {
        "U":         U,
        "residuals": np.asarray(residuals),
        "thresholds": np.asarray(thresholds),
        "thetas":    np.asarray(thetas),
        "iters":     len(residuals),
    }


# ---------------------------------------------------------------------------
# Smoothed semismooth Newton (Chen-Mangasarian sigmoid smoothing)
# ---------------------------------------------------------------------------
def _stable_sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically-stable sigmoid 1/(1+exp(-x))."""
    if np.isscalar(x):
        if x >= 0.0:
            e = np.exp(-x)
            return 1.0 / (1.0 + e)
        else:
            e = np.exp(x)
            return e / (1.0 + e)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0.0
    e_pos = np.exp(-x[pos])
    out[pos] = 1.0 / (1.0 + e_pos)
    e_neg = np.exp(x[~pos])
    out[~pos] = e_neg / (1.0 + e_neg)
    return out


def smoothed_newton(sys_: NystromSystem, c: float,
                    U0: np.ndarray,
                    eps_init: float = 1e-1,
                    eps_min: float = 1e-14,
                    eps_factor: float = 0.1,
                    tol: float = 1e-13,
                    nmax_inner: int = 50,
                    nmax_outer: int = 20) -> Dict[str, np.ndarray]:
    """
    Smoothed semismooth Newton for the discrete threshold equation

        F(U) = U - A_N U - G - phi_eps(w . U - c) Q = 0,

    with Chen-Mangasarian sigmoid smoothing
        phi_eps(s) = 1 / (1 + exp(-s / eps)).

    Each Newton step solves
        J(U) du = -F(U),    J(U) = M - phi_eps'(s) Q w^T,
    where M = I - A_N and the rank-one update is inverted via the
    Sherman-Morrison formula from a single LU factorisation of M.

    The smoothing parameter eps is reduced geometrically
    (homotopy continuation) until eps < eps_min or until convergence at
    the deepest level is reached. Both the smoothed residual
    ||F(U; eps)|| and the true residual ||F(U; eps -> 0)|| are returned.

    The Heaviside is not Lipschitz, so genuine semismooth Newton in the
    sense of Qi-Sun does not apply directly; the sigmoid smoothing is
    the canonical practical surrogate (Chen & Mangasarian 1995;
    Facchinei & Pang 2003, Ch.~9).
    """
    A, G, Q, w = sys_.A, sys_.G, sys_.Q, sys_.w
    N = G.size
    M  = np.eye(N) - A
    # LU factorisation done once; subsequent solves are O(N^2).
    from scipy.linalg import lu_factor, lu_solve
    LU = lu_factor(M)
    z  = lu_solve(LU, Q)        # M^{-1} Q, cached for Sherman-Morrison
    wTz = float(w @ z)

    U = np.asarray(U0, dtype=float).copy()

    res_smooth_hist: list[float] = []
    res_true_hist:   list[float] = []
    eps_hist:        list[float] = []
    total_iters = 0

    eps = float(eps_init)
    converged_overall = False

    for outer in range(nmax_outer):
        # Inner Newton loop at fixed eps.
        for inner in range(nmax_inner):
            s    = float(w @ U) - c
            phi  = _stable_sigmoid(s / eps)
            dphi = phi * (1.0 - phi) / eps

            F_smooth = U - A @ U - G - phi * Q
            res_smooth = float(np.max(np.abs(F_smooth)))

            theta_true = 1.0 if s > 0.0 else 0.0
            F_true = U - A @ U - G - theta_true * Q
            res_true = float(np.max(np.abs(F_true)))

            res_smooth_hist.append(res_smooth)
            res_true_hist.append(res_true)
            eps_hist.append(eps)
            total_iters += 1

            if res_true < tol and res_smooth < tol:
                converged_overall = True
                break

            # Sherman-Morrison Newton step (each lu_solve is O(N^2)).
            y0    = lu_solve(LU, -F_smooth)
            beta  = float(w @ y0)
            denom = 1.0 - dphi * wTz
            if abs(denom) < 1.0e-14:
                # Jacobian (numerically) singular: fall back to the
                # underlying linear correction.
                du = y0
            else:
                du = y0 + (dphi * beta / denom) * z
            U = U + du

        if converged_overall:
            break
        if eps <= eps_min:
            break
        eps = max(eps * eps_factor, eps_min)

    return {
        "U":                  U,
        "residuals_smooth":   np.asarray(res_smooth_hist),
        "residuals_true":     np.asarray(res_true_hist),
        "eps_history":        np.asarray(eps_hist),
        "total_iterations":   total_iters,
        "final_eps":          eps,
        "converged":          bool(converged_overall),
    }


# ---------------------------------------------------------------------------
# Two-sided Picard sweep (monotone-iteration certificate of uniqueness)
# ---------------------------------------------------------------------------
def two_sided_picard(sys_: NystromSystem, c: float,
                     U_lower0: np.ndarray, U_upper0: np.ndarray,
                     tol: float = 1e-13,
                     nmax: int = 500) -> Dict[str, np.ndarray]:
    """
    Run two simultaneous Picard--Nystroem sweeps from a sub-solution
    U_lower0 and a super-solution U_upper0 of the discrete equation,

        U^{n+1}_- = A_N U^{n}_- + G + theta^-_n Q,
        U^{n+1}_+ = A_N U^{n}_+ + G + theta^+_n Q,

    with theta^pm_n = H( w . U^{n}_pm - c ).

    The diagnostic separation history sigma_k = ||U^{k}_+ - U^{k}_-||_inf
    serves as an empirical certificate of uniqueness:
      * sigma_k -> 0           => unique fixed point (contractive regime);
      * sigma_k -> sigma_* > 0 => multiplicity (bistable regime).

    The method requires only a positive shift function q >= 0 (componentwise)
    and a positive kernel K >= 0 to preserve the natural order of the
    iterates; the function does not enforce these conditions but the
    interpretation as a sub/super-solution sweep is valid only under them.
    """
    A, G, Q, w = sys_.A, sys_.G, sys_.Q, sys_.w
    N = G.size
    U_lo = np.asarray(U_lower0, dtype=float).copy()
    U_hi = np.asarray(U_upper0, dtype=float).copy()
    if U_lo.size != N or U_hi.size != N:
        raise ValueError("U_lower0 and U_upper0 must have length N.")

    sigmas, res_lo, res_hi = [], [], []
    theta_lo_hist, theta_hi_hist = [], []

    for _ in range(nmax):
        ell_lo = float(w @ U_lo)
        ell_hi = float(w @ U_hi)
        th_lo  = 1.0 if ell_lo > c else 0.0
        th_hi  = 1.0 if ell_hi > c else 0.0

        U_lo_new = A @ U_lo + G + th_lo * Q
        U_hi_new = A @ U_hi + G + th_hi * Q

        r_lo = float(np.max(np.abs(U_lo_new - U_lo)))
        r_hi = float(np.max(np.abs(U_hi_new - U_hi)))
        sig  = float(np.max(np.abs(U_hi_new - U_lo_new)))

        sigmas.append(sig)
        res_lo.append(r_lo)
        res_hi.append(r_hi)
        theta_lo_hist.append(th_lo)
        theta_hi_hist.append(th_hi)

        U_lo, U_hi = U_lo_new, U_hi_new
        if max(r_lo, r_hi) < tol:
            break

    return {
        "U_lower":         U_lo,
        "U_upper":         U_hi,
        "sigmas":          np.asarray(sigmas),
        "residuals_lower": np.asarray(res_lo),
        "residuals_upper": np.asarray(res_hi),
        "thetas_lower":    np.asarray(theta_lo_hist),
        "thetas_upper":    np.asarray(theta_hi_hist),
        "iters":           len(sigmas),
    }


# ---------------------------------------------------------------------------
# Benchmark problem (Example 4.5 of the paper)
# ---------------------------------------------------------------------------
def benchmark_parameters() -> Dict[str, float]:
    return dict(alpha=0.10, beta=1.00, rho=0.00, sigma=0.05, c=0.50, R=2.00)


def exact_upper_branch(t: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    """Analytical fixed point on the upper branch (H = 1)."""
    alpha, beta, rho, sigma = p["alpha"], p["beta"], p["rho"], p["sigma"]
    m = ((beta + sigma) / 2.0 + (rho + sigma) / 3.0) / (1.0 - alpha / 3.0)
    return (beta + sigma) + (rho + sigma + alpha * m) * t


def build_benchmark_functions(p: Dict[str, float]):
    alpha, beta, rho, sigma = p["alpha"], p["beta"], p["rho"], p["sigma"]
    K = lambda t, s: alpha * t * s
    g = lambda t: beta + rho * t
    q = lambda t: sigma * (1.0 + t)
    return K, g, q


# ---------------------------------------------------------------------------
# Non-separable exponential kernel benchmark (Section 7.7 of the paper)
# ---------------------------------------------------------------------------
def nonseparable_parameters() -> Dict[str, float]:
    """
    Reference parameters for the non-separable exponential benchmark

        K(t,s) = alpha * exp(-|t-s|),
        g(t)   = beta * cos(pi t / 2),
        q(t)   = sigma (1 + t).
    """
    return dict(alpha=0.10, beta=1.50, sigma=0.05, c=0.50, R=2.00)


def build_nonseparable_functions(p: Dict[str, float]):
    alpha, beta, sigma = p["alpha"], p["beta"], p["sigma"]
    K = lambda t, s: alpha * np.exp(-np.abs(t - s))
    g = lambda t: beta * np.cos(np.pi * t / 2.0)
    q = lambda t: sigma * (1.0 + t)
    return K, g, q


def nonseparable_continuous_diagnostics(p: Dict[str, float]) -> Dict[str, float]:
    """
    Analytical continuous Kannan quantities for the non-separable benchmark:
        kappa  = 2 alpha (1 - e^{-1/2})
        mu_K   = 2 alpha e^{-1}
        int g  = 2 beta / pi
        ||g||  = beta     (attained at t = 0)
        ||q||  = 2 |sigma|
    """
    alpha, beta, sigma, c, R = p["alpha"], p["beta"], p["sigma"], p["c"], p["R"]
    kappa   = 2.0 * abs(alpha) * (1.0 - np.exp(-0.5))
    mu_K    = 2.0 * abs(alpha) * np.exp(-1.0)
    g_int   = 2.0 * beta / np.pi
    normg   = abs(beta)
    normq   = 2.0 * abs(sigma)
    delta_R = g_int - c - mu_K * R
    inv_lhs = kappa * R + normg + normq
    kannan_lhs = 3.0 * kappa + 2.0 * normq / delta_R if delta_R > 0 else float("inf")
    lam = (kappa + normq / delta_R) / (1.0 - kappa) \
          if (kappa < 1 and delta_R > 0) else float("inf")
    return dict(kappa=kappa, mu_K=mu_K, g_int=g_int,
                norm_g=normg, norm_q=normq, delta_R=delta_R,
                invariance_lhs=inv_lhs, invariance_rhs=R,
                kannan_lhs=kannan_lhs, lambda_cont=lam,
                admissible=(inv_lhs <= R and delta_R > 0
                            and kannan_lhs < 1 and lam < 0.5))


def nystrom_extension(t_eval: np.ndarray,
                      U_star: np.ndarray,
                      sys_: NystromSystem,
                      theta: float,
                      K: Callable[[np.ndarray, np.ndarray], np.ndarray],
                      g: Callable[[np.ndarray], np.ndarray],
                      q: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Nystroem continuous extension of the discrete fixed point:

        u_N(t) = sum_j w_j K(t, t_j) U_j + g(t) + theta q(t).

    This evaluates the Picard--Nystroem fixed point on an arbitrary grid
    without recomputing the iteration, and is the standard device to
    compare discretizations of different resolution N.
    """
    # Build K(t_eval, t_j) as a (len(t_eval), N) matrix via broadcasting
    T_eval = t_eval[:, np.newaxis]
    S_nod  = sys_.t[np.newaxis, :]
    Kmat   = K(T_eval, S_nod)
    Avec   = Kmat @ (sys_.w * U_star)
    return Avec + g(t_eval) + theta * q(t_eval)


# ---------------------------------------------------------------------------
# Applied case study: spatial population model with threshold harvesting
# ---------------------------------------------------------------------------
# Steady-state population equation with non-local dispersal and a regulator
# that activates harvesting whenever the integrated density exceeds a
# carrying-capacity threshold:
#
#   u(x) = int_0^1 K(x,y) u(y) dy + g(x) - h(x) H(int_0^1 u(y) dy - c).
#
# This is exactly the threshold integral equation (4.2) with q := -h,
# negative jump. References: Cantrell and Cosner (2003), Spatial Ecology
# via Reaction-Diffusion Equations, Wiley, Chapter 4.
# ---------------------------------------------------------------------------
def population_parameters() -> Dict[str, float]:
    """Reference parameters of the threshold-harvesting population model."""
    return dict(alpha=0.05, beta=1.5, eta=0.10, c=0.4, R=2.0)


def build_population_functions(p: Dict[str, float]):
    """
    Returns (K, g, q) for the population model with:
        K(x,y)    = alpha * exp(-|x-y|),
        g(x)      = (beta/2) * (1 + cos(pi x)),
        h(x)      = eta * x,         q(x) := -h(x).
    """
    alpha, beta, eta = p["alpha"], p["beta"], p["eta"]
    K = lambda x, y: alpha * np.exp(-np.abs(x - y))
    g = lambda x: 0.5 * beta * (1.0 + np.cos(np.pi * x))
    q = lambda x: -eta * x
    return K, g, q


def population_continuous_diagnostics(p: Dict[str, float]) -> Dict[str, float]:
    """Analytical Kannan quantities for the harvesting benchmark."""
    alpha, beta, eta, c, R = p["alpha"], p["beta"], p["eta"], p["c"], p["R"]
    kappa   = 2.0 * abs(alpha) * (1.0 - np.exp(-0.5))
    mu_K    = 2.0 * abs(alpha) * np.exp(-1.0)
    g_int   = 0.5 * beta                    # int (1+cos(pi x))/2 dx = 1/2
    Lq      = -0.5 * eta                    # -int eta x dx = -eta/2
    norm_g  = abs(beta)                      # max at x=0
    norm_q  = abs(eta)                       # max at x=1
    delta_R = g_int - c - mu_K * R
    inv_lhs = kappa * R + norm_g + norm_q
    kannan_lhs = 3.0 * kappa + 2.0 * norm_q / delta_R if delta_R > 0 else float("inf")
    lambda_R = (kappa + norm_q / delta_R) / (1.0 - kappa) \
               if (kappa < 1 and delta_R > 0) else float("inf")
    return dict(kappa=kappa, mu_K=mu_K, g_int=g_int, Lq=Lq,
                norm_g=norm_g, norm_q=norm_q, delta_R=delta_R,
                invariance_lhs=inv_lhs, invariance_rhs=R,
                kannan_lhs=kannan_lhs, lambda_R=lambda_R,
                admissible=(inv_lhs <= R and delta_R > 0
                            and kannan_lhs < 1 and lambda_R < 0.5))


def run_population_application(N: int = 128) -> Dict[str, object]:
    """Solve the threshold-harvesting model with the nominal parameters."""
    p = population_parameters()
    K, g, q = build_population_functions(p)
    sys_ = assemble(N, K, g, q, rule="gauss")
    diag_cont = population_continuous_diagnostics(p)
    diag_disc = kannan_diagnostics(sys_, p["c"], p["R"])
    U0 = np.zeros(sys_.t.size)        # start with empty population
    out = picard_nystrom(sys_, p["c"], U0=U0, tol=1e-13, nmax=300)

    # Also compute the harvest-free reference equilibrium for comparison
    # u_0 = (I - A)^{-1} g  (hypothetical "no harvesting" baseline)
    I = np.eye(sys_.t.size)
    U_no_harvest = np.linalg.solve(I - sys_.A, sys_.G)
    ell_no_harvest = float(sys_.w @ U_no_harvest)

    ell_star = float(sys_.w @ out["U"])
    return {
        "params": p, "N": N,
        "diagnostics_cont": diag_cont,
        "diagnostics_disc": diag_disc,
        "iterations": out["iters"],
        "final_residual": float(out["residuals"][-1]),
        "u_star": out["U"],
        "u_no_harvest": U_no_harvest,
        "ell_star": ell_star,
        "ell_no_harvest": ell_no_harvest,
        "harvest_reduction": ell_no_harvest - ell_star,
        "regulator_active_margin": ell_star - p["c"],
        "t_nodes": sys_.t,
        "residual_history": out["residuals"].tolist(),
        "threshold_history": out["thresholds"].tolist(),
    }


def run_population_sweep(eta_list, N: int = 128) -> Dict[str, object]:
    """Sweep harvesting effort eta and record equilibrium total population."""
    rows = []
    for eta in eta_list:
        p = population_parameters(); p["eta"] = float(eta)
        K, g, q = build_population_functions(p)
        sys_ = assemble(N, K, g, q, rule="gauss")
        diag_cont = population_continuous_diagnostics(p)
        if not diag_cont["admissible"]:
            rows.append({"eta": float(eta), "admissible": False,
                         "lambda_R": diag_cont["lambda_R"],
                         "ell_star": None, "iters": None})
            continue
        U0 = np.zeros(sys_.t.size)
        out = picard_nystrom(sys_, p["c"], U0=U0, tol=1e-13, nmax=300)
        rows.append({"eta": float(eta), "admissible": True,
                     "lambda_R": diag_cont["lambda_R"],
                     "ell_star": float(sys_.w @ out["U"]),
                     "iters": int(out["iters"])})
    return {"N": N, "rows": rows}


def plot_population_application(result, sweep, filename: str) -> None:
    """Two-panel figure for the harvesting benchmark."""
    p = result["params"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    # ---- left: equilibrium density profiles ----
    t = result["t_nodes"]
    axes[0].plot(t, result["u_no_harvest"], "--", color="#7f8c8d", lw=1.7,
                 label=r"No harvesting: $(I-A)^{-1}g$")
    axes[0].plot(t, result["u_star"], "-", color="#16a085", lw=2.0,
                 label=r"With harvesting: $u_\ast$")
    # source g and harvesting profile h, scaled for visualisation
    axes[0].plot(t, 0.5 * p["beta"] * (1 + np.cos(np.pi * t)), ":",
                 color="#1f3a93", lw=1.2, label=r"Source $g(x)$")
    axes[0].plot(t, p["eta"] * t, "-.", color="#c0392b", lw=1.2,
                 label=r"Harvest $h(x)$")
    axes[0].axhline(0, color="gray", lw=0.5, alpha=0.5)
    axes[0].set_xlabel(r"Spatial position $x$")
    axes[0].set_ylabel(r"Population density $u(x)$")
    axes[0].set_title("Threshold-harvesting equilibrium")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=True, fancybox=False, framealpha=0.92,
                   edgecolor="0.7", loc="upper right")

    # ---- right: sweep of total population vs harvesting effort eta ----
    etas = [r["eta"] for r in sweep["rows"] if r["admissible"]]
    ells = [r["ell_star"] for r in sweep["rows"] if r["admissible"]]
    axes[1].plot(etas, ells, "o-", color="#16a085", lw=1.5, ms=5,
                 label="$\\ell_N(u_\\ast)$ (Picard\u2013Nystr\u00f6m)")
    axes[1].axhline(p["c"], ls="--", color="#c0392b", lw=1.3,
                    label=fr"Threshold $c={p['c']}$")
    axes[1].set_xlabel(r"Harvesting effort $\eta$")
    axes[1].set_ylabel(r"Equilibrium total $\ell_N(u_\ast)$")
    axes[1].set_title("Sensitivity to harvesting effort")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=True, fancybox=False, framealpha=0.92,
                   edgecolor="0.7", loc="upper right")

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reference baseline: direct linear solve with branch test (DLS-BT)
# ---------------------------------------------------------------------------
def direct_solve_branch_test(sys_: NystromSystem, c: float,
                             compute_cond: bool = False) -> Dict[str, object]:
    """
    Reference solver for the discrete threshold problem
        U = A_N U + G + Q H(ell_N(U) - c).

    The method enumerates the two branches:
      (upper)  solve  (I - A_N) U_+ = G + Q,    accept if ell_N(U_+) > c;
      (lower)  solve  (I - A_N) U_- = G,        accept if ell_N(U_-) <= c.

    The optional flag compute_cond enables the (expensive) computation of
    the 2-norm condition number of I - A_N, which involves an O(N^3) SVD
    and is intended for analysis rather than timing comparisons.
    """
    N = sys_.G.size
    I = np.eye(N)
    M = I - sys_.A
    # Single LU factorisation reused for both right-hand sides
    from scipy.linalg import lu_factor, lu_solve
    lu_piv = lu_factor(M)
    U_plus  = lu_solve(lu_piv, sys_.G + sys_.Q)
    U_minus = lu_solve(lu_piv, sys_.G)
    # Condition number is opt-in: it triggers an SVD that is irrelevant
    # to the timing benchmark and would dominate the DS wall-clock for
    # large N.
    cond_M = float(np.linalg.cond(M)) if compute_cond else float("nan")

    ell_plus  = float(sys_.w @ U_plus)
    ell_minus = float(sys_.w @ U_minus)
    upper_admissible = ell_plus  > c
    lower_admissible = ell_minus <= c

    if upper_admissible and not lower_admissible:
        accepted = "upper"; U_star = U_plus
    elif lower_admissible and not upper_admissible:
        accepted = "lower"; U_star = U_minus
    elif upper_admissible and lower_admissible:
        accepted = "both"; U_star = U_plus       # ambiguous: both branches valid
    else:
        accepted = "none"; U_star = None         # neither: scheme breaks down

    # Verify residual ||U - T_N U||_inf
    if U_star is not None:
        ell_star = float(sys_.w @ U_star)
        theta_star = 1.0 if ell_star > c else 0.0
        T_star = sys_.A @ U_star + sys_.G + theta_star * sys_.Q
        residual = float(np.max(np.abs(U_star - T_star)))
    else:
        residual = float("inf")

    return {
        "U_star":      U_star,
        "U_plus":      U_plus,
        "U_minus":     U_minus,
        "ell_plus":    ell_plus,
        "ell_minus":   ell_minus,
        "accepted":    accepted,
        "residual":    residual,
        "cond_I_minus_AN": cond_M,
    }


def compare_picard_vs_direct(N_list=(32, 64, 128, 256),
                             benchmarks=("integral", "bvp", "nonseparable"),
                             tol: float = 1e-13,
                             repeats: int = 5000,
                             warmup: int = 50) -> Dict[str, object]:
    """
    Side-by-side comparison of Picard--Nystroem and DLS-BT on three benchmarks
    at multiple resolutions. For each (benchmark, N) combination, both methods
    are timed by performing `warmup` discarded warmup runs (to flush cold-cache
    effects) followed by `repeats` measured runs. The full set of measured
    samples is summarised by min/p1/q1/median/q3/p99/max, mean, std and IQR;
    these enable a complete distributional report and tail-sensitivity check.
    Pairwise sup-norm differences and residuals are also recorded.
    """
    import time
    rows = []
    for bench in benchmarks:
        # Build benchmark
        if bench == "integral":
            p = benchmark_parameters()
            K, g, q = build_benchmark_functions(p)
            c, R = p["c"], p["R"]
            exact = lambda t: exact_upper_branch(t, p)
        elif bench == "bvp":
            p = bvp_parameters()
            c, R = p["c"], p["R"]
            exact = lambda t: bvp_exact_solution(t, p)
        elif bench == "nonseparable":
            p = nonseparable_parameters()
            K, g, q = build_nonseparable_functions(p)
            c, R = p["c"], p["R"]
            exact = None  # no closed form
        else:
            raise ValueError(bench)

        for N in N_list:
            # Assemble
            if bench == "bvp":
                sys_ = assemble_bvp(N, p)
            else:
                sys_ = assemble(N, K, g, q, rule="gauss")

            # ----- Picard--Nystroem (warmup + timed) -------------------
            U0 = np.zeros(N) if bench == "bvp" else np.full(N, c / 2.0)
            for _ in range(warmup):
                _ = picard_nystrom(sys_, c, U0=U0, tol=tol, nmax=300)
            picard_times = np.empty(repeats, dtype=float)
            for k in range(repeats):
                t0 = time.perf_counter()
                out = picard_nystrom(sys_, c, U0=U0, tol=tol, nmax=300)
                picard_times[k] = time.perf_counter() - t0
            picard_time      = float(np.median(picard_times))
            picard_q1        = float(np.percentile(picard_times, 25))
            picard_q3        = float(np.percentile(picard_times, 75))
            picard_iqr       = picard_q3 - picard_q1
            picard_p1        = float(np.percentile(picard_times,  1))
            picard_p99       = float(np.percentile(picard_times, 99))
            picard_min       = float(np.min(picard_times))
            picard_max       = float(np.max(picard_times))
            picard_mean      = float(np.mean(picard_times))
            picard_std       = float(np.std(picard_times, ddof=1))
            U_pn = out["U"]
            res_pn = float(out["residuals"][-1])
            iters_pn = int(out["iters"])

            # ----- Direct solve + branch test (warmup + timed) ---------
            for _ in range(warmup):
                _ = direct_solve_branch_test(sys_, c, compute_cond=False)
            direct_times = np.empty(repeats, dtype=float)
            for k in range(repeats):
                t0 = time.perf_counter()
                ds = direct_solve_branch_test(sys_, c, compute_cond=False)
                direct_times[k] = time.perf_counter() - t0
            direct_time      = float(np.median(direct_times))
            direct_q1        = float(np.percentile(direct_times, 25))
            direct_q3        = float(np.percentile(direct_times, 75))
            direct_iqr       = direct_q3 - direct_q1
            direct_p1        = float(np.percentile(direct_times,  1))
            direct_p99       = float(np.percentile(direct_times, 99))
            direct_min       = float(np.min(direct_times))
            direct_max       = float(np.max(direct_times))
            direct_mean      = float(np.mean(direct_times))
            direct_std       = float(np.std(direct_times, ddof=1))
            U_ds = ds["U_star"]
            res_ds = ds["residual"]
            # Recompute the diagnostics with cond once (untimed)
            ds_with_cond = direct_solve_branch_test(sys_, c, compute_cond=True)
            ds["cond_I_minus_AN"] = ds_with_cond["cond_I_minus_AN"]

            # Pairwise consistency check
            if U_ds is not None:
                diff_pn_ds = float(np.max(np.abs(U_pn - U_ds)))
            else:
                diff_pn_ds = float("inf")

            # Error vs analytical exact (where available)
            if exact is not None:
                err_pn = float(np.max(np.abs(U_pn - exact(sys_.t))))
                err_ds = float(np.max(np.abs(U_ds - exact(sys_.t)))) if U_ds is not None else float("inf")
            else:
                err_pn = err_ds = None

            rows.append({
                "benchmark":         bench,
                "N":                 N,
                "picard_iters":      iters_pn,
                # Picard timing distribution
                "picard_time_s":     picard_time,    # median
                "picard_iqr_s":      picard_iqr,
                "picard_q1_s":       picard_q1,
                "picard_q3_s":       picard_q3,
                "picard_p1_s":       picard_p1,
                "picard_p99_s":      picard_p99,
                "picard_min_s":      picard_min,
                "picard_max_s":      picard_max,
                "picard_mean_s":     picard_mean,
                "picard_std_s":      picard_std,
                "picard_residual":   res_pn,
                "picard_error":      err_pn,
                # Direct timing distribution
                "direct_time_s":     direct_time,    # median
                "direct_iqr_s":      direct_iqr,
                "direct_q1_s":       direct_q1,
                "direct_q3_s":       direct_q3,
                "direct_p1_s":       direct_p1,
                "direct_p99_s":      direct_p99,
                "direct_min_s":      direct_min,
                "direct_max_s":      direct_max,
                "direct_mean_s":     direct_mean,
                "direct_std_s":      direct_std,
                "direct_residual":   res_ds,
                "direct_error":      err_ds,
                # Branch / consistency
                "direct_branch":     ds["accepted"],
                "direct_ell_plus":   ds["ell_plus"],
                "direct_ell_minus":  ds["ell_minus"],
                "cond_I_minus_AN":   ds["cond_I_minus_AN"],
                "max_diff_PN_DS":    diff_pn_ds,
                "speedup_DS_over_PN": picard_time / direct_time,
            })
    return {"rows": rows, "tol": tol, "repeats": repeats, "warmup": warmup}


def run_singlestep_experiment(N: int = 128,
                              sigma: float = 0.058,
                              n_random_per_family: int = 1000,
                              seed: int = 42) -> Dict[str, object]:
    """
    Experiment 6 of the paper: empirical investigation of the single-step
    transient property (Proposition 3.5) and of the operational sharpness
    of the Kannan envelope k_N = lambda_N / (1 - lambda_N).

    Two complementary tests on the non-separable benchmark of Section 7.7:

    (A) RANDOM SWEEP. n_random_per_family random initial conditions are drawn
        from each of four families (uniform, two-piece step, sinusoidal,
        high-frequency noise around +/-R), and the maximum number of
        threshold-indicator transitions theta_n != theta_{n-1} is recorded.

    (B) PARAMETRIC SEARCH. Within the lower-branch family
            U^0 = alpha + beta*t + gamma*sin(pi*t),  ell_N(U^0) <= c,
        a 25 x 21 x 11 grid is exhausted to maximise the empirical
        single-step ratio r_1/r_0; the maximum is compared with k_N.
    """
    p = nonseparable_parameters()
    p["sigma"] = float(sigma)
    K, g_fun, q_fun = build_nonseparable_functions(p)
    sys_ = assemble(N, K, g_fun, q_fun, rule="gauss")
    diag = kannan_diagnostics(sys_, p["c"], p["R"])
    lambda_N = float(diag["lambda_N"])
    kappa_N  = float(diag["kappa_N"])
    delta_N  = float(diag["delta"])
    k_N      = lambda_N / (1.0 - lambda_N) if lambda_N < 1 else float("inf")
    R = float(p["R"]); c = float(p["c"])
    t = sys_.t
    A, G, Q, w = sys_.A, sys_.G, sys_.Q, sys_.w

    # --------------- (A) Random sweep ---------------
    rng = np.random.default_rng(seed)
    families = ["uniform", "step", "sinusoidal", "noisy_extreme"]
    family_results = {}
    for family in families:
        max_sw = 0
        for _ in range(n_random_per_family):
            if family == "uniform":
                U0 = rng.uniform(-R, R, sys_.t.size)
            elif family == "step":
                split = rng.uniform(0, 1)
                v1, v2 = rng.uniform(-R, R, 2)
                U0 = np.where(t < split, v1, v2)
            elif family == "sinusoidal":
                k = rng.integers(1, 6)
                amp = rng.uniform(0, R)
                phase = rng.uniform(0, 2 * np.pi)
                U0 = amp * np.sin(k * np.pi * t + phase)
            else:  # noisy_extreme
                base = rng.choice([-R, R])
                noise = rng.uniform(-R / 2, R / 2, sys_.t.size)
                U0 = np.clip(base + noise, -R, R)
            U0 = np.clip(U0, -R, R)
            out = picard_nystrom(sys_, c, U0=U0, tol=1e-13, nmax=300)
            n_sw = int(np.sum(np.abs(np.diff(out["thetas"])) > 0.5))
            max_sw = max(max_sw, n_sw)
        family_results[family] = max_sw

    # --------------- (B) Parametric search for max r_1/r_0 ---------------
    def one_step_ratio(U0):
        U1 = A @ U0 + G                                        # theta=0 branch
        r0 = float(np.max(np.abs(U1 - U0)))
        ell1 = float(w @ U1)
        theta1 = 1.0 if ell1 > c else 0.0
        U2 = A @ U1 + G + theta1 * Q
        r1 = float(np.max(np.abs(U2 - U1)))
        return r0, r1, theta1, ell1

    grid_alpha = np.linspace(-R, c - 1e-3, 25)
    grid_beta  = np.linspace(-R, R, 21)
    grid_gamma = np.linspace(-R / 2, R / 2, 11)

    best = {"ratio": -1.0}
    n_admissible = 0
    for alpha in grid_alpha:
        for beta in grid_beta:
            for gamma in grid_gamma:
                U0 = alpha + beta * t + gamma * np.sin(np.pi * t)
                if np.max(np.abs(U0)) > R: continue
                ell0 = float(w @ U0)
                if ell0 > c: continue
                n_admissible += 1
                r0, r1, theta1, ell1 = one_step_ratio(U0)
                if r0 < 1e-10: continue
                ratio = r1 / r0
                if ratio > best["ratio"]:
                    best = dict(ratio=ratio, alpha=alpha, beta=beta, gamma=gamma,
                                r0=r0, r1=r1, theta1=theta1, ell0=ell0, ell1=ell1)

    return {
        "params": {"N": N, "sigma": sigma, "c": c, "R": R, "seed": seed},
        "diagnostics": {
            "lambda_N": lambda_N, "kappa_N": kappa_N,
            "k_N_envelope": k_N, "delta_N": delta_N,
        },
        "random_sweep": {
            "n_per_family": n_random_per_family,
            "max_switches_per_family": family_results,
            "global_max_switches": max(family_results.values()),
        },
        "parametric_search": {
            "n_admissible_initial_data": n_admissible,
            "best_one_step_ratio": best["ratio"],
            "best_alpha": best["alpha"],
            "best_beta": best["beta"],
            "best_gamma": best["gamma"],
            "best_r0": best["r0"],
            "best_r1": best["r1"],
            "best_ell0": best["ell0"],
            "best_ell1": best["ell1"],
            "best_theta1": int(best["theta1"]),
        },
    }


def run_kannan_envelope_experiment(N: int = 128,
                                   sigma: float = 0.058) -> Dict[str, object]:
    """
    Probe the operational role of the Kannan envelope k_N = lambda_N/(1-lambda_N)
    in the transient regime, on the non-separable benchmark with sigma chosen
    near the Kannan boundary (lambda_N close to 1/2).

    Two complementary searches:
    (a) Maximize r_1/r_0 over a parametric family of lower-branch initial data
        U^0 = alpha + beta*t + gamma*sin(pi*t), measuring how close the empirical
        single-step transient ratio comes to the theoretical Kannan envelope k_N.
    (b) Count threshold-switch flips for a wide sweep of initial data, including
        configurations designed to encourage oscillations (extreme values, both
        signs, t-dependent profiles).
    """
    p = nonseparable_parameters()
    p["sigma"] = float(sigma)
    K, g_fun, q_fun = build_nonseparable_functions(p)
    sys_ = assemble(N, K, g_fun, q_fun, rule="gauss")
    diag = kannan_diagnostics(sys_, p["c"], p["R"])
    lambda_N = float(diag["lambda_N"])
    kappa_N  = float(diag["kappa_N"])
    delta_N  = float(diag["delta"])
    k_N      = lambda_N / (1.0 - lambda_N) if lambda_N < 1 else float("inf")

    # ----- Part (a): maximize r_1/r_0 over lower-branch initial data -------
    R = float(p["R"]); c = float(p["c"])
    t = sys_.t
    A, G, Q, w = sys_.A, sys_.G, sys_.Q, sys_.w

    def one_step_residuals(U0):
        # First Picard step: U0 in lower branch (theta=0) by construction.
        U1 = A @ U0 + G
        r0 = float(np.max(np.abs(U1 - U0)))
        ell1 = float(w @ U1)
        theta1 = 1.0 if ell1 > c else 0.0
        U2 = A @ U1 + G + theta1 * Q
        r1 = float(np.max(np.abs(U2 - U1)))
        return r0, r1, theta1, ell1

    grid_alpha = np.linspace(-R, c - 1e-3, 25)
    grid_beta  = np.linspace(-R, R, 21)
    grid_gamma = np.linspace(-R/2, R/2, 11)

    best = {"ratio": -1.0}
    for alpha in grid_alpha:
        for beta in grid_beta:
            for gamma in grid_gamma:
                U0 = alpha + beta * t + gamma * np.sin(np.pi * t)
                if np.max(np.abs(U0)) > R: continue
                ell0 = float(w @ U0)
                if ell0 > c: continue                   # require lower branch
                r0, r1, theta1, ell1 = one_step_residuals(U0)
                if r0 < 1e-10: continue                  # avoid numerical noise
                ratio = r1 / r0
                if ratio > best["ratio"]:
                    best = dict(ratio=ratio, alpha=alpha, beta=beta, gamma=gamma,
                                r0=r0, r1=r1, theta1=theta1, ell0=ell0, ell1=ell1,
                                U0=U0.copy())

    # Run full iteration from the best initial datum to verify convergence
    out_best = picard_nystrom(sys_, c, U0=best["U0"], tol=1e-13, nmax=300)

    # ----- Part (b): exhaustive switch-count search ------------------------
    switch_records = []
    for alpha in np.linspace(-R, R, 21):
        for beta in np.linspace(-R, R, 21):
            U0 = alpha + beta * t
            if np.max(np.abs(U0)) > R: continue
            out = picard_nystrom(sys_, c, U0=U0, tol=1e-13, nmax=300)
            n_switches = int(np.sum(np.abs(np.diff(out["thetas"])) > 0.5))
            switch_records.append({
                "alpha": float(alpha), "beta": float(beta),
                "ell0": float(w @ U0), "n_switches": n_switches,
                "iters": out["iters"],
            })

    max_switches = max(r["n_switches"] for r in switch_records)
    n_with_max   = sum(1 for r in switch_records if r["n_switches"] == max_switches)

    return {
        "params": p,
        "lambda_N": lambda_N, "kappa_N": kappa_N, "k_N": k_N, "delta_N": delta_N,
        "best": {k: v for k, v in best.items() if k != "U0"},
        "best_iters": out_best["iters"],
        "best_residuals": out_best["residuals"].tolist(),
        "best_thresholds": out_best["thresholds"].tolist(),
        "best_thetas": out_best["thetas"].tolist(),
        "max_switches_observed": max_switches,
        "n_initial_data_with_max_switches": n_with_max,
        "switch_distribution": {s: sum(1 for r in switch_records if r["n_switches"] == s)
                                 for s in range(max_switches + 1)},
    }


def run_quadrature_comparison(Ns: tuple = (9, 17, 33, 65, 129)) -> Dict[str, object]:
    """
    Compare Gauss-Legendre, composite Simpson and composite trapezoidal rules
    on the non-separable exponential benchmark. For each rule and each
    resolution N, report the absolute errors of the Kannan diagnostics
    against their analytical continuous counterparts, plus the Picard-Nystroem
    iteration count to tolerance 1e-13.
    """
    p = nonseparable_parameters()
    K, g_fun, q_fun = build_nonseparable_functions(p)
    cont = nonseparable_continuous_diagnostics(p)

    table = {}
    for rule in ("gauss", "simpson", "trapezoidal"):
        rows = []
        for N in Ns:
            sys_ = assemble(N, K, g_fun, q_fun, rule=rule)
            d   = kannan_diagnostics(sys_, p["c"], p["R"])
            U0  = np.full(sys_.t.size, p["c"] / 2.0)
            out = picard_nystrom(sys_, p["c"], U0=U0, tol=1e-13, nmax=300)
            rows.append({
                "N":      N,
                "kappa_err":  abs(d["kappa_N"] - cont["kappa"]),
                "mu_err":     abs(d["mu_N"]    - cont["mu_K"]),
                "ellG_err":   abs(d["ell_G"]   - cont["g_int"]),
                "delta_err":  abs(d["delta"]   - cont["delta_R"]),
                "lambda_err": abs(d["lambda_N"]- cont["lambda_cont"]),
                "lambda_N":   d["lambda_N"],
                "iters":      out["iters"],
                "final_res":  float(out["residuals"][-1]),
            })
        table[rule] = rows

    # Observed convergence orders from the last two rows (finest grids)
    orders = {}
    for rule, rows in table.items():
        if len(rows) < 2:
            continue
        r_c, r_f = rows[-2], rows[-1]
        N_c, N_f = r_c["N"], r_f["N"]
        ratio_N = np.log(N_f / N_c)
        def order(ec, ef):
            if ec <= 0 or ef <= 0 or ec == ef:
                return float("nan")
            return float(-np.log(ef / ec) / ratio_N)
        orders[rule] = {
            "kappa_order":  order(r_c["kappa_err"], r_f["kappa_err"]),
            "mu_order":     order(r_c["mu_err"],    r_f["mu_err"]),
            "ellG_order":   order(r_c["ellG_err"],  r_f["ellG_err"]),
            "lambda_order": order(r_c["lambda_err"],r_f["lambda_err"]),
        }

    return {"continuous": cont, "table": table, "observed_orders": orders}


def run_nonseparable_experiment(Ns: tuple = (32, 64, 128, 256),
                                output_dir: str = ".") -> Dict[str, object]:
    """
    Execute the non-separable benchmark at multiple resolutions, certifying
    convergence via Nystroem mesh refinement.
    """
    import os
    p = nonseparable_parameters()
    c, R = p["c"], p["R"]
    K, g, q = build_nonseparable_functions(p)
    diag_cont = nonseparable_continuous_diagnostics(p)
    assert diag_cont["admissible"], f"Continuous hypotheses fail: {diag_cont}"

    # Common reference grid for the Nystroem extension
    t_ref = np.linspace(0.0, 1.0, 401)

    results = []
    u_extensions = {}
    for N in Ns:
        sys_ = assemble(N, K, g, q, rule="gauss")
        diag_N = kannan_diagnostics(sys_, c, R)
        assert diag_N["admissible"], f"Discrete Kannan fails at N={N}: {diag_N}"
        U0 = np.full(sys_.t.size, c / 2.0)
        out = picard_nystrom(sys_, c, U0=U0, tol=1e-13, nmax=300)
        theta_star = float(out["thetas"][-1])
        u_ext = nystrom_extension(t_ref, out["U"], sys_, theta_star, K, g, q)
        u_extensions[N] = u_ext
        results.append({
            "N": N,
            "kappa_N": diag_N["kappa_N"],
            "mu_N": diag_N["mu_N"],
            "ell_G": diag_N["ell_G"],
            "delta": diag_N["delta"],
            "lambda_N": diag_N["lambda_N"],
            "iters": out["iters"],
            "final_residual": float(out["residuals"][-1]),
            "theta_star": theta_star,
        })

    # Mesh-refinement self-consistency: sup differences between N and 2N
    refinement = []
    for a, b in zip(Ns[:-1], Ns[1:]):
        diff = float(np.max(np.abs(u_extensions[a] - u_extensions[b])))
        refinement.append({"N_coarse": a, "N_fine": b, "sup_diff": diff})

    return {
        "params": p,
        "continuous_diagnostics": diag_cont,
        "resolution_results": results,
        "mesh_refinement": refinement,
        "t_ref": t_ref,
        "u_extensions": u_extensions,
    }


def run_stress_sweep(sigmas: tuple = (0.010, 0.030, 0.050, 0.058, 0.060, 0.065),
                     N: int = 128) -> Dict[str, object]:
    """
    Stress sweep for the non-separable benchmark: vary sigma, measure
    discrete Kannan constant lambda_N, empirical successive-residual rate
    k_emp := geometric mean of r_n / r_{n-1} over the tail of the history,
    and compare with the theoretical bound k_N = lambda_N / (1 - lambda_N).
    """
    p0 = nonseparable_parameters()
    rows = []
    for sigma in sigmas:
        p = dict(p0); p["sigma"] = float(sigma)
        K, g, q = build_nonseparable_functions(p)
        sys_ = assemble(N, K, g, q, rule="gauss")
        diag_N = kannan_diagnostics(sys_, p["c"], p["R"])
        U0 = np.full(sys_.t.size, p["c"] / 2.0)
        out = picard_nystrom(sys_, p["c"], U0=U0, tol=1e-13, nmax=500)
        r = out["residuals"]
        # Geometric mean of successive ratios over the tail (after switch settled)
        if r.size >= 4:
            tail = r[-min(5, r.size - 1):]
            ratios = tail[1:] / tail[:-1]
            ratios = ratios[(ratios > 0) & (ratios < 1)]
            k_emp = float(np.exp(np.mean(np.log(ratios)))) if ratios.size else float("nan")
        else:
            k_emp = float("nan")
        lam = diag_N["lambda_N"]
        k_N = lam / (1.0 - lam) if lam < 1 else float("inf")
        rows.append({
            "sigma": sigma,
            "lambda_N": lam,
            "admissible": diag_N["admissible"],
            "k_N_theory": k_N,
            "k_empirical": k_emp,
            "iterations": out["iters"],
            "final_residual": float(out["residuals"][-1]),
        })
    return {"N": N, "rows": rows}


def plot_nonseparable(ns_result: Dict[str, object], filename: str) -> None:
    """
    Two-panel figure for the non-separable benchmark:
      (left)  converged u_N(t) for different N superimposed;
      (right) mesh-refinement sup-norm decay on log-log axes.
    """
    t_ref = ns_result["t_ref"]
    u_ext = ns_result["u_extensions"]
    ref   = ns_result["mesh_refinement"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # --- left panel: Nystroem-extension solutions overlap ---
    colors = ["#1f3a93", "#16a085", "#c0392b", "#8e44ad"]
    for (N, u), col in zip(u_ext.items(), colors):
        axes[0].plot(t_ref, u, lw=1.3, color=col, label=f"$N={N}$")
    axes[0].set_xlabel(r"$t$")
    axes[0].set_ylabel("$u_N(t)$  (Nystr\u00f6m extension)")
    axes[0].set_title("Converged fixed point at several resolutions")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=True, fancybox=False, framealpha=0.92,
                   edgecolor="0.7", loc="upper right")

    # --- right panel: log-log mesh refinement ---
    N_fine = np.array([r["N_fine"] for r in ref])
    diffs  = np.array([r["sup_diff"] for r in ref])
    axes[1].loglog(N_fine, diffs, "o-", lw=1.3, ms=6, color="#1f3a93",
                   label=r"$\|u_{2N}-u_N\|_\infty$ (sup-norm)")
    # Reference power laws for visual guidance
    ref_line = diffs[0] * (N_fine[0] / N_fine) ** 2
    axes[1].loglog(N_fine, ref_line, "--", lw=1.0, color="#7f8c8d",
                   label=r"reference slope $N^{-2}$")
    axes[1].set_xlabel(r"Finer resolution $2N$")
    axes[1].set_ylabel("Nystr\u00f6m mesh consistency")
    axes[1].set_title("Self-consistency by mesh refinement")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(frameon=True, fancybox=False, framealpha=0.92,
                   edgecolor="0.7", loc="lower left")

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------
def plot_residuals(out: Dict[str, np.ndarray],
                   diag: Dict[str, float],
                   filename: str) -> None:
    """Semilog plot of the Picard--Nystroem residuals with Kannan envelope."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    n = np.arange(1, out["iters"] + 1)
    ax.semilogy(n, out["residuals"], "o-", lw=1.3, ms=4.5,
                color="#1f3a93",
                label=r"$\|U^{(n+1)}-U^{(n)}\|_\infty$")

    lam = diag["lambda_N"]
    if np.isfinite(lam) and lam < 1.0:
        k = lam / (1.0 - lam)
        if k > 0.0 and k < 1.0 and out["iters"] >= 2:
            r0 = out["residuals"][0]
            env = r0 * k ** (n - 1)
            ax.semilogy(n, env, "--", lw=1.1, color="#c0392b",
                        label=rf"Kannan bound, $\lambda_N={lam:.3f}$")

    ax.set_xlabel(r"Iteration $n$")
    ax.set_ylabel(r"Successive residual (log scale)")
    ax.set_title("Picard\u2013Nystr\u00f6m convergence")
    ax.grid(True, which="both", alpha=0.3)
    # Both curves occupy the diagonal from upper-left to lower-right of the
    # log plot; the lower-left corner is the only region free of data.
    ax.legend(frameon=True, fancybox=False, framealpha=0.92,
              edgecolor="0.7", loc="lower left")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def plot_solution_and_threshold(out: Dict[str, np.ndarray],
                                sys_: NystromSystem,
                                p: Dict[str, float],
                                c: float,
                                filename: str) -> None:
    """Two-panel figure: converged solution and threshold activation."""
    tt = np.linspace(0.0, 1.0, 400)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # ---- left: converged solution ----
    u_ex = exact_upper_branch(tt, p)
    axes[0].plot(tt, u_ex, "-", color="#16a085", lw=2.0,
                 label=r"Exact upper-branch $u_*(t)$")
    axes[0].plot(sys_.t, out["U"], "o", ms=4, color="#1f3a93",
                 label=f"Nystr\u00f6m iterate, $N={sys_.t.size}$")
    # also overlay the (hypothetical) lower-branch fixed point (I-A)^{-1} g
    I = np.eye(sys_.A.shape[0])
    U_lower = np.linalg.solve(I - sys_.A, sys_.G)
    axes[0].plot(sys_.t, U_lower, "s", ms=3, mfc="none", color="#7f8c8d",
                 label=r"Lower-branch candidate $(I-A)^{-1}g$")
    axes[0].set_xlabel(r"$t$")
    axes[0].set_ylabel(r"$u(t)$")
    axes[0].set_title(r"Converged solution (jump captured)")
    axes[0].grid(True, alpha=0.3)
    # Upper-left is the only region free of data (upper branch starts at u~1.05,
    # lower branch starts at u~1.00); legend fits above the upper branch.
    axes[0].legend(frameon=True, fancybox=False, framealpha=0.92,
                   edgecolor="0.7", loc="upper left")

    # ---- right: threshold history ----
    axes[1].plot(np.arange(out["iters"]), out["thresholds"], "o-",
                 lw=1.2, ms=4.5, color="#2c3e50",
                 label=r"$\ell_N(U^{(n)})$")
    axes[1].axhline(c, ls="--", color="#c0392b", lw=1.3,
                    label=rf"threshold $c={c}$")

    # vertical markers when the switch activates/deactivates
    thetas = out["thetas"]
    switch_idx = np.where(np.diff(thetas) != 0)[0]
    for i, idx in enumerate(switch_idx):
        axes[1].axvline(idx + 1, color="#e67e22", alpha=0.5, ls=":",
                        label="Heaviside switch" if i == 0 else None)

    axes[1].set_xlabel(r"Iteration $n$")
    axes[1].set_ylabel(r"$\ell_N(U^{(n)}) = w \cdot U^{(n)}$")
    axes[1].set_title(r"Nonlocal threshold and Heaviside activation")
    axes[1].grid(True, alpha=0.3)
    # Data stabilises at y~1.1 from n>=2; threshold line at y=0.5; the
    # mid-right region (y~0.6-1.0) is free of data.
    axes[1].legend(frameon=True, fancybox=False, framealpha=0.92,
                   edgecolor="0.7", loc="center right")

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Threshold boundary value problem (Section 5 of the paper)
# ---------------------------------------------------------------------------
def green_kernel(T: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    Vectorised Green kernel of the two-point BVP -u''=f, u(0)=u(1)=0:

        G(t,s) = t (1-s)   if t <= s,
                 s (1-t)   if t >  s.
    """
    return np.where(T <= S, T * (1.0 - S), S * (1.0 - T))


def bvp_parameters() -> Dict[str, float]:
    """
    Reference parameters of the threshold BVP of Section 5:
        -u''(t) = a_0 u(t) + b_0 + d_0 H(int u - c),   u(0)=u(1)=0.
    The constants below verify the hypotheses of Theorem 5.1 strictly.
    """
    return dict(a0=0.5, b0=1.0, d0=0.1, c=0.0, R=0.5)


def assemble_bvp(N: int, p: Dict[str, float]) -> NystromSystem:
    """
    Assemble the Nystroem system associated with the Green formulation

        u(t) = int_0^1 G(t,s) a(s) u(s) ds + g(t) + q(t) H(int u - c),
        g(t) = int G(t,s) f_0(s) ds,  q(t) = int G(t,s) f_1(s) ds.

    For constant coefficients a(t)=a_0, f_0=b_0, f_1=d_0, the analytical
    identities reduce the assembly to explicit quadrature evaluations.
    """
    t, w = gauss_legendre_01(N)
    T_grid, S_grid = np.meshgrid(t, t, indexing="ij")
    G = green_kernel(T_grid, S_grid)
    A = w[np.newaxis, :] * G * p["a0"]            # (A_N)_{ij} = w_j G(t_i,t_j) a(t_j)
    Gvec = (w[np.newaxis, :] * G).sum(axis=1) * p["b0"]   # g(t_i)
    Qvec = (w[np.newaxis, :] * G).sum(axis=1) * p["d0"]   # q(t_i)
    return NystromSystem(t=t, w=w, A=A, G=Gvec, Q=Qvec)


def bvp_exact_solution(t: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    """
    Closed-form fixed point on the upper branch (H=1) of the BVP:
        u''(t) + a_0 u(t) = -(b_0 + d_0),  u(0)=u(1)=0.
    """
    a0, b0, d0 = p["a0"], p["b0"], p["d0"]
    omega = np.sqrt(a0) if a0 > 0 else np.sqrt(-a0)
    gamma = b0 + d0
    if a0 > 0:
        A = gamma / a0
        B = A * (1.0 - np.cos(omega)) / np.sin(omega)
        return A * np.cos(omega * t) + B * np.sin(omega * t) - A
    elif a0 < 0:
        # hyperbolic case: u'' - |a0| u = -(b0+d0) (not used here)
        A = -gamma / a0
        B = A * (1.0 - np.cosh(omega)) / np.sinh(omega)
        return A * np.cosh(omega * t) + B * np.sinh(omega * t) - A
    else:
        # a0 = 0: -u'' = gamma, u = gamma t (1-t)/2
        return 0.5 * gamma * t * (1.0 - t)


def bvp_continuous_diagnostics(p: Dict[str, float]) -> Dict[str, float]:
    """Return the analytical Kannan constants for the BVP benchmark."""
    a0, b0, d0, c, R = p["a0"], p["b0"], p["d0"], p["c"], p["R"]
    kappa_G = abs(a0) / 8.0
    mu_G    = abs(a0) / 12.0
    g_int   = b0 / 12.0
    normg   = b0 / 8.0
    normq   = d0 / 8.0
    delta_R = g_int - c - mu_G * R
    invariance_lhs = kappa_G * R + normg + normq
    kannan_lhs = 3.0 * kappa_G + 2.0 * normq / delta_R if delta_R > 0 else float("inf")
    lambda_G = (kappa_G + normq / delta_R) / (1.0 - kappa_G) \
               if (kappa_G < 1 and delta_R > 0) else float("inf")
    return dict(kappa_G=kappa_G, mu_G=mu_G, g_int=g_int,
                norm_g=normg, norm_q=normq, delta_R=delta_R,
                invariance_lhs=invariance_lhs, invariance_rhs=R,
                kannan_lhs=kannan_lhs, lambda_G=lambda_G,
                admissible=(invariance_lhs <= R and delta_R > 0 and kannan_lhs < 1 and lambda_G < 0.5))


def plot_bvp(out: Dict[str, np.ndarray],
             sys_: NystromSystem,
             p: Dict[str, float],
             diag_N: Dict[str, float],
             filename: str) -> None:
    """
    Two-panel figure for the BVP experiment:
      (left)  converged u(t) against the analytical solution;
      (right) residual decay with Kannan envelope.
    """
    tt = np.linspace(0.0, 1.0, 400)
    u_ex = bvp_exact_solution(tt, p)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # Left panel: solution
    axes[0].plot(tt, u_ex, "-", color="#16a085", lw=2.0,
                 label="Analytical $u_*(t)$ (upper branch)")
    axes[0].plot(sys_.t, out["U"], "o", ms=4, color="#1f3a93",
                 label=f"Nystr\u00f6m iterate, $N={sys_.t.size}$")
    # Boundary conditions marker
    axes[0].axhline(0.0, color="gray", lw=0.5, alpha=0.5)
    axes[0].set_xlabel(r"$t$")
    axes[0].set_ylabel(r"$u(t)$")
    axes[0].set_title(f"BVP: $a_0={p['a0']}$, $b_0={p['b0']}$, $d_0={p['d0']}$, $c={p['c']}$")
    axes[0].grid(True, alpha=0.3)
    # The BVP solution is a bump with u~0 near t=0 and t=1; the lower-centre
    # of the axes is always below the curve.
    axes[0].legend(frameon=True, fancybox=False, framealpha=0.92,
                   edgecolor="0.7", loc="lower center")

    # Right panel: residual with Kannan envelope
    n = np.arange(1, out["iters"] + 1)
    axes[1].semilogy(n, out["residuals"], "o-", lw=1.2, ms=4.5,
                     color="#1f3a93",
                     label=r"$\|U^{(n+1)}-U^{(n)}\|_\infty$")
    lam = diag_N["lambda_N"]
    if np.isfinite(lam) and lam < 1.0:
        k = lam / (1.0 - lam)
        if 0 < k < 1:
            r0 = out["residuals"][0]
            env = r0 * k ** (n - 1)
            axes[1].semilogy(n, env, "--", lw=1.1, color="#c0392b",
                             label=rf"Kannan bound, $\lambda_N={lam:.3f}$")
    axes[1].set_xlabel(r"Iteration $n$")
    axes[1].set_ylabel(r"Successive residual (log scale)")
    axes[1].set_title("Picard\u2013Nystr\u00f6m convergence (BVP)")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(frameon=True, fancybox=False, framealpha=0.92,
                   edgecolor="0.7", loc="lower left")

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def run_bvp_experiment(N: int = 64,
                       output_dir: str = ".") -> Dict[str, object]:
    import os
    p = bvp_parameters()
    diag_cont = bvp_continuous_diagnostics(p)
    assert diag_cont["admissible"], (
        f"BVP parameters violate the Kannan hypotheses: {diag_cont}")

    sys_ = assemble_bvp(N, p)
    diag_N = kannan_diagnostics(sys_, p["c"], p["R"])
    assert diag_N["admissible"], f"Discrete Kannan fails: {diag_N}"

    U0 = np.zeros(sys_.t.size)  # start at zero (lower branch: int U0 = 0 = c)
    out = picard_nystrom(sys_, p["c"], U0=U0, tol=1e-13, nmax=300)

    err = float(np.max(np.abs(out["U"] - bvp_exact_solution(sys_.t, p))))

    fig_bvp = os.path.join(output_dir, "fig_bvp.pdf")
    plot_bvp(out, sys_, p, diag_N, fig_bvp)

    return {
        "params": p,
        "continuous_diagnostics": diag_cont,
        "discrete_diagnostics": diag_N,
        "iterations": out["iters"],
        "final_residual": float(out["residuals"][-1]),
        "sup_error_vs_exact": err,
        "figure": fig_bvp,
        "history": {
            "residuals": out["residuals"].tolist(),
            "thresholds": out["thresholds"].tolist(),
            "thetas": out["thetas"].tolist(),
        },
    }


def run_full_experiment(N: int = 64,
                        output_dir: str = ".") -> Dict[str, object]:
    import os
    p = benchmark_parameters()
    c, R = p["c"], p["R"]
    K, g, q = build_benchmark_functions(p)

    sys_ = assemble(N, K, g, q, rule="gauss")
    diag = kannan_diagnostics(sys_, c, R)

    # start strictly below the threshold: the switch begins at H=0
    U0 = np.full(sys_.t.size, c / 2.0)
    out = picard_nystrom(sys_, c, U0=U0, tol=1e-13, nmax=200)

    u_ex_nodes = exact_upper_branch(sys_.t, p)
    err = float(np.max(np.abs(out["U"] - u_ex_nodes)))

    fig_res = os.path.join(output_dir, "fig_residual.pdf")
    fig_sol = os.path.join(output_dir, "fig_solution_threshold.pdf")
    plot_residuals(out, diag, fig_res)
    plot_solution_and_threshold(out, sys_, p, c, fig_sol)

    summary = {
        "params": p,
        "diagnostics": diag,
        "iterations": out["iters"],
        "final_residual": float(out["residuals"][-1]),
        "sup_error_vs_exact": err,
        "figures": [fig_res, fig_sol],
    }
    return summary


# ---------------------------------------------------------------------------
# Bistable regime experiment (Section 7.11 of the paper)
# ---------------------------------------------------------------------------
def bistability_parameters() -> Dict[str, float]:
    """
    Reference parameters in the bistable regime for the separable benchmark
        K(t,s) = alpha t s, g(t) = beta + rho t, q(t) = sigma (1 + t).

    Choice rationale:
      alpha = 0.5  -> ||A||_inf = alpha/2 = 0.25 < 1 (Banach contraction
                                                      on each branch)
      beta  = 0.6, rho = 0.0
      sigma = 0.6
      c     = 1.0  -> placed in the bistable interval [L u^-, L u^+]
      R     = 3.0  -> ensures invariance B_R for both branch fixed points

    With these data the global Kannan diagnostic (Theorem 4.1) is
    inadmissible (delta < 0), but Theorem 3.10 yields two coexisting
    fixed points u^- and u^+ with positive basin radii (Cor. 3.11).
    """
    return dict(alpha=0.5, beta=0.6, rho=0.0, sigma=0.6, c=1.0, R=3.0)


def bistability_branch_diagnostics(sys_: NystromSystem,
                                   c: float) -> Dict[str, object]:
    """
    Compute branch fixed points u^- = (I-A_N)^{-1} G and
    u^+ = (I-A_N)^{-1} (G+Q), and the discrete branch margins
        m^- = ell_N(u^-) - c,    m^+ = ell_N(u^+) - c,
    together with the basin radii r^- = -m^- / ||ell_N||,
    r^+ = m^+ / ||ell_N|| of Corollary 3.11.

    All quantities are exact (machine precision) for the discrete system.
    """
    A, G, Q, w = sys_.A, sys_.G, sys_.Q, sys_.w
    N = G.size

    M = np.eye(N) - A
    u_minus = np.linalg.solve(M, G)
    u_plus  = np.linalg.solve(M, G + Q)

    Lu_minus = float(w @ u_minus)
    Lu_plus  = float(w @ u_plus)
    m_minus  = Lu_minus - c
    m_plus   = Lu_plus  - c

    norm_L  = float(np.sum(np.abs(w)))   # ||ell_N|| in the dual sup-norm
    r_minus = (-m_minus) / norm_L if m_minus <= 0.0 else 0.0
    r_plus  = m_plus  / norm_L if m_plus  >  0.0 else 0.0

    width_bistable = Lu_plus - Lu_minus    # = L (I-A)^{-1} q

    return {
        "u_minus":         u_minus,
        "u_plus":          u_plus,
        "Lu_minus":        Lu_minus,
        "Lu_plus":         Lu_plus,
        "m_minus":         m_minus,
        "m_plus":          m_plus,
        "norm_L":          norm_L,
        "r_minus_basin":   r_minus,
        "r_plus_basin":    r_plus,
        "width_bistable":  width_bistable,
        "bistable":        (m_minus <= 0.0) and (m_plus > 0.0),
    }


def _classify_attractor(U: np.ndarray, branch: Dict[str, object]) -> tuple:
    """Return (label, sup_distance_to_lower, sup_distance_to_upper)."""
    d_minus = float(np.max(np.abs(U - branch["u_minus"])))
    d_plus  = float(np.max(np.abs(U - branch["u_plus"])))
    label   = "lower" if d_minus < d_plus else "upper"
    return label, d_minus, d_plus


def run_bistability_experiment(N: int = 128,
                               n_alpha: int = 41,
                               alpha_max: float = 2.0) -> Dict[str, object]:
    """
    Experiment 8 (Section 7.11):
      Empirical demonstration of Theorem 3.10 (bistable regime) and
      Corollary 3.11 (basin radii) on the separable benchmark.

    Outputs:
      - branch fixed points u^- and u^+ (computed by direct solve);
      - branch margins m^-, m^+ and basin radii r^-, r^+;
      - two reference Picard runs from x_0 = 0 (lower basin candidate)
        and x_0 = 2 * 1 (upper basin candidate);
      - sweep over the 1-parameter family x_0 = alpha * 1, alpha in
        [0, alpha_max], reporting attractor, iterations, branch crossings;
      - the global Kannan diagnostic (Theorem 4.1) confirming that the
        contractive regime is NOT applicable.
    """
    p = bistability_parameters()
    K, g_func, q_func = build_benchmark_functions(p)
    sys_ = assemble(N, K, g_func, q_func, rule="gauss")

    branch = bistability_branch_diagnostics(sys_, p["c"])

    # Two reference runs
    run_low  = picard_nystrom(sys_, p["c"], U0=np.zeros(N),
                              tol=1e-13, nmax=300)
    run_high = picard_nystrom(sys_, p["c"], U0=2.0 * np.ones(N),
                              tol=1e-13, nmax=300)
    low_attr,  low_dm,  low_dp  = _classify_attractor(run_low["U"], branch)
    high_attr, high_dm, high_dp = _classify_attractor(run_high["U"], branch)

    # Sweep over constant initial conditions x_0 = alpha * 1
    alphas = np.linspace(0.0, alpha_max, n_alpha)
    sweep_rows = []
    for a in alphas:
        U0 = float(a) * np.ones(N)
        res = picard_nystrom(sys_, p["c"], U0=U0, tol=1e-13, nmax=300)
        attr, dm, dp = _classify_attractor(res["U"], branch)
        n_cross = int(np.sum(np.abs(np.diff(res["thetas"])) > 0.5))
        sweep_rows.append({
            "alpha":             float(a),
            "iters":             int(res["iters"]),
            "attractor":         attr,
            "n_crossings":       n_cross,
            "final_diff_lower":  dm,
            "final_diff_upper":  dp,
            "final_residual":    float(res["residuals"][-1]),
        })

    # Global Kannan diagnostic (must be inadmissible: delta < 0 or lambda > 1/2)
    kannan = kannan_diagnostics(sys_, p["c"], p["R"])

    return {
        "params":            p,
        "N":                 N,
        "branch":            branch,
        "run_low": {
            "iters":            int(run_low["iters"]),
            "final_residual":   float(run_low["residuals"][-1]),
            "thresholds":       run_low["thresholds"].tolist(),
            "thetas":           run_low["thetas"].tolist(),
            "attractor":        low_attr,
            "diff_lower":       low_dm,
            "diff_upper":       low_dp,
        },
        "run_high": {
            "iters":            int(run_high["iters"]),
            "final_residual":   float(run_high["residuals"][-1]),
            "thresholds":       run_high["thresholds"].tolist(),
            "thetas":           run_high["thetas"].tolist(),
            "attractor":        high_attr,
            "diff_lower":       high_dm,
            "diff_upper":       high_dp,
        },
        "sweep":             sweep_rows,
        "global_kannan":     kannan,
        "t":                 sys_.t.tolist(),
    }


def plot_bistability(result: Dict[str, object], filename: str) -> None:
    """
    Generate fig_bistability.pdf with four panels:
      (a) coexisting branch profiles u^- and u^+
      (b) Picard orbits of ell(x_n) from x_0 = 0 and x_0 = 2*1
      (c) basin separation in the 1-parameter family x_0 = alpha * 1
      (d) iterations and crossings vs alpha
    """
    from matplotlib.gridspec import GridSpec

    p      = result["params"]
    branch = result["branch"]
    t      = np.array(result["t"])
    sweep  = result["sweep"]

    # Slightly taller figure + more horizontal/vertical breathing room
    # to host the legends placed BELOW panels (b) and (d).
    fig = plt.figure(figsize=(11.0, 8.6))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.30)

    # ---------- (a) Branch profiles ------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    u_m = branch["u_minus"]
    u_p = branch["u_plus"]
    ax_a.plot(t, u_m, color="C0", lw=1.8, label=r"$u^-(t)$ (lower branch)")
    ax_a.plot(t, u_p, color="C3", lw=1.8, label=r"$u^+(t)$ (upper branch)")
    ax_a.fill_between(t, u_m, u_p, alpha=0.10, color="gray")
    ax_a.set_xlabel(r"$t$")
    ax_a.set_ylabel(r"$u(t)$")
    ax_a.set_title(r"(a) Coexisting branch fixed points $u^\pm$")
    ax_a.legend(loc="upper left", framealpha=0.92)
    ax_a.grid(True, alpha=0.3)

    # ---------- (b) Picard orbits of ell(x_n) --------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    rl = result["run_low"]
    rh = result["run_high"]
    n_low  = len(rl["thresholds"])
    n_high = len(rh["thresholds"])
    ax_b.plot(range(n_low),  rl["thresholds"], "o-", color="C0",
              lw=1.4, ms=4, label=r"$x_0=0\;\to\;u^-$")
    ax_b.plot(range(n_high), rh["thresholds"], "s-", color="C3",
              lw=1.4, ms=4, label=r"$x_0=2\,\mathbf{1}\;\to\;u^+$")
    # Asymptotes ell(u^-) and ell(u^+) are drawn but NOT in legend
    # (they are described in the figure caption, no need to crowd the legend).
    ax_b.axhline(branch["Lu_minus"], color="C0", ls=":", lw=1.0, alpha=0.55)
    ax_b.axhline(branch["Lu_plus"],  color="C3", ls=":", lw=1.0, alpha=0.55)
    ax_b.axhline(p["c"], color="k", ls="--", lw=1.1, label=r"threshold $c$")
    ax_b.set_xlabel(r"iteration $n$")
    ax_b.set_ylabel(r"$\ell(x_n)=\int_0^1 x_n(s)\,ds$")
    ax_b.set_title(r"(b) Two Picard orbits, two basins")
    # Legend BELOW the axes box, horizontal, no overlap with data:
    ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                ncol=3, framealpha=0.95, fontsize=9, handlelength=2.2,
                columnspacing=1.2)
    ax_b.grid(True, alpha=0.3)
    ax_b.set_xlim(left=-0.5)

    # ---------- (c) Basin separation in 1-parameter family -------------
    ax_c = fig.add_subplot(gs[1, 0])
    alphas = np.array([r["alpha"] for r in sweep])
    is_lower = np.array([r["attractor"] == "lower" for r in sweep])
    final_ell = np.where(is_lower, branch["Lu_minus"], branch["Lu_plus"])
    ax_c.plot(alphas[is_lower],  final_ell[is_lower],  "o", color="C0",
              ms=2.5, mew=0.5, label=r"basin of $u^-$")
    ax_c.plot(alphas[~is_lower], final_ell[~is_lower], "s", color="C3",
              ms=2.5, mew=0.5, label=r"basin of $u^+$")
    ax_c.axhline(p["c"], color="k", ls="--", lw=1.1, label=r"threshold $c$")
    ax_c.set_xlabel(r"$\alpha$ (initial condition $x_0=\alpha\,\mathbf{1}$)")
    ax_c.set_ylabel(r"$\ell$ at attained fixed point")
    ax_c.set_title(r"(c) Basin assignment vs $\alpha$")
    ax_c.legend(loc="center right", framealpha=0.92)
    ax_c.grid(True, alpha=0.3)

    # ---------- (d) Cost & crossings vs alpha --------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    iters_arr   = np.array([r["iters"]       for r in sweep], dtype=float)
    n_cross_arr = np.array([r["n_crossings"] for r in sweep], dtype=float)

    ln1 = ax_d.plot(alphas, iters_arr, "-", color="C2", lw=1.6,
                    label=r"iterations to $10^{-13}$")
    ax_d2 = ax_d.twinx()
    ln2 = ax_d2.plot(alphas, n_cross_arr, "--", color="C1", lw=1.6,
                     label=r"# branch crossings")

    # Vertical reference lines (threshold and ell(u^pm)) drawn but NOT in legend
    ax_d.axvline(branch["Lu_minus"], color="C0", ls=":", lw=1.0, alpha=0.55)
    ax_d.axvline(branch["Lu_plus"],  color="C3", ls=":", lw=1.0, alpha=0.55)
    ax_d.axvline(p["c"],             color="k",  ls="--", lw=1.0)
    ax_d.set_xlabel(r"$\alpha$ (initial condition $x_0=\alpha\,\mathbf{1}$)")
    ax_d.set_ylabel(r"# iterations", color="C2")
    ax_d2.set_ylabel(r"# branch crossings", color="C1")
    ax_d.tick_params(axis="y", labelcolor="C2")
    ax_d2.tick_params(axis="y", labelcolor="C1")
    ax_d.set_title(r"(d) Cost and crossings along the sweep")
    ax_d.grid(True, alpha=0.3)

    # Combined legend BELOW the axes box:
    lines = ln1 + ln2
    labels = [ln.get_label() for ln in lines]
    ax_d.legend(lines, labels, loc="upper center",
                bbox_to_anchor=(0.5, -0.18), ncol=2,
                framealpha=0.95, fontsize=9, handlelength=2.2,
                columnspacing=1.5)

    fig.savefig(filename)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Filippov bifurcation experiment (Section 7.12 of the paper)
# ---------------------------------------------------------------------------
def filippov_solution_set(sys_: NystromSystem,
                          c: float) -> Dict[str, object]:
    """
    Compute the Filippov solution set F(c) at a fixed threshold c, for the
    discretised operator T_N (Proposition 3.15).

    F(c) consists of up to three points:
      - u^- = (I-A_N)^{-1} G              admissible iff L u^- <= c
      - u^+ = (I-A_N)^{-1} (G+Q)          admissible iff L u^+ >= c
      - u^d = (I-A_N)^{-1} (G + alpha* Q) admissible iff alpha* in [0,1],
        with alpha* = (c - L u^-) / (L u^+ - L u^-)
        and L u^d = c by construction.

    Returns a dictionary with the three candidate points and admissibility
    flags, plus the computed alpha* for the boundary solution.
    """
    A, G, Q, w = sys_.A, sys_.G, sys_.Q, sys_.w
    N = G.size

    M = np.eye(N) - A
    u_minus = np.linalg.solve(M, G)
    u_plus  = np.linalg.solve(M, G + Q)

    Lu_m = float(w @ u_minus)
    Lu_p = float(w @ u_plus)

    # alpha* well-defined as long as Lu_p != Lu_m (assumed L (I-A)^-1 q > 0)
    if Lu_p > Lu_m:
        alpha_star = (c - Lu_m) / (Lu_p - Lu_m)
    else:
        alpha_star = float("nan")

    # Boundary solution u^d, only defined when alpha* in [0, 1]
    if 0.0 <= alpha_star <= 1.0:
        u_boundary = np.linalg.solve(M, G + alpha_star * Q)
        Lu_b       = float(w @ u_boundary)
    else:
        u_boundary = None
        Lu_b       = None

    return {
        "u_minus":          u_minus,
        "u_plus":           u_plus,
        "u_boundary":       u_boundary,
        "Lu_minus":         Lu_m,
        "Lu_plus":          Lu_p,
        "Lu_boundary":      Lu_b,
        "alpha_star":       alpha_star,
        "lower_admissible": Lu_m <= c,
        "upper_admissible": Lu_p >= c,
        "boundary_admissible": (0.0 <= alpha_star <= 1.0),
    }


def run_filippov_bifurcation(N: int = 128,
                             n_c: int = 81,
                             c_min: float = 0.30,
                             c_max: float = 2.10) -> Dict[str, object]:
    """
    Experiment 9 (Section 7.12): Filippov solution set as a function of c.

    Sweeps the threshold c in [c_min, c_max] for the bistability benchmark
    parameters (alpha=0.5, beta=0.6, rho=0, sigma=0.6) and reports, at each
    c, the membership of u^- / u^+ / u^d in the Filippov set F(c), together
    with the Picard fixed point obtained from x_0 = 0 (lower init) and
    x_0 = 2*1 (upper init).

    Goal: empirically reproduce the regime decomposition of Proposition 3.15
    and verify that Picard never selects the boundary solution u^d.
    """
    p = bistability_parameters()   # alpha=0.5, beta=0.6, rho=0, sigma=0.6
    K, g_func, q_func = build_benchmark_functions(p)
    sys_ = assemble(N, K, g_func, q_func, rule="gauss")

    # Branch fixed points and their L-images are independent of c
    M = np.eye(N) - sys_.A
    u_minus = np.linalg.solve(M, sys_.G)
    u_plus  = np.linalg.solve(M, sys_.G + sys_.Q)
    Lu_m = float(sys_.w @ u_minus)
    Lu_p = float(sys_.w @ u_plus)

    cs = np.linspace(c_min, c_max, n_c)
    rows = []
    for c in cs:
        F = filippov_solution_set(sys_, float(c))
        # Picard runs from low and high initial conditions
        run_low  = picard_nystrom(sys_, float(c), U0=np.zeros(N),
                                  tol=1e-13, nmax=300)
        run_high = picard_nystrom(sys_, float(c), U0=2.0 * np.ones(N),
                                  tol=1e-13, nmax=300)
        # Identify the Picard limit by sup-distance to candidates
        d_low_to_minus = float(np.max(np.abs(run_low["U"]  - u_minus)))
        d_low_to_plus  = float(np.max(np.abs(run_low["U"]  - u_plus )))
        d_high_to_minus = float(np.max(np.abs(run_high["U"] - u_minus)))
        d_high_to_plus  = float(np.max(np.abs(run_high["U"] - u_plus )))
        attr_low  = "lower" if d_low_to_minus  < d_low_to_plus  else "upper"
        attr_high = "lower" if d_high_to_minus < d_high_to_plus else "upper"

        # Distance of Picard limit from the boundary solution (if defined)
        if F["u_boundary"] is not None:
            d_low_to_bnd  = float(np.max(np.abs(run_low["U"]  - F["u_boundary"])))
            d_high_to_bnd = float(np.max(np.abs(run_high["U"] - F["u_boundary"])))
        else:
            d_low_to_bnd = d_high_to_bnd = None

        # Classify regime
        if F["lower_admissible"] and F["upper_admissible"]:
            regime = "bistable"
        elif F["upper_admissible"]:
            regime = "contractive"
        elif F["lower_admissible"]:
            regime = "lower-monostable"
        else:
            regime = "none"

        rows.append({
            "c":                  float(c),
            "regime":             regime,
            "alpha_star":         F["alpha_star"],
            "Lu_boundary":        F["Lu_boundary"],
            "lower_admissible":   F["lower_admissible"],
            "upper_admissible":   F["upper_admissible"],
            "boundary_admissible":F["boundary_admissible"],
            "picard_low_attr":    attr_low,
            "picard_high_attr":   attr_high,
            "picard_low_d_to_boundary":  d_low_to_bnd,
            "picard_high_d_to_boundary": d_high_to_bnd,
            "picard_low_iters":  int(run_low["iters"]),
            "picard_high_iters": int(run_high["iters"]),
        })

    return {
        "params":  p,
        "N":       N,
        "Lu_minus": Lu_m,
        "Lu_plus":  Lu_p,
        "cs":      cs.tolist(),
        "rows":    rows,
    }


def plot_filippov_bifurcation(result: Dict[str, object],
                              filename: str) -> None:
    """
    Generate fig_filippov.pdf with two panels:
      (a) The three Filippov branches (Lu^-, Lu^d, Lu^+) as functions of c,
          showing where each branch is admissible (solid) vs not (dashed),
          with the bifurcation values c = Lu^- and c = Lu^+ marked.
      (b) Picard selection from x_0 = 0 (low init) and x_0 = 2*1 (high init)
          as functions of c: in the contractive regime both runs converge
          to u^+; in the bistable regime they split (low->u^-, high->u^+);
          in the lower-monostable regime both converge to u^-. The boundary
          solution u^d is NEVER selected.
    """
    p   = result["params"]
    Lum = result["Lu_minus"]
    Lup = result["Lu_plus"]
    cs  = np.array(result["cs"])
    rows = result["rows"]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    # ------- (a) Filippov branches -----------------------------------
    # u^- (constant in c, admissible iff Lu^- <= c, i.e. c >= Lum)
    lower_adm   = cs >= Lum
    upper_adm   = cs <= Lup    # Lu^+ >= c iff c <= Lup
    Lu_b_arr    = np.where((cs >= Lum) & (cs <= Lup), cs, np.nan)
    # Solid where admissible, dashed where not
    ax_a.plot(cs, np.full_like(cs, Lum), color="C0",
              ls=(0, (4, 3)), lw=1.2, alpha=0.5)
    ax_a.plot(cs[lower_adm], np.full_like(cs[lower_adm], Lum),
              color="C0", lw=2.2, label=r"$\ell(u^-)$ (admissible)")
    ax_a.plot(cs, np.full_like(cs, Lup), color="C3",
              ls=(0, (4, 3)), lw=1.2, alpha=0.5)
    ax_a.plot(cs[upper_adm], np.full_like(cs[upper_adm], Lup),
              color="C3", lw=2.2, label=r"$\ell(u^+)$ (admissible)")
    ax_a.plot(cs, Lu_b_arr, color="C2", lw=2.2, ls="-",
              label=r"$\ell(u^\partial)=c$ (boundary)")
    ax_a.plot(cs, cs, color="grey", ls=":", lw=0.8, alpha=0.5)  # diagonal c=c
    # Bifurcation markers
    ax_a.axvline(Lum, color="C0", ls=":", lw=1.0, alpha=0.6)
    ax_a.axvline(Lup, color="C3", ls=":", lw=1.0, alpha=0.6)
    ax_a.text(Lum, ax_a.get_ylim()[1] * 0.96, r"$c=\ell(u^-)$",
              color="C0", ha="center", va="top", fontsize=9,
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
    ax_a.text(Lup, ax_a.get_ylim()[1] * 0.96, r"$c=\ell(u^+)$",
              color="C3", ha="center", va="top", fontsize=9,
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
    ax_a.set_xlabel(r"threshold $c$")
    ax_a.set_ylabel(r"$\ell(u)$ for $u\in\mathcal{F}(c)$")
    ax_a.set_title(r"(a) Filippov solution set $\mathcal{F}(c)$")
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                ncol=3, framealpha=0.95, fontsize=9, handlelength=2.3,
                columnspacing=1.2)
    ax_a.grid(True, alpha=0.3)

    # ------- (b) Picard selection ------------------------------------
    # Plot ell(picard_limit) for each c, for both initial conditions
    sel_low  = np.array([Lum if r["picard_low_attr"]  == "lower" else Lup
                         for r in rows])
    sel_high = np.array([Lum if r["picard_high_attr"] == "lower" else Lup
                         for r in rows])
    ax_b.plot(cs, sel_low,  "o", color="C0", ms=3.0, mew=0.5,
              label=r"$x_0=0\;\to\;$Picard limit")
    ax_b.plot(cs, sel_high, "s", color="C3", ms=3.0, mew=0.5,
              label=r"$x_0=2\,\mathbf{1}\;\to\;$Picard limit")
    # Boundary solution would be here but is never selected
    ax_b.plot(cs, Lu_b_arr, color="C2", lw=1.8, ls="--", alpha=0.7,
              label=r"$\ell(u^\partial)$ (never selected)")
    ax_b.axvline(Lum, color="C0", ls=":", lw=1.0, alpha=0.6)
    ax_b.axvline(Lup, color="C3", ls=":", lw=1.0, alpha=0.6)
    ax_b.set_xlabel(r"threshold $c$")
    ax_b.set_ylabel(r"$\ell$ at Picard limit")
    ax_b.set_title(r"(b) Picard selects no $u^\partial$")
    ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                ncol=3, framealpha=0.95, fontsize=9, handlelength=2.3,
                columnspacing=1.2)
    ax_b.grid(True, alpha=0.3)

    fig.savefig(filename)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Applied case study II (Section 7.13): relay control with integral observation
# Plant: y(t) = alpha int_0^1 K(t,s) y(s) ds + g(t) + q(t) H(<y> - c)
# with K(t,s) = exp(-2|t-s|), g(t) = beta cos(pi t / 2), q(t) = -sigma t
# (negative jump = "extraction" relay activating when integrated output exceeds
# setpoint c). Standard formulation in relay control with integral observation
# (Tsypkin 1984; Goebel--Sanfelice--Teel 2012; Utkin 1992).
# ---------------------------------------------------------------------------
def relay_control_parameters() -> Dict[str, float]:
    """
    Reference parameters of the relay control benchmark.

    Choice rationale (alpha=0.08, beta=1.5, sigma=0.10, c=0.5, R=2):
      kappa  = alpha (1 - exp(-1))           ~ 0.0506
      mu_K   = alpha [1 - 0.5 (1 - exp(-2))] ~ 0.0454
      Lg     = 2 beta / pi                    ~ 0.955
      Lq     = -sigma / 2                     = -0.050
      ||q||_inf = sigma                       = 0.100
      m_R    = Lg - c - mu_K R                ~ 0.364
      gamma_R = ||q||/m_R                     ~ 0.275
      smallness 3 kappa + 2 gamma_R           ~ 0.701 < 1
      lambda_R = (kappa + gamma_R)/(1-kappa)  ~ 0.343 < 1/2
      single-step margin Lq + m_R             ~ +0.314 > 0
    The relay activates (extraction effective) when ell(y) > c, i.e., the plant
    output exceeds the setpoint; the equilibrium settles in the upper branch
    (relay active) for the nominal data.
    """
    return dict(alpha=0.08, beta=1.5, sigma=0.10, c=0.5, R=2.0)


def build_relay_control_functions(p: Dict[str, float]):
    """
    Returns (K, g, q) for the relay-control benchmark with:
        K(t,s) = alpha * exp(-2|t-s|)   (plant coupling, range ~ 1/2)
        g(t)   = beta * cos(pi t / 2)   (reference signal, decays on [0,1])
        q(t)   = -sigma * t             (extraction profile; negative jump
                                         corresponds to actuation REMOVING
                                         output mass when relay is active).
    """
    alpha, beta, sigma = p["alpha"], p["beta"], p["sigma"]
    K = lambda t, s: alpha * np.exp(-2.0 * np.abs(t - s))
    g = lambda t: beta * np.cos(0.5 * np.pi * t)
    q = lambda t: -sigma * t
    return K, g, q


def relay_control_continuous_diagnostics(p: Dict[str, float]) -> Dict[str, float]:
    """
    Analytical continuous Kannan diagnostics for the relay-control benchmark.

    All identities are derived in closed form from the explicit kernel and
    forcings: kappa = alpha (1 - exp(-1)), mu_K = alpha (1 - (1 - exp(-2))/2),
    Lg = 2 beta / pi, Lq = -sigma/2, ||q||_inf = sigma.
    """
    alpha, beta, sigma = p["alpha"], p["beta"], p["sigma"]
    c, R = p["c"], p["R"]
    kappa  = alpha * (1.0 - np.exp(-1.0))
    mu_K   = alpha * (1.0 - 0.5 * (1.0 - np.exp(-2.0)))
    Lg     = 2.0 * beta / np.pi
    Lq     = -0.5 * sigma
    norm_q = sigma
    norm_g = abs(beta)
    delta_R    = Lg - c - mu_K * R
    inv_lhs    = kappa * R + norm_g + norm_q
    if delta_R > 0 and kappa < 1:
        kannan_lhs = 3.0 * kappa + 2.0 * norm_q / delta_R
        lambda_R   = (kappa + norm_q / delta_R) / (1.0 - kappa)
    else:
        kannan_lhs = float("inf")
        lambda_R   = float("inf")
    return dict(
        kappa=kappa, mu_K=mu_K, Lg=Lg, Lq=Lq,
        norm_g=norm_g, norm_q=norm_q,
        delta_R=delta_R,
        invariance_lhs=inv_lhs, invariance_rhs=R,
        kannan_lhs=kannan_lhs, lambda_R=lambda_R,
        single_step_margin=Lq + delta_R,
        admissible=(inv_lhs <= R and delta_R > 0
                    and kannan_lhs < 1 and lambda_R < 0.5),
    )


def run_relay_control_application(N: int = 128) -> Dict[str, object]:
    """
    Solve the relay-control benchmark with the nominal parameters of
    relay_control_parameters().
    """
    p = relay_control_parameters()
    K, g, q = build_relay_control_functions(p)
    sys_ = assemble(N, K, g, q, rule="gauss")
    diag_cont = relay_control_continuous_diagnostics(p)
    diag_disc = kannan_diagnostics(sys_, p["c"], p["R"])

    # Run Picard from y_0 = 0 (relay initially OFF since ell(0) = 0 < c)
    out = picard_nystrom(sys_, p["c"], U0=np.zeros(N), tol=1e-13, nmax=300)

    # Open-loop equilibrium (no relay): y_open = (I - A)^{-1} g
    Imat = np.eye(N)
    U_open = np.linalg.solve(Imat - sys_.A, sys_.G)
    ell_open = float(sys_.w @ U_open)

    ell_star = float(sys_.w @ out["U"])
    return {
        "params":            p,
        "N":                 N,
        "diagnostics_cont":  diag_cont,
        "diagnostics_disc":  diag_disc,
        "iterations":        out["iters"],
        "final_residual":    float(out["residuals"][-1]),
        "y_star":            out["U"],
        "y_open_loop":       U_open,
        "ell_star":          ell_star,
        "ell_open_loop":     ell_open,
        "regulation_drop":   ell_open - ell_star,
        "active_margin":     ell_star - p["c"],
        "t_nodes":           sys_.t,
        "residual_history":  out["residuals"].tolist(),
        "threshold_history": out["thresholds"].tolist(),
    }


def run_relay_setpoint_sweep(c_list, N: int = 128) -> Dict[str, object]:
    """
    Sweep the setpoint c and record equilibrium ell(y_*), iteration count,
    residual, and Kannan admissibility.
    """
    p_ref = relay_control_parameters()
    K, g, q = build_relay_control_functions(p_ref)
    sys_ = assemble(N, K, g, q, rule="gauss")

    # Open-loop equilibrium (independent of c)
    Imat = np.eye(N)
    U_open = np.linalg.solve(Imat - sys_.A, sys_.G)
    ell_open = float(sys_.w @ U_open)
    Imat_full = np.eye(N)
    U_closed = np.linalg.solve(Imat_full - sys_.A, sys_.G + sys_.Q)
    ell_closed = float(sys_.w @ U_closed)  # always-active equilibrium

    rows = []
    for c in c_list:
        diag = kannan_diagnostics(sys_, float(c), p_ref["R"])
        if diag["admissible"]:
            out = picard_nystrom(sys_, float(c), U0=np.zeros(N),
                                  tol=1e-13, nmax=300)
            ell_star = float(sys_.w @ out["U"])
            n_cross  = int(np.sum(np.abs(np.diff(out["thetas"])) > 0.5))
            theta_eq = float(out["thetas"][-1])
            rows.append({
                "c": float(c),
                "lambda_N": diag["lambda_N"],
                "delta_NR": diag["delta"],
                "admissible": True,
                "iters": int(out["iters"]),
                "final_residual": float(out["residuals"][-1]),
                "ell_star": ell_star,
                "theta_eq": theta_eq,
                "n_crossings": n_cross,
            })
        else:
            rows.append({
                "c": float(c),
                "lambda_N": diag["lambda_N"],
                "delta_NR": diag["delta"],
                "admissible": False,
                "iters": None,
                "final_residual": None,
                "ell_star": None,
                "theta_eq": None,
                "n_crossings": None,
            })
    return {
        "params":     p_ref,
        "N":          N,
        "ell_open":   ell_open,    # ell(y) when relay always OFF
        "ell_closed": ell_closed,  # ell(y) when relay always ON
        "rows":       rows,
    }


def plot_relay_control(result, sweep, filename: str) -> None:
    """
    Two-panel publication figure for Section 7.13:
      Left:  equilibrium output y_*(t) (relay closed-loop) vs open-loop
             reference y_open(t); g(t) and q(t) profiles for context.
      Right: setpoint sweep ell(y_*) vs c, with the open-loop and closed-loop
             saturation horizontals and the Kannan-admissible range marked.
    """
    p = result["params"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    # Left: profiles in space
    ax_L = axes[0]
    t  = result["t_nodes"]
    y_star = result["y_star"]
    y_open = result["y_open_loop"]
    g_t = p["beta"] * np.cos(0.5 * np.pi * t)
    h_t = p["sigma"] * t                       # extraction profile (positive sign)
    ax_L.plot(t, y_star,  color="C2", lw=2.0,
              label=r"$y_\ast(t)$ (closed-loop, relay active)")
    ax_L.plot(t, y_open,  color="0.45", ls="--", lw=1.4,
              label=r"$y_{\rm open}(t)$ (open-loop, no relay)")
    ax_L.plot(t, g_t,     color="C0", ls=":", lw=1.2,
              label=r"$g(t) = \beta\cos(\pi t/2)$ (reference)")
    ax_L.plot(t, h_t,     color="C3", ls="-.", lw=1.2,
              label=r"$h(t) = \sigma\,t$ (extraction profile)")
    ax_L.axhline(p["c"], color="k", ls="--", lw=1.0, alpha=0.5)
    ax_L.text(0.02, p["c"] + 0.03, rf"$c={p['c']}$",
              color="k", fontsize=9)
    ax_L.set_xlabel(r"$t$")
    ax_L.set_ylabel(r"output / forcing")
    ax_L.set_title("Equilibrium output and forcings")
    ax_L.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    ax_L.grid(True, alpha=0.3)

    # Right: setpoint sweep
    ax_R = axes[1]
    cs = np.array([r["c"] for r in sweep["rows"]])
    adm = np.array([r["admissible"] for r in sweep["rows"]])
    ell_arr = np.array([r["ell_star"] if r["ell_star"] is not None else np.nan
                        for r in sweep["rows"]])
    ax_R.plot(cs[adm], ell_arr[adm], "o-", color="C2", ms=4, lw=1.5,
              label=r"$\ell_N(y_\ast)$ at equilibrium")
    ax_R.axhline(sweep["ell_open"],   color="0.45", ls="--", lw=1.2,
                 label=rf"$\ell_N(y_{{\rm open}})={sweep['ell_open']:.3f}$"
                       r" (relay never on)")
    ax_R.axhline(sweep["ell_closed"], color="C3",   ls="-.", lw=1.2,
                 label=rf"$\ell_N(y_{{\rm closed}})={sweep['ell_closed']:.3f}$"
                       r" (relay always on)")
    ax_R.plot(cs, cs, color="k", ls=":", lw=0.9, alpha=0.6,
              label=r"$\ell=c$ (diagonal)")
    ax_R.set_xlabel(r"setpoint $c$")
    ax_R.set_ylabel(r"$\ell_N(y_\ast)$")
    ax_R.set_title("Setpoint sweep")
    ax_R.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
    ax_R.grid(True, alpha=0.3)

    fig.savefig(filename)
    plt.close(fig)


if __name__ == "__main__":
    import json
    integral_summary = run_full_experiment(N=64, output_dir=".")
    bvp_summary      = run_bvp_experiment(N=64, output_dir=".")
    nonsep_summary   = run_nonseparable_experiment(
        Ns=(32, 64, 128, 256), output_dir=".")
    stress_summary   = run_stress_sweep(
        sigmas=(0.010, 0.030, 0.050, 0.058, 0.060, 0.065), N=128)
    quad_summary     = run_quadrature_comparison(Ns=(9, 17, 33, 65, 129))

    # Generate the non-separable figure
    plot_nonseparable(nonsep_summary, "./fig_nonseparable.pdf")

    print("=" * 70)
    print("Threshold Fredholm integral equation (Sections 7.3--7.5)")
    print("=" * 70)
    print(json.dumps(integral_summary, indent=2, default=str))

    print()
    print("=" * 70)
    print("Threshold boundary value problem (Section 7.6)")
    print("=" * 70)
    compact = {k: v for k, v in bvp_summary.items() if k != "history"}
    print(json.dumps(compact, indent=2, default=str))

    print()
    print("=" * 70)
    print("Non-separable exponential kernel benchmark (Section 7.7)")
    print("=" * 70)
    compact_ns = {k: v for k, v in nonsep_summary.items()
                  if k not in ("t_ref", "u_extensions")}
    print(json.dumps(compact_ns, indent=2, default=str))

    print()
    print("=" * 70)
    print("Stress sweep in sigma (Section 7.7, continued)")
    print("=" * 70)
    print(json.dumps(stress_summary, indent=2, default=str))

    print()
    print("=" * 70)
    print("Quadrature consistency comparison (Section 7.7, continued)")
    print("=" * 70)
    for rule, rows in quad_summary["table"].items():
        print(f"\n-- {rule.upper()} --")
        print(f'{"N":>4}  {"|kappa err|":>12}  {"|mu err|":>12}  '
              f'{"|ellG err|":>12}  {"|lambda err|":>13}  {"iters":>6}  {"final_res":>11}')
        for r in rows:
            print(f'{r["N"]:>4}  {r["kappa_err"]:>12.3e}  {r["mu_err"]:>12.3e}  '
                  f'{r["ellG_err"]:>12.3e}  {r["lambda_err"]:>13.3e}  '
                  f'{r["iters"]:>6d}  {r["final_res"]:>11.3e}')
    print("\n-- Observed convergence orders (from the two finest grids) --")
    print(f'{"rule":>13}  {"kappa":>8}  {"mu":>8}  {"ellG":>8}  {"lambda":>8}')
    for rule, o in quad_summary["observed_orders"].items():
        print(f'{rule:>13}  {o["kappa_order"]:>8.3f}  {o["mu_order"]:>8.3f}  '
              f'{o["ellG_order"]:>8.3f}  {o["lambda_order"]:>8.3f}')
