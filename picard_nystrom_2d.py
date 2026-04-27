"""
picard_nystrom_2d.py
====================
Two-dimensional extension of the Picard--Nyström framework for threshold
Fredholm integral equations on [0,1]^2.

Continuous setting:
    Au(x,y) = int_0^1 int_0^1 K(x,y; xi, eta) u(xi, eta) d(xi) d(eta)
    Lu      = int_0^1 int_0^1 u(xi, eta) d(xi) d(eta)
    Tu      = Au + g + q * H(Lu - c)

Discretisation: tensor-product Gauss--Legendre quadrature on [0,1]^2 with
N nodes per direction, giving an N^2 x N^2 linear-algebraic system. The
flat-index convention is k = i*N + j  <->  node (t_i, t_j).

This module REUSES the dataclass NystromSystem, picard_nystrom and
kannan_diagnostics from picard_nystrom.py without modification: the 2D
problem is encoded as an N^2 x N^2 system fitting the existing API.

Two benchmarks are provided:
  A. Separable bilinear kernel with closed-form fixed point;
  B. Non-separable exponential kernel (cusp at x=xi and y=eta), validated
     by mesh-refinement self-consistency.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

from picard_nystrom import (
    NystromSystem,
    gauss_legendre_01,
    kannan_diagnostics,
    picard_nystrom,
)


# ---------------------------------------------------------------------------
# Tensor-product Gauss--Legendre nodes
# ---------------------------------------------------------------------------
def tensor_product_nodes_2d(N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (t_x, t_y, w_2, t_1d, w_1d) where:
      t_x, t_y  : flat arrays of length N^2 with the (x, y) coordinates of
                  the tensor-product Gauss--Legendre nodes on [0,1]^2;
      w_2       : flat array of length N^2 of tensor-product weights w_i w_j;
      t_1d, w_1d: 1D Gauss--Legendre nodes/weights on [0,1] (length N).
    Index convention: flat index k = i*N + j corresponds to (t_1d[i], t_1d[j]).
    """
    t, w = gauss_legendre_01(N)
    T_x, T_y = np.meshgrid(t, t, indexing="ij")
    W = np.outer(w, w)
    return T_x.ravel(), T_y.ravel(), W.ravel(), t, w


# ---------------------------------------------------------------------------
# 2D Nyström assembly
# ---------------------------------------------------------------------------
def assemble_2d(N: int,
                K: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
                g: Callable[[np.ndarray, np.ndarray], np.ndarray],
                q: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> NystromSystem:
    """
    Assemble the 2D Nyström system for tensor-product Gauss--Legendre nodes.
    The result is encoded as a NystromSystem (compatible with picard_nystrom
    and kannan_diagnostics) with the FLAT representation of dimension N^2:
      A[i*N+j, k*N+l] = w_k * w_l * K(t_i, t_j, t_k, t_l)
      G[i*N+j]        = g(t_i, t_j)
      Q[i*N+j]        = q(t_i, t_j)
      w               = tensor-product weights w_i * w_j (length N^2)
      t               = x-coordinates of flat nodes (length N^2)

    The downstream Picard iteration U^{n+1} = A U^n + G + theta Q with
    theta = H(w . U^n - c) is identical in form to the 1D scheme.

    Vectorised: no Python-level loops; uses broadcasting in K(...).
    """
    t_x, t_y, w_2, _, _ = tensor_product_nodes_2d(N)

    # K_mat[i_flat, k_flat] = K(t_x[i], t_y[i], t_x[k], t_y[k]),
    # via (N^2, 1) x (1, N^2) broadcasting; shape (N^2, N^2).
    K_mat = K(
        t_x[:, np.newaxis],
        t_y[:, np.newaxis],
        t_x[np.newaxis, :],
        t_y[np.newaxis, :],
    )
    # Apply column weights w_k * w_l (i.e. w_2[k]):
    A_mat = K_mat * w_2[np.newaxis, :]

    G_vec = g(t_x, t_y)
    Q_vec = q(t_x, t_y)

    return NystromSystem(t=t_x, w=w_2, A=A_mat, G=G_vec, Q=Q_vec)


# ---------------------------------------------------------------------------
# Benchmark A: separable bilinear (analytical fixed point)
# ---------------------------------------------------------------------------
def benchmark_separable_2d_parameters() -> Dict[str, float]:
    """
    Reference parameters for the separable bilinear 2D benchmark
        K(x,y; xi, eta) = (alpha/2) (x*xi + y*eta),
        g(x,y) = beta,
        q(x,y) = sigma.

    Choice rationale (alpha=0.3, beta=0.5, sigma=0.02, c=0.2, R=2):
      kappa_2 = alpha/2 = 0.15
      mu_K_2  = alpha/4 = 0.075
      m_R_2   = beta - c - mu_K_2 * R = 0.15
      gamma_R = sigma / m_R_2 = 0.13333...
      Smallness 3*kappa_2 + 2*gamma_R = 0.71666... < 1
      lambda_R = (kappa_2 + gamma_R) / (1 - kappa_2) = 0.33333... < 1/2

    Closed-form fixed point u^+(x,y) = (beta + sigma) + b (x + y) with
        b = 6*alpha*(beta+sigma) / (24 - 7*alpha) ~ 0.04192 (for alpha=0.3).
    Gauss--Legendre is exact for polynomial integrands of degree <= 2N-1
    in each variable, so the discrete fixed point matches the analytical
    one to machine precision for every N >= 1.
    """
    return dict(alpha=0.3, beta=0.5, sigma=0.02, c=0.2, R=2.0)


def build_separable_2d_functions(p: Dict[str, float]):
    """Return (K, g, q) for the separable bilinear 2D benchmark."""
    alpha, beta, sigma = p["alpha"], p["beta"], p["sigma"]
    K = lambda x, y, xi, eta: 0.5 * alpha * (x * xi + y * eta)
    g = lambda x, y: beta * np.ones_like(x)
    q = lambda x, y: sigma * np.ones_like(x)
    return K, g, q


def exact_separable_2d(t_x: np.ndarray, t_y: np.ndarray,
                       p: Dict[str, float]) -> np.ndarray:
    """
    Closed-form upper-branch fixed point for the separable bilinear benchmark:
        u^+(x,y) = (beta + sigma) + b (x + y),
        b = 6*alpha*(beta+sigma) / (24 - 7*alpha).
    """
    alpha, beta, sigma = p["alpha"], p["beta"], p["sigma"]
    a = beta + sigma
    b = 6.0 * alpha * a / (24.0 - 7.0 * alpha)
    return a + b * (t_x + t_y)


def separable_2d_continuous_diagnostics(p: Dict[str, float]) -> Dict[str, float]:
    """
    Analytical continuous Kannan diagnostics for the separable bilinear
    2D benchmark.
    """
    alpha, beta, sigma = p["alpha"], p["beta"], p["sigma"]
    c, R = p["c"], p["R"]
    kappa_2 = alpha / 2.0
    mu_K_2  = alpha / 4.0
    Lg      = beta
    Lq      = sigma
    delta   = Lg - c - mu_K_2 * R
    if delta <= 0.0 or kappa_2 >= 1.0:
        gamma = float("inf")
        lam   = float("inf")
    else:
        gamma = sigma / delta
        lam   = (kappa_2 + gamma) / (1.0 - kappa_2)
    return {
        "kappa_2":       kappa_2,
        "mu_K_2":        mu_K_2,
        "Lg":            Lg,
        "Lq":            Lq,
        "delta_R_2":     delta,
        "gamma_R_2":     gamma,
        "lambda_R_2":    lam,
        "smallness_lhs": 3.0 * kappa_2 + 2.0 * gamma,
        "admissible":    (3.0 * kappa_2 + 2.0 * gamma < 1.0) and (delta > 0.0),
    }


# ---------------------------------------------------------------------------
# Benchmark B: non-separable exponential (no closed-form FP)
# ---------------------------------------------------------------------------
def benchmark_nonseparable_2d_parameters() -> Dict[str, float]:
    """
    Reference parameters for the non-separable 2D exponential benchmark
        K(x,y; xi, eta) = alpha * exp(-|x-xi|) * exp(-|y-eta|),
        g(x,y) = beta * cos(pi x/2) * cos(pi y/2),
        q(x,y) = sigma * (1 + x) * (1 + y) / 4.
    The kernel is non-smooth on {x=xi} U {y=eta}, so the Nyström
    convergence rate is O(N^{-2}) (2D analogue of Section 7.7).
    """
    return dict(alpha=0.10, beta=1.50, sigma=0.05, c=0.30, R=2.0)


def build_nonseparable_2d_functions(p: Dict[str, float]):
    """Return (K, g, q) for the non-separable 2D exponential benchmark."""
    alpha, beta, sigma = p["alpha"], p["beta"], p["sigma"]
    K = lambda x, y, xi, eta: alpha * np.exp(-np.abs(x - xi)) * np.exp(-np.abs(y - eta))
    g = lambda x, y: beta * np.cos(np.pi * x / 2.0) * np.cos(np.pi * y / 2.0)
    q = lambda x, y: sigma * (1.0 + x) * (1.0 + y) / 4.0
    return K, g, q


# ---------------------------------------------------------------------------
# Nyström extension to a reference grid (for cross-resolution comparison)
# ---------------------------------------------------------------------------
def nystrom_extension_2d(x_ref: np.ndarray, y_ref: np.ndarray,
                         t_1d: np.ndarray, w_1d: np.ndarray,
                         U_grid: np.ndarray,
                         K: Callable, g: Callable, q: Callable,
                         c: float) -> np.ndarray:
    """
    Compute the Nyström extension of a discrete fixed point to a reference
    grid (x_ref, y_ref). The extension is

        u_N(x, y) = sum_{k,l} w_k w_l K(x, y, t_k, t_l) U[k, l]
                  + g(x, y) + theta * q(x, y),

    where theta = H(w_2 . U_flat - c) is the discrete threshold indicator
    of the iterate (computed once from the flat representation).

    Inputs:
      x_ref, y_ref : 1D arrays of length n_ref defining a tensor-product grid
      t_1d, w_1d   : 1D Gauss--Legendre nodes/weights of the discretisation
      U_grid       : the discrete fixed point on the (N, N) grid
      K, g, q      : the operator data
      c            : threshold

    Output: u_ext of shape (n_ref, n_ref).
    """
    Xr, Yr = np.meshgrid(x_ref, y_ref, indexing="ij")
    T_k, T_l = np.meshgrid(t_1d, t_1d, indexing="ij")
    WW       = np.outer(w_1d, w_1d)

    # K_4d shape (n_ref, n_ref, N, N): K(x, y, t_k, t_l)
    K_4d = K(Xr[:, :, None, None], Yr[:, :, None, None],
             T_k[None, None, :, :], T_l[None, None, :, :])
    # Apply tensor weights and the iterate
    Au_ext = np.einsum("ijkl,kl->ij", K_4d, WW * U_grid)

    # Threshold indicator from the iterate
    ell_U = float(np.sum(WW * U_grid))
    theta = 1.0 if ell_U > c else 0.0

    return Au_ext + g(Xr, Yr) + theta * q(Xr, Yr)


# ---------------------------------------------------------------------------
# Driver experiments
# ---------------------------------------------------------------------------
def run_separable_2d_experiment(N: int = 32) -> Dict[str, object]:
    """
    Experiment 10: validation of the 2D Picard--Nyström on the separable
    bilinear benchmark with closed-form fixed point.
    """
    p = benchmark_separable_2d_parameters()
    K, g, q = build_separable_2d_functions(p)
    sys_ = assemble_2d(N, K, g, q)

    diag_cont = separable_2d_continuous_diagnostics(p)
    diag_disc = kannan_diagnostics(sys_, p["c"], p["R"])

    out = picard_nystrom(sys_, p["c"], U0=np.zeros(N * N),
                         tol=1e-13, nmax=200)

    # Analytical fixed point at flat nodes
    t_1d, _ = gauss_legendre_01(N)
    T_x, T_y = np.meshgrid(t_1d, t_1d, indexing="ij")
    u_exact = exact_separable_2d(T_x.ravel(), T_y.ravel(), p)

    err = float(np.max(np.abs(out["U"] - u_exact)))
    Lu_disc = float(sys_.w @ out["U"])
    Lu_anal = float(sys_.w @ u_exact)

    return {
        "params":           p,
        "N":                N,
        "diag_cont":        diag_cont,
        "diag_disc":        diag_disc,
        "iterations":       int(out["iters"]),
        "final_residual":   float(out["residuals"][-1]),
        "sup_error":        err,
        "ell_disc":         Lu_disc,
        "ell_analytical":   Lu_anal,
        "U":                out["U"],
        "U_exact":          u_exact,
        "t_1d":             t_1d,
    }


def run_nonseparable_2d_experiment(Ns: tuple = (8, 16, 32, 64),
                                   n_ref: int = 41) -> Dict[str, object]:
    """
    Experiment 11: mesh-refinement self-consistency on the non-separable 2D
    exponential benchmark. For each N in Ns the Picard--Nyström is solved
    and its Nyström extension to a common (n_ref x n_ref) reference grid
    is computed; cross-resolution sup-norm differences are reported.
    """
    p = benchmark_nonseparable_2d_parameters()
    K, g, q = build_nonseparable_2d_functions(p)
    x_ref = np.linspace(0.0, 1.0, n_ref)

    res = []
    extensions = {}
    for N in Ns:
        sys_ = assemble_2d(N, K, g, q)
        diag = kannan_diagnostics(sys_, p["c"], p["R"])
        out = picard_nystrom(sys_, p["c"], U0=np.zeros(N * N),
                             tol=1e-13, nmax=300)

        t_1d, w_1d = gauss_legendre_01(N)
        U_grid = out["U"].reshape(N, N)
        u_ext = nystrom_extension_2d(x_ref, x_ref, t_1d, w_1d, U_grid,
                                     K, g, q, p["c"])
        extensions[N] = u_ext

        res.append({
            "N":              N,
            "kappa_N":        diag["kappa_N"],
            "mu_N":           diag["mu_N"],
            "ell_G":          diag["ell_G"],
            "delta":          diag["delta"],
            "lambda_N":       diag["lambda_N"],
            "iters":          int(out["iters"]),
            "final_residual": float(out["residuals"][-1]),
            "ell_U":          float(sys_.w @ out["U"]),
        })

    Ns_sorted = sorted(extensions.keys())
    refinement = []
    for i in range(1, len(Ns_sorted)):
        Nc = Ns_sorted[i-1]
        Nf = Ns_sorted[i]
        diff = float(np.max(np.abs(extensions[Nf] - extensions[Nc])))
        refinement.append({"N_coarse": Nc, "N_fine": Nf, "sup_diff": diff})

    return {
        "params":          p,
        "resolutions":     res,
        "extensions":      extensions,
        "x_ref":           x_ref,
        "mesh_refinement": refinement,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_2d_overview(separable_result: Dict[str, object],
                     nonsep_result:    Dict[str, object],
                     filename: str) -> None:
    """
    Three-panel publication figure for Section 8.4:
      (a) Heatmap of the analytical fixed point u^+(x,y) of the separable
          benchmark (Experiment 10).
      (b) Heatmap of the discrete fixed point u_N at N=64 of the non-
          separable benchmark (Experiment 11).
      (c) Mesh-refinement log-log plot for the non-separable benchmark
          showing the empirical O(N^{-2}) decay.
    """
    p_sep = separable_result["params"]
    t_1d  = separable_result["t_1d"]
    N_sep = separable_result["N"]
    T_x, T_y = np.meshgrid(t_1d, t_1d, indexing="ij")
    u_sep = exact_separable_2d(T_x.ravel(), T_y.ravel(), p_sep).reshape(N_sep, N_sep)

    nonsep_extensions = nonsep_result["extensions"]
    Nmax = max(nonsep_extensions.keys())
    u_nonsep = nonsep_extensions[Nmax]
    refinement = nonsep_result["mesh_refinement"]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

    # (a) Separable analytical fixed point
    ax_a = axes[0]
    im_a = ax_a.imshow(u_sep, extent=[0, 1, 0, 1], origin="lower",
                       cmap="viridis", aspect="auto")
    cbar_a = plt.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
    cbar_a.set_label(r"$u^+(x,y)$")
    ax_a.set_xlabel(r"$x$")
    ax_a.set_ylabel(r"$y$")
    ax_a.set_title(rf"(a) Separable benchmark: analytical $u^+$ ($N={N_sep}$)")

    # (b) Non-separable discrete fixed point at finest N
    ax_b = axes[1]
    im_b = ax_b.imshow(u_nonsep, extent=[0, 1, 0, 1], origin="lower",
                       cmap="viridis", aspect="auto")
    cbar_b = plt.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
    cbar_b.set_label(r"$u_N(x,y)$")
    ax_b.set_xlabel(r"$x$")
    ax_b.set_ylabel(r"$y$")
    ax_b.set_title(rf"(b) Non-separable benchmark ($N={Nmax}$)")

    # (c) Mesh refinement
    ax_c = axes[2]
    Nfines = np.array([r["N_fine"]   for r in refinement])
    diffs  = np.array([r["sup_diff"] for r in refinement])
    ax_c.loglog(Nfines, diffs, "o-", color="C0", lw=1.8, ms=7,
                label=r"$\|u_{2N}-u_N\|_\infty$")
    if len(Nfines) > 0 and diffs[0] > 0:
        ref = diffs[0] * (Nfines[0] / Nfines)**2
        ax_c.loglog(Nfines, ref, "--", color="C3", lw=1.0, alpha=0.75,
                    label=r"$\mathcal{O}(N^{-2})$ slope")
    ax_c.set_xlabel(r"$N$ (finer of pair)")
    ax_c.set_ylabel(r"sup-norm difference (reference grid)")
    ax_c.set_title(r"(c) Mesh refinement, non-separable benchmark")
    ax_c.grid(True, which="both", alpha=0.3)
    ax_c.legend(loc="upper right")

    fig.savefig(filename)
    plt.close(fig)
