"""
run_scaling_study.py
====================
Systematic scaling study for the Picard-Nystroem iteration on the
contractive Kannan benchmark of Section 7.2. For each
N in {32, 64, 128, 256, 512, 1024, 2048, 4096} we measure:

  (a) Wall-clock time for matrix assembly,
  (b) Wall-clock time for the full Picard iteration,
  (c) Wall-clock time for the full smoothed-Newton iteration,
  (d) Iteration counts for both methods,
  (e) Sup-norm condition number cond_inf(I - A_N).

The output is a 2x2 publication-quality figure fig_scaling.pdf and a
JSON summary suitable for the manuscript table. The empirical claims
verified are:

  - Picard iteration count is essentially constant in N (Theorem 6.X);
  - per-iteration cost scales as O(N^2);
  - cond_inf(I - A_N) remains bounded as N grows, in line with the
    operator-norm convergence A_N -> A and the invertibility of I - A.

Usage:
    python3 run_scaling_study.py
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from picard_nystrom import (
    benchmark_parameters,
    build_benchmark_functions,
    assemble,
    picard_nystrom,
    smoothed_newton,
)


# ---------- timing utilities -------------------------------------------------
def _time_block(callable_, nrep: int) -> float:
    """Mean wall-clock seconds over nrep repetitions."""
    t0 = time.perf_counter()
    for _ in range(nrep):
        callable_()
    return (time.perf_counter() - t0) / nrep


def measure_one(N: int, p: dict) -> dict:
    K, g, q = build_benchmark_functions(p)
    U0 = np.full(N, p["c"] / 2.0)

    # Sample sizes adapted to N to keep total runtime bounded.
    if N <= 256:
        nrep_assembly, nrep_iter = 30, 200
    elif N <= 1024:
        nrep_assembly, nrep_iter = 10, 50
    else:
        nrep_assembly, nrep_iter = 3, 10

    # Assembly time.
    t_assembly = _time_block(
        lambda: assemble(N, K, g, q, rule="gauss"), nrep_assembly
    )

    # Build the system once for the iteration timings.
    sys_ = assemble(N, K, g, q, rule="gauss")

    # Warm-up.
    picard_nystrom(sys_, p["c"], U0=U0, tol=1e-13, nmax=200)
    smoothed_newton(sys_, p["c"], U0=U0,
                    eps_init=1e-1, eps_min=1e-14, eps_factor=0.1,
                    tol=1e-13, nmax_inner=20, nmax_outer=20)

    # Picard total time.
    t_pic = _time_block(
        lambda: picard_nystrom(sys_, p["c"], U0=U0, tol=1e-13, nmax=200),
        nrep_iter,
    )

    # Newton total time.
    nrep_newton = max(nrep_iter // 4, 3)
    t_nw = _time_block(
        lambda: smoothed_newton(sys_, p["c"], U0=U0,
                                eps_init=1e-1, eps_min=1e-14,
                                eps_factor=0.1, tol=1e-13,
                                nmax_inner=20, nmax_outer=20),
        nrep_newton,
    )

    # Iteration counts (single calls, deterministic).
    pic = picard_nystrom(sys_, p["c"], U0=U0, tol=1e-13, nmax=200)
    nw  = smoothed_newton(sys_, p["c"], U0=U0,
                          eps_init=1e-1, eps_min=1e-14, eps_factor=0.1,
                          tol=1e-13, nmax_inner=20, nmax_outer=20)

    # Sup-norm condition number of M = I - A_N.
    M = np.eye(N) - sys_.A
    cond_inf = float(np.linalg.cond(M, p=np.inf))

    return {
        "N":               N,
        "assembly_s":      t_assembly,
        "picard_total_s":  t_pic,
        "newton_total_s":  t_nw,
        "picard_iters":    int(pic["iters"]),
        "newton_iters":    int(nw["total_iterations"]),
        "picard_per_iter_s": t_pic / max(pic["iters"], 1),
        "cond_inf":        cond_inf,
    }


# ---------- plotting ---------------------------------------------------------
def plot_scaling(rows: list[dict], filename: str = "./fig_scaling.pdf") -> None:
    Ns      = np.array([r["N"] for r in rows], dtype=float)
    asm_s   = np.array([r["assembly_s"] for r in rows]) * 1e3   # ms
    pic_s   = np.array([r["picard_total_s"] for r in rows]) * 1e3
    nw_s    = np.array([r["newton_total_s"] for r in rows]) * 1e3
    pic_pi  = np.array([r["picard_per_iter_s"] for r in rows]) * 1e3
    pic_it  = np.array([r["picard_iters"] for r in rows], dtype=float)
    nw_it   = np.array([r["newton_iters"] for r in rows], dtype=float)
    cond    = np.array([r["cond_inf"] for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)

    # (a) Assembly time vs N (log-log) with O(N^2) reference line
    ax = axes[0, 0]
    ax.loglog(Ns, asm_s, "o-", color="C0", lw=1.6, ms=5, label="assembly")
    Ns_ref = np.array([Ns[0], Ns[-1]])
    asm_ref = asm_s[0] * (Ns_ref / Ns[0]) ** 2
    ax.loglog(Ns_ref, asm_ref, "k--", lw=1.0, alpha=0.7,
              label=r"$\mathcal{O}(N^2)$ reference")
    ax.set_xlabel(r"$N$ (Gauss--Legendre nodes)")
    ax.set_ylabel("wall-clock (ms)")
    ax.set_title("(a) Matrix assembly")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)

    # (b) Total wall-clock Picard vs Newton (log-log)
    ax = axes[0, 1]
    ax.loglog(Ns, pic_s, "o-", color="C2", lw=1.6, ms=5, label="Picard total")
    ax.loglog(Ns, nw_s, "s-", color="C3", lw=1.6, ms=5, label="Newton total")
    ax.loglog(Ns_ref, pic_s[0] * (Ns_ref / Ns[0]) ** 2,
              "k--", lw=1.0, alpha=0.6, label=r"$\mathcal{O}(N^2)$ reference")
    ax.loglog(Ns_ref, nw_s[0] * (Ns_ref / Ns[0]) ** 3,
              "k:",  lw=1.0, alpha=0.6, label=r"$\mathcal{O}(N^3)$ reference")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel("wall-clock (ms)")
    ax.set_title("(b) Total iteration time")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)

    # (c) Iteration counts vs N (semi-log x; should be flat)
    ax = axes[1, 0]
    ax.semilogx(Ns, pic_it, "o-", color="C2", lw=1.6, ms=5,
                label="Picard")
    ax.semilogx(Ns, nw_it, "s-", color="C3", lw=1.6, ms=5,
                label="smoothed Newton")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel("iterations to $10^{-13}$")
    ax.set_title("(c) Iteration counts (resolution-independent)")
    ax.set_ylim(0, max(nw_it.max(), pic_it.max()) * 1.4)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)

    # (d) Condition number cond_inf(I - A_N) vs N
    ax = axes[1, 1]
    ax.semilogx(Ns, cond, "o-", color="C4", lw=1.6, ms=5,
                label=r"$\mathrm{cond}_\infty(I - A_N)$")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"sup-norm condition number")
    ax.set_title("(d) Conditioning of $I - A_N$")
    cond_max = cond.max()
    ax.set_ylim(0.95 * cond.min(), 1.10 * cond_max)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)

    fig.savefig(filename)
    plt.close(fig)


# ---------- main -------------------------------------------------------------
def main() -> int:
    print("=" * 90)
    print("  Scaling study: Picard-Nystrom on the contractive benchmark")
    print("=" * 90)

    p = benchmark_parameters()
    print(f"  Reference parameters: {p}")
    print(f"  Benchmark: separable kernel K(t,s) = alpha t s, sec.~7.2.")

    Ns = (32, 64, 128, 256, 512, 1024, 2048, 4096)
    rows: list[dict] = []
    print(f"\n  {'N':>5s}  {'asm (ms)':>9s}  "
          f"{'Pic iters':>9s}  {'Pic/it (ms)':>11s}  {'Pic tot (ms)':>12s}  "
          f"{'New iters':>9s}  {'New tot (ms)':>12s}  {'cond_inf':>9s}")
    for N in Ns:
        r = measure_one(N, p)
        rows.append(r)
        print(f"  {r['N']:>5d}  {r['assembly_s']*1e3:>9.3f}  "
              f"{r['picard_iters']:>9d}  "
              f"{r['picard_per_iter_s']*1e3:>11.4f}  "
              f"{r['picard_total_s']*1e3:>12.4f}  "
              f"{r['newton_iters']:>9d}  "
              f"{r['newton_total_s']*1e3:>12.4f}  "
              f"{r['cond_inf']:>9.4f}")

    # Empirical scaling exponents (last 4 points only, asymptotic regime).
    Ns_tail   = np.array([r["N"] for r in rows[-4:]], dtype=float)
    pic_tail  = np.array([r["picard_total_s"] for r in rows[-4:]])
    nw_tail   = np.array([r["newton_total_s"]  for r in rows[-4:]])
    asm_tail  = np.array([r["assembly_s"] for r in rows[-4:]])
    p_pic = float(np.polyfit(np.log(Ns_tail), np.log(pic_tail), 1)[0])
    p_nw  = float(np.polyfit(np.log(Ns_tail), np.log(nw_tail),  1)[0])
    p_asm = float(np.polyfit(np.log(Ns_tail), np.log(asm_tail), 1)[0])

    cond_min = min(r["cond_inf"] for r in rows)
    cond_max = max(r["cond_inf"] for r in rows)

    print(f"\n  Empirical scaling exponents (last 4 N values, asymptotic):")
    print(f"    Picard total : N^{p_pic:.3f}  (theoretical: 2)")
    print(f"    Newton total : N^{p_nw:.3f}  (theoretical: 3, LU-dominated)")
    print(f"    Assembly     : N^{p_asm:.3f}  (theoretical: 2)")
    print(f"  cond_inf(I - A_N) range over N in [{Ns[0]}, {Ns[-1]}]:")
    print(f"    [{cond_min:.4f}, {cond_max:.4f}]   relative variation: "
          f"{(cond_max - cond_min) / cond_min * 100:.3f}%")

    plot_scaling(rows, "./fig_scaling.pdf")
    print(f"\n  Figure written to ./fig_scaling.pdf")

    print("\n" + "=" * 90)
    print("  JSON dump (compact)")
    print("=" * 90)
    out = {
        "params":           p,
        "rows":             rows,
        "exponents": {
            "picard_total":  p_pic,
            "newton_total":  p_nw,
            "assembly":      p_asm,
        },
        "cond_inf_range":   [cond_min, cond_max],
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
