"""
run_experiment7.py
==================
Driver script for Experiment 7 of the manuscript:
  "Comparison with direct linear-system solve (DLS-BT)".

For each of the three benchmarks (separable integral, BVP via Green,
non-separable exponential) and each resolution N in {32, 64, 128, 256},
this script measures:

  - Picard-Nystroem iteration count and wall-clock time.
  - Direct solve  (I - A_N) U = ...  + branch test wall-clock time.
  - Final residuals achieved by both.
  - Sup-norm error against the analytical fixed point (when available).
  - Branch identified by DLS-BT (must be 'upper' under our hypotheses).
  - Pairwise consistency: ||U_PN - U_DS||_inf.

Each timing is the median of 5 runs (to reduce OS scheduler noise).

Usage from a terminal in the project folder:
    python3 run_experiment7.py
"""
from __future__ import annotations

import json
import sys

from picard_nystrom import compare_picard_vs_direct


def _row_table_line(r):
    return (f"  {r['benchmark']:>12s}  N={r['N']:>4d}  "
            f"PN: {r['picard_iters']:>2d}it  "
            f"med={r['picard_time_s']*1e3:>7.3f} IQR={r['picard_iqr_s']*1e3:>6.3f} ms  "
            f"DS: med={r['direct_time_s']*1e3:>7.3f} IQR={r['direct_iqr_s']*1e3:>6.3f} ms  "
            f"PN/DS={r['speedup_DS_over_PN']:>5.2f}")


def _row_distribution_line(r, method_prefix):
    """Detailed distribution line for one timing distribution (PN or DS)."""
    return (f"  {r['benchmark']:>12s} N={r['N']:>4d} {method_prefix} "
            f"min={r[method_prefix.lower() + '_min_s']*1e3:>7.3f}  "
            f"p1={r[method_prefix.lower() + '_p1_s']*1e3:>7.3f}  "
            f"q1={r[method_prefix.lower() + '_q1_s']*1e3:>7.3f}  "
            f"med={r[method_prefix.lower() + '_time_s']*1e3:>7.3f}  "
            f"q3={r[method_prefix.lower() + '_q3_s']*1e3:>7.3f}  "
            f"p99={r[method_prefix.lower() + '_p99_s']*1e3:>7.3f}  "
            f"max={r[method_prefix.lower() + '_max_s']*1e3:>7.3f}  "
            f"mean={r[method_prefix.lower() + '_mean_s']*1e3:>7.3f}  "
            f"std={r[method_prefix.lower() + '_std_s']*1e3:>6.3f} ms")


def main() -> int:
    print("=" * 110)
    print("  Experiment 7 — Picard-Nystroem vs direct linear solve with branch test (DLS-BT)")
    print("=" * 110)
    print()
    print("  Both methods compute the same discrete fixed point U_*; we compare")
    print("  wall-clock time, final residual, and sup-norm consistency.")
    print()

    res = compare_picard_vs_direct(
        N_list=(32, 64, 128, 256),
        benchmarks=("integral", "bvp", "nonseparable"),
        tol=1e-13,
        repeats=5000,
        warmup=50,
    )

    print(f"  Tolerance: {res['tol']}; timing protocol = "
          f"{res['warmup']} warmup runs (discarded) + "
          f"{res['repeats']} measured runs per (benchmark, N) cell.")
    print(f"  Reported per cell: median, IQR (q3-q1), p1, p99, min, max, "
          f"mean, std.")
    print(f"  Estimated wall-clock: ~10-30 minutes on Apple M3 (depends on")
    print(f"  background load; mostly limited by N=256 BLAS calls).")
    print()
    print("  Per-row summary (median + IQR):")
    for r in res["rows"]:
        print(_row_table_line(r))

    print()
    print("=" * 130)
    print("  Detailed timing distribution per cell (Picard-Nystrom and DLS-BT, ms)")
    print("=" * 130)
    for r in res["rows"]:
        print(_row_distribution_line(r, "Picard"))
        print(_row_distribution_line(r, "Direct"))

    print()
    print("=" * 110)
    print("  Sup-norm consistency between methods (||U_PN - U_DS||_inf)")
    print("=" * 110)
    print(f"  {'benchmark':>12s} {'N':>5s} {'||U_PN - U_DS||_inf':>22s}")
    for r in res["rows"]:
        print(f"  {r['benchmark']:>12s} {r['N']:>5d} {r['max_diff_PN_DS']:>22.3e}")

    print()
    print("=" * 110)
    print("  Branch dichotomy (DLS-BT must accept 'upper' under m_R + Lq > 0)")
    print("=" * 110)
    print(f"  {'benchmark':>12s} {'N':>5s} {'ell_plus':>12s} {'ell_minus':>12s} {'accepted':>10s}")
    for r in res["rows"]:
        print(f"  {r['benchmark']:>12s} {r['N']:>5d} "
              f"{r['direct_ell_plus']:>12.6f} {r['direct_ell_minus']:>12.6f} "
              f"{r['direct_branch']:>10s}")

    print()
    print("=" * 110)
    print("  Sup-norm error vs analytical fixed point  (only for benchmarks with closed form)")
    print("=" * 110)
    print(f"  {'benchmark':>12s} {'N':>5s} {'PN error':>14s} {'DS error':>14s}")
    for r in res["rows"]:
        if r["picard_error"] is None: continue
        print(f"  {r['benchmark']:>12s} {r['N']:>5d} "
              f"{r['picard_error']:>14.3e} {r['direct_error']:>14.3e}")

    print()
    print("=" * 110)
    print("  JSON dump (for archival / programmatic use)")
    print("=" * 110)
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
