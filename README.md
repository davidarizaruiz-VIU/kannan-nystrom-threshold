# kannan-nystrom-threshold

Reference implementation and experimental code for the manuscript

> **Kannan-stable Picard–Nyström iteration for discontinuous threshold integral equations: theory, computation, and applications**
> David Ariza-Ruiz. Submitted to the *Journal of Computational and Applied Mathematics* (Elsevier), 2026.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19814083.svg)](https://doi.org/10.5281/zenodo.19814083)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python ≥3.10](https://img.shields.io/badge/python-%E2%89%A53.10-blue.svg)](https://www.python.org/)
[![Status: companion code](https://img.shields.io/badge/status-paper%20companion-green)](#)

This repository accompanies the manuscript and contains every piece of code, every benchmark dataset and every driver script needed to reproduce the tables and figures of Sections 4–8 in their default configuration.

## Scope

The library implements:

- **Picard–Nyström iteration** for the threshold integral equation
  `u = A u + g + q · H(ℓ(u) − c)` on `Ω = [0,1]` and `[0,1]²`, with Gauss–Legendre, composite trapezoidal and composite Simpson quadrature.
- **Kannan diagnostics** (`κ_N, μ_N, λ_N`, invariance margin `m_R`) at both the continuous and discrete level.
- **Smoothed semismooth Newton** comparison via Sherman–Morrison rank-one updates (Chen–Mangasarian sigmoid).
- **Two-sided monotone Picard sweep** producing certified upper/lower envelopes of the fixed point.
- **Bistable regime** existence test and basin certification (Theorem 3.6 of the paper).
- **Filippov set-valued regularisation** of the discontinuous switch.
- **Green-kernel BVP** formulation (Theorem 5.1).
- **2-D extension** for non-separable kernels on the unit square.

## Installation

The library targets a clean scientific-Python stack:

```bash
git clone https://github.com/davidarizaruiz-VIU/kannan-nystrom-threshold.git
cd kannan-nystrom-threshold
python3 -m pip install -r requirements.txt
```

A pip-installable package (`kannan-nystrom`) is planned for the v1.1 release.

## Quick start

```python
import numpy as np
from picard_nystrom import (
    benchmark_parameters, build_benchmark_functions,
    assemble, kannan_diagnostics, picard_nystrom,
)

# Reference benchmark of Section 7 (separable kernel, contractive Kannan regime)
p = benchmark_parameters()
K, g, q = build_benchmark_functions(p)
sys_ = assemble(N=128, K=K, g=g, q=q, rule="gauss")

# Discrete Kannan smallness 3 κ_N + 2 ‖q‖/m_R < 1 must hold
diag = kannan_diagnostics(sys_, p["c"], p["R"])
assert diag["admissible"], diag

# Picard–Nyström iteration to machine precision
result = picard_nystrom(sys_, p["c"],
                        U0=np.full(sys_.t.size, p["c"] / 2.0),
                        tol=1e-13, nmax=200)
print(f"converged in {result['iters']} iterations, "
      f"final residual {result['residuals'][-1]:.2e}")
```

## Reproducibility map

Each driver script reproduces a specific manuscript object with **bit-identical** output on the reference hardware (Apple M3, macOS Sequoia, Python 3.12, NumPy 1.26 + Accelerate BLAS, SciPy 1.13). Wall-clock timings rounded.

| Driver script                       | Reproduces                                            | Wall-clock |
|-------------------------------------|-------------------------------------------------------|-----------:|
| `run_experiment6.py`                | Tables 1, 3; Figs. residual + solution-threshold      | ~2 s       |
| `run_bvp_experiment.py`             | Tables BVP-diagnostics, BVP-history; Fig. BVP         | ~1 s       |
| `run_experiment5_highres.py`        | Non-separable diagnostics + refinement; Fig. nonsep   | ~10 s      |
| `run_experiment7.py`                | Picard vs Newton table (5000 reps + 50 warmup)        | ~20 s      |
| `run_application.py`                | Fig. population case study                            | ~1 s       |
| `run_bistability_experiment.py`     | Fig. bistability                                      | ~2 s       |
| `run_filippov_experiment.py`        | Fig. Filippov                                         | ~3 s       |
| `run_2d_experiment.py`              | 2D separable + non-separable tables; Fig. 2D          | ~3 s       |
| `run_relay_experiment.py`           | Fig. relay control case study                         | ~1 s       |
| `run_stability_experiment.py`       | Empirical validation of the stability theorem         | ~1 s       |
| `run_two_sided_iteration.py`        | Two-sided monotone sweep table                        | ~1 s       |
| `run_newton_comparison.py`          | Newton-vs-Picard comparison + scaling tables          | ~5 s       |
| `run_scaling_study.py`              | Fig. scaling diagnostics                              | ~90 s      |
| `run_real_data_application.py`      | Fig. real-data fit (Royama et al. 2017)               | ~20 s      |
| `run_convergence_sweep.py`          | Multi-regularity convergence table; Fig. convergence  | ~15 s      |

Run, for example:

```bash
python3 run_convergence_sweep.py
```

## Datasets

The folder `data/` contains six CSV files derived from the public *Choristoneura fumiferana* (spruce budworm) dataset of Royama, Eveleigh, Morin, Pothier, Tosh, Filotas & Ostaff (2017), distributed under CC0 via Dryad. Each file lists the fourth-instar larval density on Plot 1 of the Green River study area for one calendar year, sampled along the budworm phenological window. See `data/README.md` for full provenance and citation.

These data are used by `run_real_data_application.py` to fit the 7-parameter bistable threshold model jointly to the 1985 outbreak year and the 1988 endemic year, reproducing the cusp catastrophe of Ludwig, Jones & Holling (1978) on real observations.

## Layout

```
.
├── picard_nystrom.py             core 1-D module
├── picard_nystrom_2d.py          non-separable 2-D extension
├── run_*.py                      15 driver scripts (see table above)
├── data/                         Royama-2017 CSV extracts (CC0)
├── requirements.txt              pinned scientific-Python stack
├── LICENSE                       MIT
├── CITATION.cff                  GitHub citation metadata
├── CHANGELOG.md                  release notes
└── README.md                     this file
```

## How to cite

Both the software and the paper should be cited. A `CITATION.cff` is provided so that GitHub's "Cite this repository" button generates the correct BibTeX automatically.

```bibtex
@article{ArizaRuiz2026KannanNystrom,
  author       = {David Ariza-Ruiz},
  title        = {Kannan-stable {Picard}--{Nystr\"om} iteration for
                  discontinuous threshold integral equations:
                  theory, computation, and applications},
  journal      = {Journal of Computational and Applied Mathematics},
  year         = {2026},
  note         = {Submitted}
}

@software{ArizaRuiz2026KannanNystromCode,
  author       = {David Ariza-Ruiz},
  title        = {kannan-nystrom-threshold: reference implementation
                  of the Kannan-stable {Picard}--{Nystr\"om} iteration
                  for discontinuous threshold integral equations},
  year         = {2026},
  version      = {v1.0.0},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19814083},
  url          = {https://doi.org/10.5281/zenodo.19814083}
}
```

The DOI `10.5281/zenodo.19814083` is the *concept DOI* that always resolves to the latest version; the *version DOI* `10.5281/zenodo.19814084` resolves specifically to the v1.0.0 release accompanying the JCAM submission. The release is additionally indexed in OpenAIRE and archived in Software Heritage.

## License

The source code is released under the MIT License (see `LICENSE`). The Royama et al. (2017) dataset extracts in `data/` are redistributed under the original CC0 dedication; please refer to `data/README.md` for the canonical Dryad citation.

## Author

David Ariza-Ruiz — Universidad Internacional de Valencia (VIU). ORCID: [0000-0001-8782-7219](https://orcid.org/0000-0001-8782-7219).
