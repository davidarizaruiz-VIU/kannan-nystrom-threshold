# Changelog

All notable changes to `kannan-nystrom-threshold` are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-04-27

First public release accompanying the submission of the manuscript

> *Kannan-stable Picard–Nyström iteration for discontinuous threshold integral equations: theory, computation, and applications.*
> D. Ariza-Ruiz, *Journal of Computational and Applied Mathematics*, submitted.

### Added
- Core 1-D module `picard_nystrom.py`: Gauss–Legendre / trapezoidal / Simpson assembly, Kannan diagnostics (`κ_N, μ_N, λ_N`), Picard–Nyström iteration with full residual and threshold-state history, smoothed semismooth Newton via Sherman–Morrison rank-one updates, two-sided monotone sweep, Green-kernel BVP machinery and Filippov-regularised fixed point.
- 2-D extension `picard_nystrom_2d.py` for non-separable kernels on the unit square.
- Fifteen driver scripts that regenerate every table and figure of the manuscript with bit-identical output on the reference hardware.
- Six CSV extracts of the Royama et al. (2017) spruce-budworm dataset (Plot 1, years 1984–1989), redistributed under the original CC0 dedication for the bistable real-data fit of `run_real_data_application.py`.
- MIT license, `CITATION.cff`, pinned `requirements.txt`, and reproducibility map in the README.

[1.0.0]: https://github.com/davidarizaruiz-VIU/kannan-nystrom-threshold/releases/tag/v1.0.0
