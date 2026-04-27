# Datasets

This folder contains six CSV files, one per calendar year (1984–1989), derived from the public dataset

> **Royama, T., Eveleigh, D. S., Morin, J. R. B., Pothier, S. J., Tosh, D. R., Filotas, E. & Ostaff, D. P. (2017).**
> *Mechanisms underlying spruce budworm outbreak processes as elucidated by a 14-year study in New Brunswick, Canada.*
> *Ecological Monographs* 87 (4), 600–631. doi:10.1002/ecm.1270.
> Dryad data package: doi:10.5061/dryad.94vt5 — distributed under the **CC0 Public Domain Dedication**.

## File contents

Each file lists the **fourth-instar larval density of *Choristoneura fumiferana*** (spruce budworm), in larvae per square metre of branch surface, on **Plot 1 of the Green River study area** (New Brunswick, Canada) along the larval phenological window of one calendar year. The two columns are:

| Column                  | Type    | Description                                                                       |
|-------------------------|---------|-----------------------------------------------------------------------------------|
| `ordinal_date`          | int     | Day of the year (1–366) on which the sample was taken.                            |
| `larva_per_m2_branch`   | float   | Sampled fourth-instar larval density (larvae · m⁻² of branch surface).            |

## Provenance and processing

These CSV files are tabular extracts of the original Dryad data package, obtained as follows. The Royama et al. (2017) data were tabulated by year and plot, the Plot 1 entries were retained, and the corresponding `(ordinal_date, larva_per_m2_branch)` pairs were exported to CSV with no rounding, smoothing or imputation. No data points were added, removed, or reweighted. The original units, sampling dates and density values are preserved verbatim.

## Intended use

In the companion repository these files feed `run_real_data_application.py`, which performs a **joint bistable fit** of the seven-parameter threshold model

```
u(t) = α (B u)(t) + g(t; β, γ) + σ q(t; δ, μ) · H(ℓ(u) − c)
```

simultaneously on the 1985 outbreak year (upper branch `u⁺`) and the 1988 endemic year (lower branch `u⁻`). The fitted parameters reproduce the cusp catastrophe of Ludwig, Jones & Holling (1978) on real observations, with bistability margins `m⁻ < 0 < m⁺` and an attainable threshold `c` strictly between the two branches.

The remaining years (1984, 1986, 1987, 1989) are provided for completeness and out-of-sample sanity checks; they are not used in the joint fit reported in the manuscript.

## Citation

If you use these files in derivative work, cite the original dataset:

```bibtex
@article{Royama2017spruce,
  author       = {Royama, T. and Eveleigh, D. S. and Morin, J. R. B. and
                  Pothier, S. J. and Tosh, D. R. and Filotas, E. and
                  Ostaff, D. P.},
  title        = {Mechanisms underlying spruce budworm outbreak processes
                  as elucidated by a 14-year study in New Brunswick, Canada},
  journal      = {Ecological Monographs},
  volume       = {87},
  number       = {4},
  pages        = {600--631},
  year         = {2017},
  doi          = {10.1002/ecm.1270}
}

@misc{Royama2017dryad,
  author       = {Royama, T. and Eveleigh, D. S. and Morin, J. R. B. and
                  Pothier, S. J. and Tosh, D. R. and Filotas, E. and
                  Ostaff, D. P.},
  title        = {Data from: Mechanisms underlying spruce budworm outbreak
                  processes as elucidated by a 14-year study in New
                  Brunswick, Canada},
  year         = {2017},
  publisher    = {Dryad Digital Repository},
  doi          = {10.5061/dryad.94vt5},
  note         = {CC0 Public Domain Dedication}
}
```

## Licence

The CSV extracts in this folder inherit the original CC0 Public Domain Dedication of the Royama et al. (2017) dataset. They may be redistributed, modified or reused without restriction; attribution to the original authors is requested as a matter of academic courtesy but is not legally required.
