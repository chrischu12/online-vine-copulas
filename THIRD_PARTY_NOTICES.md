# Third-party notices

This replication package is licensed under the GNU General Public License v3.0
(see `LICENSE`). It builds on, and in places contains code derived from, the
projects listed below. Their terms continue to apply to the corresponding parts.

---

## 1. ondil — GPL-3.0

- Upstream: https://github.com/simon-hirsch/ondil
- Authors: Simon Hirsch, Jonathan Berrisch, Florian Ziel
- License: GNU General Public License v3.0
- Used here as the git submodule `ondil/`, pinned to branch `bivariate_copula`
  (fork: https://github.com/chrischu12/ondil).

The bivariate copula distributions in `ondil/src/ondil/distributions/bicop_*.py`
and the copula code paths in `ondil/src/ondil/estimators/online_mvdistreg.py`
were added for this paper. They contain code derived from the C sources of the
R package VineCopula — see section 3.

## 2. VineCopulas — GPL-3.0

- Upstream: https://github.com/VU-IVM/VineCopulas
- Author: Judith Claassen (Institute for Environmental Studies, VU Amsterdam)
- License: GNU General Public License v3.0
- Used here as the git submodule `VineCopulas/`, pinned to branch `dev`
  (fork: https://github.com/chrischu12/VineCopulas).

The vine copula machinery used in this package is **substantially based on**
VineCopulas and was modified for this work — principally to support covariate-
dependent (conditional) pair-copula parameters and online/incremental
estimation. It is a derivative work of the upstream package and is therefore
distributed under the same GPL-3.0 terms.

As required by GPL-3.0 section 5(a), the modifications are recorded in the fork
itself: a notice at the top of `VineCopulas/README.md` and a header in each
modified file. They start from upstream version 2.0.2 (commit `07911c4`,
26 November 2024) and were made between 19 November 2025 and 26 August 2026:

- `src/vinecopulas/vinecopula.py` — `fit_vinecop`, `fit_vinecopstructure`,
  `density_vinecop` and `simulate_vinecop` added for conditional and online
  estimation; the original fitting and simulation routines replaced.
- `src/vinecopulas/bivariate.py` — reduced to the families and h-functions used
  in the paper, with covariate-dependent parameters.
- `src/vinecopulas/marginals.py` — removed; marginals are estimated with `ondil`.

The unmodified portions remain under the copyright of Judith Claassen. The
upstream README, kept below the notice, documents the original package and not
this copy.

## 3. VineCopula (R package) — GPL-2 | GPL-3

- Upstream: https://cran.r-project.org/package=VineCopula
- Authors: Thomas Nagler (maintainer), Ulf Schepsmeier, Jakob Stoeber,
  Eike Christian Brechmann, Benedikt Graeler, Tobias Erhardt, and others
- License: GPL-2 | GPL-3

The log-likelihood, score and Hessian expressions for all four bivariate copula
families in `ondil/src/ondil/distributions/bicop_{normal,studentt,gumbel,clayton}.py`
were translated into Python from this package's C sources (`deriv.c`, `deriv2.c`,
`hfunc.c`) — in particular the `diff2PDF_mod` routine, noted in the Clayton
implementation. The Gumbel h-function, its inverse and the `qcondgum` helper
come from the same sources. Each of the four files carries a header recording
this. Distributing those derivations under GPL-3.0 is consistent with the
upstream GPL-2 | GPL-3 dual option.

---

## Data

`GSPGroup_NetD_final.RData` is **not redistributed** with this package. It is
obtained from the replication material of Gioia and Fasiolo (see the README),
published on Zenodo under the **Creative Commons Attribution 4.0 International
(CC BY 4.0)** license:

> Gioia, V. and Fasiolo, M. *Code for "Additive Covariance Matrix Models:
> Modelling Regional Electricity Net-Demand in Great Britain"* (v4).
> Zenodo. https://doi.org/10.5281/zenodo.13330081

CC BY 4.0 requires attribution; the citation above serves that purpose. Note
that the data license (CC BY 4.0) is separate from, and unaffected by, the
GPL-3.0 license covering the code in this repository.
