# Replication package

This is the code for reproducing the results, figures and tables appearing in
"Online Conditional Vine Copulas: Forecasting Electricity Demand" by Jobelius Schulz,
Hanck, Hirsch and Ziel.

The code depends mostly on packages available on PyPI (see `req.txt`), with the
exception of two packages that are included as git submodules:

- **`ondil`** — implements the online distributional regression methods discussed
  in the paper, including the bivariate copula distributions and the online IRLS
  estimator for covariate-dependent dependence parameters. Pinned to branch
  `bivariate_copula`.
- **`VineCopulas`** — provides the vine copula machinery (structure selection,
  simulation, h-functions). Substantially based on the package of the same name
  by Claassen (VU-IVM) and modified here for conditional and online estimation.
  Pinned to branch `dev`.

Both submodules must be initialised and installed before proceeding further.

## Setup

Run the setup script for your platform from the repository root. It creates a
conda environment `online_copula_exp` (Python 3.12), installs `req.txt`,
initialises both submodules on the correct branches, and installs them in
editable mode:

```
setup_windows.bat        # Windows (run from an Anaconda Prompt)
bash setup_linux.sh      # Linux
```

If you are unfamiliar with git submodules, [this gist](https://gist.github.com/gitaarik/8735255)
is a useful reference.

Then choose the output root, `BASE_PATH`: the one folder the input data is read
from and every result is written to. Both studies use it, so it is set once.

Copy `application/local_path_example.py` to `application/local_path.py` and put
your path in it:

```python
from pathlib import Path

BASE_PATH = Path(r"D:\forecasting_study")
```

That file is listed in `.gitignore`, so each machine keeps its own path and no
tracked file has to be edited. Without it, `BASE_PATH` is `forecasting_study/`
in your home directory.

## Data

The application uses the regional net-demand data for Great Britain from the
replication material of Gioia and Fasiolo:

> Gioia, V. and Fasiolo, M. *Code for "Additive Covariance Matrix Models:
> Modelling Regional Electricity Net-Demand in Great Britain"* (v4).
> Zenodo. https://doi.org/10.5281/zenodo.13330081

Download `GSPGroup_NetD_final.RData` from that record and place it directly in
`BASE_PATH`. The file is not redistributed here. It contains the list
`NodeData1` with 14 elements, one per GSP group, keyed by the Great Britain
electricity market designations ("A" = E_England, ..., "P" = N_Scotland). The
first pipeline step aggregates these 14 GSP groups into the 5 macro regions used
in the paper (Scotland, Northern, Midlands, London, Southern).

Parsing the ~120 MB `.RData` takes 1–2 minutes.

## Reproducing the results

There are two independent studies, each with its own driver script. Set the
working directory to the repository root and run:

```
python simulations/00_main.py     # simulation study
python application/00_main.py     # empirical application
```

Each driver runs its files in order as subprocesses. They can also be run
individually, in the order given below.

### Simulation study — `simulations/00_main.py`

| File | Role | Paper |
|---|---|---|
| `01_vine_forecasting.py stable` | Online vs. batch forecasting, stable scenario → `all_runs_50.pkl` | Section 3.2 |
| `01_vine_forecasting.py break` | Same, structural-break scenario → `all_runs_50_break.pkl` | Section 3.2 |
| `01_vine_batch.py` | Batch reference fits (parallelised via `joblib`) | Section 3.1 |
| `02_plots.py` | Figures and LaTeX tables | Figures 2, 3, 7; Tables 2, 6 |

The simulation generates its own data, so no input file is required.
Intermediate results are written to `BASE_PATH/intermediate_simulation`.

### Application — `application/00_main.py`

| File | Role | Paper |
|---|---|---|
| `01_build_macro_regions.py` | 14 GSP groups → 5 macro regions (`macro_<region>.csv`) | Section 4, aggregation in Table 7 |
| `01_prepare_data_macro_marginals.py` | Marginal design matrices → `Xy_data_macro.pkl` | Section 4.1, covariates in Table 3 |
| `01_prepare_data_macro_copula.py` | Pairwise weather-spread features → `combined_Xy_data_copula_absspread.pkl` | Section 4.2 |
| `02_marginals.py` | Per-region marginal models; writes CDFs and PITs (`u_in_sample_*`, `u_out_sample_*`) | Section 4.1 |
| `02_vine_copula.py` | Main study: batch vs. repeated-online vine copulas | Section 4.2 |
| `02_vine_copula_full_online.py` | Pure online variant | Section 4.2 |
| `02_cache_scores.py` | Copies the scores out of each `result_*.pkl` into a slim `scores_*.pkl` | — |
| `03_plots.py` | Figures and LaTeX tables | Figures 6, 8; Tables 4, 5, 8–12 |

The three `02_*` model steps run once per updating scheme (`repeated_batch`,
`repeated_online`, `online`), each pass writing that scheme's marginals and
copulas to `BASE_PATH/intermediate_application/<scheme>/`. The remaining steps
run once, over all schemes at the end.

### Outputs

Both `02_plots.py` and `03_plots.py` write to a shared results tree:

```
BASE_PATH/results/figures/   # .pdf figures
BASE_PATH/results/tables/    # .tex tables
BASE_PATH/results/runtime_report_*.md   # resource use of the run that produced them
```

## Resource requirements

The drivers measure themselves. Every step is run through `runtime_monitor.py`,
which samples the child's process tree while it works, so a normal run records
its own wall time and peak RAM as a side effect and writes
`runtime_report_<pipeline>.md` and `.json` to `BASE_PATH/results`, alongside the
figures and tables it produced. The same run splices the tables into this file
between the markers below, so the README is the copy that travels with the package — the
tables are never edited by hand, and re-running a driver refreshes them.

### Simulation study — `simulations/00_main.py`

<!-- runtime-report:simulations:start -->
- Machine: IBES-CALC01, Windows 2022Server
- CPU: AMD64 Family 23 Model 49 Stepping 0, AuthenticAMD (128 physical / 128 logical cores)
- RAM: 511.6 GB total, 477.7 GB free at start
- Python 3.12.14, run started 2026-09-30T15:05:07

| Step | Wall time | Peak RAM | Parallelization | Processes seen |
|---|---:|---:|---|---:|
| `01_vine_forecasting.py stable` | 20.54 h | 21.02 GB | loky, n_jobs=50; loky, n_jobs=min(20, cpu_count) | 65 |
| `01_vine_forecasting.py break` | 20.75 h | 21.04 GB | loky, n_jobs=50; loky, n_jobs=min(20, cpu_count) | 57 |
| `01_vine_batch.py` | 3.2 min | 17.36 GB | loky, n_jobs=min(50, cpu_count) | 56 |
| `02_plots.py` | 33 s | 0.76 GB | sequential | 2 |
| **Total** | **41.34 h** | | | |
<!-- runtime-report:simulations:end -->

### Application — `application/00_main.py`

<!-- runtime-report:application:start -->
- Machine: IBES-CALC01, Windows 2022Server
- CPU: AMD64 Family 23 Model 49 Stepping 0, AuthenticAMD (128 physical / 128 logical cores)
- RAM: 511.6 GB total, 478.0 GB free at start
- Python 3.12.14, run started 2026-09-28T16:51:02

| Step | Wall time | Peak RAM | Parallelization | Processes seen |
|---|---:|---:|---|---:|
| `01_build_macro_regions.py` | 4.1 min | 2.57 GB | sequential | 2 |
| `01_prepare_data_macro_marginals.py` | 48 s | 3.65 GB | sequential | 2 |
| `01_prepare_data_macro_copula.py` | 9 s | 0.24 GB | sequential | 2 |
| `02_marginals.py [repeated_batch]` | 75.0 min | 14.64 GB | threading, n_jobs=min(len(regions), cpu_count) | 2 |
| `02_vine_copula.py [repeated_batch]` | 3.29 h | 35.48 GB | loky, n_jobs=12 | 15 |
| `02_vine_copula_full_online.py [repeated_batch]` | 64 s | 2.98 GB | sequential | 2 |
| `02_marginals.py [repeated_online]` | 6.87 h | 11.73 GB | threading, n_jobs=min(len(regions), cpu_count) | 2 |
| `02_vine_copula.py [repeated_online]` | 4.50 h | 35.50 GB | loky, n_jobs=12 | 15 |
| `02_vine_copula_full_online.py [repeated_online]` | 55 s | 2.98 GB | sequential | 2 |
| `02_marginals.py [online]` | 4.42 h | 11.36 GB | threading, n_jobs=min(len(regions), cpu_count) | 2 |
| `02_vine_copula.py [online]` | 58 s | 3.00 GB | loky, n_jobs=12 | 2 |
| `02_vine_copula_full_online.py [online]` | 51.92 h | 4.22 GB | sequential | 2 |
| `02_cache_scores.py` | 82 s | 1.19 GB | sequential | 2 |
| `03_plots.py` | 40.1 min | 9.52 GB | sequential | 2 |
| **Total** | **73.08 h** | | | |
<!-- runtime-report:application:end -->

### Disk

Sizes of the generated output, from a complete run of both studies:

- `BASE_PATH/intermediate_simulation` — **110 MB**
  - `all_runs_50.pkl`, `all_runs_50_break.pkl`: 56 MB and 55 MB
- `BASE_PATH/intermediate_application` — **~10 GB**
  - `Xy_data_macro.pkl` (marginal design matrices): 2.6 GB
  - `combined_Xy_data_copula_absspread.pkl`: 20 MB; `macro_regions_out/`: 88 MB
  - one folder per updating scheme, 2.4 GB each: three `result_*.pkl` at
    ~650 MB, the PIT store `cdf_all_macro_forget.pkl` at 388 MB, and three slim
    `scores_*.pkl` at 18 MB
- `BASE_PATH/results` — **17 MB**

The `scores_*.pkl` caches exist so that a table-only pass (`TABLES_ONLY=1
python application/03_plots.py`) can rebuild every LaTeX table without reading
the multi-hundred-megabyte result files.

## License

This package is licensed under the **GNU General Public License v3.0** — see
`LICENSE`. Both submodules are themselves GPL-3.0, and the vine copula code is a
derivative work of the VineCopulas package, so GPL-3.0 applies to the combined
work.

The bivariate copula log-likelihoods, scores and Hessians were translated into
Python from the C sources of the GPL-licensed R package `VineCopula`. The
required notices, together with the licensing of the input data (CC BY 4.0), are
recorded in `THIRD_PARTY_NOTICES.md`.
