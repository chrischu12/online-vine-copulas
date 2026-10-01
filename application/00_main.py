#%%

import os
import sys
from pathlib import Path

# This file lives in <checkout>/application, so the checkout is one level up. In an
# interactive cell `__file__` is undefined; there the checkout is whichever folder
# around the working directory holds `ondil`. Its parent goes first on sys.path, so
# the imports below come from this checkout and not from another copy.
try:
    _PKG_DIR = Path(__file__).resolve().parent.parent
except NameError:
    _cwd = Path.cwd().resolve()
    _PKG_DIR = next(
        p
        for p in (_cwd, _cwd.parent, _cwd / "online_copula_experiments")
        if (p / "ondil").is_dir()
    )
sys.path.insert(0, str(_PKG_DIR.parent))

import online_copula_experiments.application as _application
from online_copula_experiments.application import BASE_PATH
from online_copula_experiments.runtime_monitor import run_and_measure, write_report

FOLDER = _PKG_DIR / "application"
# Directory that contains the `online_copula_experiments` package, so the
# subprocesses can `import online_copula_experiments...` (they otherwise only
# get their own folder on sys.path).
PKG_PARENT = _PKG_DIR.parent

# Import search path handed to every step: this checkout's own `ondil` and
# `VineCopulas` sources come first, so the subprocess cannot resolve them through
# site-packages to an editable install made from a different clone.
_CHECKOUT_PYTHONPATH = os.pathsep.join(
    [
        str(_PKG_DIR / "ondil" / "src"),
        str(_PKG_DIR / "VineCopulas" / "src"),
        str(PKG_PARENT),
    ]
)

# Everything this run depends on, so it can be checked at a glance.
print(f"[00_main] checkout   : {_PKG_DIR}")
print(f"[00_main] ondil      : {_PKG_DIR / 'ondil' / 'src'}")
print(f"[00_main] VineCopulas: {_PKG_DIR / 'VineCopulas' / 'src'}")
print(f"[00_main] PYTHONPATH : {_CHECKOUT_PYTHONPATH}")
print(f"[00_main] BASE_PATH  : {BASE_PATH}")
print(f"[00_main] interpreter: {sys.executable}")


def make_folder(path):
    try:
        os.makedirs(path, exist_ok=False)
    except OSError:
        print(path, "exists")
        print(os.listdir(path))


# Runtime, peak RAM and parallelization strategy of every step, collected as a
# side effect of the run and written to runtime_report_application.{md,json}.
RUNTIME_RECORDS = []


def run_file(file: str, folder: Path = FOLDER, scheme: str = None):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (_CHECKOUT_PYTHONPATH, env.get("PYTHONPATH", "")) if p
    )
    # Pin the switch explicitly so a value left over in the shell cannot leak in.
    env["SCHEME"] = scheme or "repeated_online"
    # Stream the subprocess output to the console (no log files).
    record = run_and_measure(file, folder, env)
    if scheme:
        record["step"] = f"{record['step']} [{scheme}]"
    RUNTIME_RECORDS.append(record)


# To reproduce: place GSPGroup_NetD_final.RData in BASE_PATH. Everything else
# (macro_regions_out/, the *.pkl design matrices and PITs, result_*.pkl) is
# created inside BASE_PATH.
make_folder(BASE_PATH)


# Run all files sequentially
# Data preparation
run_file("01_build_macro_regions.py")          # raw GSP data -> macro_<region>.csv
run_file("01_prepare_data_macro_marginals.py")  # -> Xy_data_macro.pkl
run_file("01_prepare_data_macro_copula.py")     # -> combined_Xy_data_copula_absspread.pkl

# ---------------------------------------------------------------------------
# One pass per updating scheme
# ---------------------------------------------------------------------------
# The marginal models and the copula always use the same scheme, so each pass
# produces three complete forecasting models (one per copula specification) and
# writes them to intermediate_application/<scheme>/. 02_vine_copula.py handles
# the two schemes that re-fit monthly, 02_vine_copula_full_online.py the one
# that never does; each exits immediately for the schemes that are not its own.
# 02_marginals.py also writes that scheme's marginal log-likelihood, so the log
# score can be reported for the full predictive density.
for scheme in ("repeated_batch", "repeated_online", "online"):
    run_file("02_marginals.py", scheme=scheme)
    run_file("02_vine_copula.py", scheme=scheme)
    run_file("02_vine_copula_full_online.py", scheme=scheme)

# Slim score cache, so that later table-only runs (TABLES_ONLY=1) need not read
# the multi-hundred-megabyte result files again.
run_file("02_cache_scores.py")

# Figures and LaTeX tables, over all scheme folders at once
run_file("03_plots.py")

# Into the results folder that goes with the figures and tables, so the
# resource requirements are handed on together with the output. The same call
# refreshes the Resource requirements section of the README, which keeps these
# tables between markers.
reports = write_report(
    RUNTIME_RECORDS,
    Path(BASE_PATH) / "results",
    "application",
    readme=_PKG_DIR / "README.md",
)
for report in reports:
    print(f"Wrote runtime report to {report}.")

print("Successfully finished the application and created all reproduction files.")
