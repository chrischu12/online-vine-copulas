#%%

import os
import sys
from pathlib import Path

# This file lives in <checkout>/simulations, so the checkout is one level up. In an
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

import online_copula_experiments
from online_copula_experiments.application import BASE_PATH
from online_copula_experiments.runtime_monitor import run_and_measure, write_report


FOLDER = _PKG_DIR / "simulations"
# Directory that contains the `online_copula_experiments` package

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
# side effect of the run and written to runtime_report_simulations.{md,json}.
RUNTIME_RECORDS = []


def run_file(file: str, folder: Path = FOLDER, args=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (_CHECKOUT_PYTHONPATH, env.get("PYTHONPATH", "")) if p
    )
    # Stream the subprocess output to the console (no log files).
    RUNTIME_RECORDS.append(run_and_measure(file, folder, env, args))


# The simulation study generates its own data, so no input file is required.
# Results (all_runs_*.pkl and score/timing tables) are written to BASE_PATH.
make_folder(BASE_PATH)


# Run all files sequentially
# Forecasting study: run both the stable and the structural-break scenario,
# each writing its own all_runs_50{,_break}.pkl.
run_file("01_vine_forecasting.py", args=["stable"])
run_file("01_vine_forecasting.py", args=["break"])
run_file("01_vine_batch.py")

# Figures for the paper
run_file("02_plots.py")


# Into the results folder that goes with the figures and tables, so the
# resource requirements are handed on together with the output. The same call
# refreshes the Resource requirements section of the README, which keeps these
# tables between markers.
reports = write_report(
    RUNTIME_RECORDS,
    Path(BASE_PATH) / "results",
    "simulations",
    readme=_PKG_DIR / "README.md",
)
for report in reports:
    print(f"Wrote runtime report to {report}.")

print("Successfully finished simulation and created all reproduction files.")
