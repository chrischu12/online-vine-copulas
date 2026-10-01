# %%
import os
import numpy as np
import pandas as pd
import pickle

from joblib import Parallel, delayed

import ondil
print("ondil loaded from:", ondil.__file__)

import vinecopulas
print("VineCopulas loaded from:", vinecopulas.__file__)
from vinecopulas.bivariate import *
from vinecopulas.vinecopula import *

from online_copula_experiments.application import BASE_PATH
from online_copula_experiments.application.const_and_helper.simulation import (
    make_X,
    generate_true_copula_matrix,
)

# Simulation intermediate results live under BASE_PATH/intermediate_simulation.
SIM_DIR = BASE_PATH / "intermediate_simulation"
SIM_DIR.mkdir(parents=True, exist_ok=True)

np.set_printoptions(precision=3, suppress=True)

##########################################
# Vine
##########################################

a = np.array([
    [1, 2, 3, 4, 4],
    [2, 3, 4, 3, np.nan],
    [3, 4, 2, np.nan, np.nan],
    [4, 1, np.nan, np.nan, np.nan],
    [0, np.nan, np.nan, np.nan, np.nan]
])


# %%
RUNS = 500  # How many independent datasets per sample size
setting_name = "batch_vine"

# Sample sizes to run the Monte Carlo study for; one set of pkl files per size.
SAMPLE_SIZES = [1000, 5000]

cops = [1, 31, 32, 33, 34, 41, 42, 43, 44]

beta = np.array([0.25, 0.25, -0.25, 0, 0, 0.25, -0.25, -0.25, 0, 0]).reshape(-1, 1)
P_COV = beta.shape[0]  # number of covariates (first six binary, rest normal)

MAX_RETRIES = 50
N_JOBS = min(50, os.cpu_count() or 1)

# Per-(size, run) seeds so results are reproducible regardless of worker count.
RUN_SEEDS = {n: [1000 * (s + 1) + run for run in range(RUNS)]
             for s, n in enumerate(SAMPLE_SIZES)}


def fit_one_run(run, n, seed):
    """Simulate one dataset of size ``n`` and fit the batch vine to it."""
    rng = np.random.default_rng(seed)

    # Random true copula-family matrix for this run.
    C_true_d = generate_true_copula_matrix(a, cops, seed=rng.integers(2**32))

    # Simulate until no NaNs (some family/parameter draws can fail).
    for attempt in range(1, MAX_RETRIES + 1):
        X_numpy_arr = make_X(n, p=P_COV, seed=rng.integers(2**32))
        y_numpy = simulate_vinecop(a, X_numpy_arr, beta, C_true_d, n, rng=rng)

        if not np.isnan(y_numpy).any():
            break

        if attempt == MAX_RETRIES:
            raise RuntimeError(
                f"simulate_vinecop produced NaNs after {MAX_RETRIES} retries "
                f"(run={run}, n={n}). Consider clipping theta/inputs or relaxing families."
            )

    X_numpy = pd.DataFrame(
        X_numpy_arr, columns=[f"x_{i}" for i in range(X_numpy_arr.shape[1])]
    )

    E, B, P, C_d, A = fit_vinecopstructure(
        u1=y_numpy,
        X_df=X_numpy,
        a=a,
        copsi=cops,
        printing=False,
        online=0,
        forget=0,
        method_tree_1="ols",
        method_tree_2plus="ols",
        scale_inputs_tree_1=False,
        scale_inputs_tree_2plus=False,
    )

    return run, B, C_d, C_true_d


def run_study(n):
    """Run the Monte Carlo batch-vine study for sample size ``n`` (runs in parallel)."""
    print(f"==== Fitting {RUNS} runs | n={n} | setting={setting_name} ====")

    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(fit_one_run)(run, n, seed)
        for run, seed in enumerate(RUN_SEEDS[n])
    )

    betas = {}
    C = {}
    C_true = {}
    for run, B, C_d, C_true_d in results:
        pred_key = (setting_name, run)
        betas[pred_key] = B
        C[pred_key] = C_d
        C_true[pred_key] = C_true_d

    with open(SIM_DIR / f"{setting_name}_betas_{n}.pkl", "wb") as pkl_file:
        pickle.dump(betas, pkl_file)

    with open(SIM_DIR / f"{setting_name}_C_{n}.pkl", "wb") as pkl_file:
        pickle.dump(C, pkl_file)

    with open(SIM_DIR / f"{setting_name}_C_true_{n}.pkl", "wb") as pkl_file:
        pickle.dump(C_true, pkl_file)

    print(f"Finished study for n={n}; wrote pkl files to {SIM_DIR}.")


for n in SAMPLE_SIZES:
    run_study(n)
