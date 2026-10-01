import time
import os
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import copy
from time import perf_counter
import traceback

import matplotlib.pyplot as plt

import ondil

from online_copula_experiments.application.const_and_helper.distribution import OnlineGaussianCopula
from online_copula_experiments.application.const_and_helper.evaluation import gaussian_copula_logpdf
from online_copula_experiments.application.const_and_helper.simulation import (
    make_X,
    make_beta_t,
    generate_true_copula_matrix,
)
print("ondil loaded from:", ondil.__file__)
from ondil.methods import ElasticNetPath, LassoPath

import vinecopulas
print("VineCopulas loaded from:", vinecopulas.__file__)
from vinecopulas.bivariate import *
from vinecopulas.vinecopula import *

from online_copula_experiments.application import BASE_PATH

from joblib import Parallel, delayed

np.set_printoptions(precision=3, suppress=True)


# =============================================================================
# ONE-STEP-AHEAD INDEXING
# =============================================================================
# At loop step i (n = N_TRAIN + i) we forecast the observation at index n.
#   * batch model: trained on y_numpy[:n]  -> rows 0..n-1   (pure training at i=0)
#   * online model: has ingested rows 0..n-1; the update at step i folds in the
#     PREVIOUS block y_numpy[n-freq:n] (row n-1 for freq=1), never the target.
# Both modes therefore train on rows 0..n-1 and forecast index n:
#   - genuine one-step-ahead,
#   - identical information sets across modes,
#   - the target observation n is never used to train the forecast scored on it.
# =============================================================================


# =============================================================================
# SETTINGS
# =============================================================================

BASE_PATH = Path(BASE_PATH)
# Simulation intermediate results live under BASE_PATH/intermediate_simulation.
SIM_DIR = BASE_PATH / "intermediate_simulation"
SIM_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(seed=42)

a = np.array([
    [1, 2, 3, 4, 4],
    [2, 3, 4, 3, np.nan],
    [3, 4, 2, np.nan, np.nan],
    [4, 1, np.nan, np.nan, np.nan],
    [0, np.nan, np.nan, np.nan, np.nan]
])

setting_name = "batch_vine"

# Scenario: "break" applies a structural break in the first first-tree coefficient
# at observation 750; "stable" keeps the coefficients constant. Passed as the first
# command-line argument (default "break"); each scenario is written to its own file.
import sys
SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "break"
assert SCENARIO in ("stable", "break"), f"unknown scenario {SCENARIO!r}"
if SCENARIO == "break":
    BREAK_POINTS, BREAK_INDICES, BREAK_VALUES = [750], [0], [-0.25]
    ALL_RUNS_FILE = "all_runs_50_break.pkl"
else:
    BREAK_POINTS = BREAK_INDICES = BREAK_VALUES = None
    ALL_RUNS_FILE = "all_runs_50.pkl"

N = 1000
N_TRAIN = 700
N_TEST = 300
Q = 20
p = 20

N_RUNS = 50
RUN_SEEDS = [1000 + r for r in range(N_RUNS)]

MAKE_PLOTS_FIRST_RUN = True
SAVE_PER_RUN_RESULTS = True

methods = ["ols", ElasticNetPath(alpha=0)]
forget = [1 / 50, 1 / 100, 1 / 200, 1 / 400, 0]
freq_list = [1]

margin_distribution = Normal()
copula = OnlineGaussianCopula()

# Copula families searched by the vine fits.
COPSI_BATCH = [1, 31, 34, 41, 44]   # full family set for the batch / repeated-batch fits
COPSI_ONLINE = [1, 31, 34, 41, 44]              # restricted set for the online one-step updates
cops = COPSI_BATCH                  # families used to draw the true data-generating vine

# Parallelism.
N_JOBS_BATCH = 50                          
N_JOBS_CONFIG = min(20, os.cpu_count() or 1)  # 20 independent (regime, freq, f, m) chains


# =============================================================================
# DATA GENERATION
# =============================================================================
# Shared DGP helpers (make_X, make_beta_t, generate_true_copula_matrix) live in
# const_and_helper.simulation and are imported at the top of the file.


# =============================================================================
# SINGLE DATASET RUN
# =============================================================================

def run_single_dataset(run_id, seed):
    print(f"\n{'='*80}")
    print(f"START RUN {run_id + 1}/{N_RUNS} | seed={seed}")
    print(f"{'='*80}")

    # -------------------------------------------------------------------------
    # Generate data
    # -------------------------------------------------------------------------
    X_numpy_raw = make_X(N, p=p, seed=seed)

    beta = np.array(
        [0.25, 0.25, -0.25, 0, 0, 0.25, -0.25, -0.25, 0, 0] + 10 * [0]
    )

    beta_t = make_beta_t(
        N,
        beta,
        break_points=BREAK_POINTS,
        break_indices=BREAK_INDICES,
        break_values=BREAK_VALUES,
    )

    C_true_d = generate_true_copula_matrix(a, cops, seed=seed + 100_000)

    if MAKE_PLOTS_FIRST_RUN and run_id == 0:
        plt.figure(figsize=(10, 4))
        plt.plot(beta_t[:, 1, 0])
        plt.xlabel("Index")
        plt.ylabel("Beta on shock")
        plt.title("Slightly time-varying coefficient")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(X_numpy_raw[:, 1])
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title("Second Column of X_numpy")
        plt.tight_layout()
        plt.show()

    y_numpy, P_true = simulate_vinecop(
        a, X_numpy_raw, beta_t, C_true_d, len(X_numpy_raw), return_P=True
    )

    if MAKE_PLOTS_FIRST_RUN and run_id == 0:
        fig, axes = plt.subplots(y_numpy.shape[1], y_numpy.shape[1], figsize=(12, 12))
        for i in range(y_numpy.shape[1]):
            for j in range(y_numpy.shape[1]):
                ax = axes[i, j]
                ax.scatter(y_numpy[:, j], y_numpy[:, i], alpha=0.3, s=1)
                if i == y_numpy.shape[1] - 1:
                    ax.set_xlabel(f"Col {j}")
                if j == 0:
                    ax.set_ylabel(f"Col {i}")
                ax.tick_params(labelsize=8)
        plt.tight_layout()
        plt.show()

    X_numpy = pd.DataFrame(X_numpy_raw, columns=[f"x_{i}" for i in range(X_numpy_raw.shape[1])])

    print("NaNs in X_numpy:", np.isnan(X_numpy).sum().sum())
    print("NaNs in y_numpy:", np.isnan(y_numpy).sum())

    # -------------------------------------------------------------------------
    # Batch estimation
    # -------------------------------------------------------------------------
    # forecast every test index 700..999 (n = N_TRAIN .. N_TRAIN+N_TEST-1)
    i_grid = list(range(0, N_TEST, 1))
    n_grid = list(range(N_TRAIN, N_TRAIN + N_TEST, 1))

    def fit_batch_one_if(i, n, f, m, method):
        try:
            t0 = perf_counter()

            Z, B_i, P_i, C_i, A = fit_vinecopstructure(
                u1=y_numpy[:n, :],
                X_df=X_numpy.iloc[:n, :],
                a=a,
                copsi=COPSI_BATCH,
                printing=False,
                online=0,
                truncation=False,
                forget=f,
                method_tree_1 = method,
                method_tree_2plus = method,
                fit_intercept = True,
                scale_inputs_tree_1 = False,
                scale_inputs_tree_2plus = False,
            )

            vine_elapsed = perf_counter() - t0

            return {
                "success": True,
                "i": i,
                "n": n,
                "f": f,
                "m": m,
                "method": str(method),
                "E": copy.deepcopy(Z),
                "C": copy.deepcopy(C_i),
                "B": copy.deepcopy(B_i),
                "vine_elapsed": vine_elapsed,
                "error_type": None,
                "error_message": None,
                "traceback": None,
            }

        except Exception as e:
            return {
                "success": False,
                "i": i,
                "n": n,
                "f": f,
                "m": m,
                "method": str(method),
                "E": None,
                "C": None,
                "B": None,
                "vine_elapsed": np.nan,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            }

    batch_E = {}
    batch_C = {}
    batch_B = {}
    batch_fit_time_vine = {}
    batch_errors = []

    batch_results = Parallel(n_jobs=N_JOBS_BATCH, backend="loky")(
        delayed(fit_batch_one_if)(i, n, f, m, method)
        for i, n in zip(i_grid, n_grid, strict=True)
        for f in forget
        for m, method in enumerate(methods)
    )

    for res in batch_results:
        if res["success"]:
            batch_E[res["i"], res["f"], res["m"]] = res["E"]
            batch_C[res["i"], res["f"], res["m"]] = res["C"]
            batch_B[res["i"], res["f"], res["m"]] = res["B"]
            batch_fit_time_vine[res["i"], res["f"], res["m"]] = res["vine_elapsed"]
        else:
            batch_errors.append({
                "stage": "batch",
                "run_id": run_id,
                "i": res["i"],
                "n": res["n"],
                "f": res["f"],
                "m": res["m"],
                "method": res["method"],
                "error_type": res["error_type"],
                "error_message": res["error_message"],
                "traceback": res["traceback"],
            })

    print(f"Successful batch fits: {len(batch_E)}")
    print(f"Failed batch fits: {len(batch_errors)}")

    if len(batch_errors) > 0:
        batch_errors_df = pd.DataFrame(batch_errors)
        print(batch_errors_df[["i", "n", "f", "m", "method", "error_type", "error_message"]].head(100))

    # -------------------------------------------------------------------------
    # Online / repeated-batch forecasting
    # -------------------------------------------------------------------------
    # One self-contained sequential chain per (regime, freq, f, m) configuration.
    # Within a chain the online updates are order-dependent (each update folds in
    # the previous block and carries state forward), so a chain cannot be split
    # across time steps -- but the 20 chains are mutually independent and run in
    # parallel. Only the vine and Gaussian log-densities at the realised
    # observation feed the downstream scores, so nothing else is stored.
    def run_one_config(regime, freq, f, m, method,
                       batch_E_fm, batch_C_fm, batch_B_fm, batch_tvine_fm):
        run = set(range(0, N_TEST, freq))

        copula_local = copy.deepcopy(copula)
        Z = None
        prev = {"C": None, "E": None}

        logpdf_c = {}
        logpdf_copula_c = {}
        fit_time_vine_c = 0.0
        fit_time_gauss_c = 0.0
        n_updates_c = 0
        errors = []

        def record(stage, i, n, exc):
            errors.append({
                "stage": stage,
                "run_id": run_id,
                "i": i,
                "n": n,
                "regime": regime,
                "freq": freq,
                "f": f,
                "m": m,
                "method": str(method),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            })

        for i, n in zip(i_grid, n_grid, strict=True):
            if i not in batch_E_fm:
                continue

            if i in run:
                if regime == 0 or i == 0:
                    # repeated-batch refit (and the shared online seed at i == 0):
                    # reuse the precomputed batch vine fit and refit the copula.
                    E_store = batch_E_fm[i]
                    C_i = batch_C_fm[i]
                    Z = E_store

                    t_g = perf_counter()
                    copula_local.fit(y_numpy[:n, :])
                    fit_time_gauss_c += perf_counter() - t_g

                    fit_time_vine_c += batch_tvine_fm[i]
                    n_updates_c += 1
                else:
                    # genuine one-step-ahead online update: ingest the PREVIOUS
                    # block (rows n-freq..n-1), never the target row n.
                    try:
                        t_v = perf_counter()
                        E_i, B_i, P_i, C_i, A = fit_vinecopstructure(
                            u1=y_numpy[n - freq:n, :],
                            X_df=X_numpy.iloc[n - freq:n, :],
                            a=a,
                            copsi=COPSI_ONLINE,
                            E=Z,
                            printing=False,
                            online=1,
                        )
                        fit_time_vine_c += perf_counter() - t_v

                        Z = E_i if E_i is not None else Z
                        E_store = Z

                        t_g = perf_counter()
                        copula_local.update(y_numpy[n - freq:n, :])
                        fit_time_gauss_c += perf_counter() - t_g

                        n_updates_c += 1

                    except Exception as e:
                        record("online", i, n, e)
                        C_i = prev["C"]
                        E_store = prev["E"]
            else:
                C_i = prev["C"]
                E_store = prev["E"]

            prev["C"] = C_i
            prev["E"] = E_store

            # realised observation we score against = index n (one-step-ahead)
            u_test = y_numpy[n, :].reshape(1, -1)
            X_test = X_numpy.to_numpy()[n, :].reshape(1, -1)

            # vine log-density: predict each edge copula parameter, then evaluate.
            try:
                mat = np.full_like(a, np.nan, dtype=float)
                for ii, jj in np.argwhere(~np.isnan(C_i)):
                    model_obj = E_store[ii, jj]
                    if isinstance(model_obj, (list, np.ndarray)):
                        model_obj = model_obj[0]
                    mat[ii, jj] = model_obj.predict(X=X_test)

                logpdf_c[i] = density_vinecop(u_test, a, mat, C_i)
            except Exception as e:
                logpdf_c[i] = np.nan
                record("vine_logpdf", i, n, e)

            # Gaussian copula log-density from the current online covariance.
            try:
                logpdf_copula_c[i] = gaussian_copula_logpdf(
                    u_test, np.asarray(copula_local.cov)
                )
            except Exception as e:
                logpdf_copula_c[i] = np.nan
                record("gaussian_logpdf", i, n, e)

        return {
            "regime": regime,
            "freq": freq,
            "f": f,
            "m": m,
            "logpdf": logpdf_c,
            "logpdf_copula": logpdf_copula_c,
            "fit_time_vine": fit_time_vine_c,
            "fit_time_gauss": fit_time_gauss_c,
            "n_updates": n_updates_c,
            "errors": errors,
        }

    start = time.time()

    logpdf = {}
    logpdf_copula = {}
    fit_time_vine = {}
    fit_time_gauss = {}
    n_updates = {}
    online_errors = []

    # Build the 20 independent chains, each handed only the batch slice it reads.
    config_tasks = []
    for m, method in enumerate(methods):
        for f in forget:
            for freq in freq_list:
                for regime in [0, 1]:
                    E_fm = {i: batch_E[i, f, m] for i in i_grid if (i, f, m) in batch_E}
                    C_fm = {i: batch_C[i, f, m] for i in i_grid if (i, f, m) in batch_C}
                    B_fm = {i: batch_B[i, f, m] for i in i_grid if (i, f, m) in batch_B}
                    T_fm = {i: batch_fit_time_vine[i, f, m] for i in i_grid if (i, f, m) in batch_fit_time_vine}
                    config_tasks.append(
                        delayed(run_one_config)(regime, freq, f, m, method, E_fm, C_fm, B_fm, T_fm)
                    )

    results = Parallel(n_jobs=N_JOBS_CONFIG, backend="loky")(config_tasks)

    for r in results:
        regime, freq, f, m = r["regime"], r["freq"], r["f"], r["m"]
        for i, v in r["logpdf"].items():
            logpdf[(i, regime, freq, f, m)] = v
        for i, v in r["logpdf_copula"].items():
            logpdf_copula[(i, regime, freq, f, m)] = v
        fit_time_vine[(regime, freq, f, m)] = r["fit_time_vine"]
        fit_time_gauss[(regime, freq, f, m)] = r["fit_time_gauss"]
        n_updates[(regime, freq, f, m)] = r["n_updates"]
        online_errors.extend(r["errors"])

    stop = time.time()
    print(f"Fitting time (config-parallel): {stop - start:.4f} seconds")
    print(f"Total online errors: {len(online_errors)}")

    if len(online_errors) > 0:
        online_errors_df = pd.DataFrame(online_errors)
        print(
            online_errors_df[
                ["i", "n", "freq", "f", "m", "method", "error_type", "error_message"]
            ].head(100)
        )

    # -------------------------------------------------------------------------
    # Per-run score table
    # -------------------------------------------------------------------------
    rows = []

    for reg in [0, 1]:
        for est in [0, 1]:
            for freq in freq_list:
                for forget_val in forget:

                    arrays_model = [
                        logpdf[(i, est, freq, forget_val, reg)].ravel()
                        for i in range(N_TEST)
                        if (i, est, freq, forget_val, reg) in logpdf
                    ]

                    arrays_gauss = [
                        logpdf_copula[(i, est, freq, forget_val, reg)].ravel()
                        for i in range(N_TEST)
                        if (i, est, freq, forget_val, reg) in logpdf_copula
                    ]

                    if not arrays_model or not arrays_gauss:
                        continue

                    vals_model = np.concatenate(arrays_model)
                    vals_gauss = np.concatenate(arrays_gauss)

                    vals_model = vals_model[np.isfinite(vals_model)]
                    vals_gauss = vals_gauss[np.isfinite(vals_gauss)]

                    vals_model = vals_model[vals_model > 0]
                    vals_gauss = vals_gauss[vals_gauss > 0]

                    if len(vals_model) == 0 or len(vals_gauss) == 0:
                        continue

                    score_model = -np.mean(np.log(vals_model))
                    score_gauss = -np.mean(np.log(vals_gauss))

                    rows.append({
                        "run_id": run_id,
                        "regularization": reg,
                        "estimation": est,
                        "freq": freq,
                        "forget": forget_val,
                        "model": score_model,
                        "gaussian": score_gauss,
                        "diff": score_model - score_gauss,
                    })

    df_scores_run = pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # Per-run timing table
    # -------------------------------------------------------------------------
    def build_time_df(fit_time_dict, n_updates_dict):
        rows_time = []

        for reg in [0, 1]:
            for est in [0, 1]:
                for freq in freq_list:
                    for forget_val in forget:
                        key = (est, freq, forget_val, reg)

                        total_t = fit_time_dict.get(key, np.nan)
                        n_upd = n_updates_dict.get(key, np.nan)
                        avg_t = total_t / n_upd if pd.notna(total_t) and pd.notna(n_upd) and n_upd > 0 else np.nan

                        rows_time.append({
                            "run_id": run_id,
                            "regularization": reg,
                            "estimation": est,
                            "freq": freq,
                            "forget": forget_val,
                            "total_time": total_t,
                            "n_updates": n_upd,
                            "avg_time": avg_t,
                        })

        return pd.DataFrame(rows_time)

    df_time_vine_run = build_time_df(fit_time_vine, n_updates)
    df_time_gauss_run = build_time_df(fit_time_gauss, n_updates)

    run_result = {
        "run_id": run_id,
        "seed": seed,
        "df_scores_run": df_scores_run,
        "df_time_vine_run": df_time_vine_run,
        "df_time_gauss_run": df_time_gauss_run,
        "fit_time_vine": fit_time_vine,
        "fit_time_gauss": fit_time_gauss,
        "n_updates": n_updates,
        "online_errors": online_errors,
        "batch_errors": batch_errors,
        "logpdf": logpdf,
        "logpdf_copula": logpdf_copula,
    }

    if SAVE_PER_RUN_RESULTS:
        with open(SIM_DIR / f"run_{run_id:03d}_{SCENARIO}_results.pkl", "wb") as f:
            pickle.dump(run_result, f)

        pd.DataFrame(batch_errors).to_csv(SIM_DIR / f"run_{run_id:03d}_{SCENARIO}_batch_errors.csv", index=False)
        pd.DataFrame(online_errors).to_csv(SIM_DIR / f"run_{run_id:03d}_{SCENARIO}_online_errors.csv", index=False)

    return run_result


# =============================================================================
# RUN ALL DATASETS
# =============================================================================

all_runs = {}

for run_id, seed in enumerate(RUN_SEEDS):
    res = run_single_dataset(run_id=run_id, seed=seed)
    all_runs[run_id] = res

with open(SIM_DIR / ALL_RUNS_FILE, "wb") as f:
    pickle.dump(all_runs, f)

print(f"[{SCENARIO}] wrote {ALL_RUNS_FILE} and per-run files to {SIM_DIR}")
