import numpy as np

SEED = 123
np.random.seed(SEED)


# %% ###########################################################################
# MAIN STUDY - Run the model batch vs online
################################################################################
from online_copula_experiments.application.const_and_helper.evaluation import (  
    dawid_sebastiani_scoore,
    energy_score_fast,
    gaussian_copula_logpdf,
)
from online_copula_experiments.application.const_and_helper.helpers import (
    _to_scalar,
    expand_3x3_object_matrix,
)

#%%
import time
import copy
import pickle
import os
from pathlib import Path
import gc

import pandas as pd
import scipy.stats as st
import scoringrules as sr
from joblib import Parallel, delayed

import ondil
print("ondil loaded from:", ondil.__file__)

from online_copula_experiments.application.const_and_helper.distribution import (  # noqa: E501
    OnlineGaussianCopula,
)
from online_copula_experiments.application import BASE_PATH
from ondil.methods import ElasticNetPath

from online_copula_experiments.application.const_and_helper.constants import (
    SCHEME,
    SCHEME_DIR,
    freq,
    f,
)
from online_copula_experiments.application.const_and_helper.data import (
    N_TRAIN,
    N_TEST,
    NEW_MONTH,
    X_numpy_copula,
    y_numpy,
    TO_SCALE_COP,
)
from online_copula_experiments.application.const_and_helper.copula_equations import (
    edge_covariate_equation,
)

# This script covers the two schemes that re-fit monthly. The fully online
# scheme never batch-fits after the training sample and is run by
# 02_vine_copula_full_online.py instead.
if SCHEME == "online":
    print(f"SCHEME={SCHEME}: handled by 02_vine_copula_full_online.py, nothing to do.")
    raise SystemExit(0)

# Load pseudo-observations u from disk, produced by the marginal stage under the
# same scheme -- they live in that scheme's folder, so a mismatch is impossible.
with open(SCHEME_DIR / "cdf_all_macro_forget.pkl", "rb") as pkl_file:
    cdf = pickle.load(pkl_file)

with open(SCHEME_DIR / "u_in_sample_macro_forget.pkl", "rb") as pkl_file:
    u_in_sample = pickle.load(pkl_file)

with open(SCHEME_DIR / "u_out_sample_macro_forget.pkl", "rb") as pkl_file:
    u_out_sample = pickle.load(pkl_file)

regions_ordered = ["Scotland", "Northern", "Midlands", "London", "Southern"]

import vinecopulas
print("VineCopulas loaded from:", vinecopulas.__file__)
from vinecopulas.bivariate import *
from vinecopulas.vinecopula import *

a = np.array([
    [1, 2, 3, 4, 4],
    [2, 3, 4, 3, np.nan],
    [3, 4, 2, np.nan, np.nan],
    [4, 1, np.nan, np.nan, np.nan],
    [0, np.nan, np.nan, np.nan, np.nan]
])

y_array = np.column_stack([
    y_numpy[r].to_numpy() for r in regions_ordered
])

copula = OnlineGaussianCopula()

# =============================================================================
# PREDEFINED SETTINGS
# =============================================================================
SETTINGS = {
    
    "gaussian_all": {
        "X_numpy_copula": X_numpy_copula,
        "method_tree_1": ElasticNetPath(alpha =0),
        "method_tree_2plus": "ols",
        "equation_tree_1": edge_covariate_equation,
        "equation_tree_2plus": {0: {0: "intercept"}},
        "scale_inputs_tree_1": TO_SCALE_COP,
        "scale_inputs_tree_2plus": False,
        "fit_intercept": True,
        "cop_list": [1],
    },
    

}

SETTINGS = {
    
    "gaussian_intercept": {
        "X_numpy_copula": X_numpy_copula,
        "method_tree_1": "ols",
        "method_tree_2plus": "ols",
        "equation_tree_1": {0: {0: "intercept"}},
        "equation_tree_2plus": {0: {0: "intercept"}},
        "scale_inputs_tree_1": False,
        "scale_inputs_tree_2plus": False,
        "fit_intercept": True,
        "cop_list": [1],
    },

    "all_intercept": {
        "X_numpy_copula": X_numpy_copula,
        "method_tree_1": "ols",
        "method_tree_2plus": "ols",
        "equation_tree_1": {0: {0: "intercept"}},
        "equation_tree_2plus": {0: {0: "intercept"}},
        "scale_inputs_tree_1": False,
        "scale_inputs_tree_2plus": False,
        "fit_intercept": True,
        "cop_list": [1, 31, 34, 41, 44],
    },

    "all_all": {
        "X_numpy_copula": X_numpy_copula,
        "method_tree_1": ElasticNetPath(alpha =0),
        "method_tree_2plus": "ols",
        "equation_tree_1": edge_covariate_equation,
        "equation_tree_2plus": {0: {0: "intercept"}},
        "scale_inputs_tree_1": TO_SCALE_COP,
        "scale_inputs_tree_2plus": False,
        "fit_intercept": True,
        "cop_list": [1, 31, 34, 41, 44],
    },
    

}





# The copula follows the same scheme as the marginals, so there is exactly one
# to run per invocation.
MODES = [SCHEME]

# --- 1) Build the (i,n) grid once ---
i_grid = list(range(0, N_TEST, freq))
n_grid = list(range(N_TRAIN, N_TRAIN + N_TEST, freq))
assert len(i_grid) == len(n_grid)

# --- 2) Month starts in i-space (test index). Ensure 0 is included ---
month_starts = [0] + [i for i in i_grid[1:] if NEW_MONTH[i] == 1]
month_ends = month_starts[1:] + [N_TEST]
#month_ends_minus_one = [x - 1 for x in month_ends]


# --- 3) Worker that runs ONE month chunk sequentially ---
def run_one_month(
    f,
    freq,
    i_start,
    i_end,
    setting_name,
    setting,
    mode,
    show_progress=False,
    max_iters=None,
    s=1000,
    out_dir=None,
    seed=SEED,
):
    
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    
    n_m = {}
    es_m, vs05_m, vs10_m, dss_m, logpdf_m = {}, {}, {}, {}, {}
    es_copula_m, vs05_copula_m, vs10_copula_m, dss_copula_m, logpdf_copula_m = {}, {}, {}, {}, {}

    # Diagnostics over time
    beta_m = {}
    copula_family_m = {}
    pred_mat_m = {}  # ADDED
    y_pred_m = {}       # ADDED

    # Timing
    fit_times = []
    update_times = []

    X_copula_local = setting["X_numpy_copula"]
    method_tree_1 = setting["method_tree_1"]
    method_tree_2plus = setting["method_tree_2plus"]
    equation_tree_1 = setting["equation_tree_1"]
    equation_tree_2plus = setting["equation_tree_2plus"]
    scale_inputs_tree_1 = setting["scale_inputs_tree_1"]
    scale_inputs_tree_2plus = setting["scale_inputs_tree_2plus"]
    fit_intercept_local = setting["fit_intercept"]
    cop_list_local = setting["cop_list"]

    copula_local = copy.deepcopy(copula)
    Z = None
    iter_count = 0

    # fixed month-start objects for repeated batch
    P_fixed, C_fixed, E_fixed, B_fixed, cop_fixed = None, None, None, None, None
    loc_fixed, cov_fixed = None, None

    for i, n in zip(i_grid, n_grid, strict=True):
        if not (i_start <= i < i_end):
            continue

        iter_count += 1
        if max_iters is not None and iter_count > max_iters:
            break

        u = np.column_stack([
            u_in_sample[i, 0, 0],
            u_in_sample[i, 1, 0],
            u_in_sample[i, 2, 0],
            u_in_sample[i, 3, 0],
            u_in_sample[i, 4, 0],
        ])

        # =========================================================
        # FIT / UPDATE BLOCK
        # =========================================================
        if i == i_start or NEW_MONTH[i] == 1:
            t0 = time.perf_counter()
            Z, B_i, P_i, C_i, A = fit_vinecopstructure(
                u1=u,
                X_df=X_copula_local.iloc[:n, :],
                a=a,
                copsi=cop_list_local,
                printing=False,
                online=0,
                truncation=False,
                method_tree_1=method_tree_1,
                method_tree_2plus=method_tree_2plus,
                equation_tree_1=equation_tree_1,
                equation_tree_2plus=equation_tree_2plus,
                scale_inputs_tree_1=scale_inputs_tree_1,
                scale_inputs_tree_2plus=scale_inputs_tree_2plus,
                fit_intercept=fit_intercept_local,
            )
            fit_times.append(time.perf_counter() - t0)
            cop_i = copula_local.fit(u)
            E_store = Z

            # freeze month-start objects for repeated batch
            P_fixed = copy.deepcopy(P_i)
            C_fixed = copy.deepcopy(C_i)
            E_fixed = copy.deepcopy(E_store)
            B_fixed = copy.deepcopy(B_i)
            cop_fixed = copy.deepcopy(cop_i)
            loc_fixed = np.asarray(copula_local.loc).copy()
            cov_fixed = np.asarray(copula_local.cov).copy()

        else:
            if mode == "repeated_online":
                t0 = time.perf_counter()
                E_i, B_i, P_i, C_i, A = fit_vinecopstructure(
                    u1=u,
                    X_df=X_copula_local.iloc[n-1:n, :],
                    a=a,
                    copsi=[1, 31],
                    E=Z,
                    printing=True,
                    online=1,
                    method_tree_1=method_tree_1,
                    method_tree_2plus=method_tree_2plus,
                    fit_intercept=fit_intercept_local,
                )
                update_times.append(time.perf_counter() - t0)
                Z = E_i if E_i is not None else Z
                E_store = Z
                cop_i = copula_local.update(u)

            elif mode == "repeated_batch":
                P_i = P_fixed
                C_i = C_fixed
                E_store = E_fixed
                B_i = B_fixed
                cop_i = cop_fixed

            else:
                raise ValueError(f"Unknown mode: {mode}")

        # =========================================================
        # STORE DIAGNOSTICS
        # =========================================================
        beta_m[i] = copy.deepcopy(B_i)
        copula_family_m[i] = np.asarray(C_i, dtype=float).copy()

        # =========================================================
        # USE CURRENT GAUSSIAN COPULA PARAMETERS
        # =========================================================
        if mode == "repeated_online":
            loc_i = np.asarray(copula_local.loc).copy()
            cov_i = np.asarray(copula_local.cov).copy()
        else:
            loc_i = loc_fixed.copy()
            cov_i = cov_fixed.copy()

        # =========================================================
        # BUILD PREDICTION MATRIX
        # =========================================================
        mat = np.full_like(a, np.nan, dtype=float)
        non_nan_indices = np.argwhere(~np.isnan(C_i))

        X_test = X_copula_local.to_numpy()[n, :].reshape(1, -1)
       
        for ii, jj in non_nan_indices:
            model_obj = E_store[ii, jj]

            if isinstance(model_obj, (list, np.ndarray)):
                model_obj = model_obj[0]

            mat[ii, jj] = model_obj.predict(X=X_test)

        pred_i = copy.deepcopy(mat)
        mat_i = expand_3x3_object_matrix(pred_i, s)
        pred_mat_m[i] = pred_i.copy()  # ADDED

        # =========================================================
        # SAMPLE FROM VINE COPULA
        # =========================================================
        sa = sample_vinecop(
            a,
            mat_i,
            C_i,
            s,
        )

        # =========================================================
        # SAMPLE FROM GAUSSIAN COPULA BASELINE
        # =========================================================
        sc = st.norm().cdf(
            st.multivariate_normal(
                mean=loc_i,
                cov=cov_i
            ).rvs(size=s, random_state=rng)
        )

        # =========================================================
        # TRANSFORM TO DATA SPACE
        # =========================================================
        samples_sxd = np.zeros((s, 5))
        samples_copula_arr = np.zeros((s, 5))

        for rn, r in enumerate(regions_ordered):
            samples_sxd[:, rn] = cdf[i, rn, 0].ppf(sa[:, rn])
            samples_copula_arr[:, rn] = cdf[i, rn, 0].ppf(sc[:, rn])

        y_pred_m[i] = samples_sxd.copy()  # ADDED

        # =========================================================
        # TEST TARGET
        # =========================================================

        u_test = np.column_stack([
            u_out_sample[i, 0, 0],
            u_out_sample[i, 1, 0],
            u_out_sample[i, 2, 0],
            u_out_sample[i, 3, 0],
            u_out_sample[i, 4, 0],
        ])

        # =========================================================
        # SCORING
        # =========================================================
        n_m[i] = n

        
        try:
            es_m[i] = energy_score_fast(y_array[n, :], samples_sxd)
            vs05_m[i] = sr.variogram_score(y_array[n, :], samples_sxd, p=0.5)
            vs10_m[i] = sr.variogram_score(y_array[n, :], samples_sxd, p=1)
            dss_m[i] = dawid_sebastiani_scoore(y_array[n, :], samples_sxd)
            logpdf_m[i] = float(density_vinecop(u_test, a, pred_i, C_i))
            es_copula_m[i] = energy_score_fast(y_array[n, :], samples_copula_arr)
            vs05_copula_m[i] = sr.variogram_score(y_array[n, :], samples_copula_arr, p=0.5)
            vs10_copula_m[i] = sr.variogram_score(y_array[n, :], samples_copula_arr, p=1)
            dss_copula_m[i] = dawid_sebastiani_scoore(y_array[n, :], samples_copula_arr)
            logpdf_copula_m[i] = float(gaussian_copula_logpdf(u_test, cov_i))
        except np.linalg.LinAlgError:
            es_m[i] = np.nan
            vs05_m[i] = np.nan
            vs10_m[i] = np.nan
            dss_m[i] = np.nan
            logpdf_m[i] = np.nan
            es_copula_m[i] = np.nan
            vs05_copula_m[i] = np.nan
            vs10_copula_m[i] = np.nan
            dss_copula_m[i] = np.nan
            logpdf_copula_m[i] = np.nan

        # local cleanup inside loop
        del pred_i, mat_i, sa, sc, samples_sxd, samples_copula_arr
        gc.collect()

    # =========================================================
    # SAVE SCORES + DIAGNOSTICS
    # =========================================================
    result = {
        "setting_name": setting_name,
        "mode": mode,
        "n_m": n_m,
        "es_m": es_m,
        "vs05_m": vs05_m,
        "vs10_m": vs10_m,
        "dss_m": dss_m,
        "logpdf_m": logpdf_m,
        "es_copula_m": es_copula_m,
        "vs05_copula_m": vs05_copula_m,
        "vs10_copula_m": vs10_copula_m,
        "dss_copula_m": dss_copula_m,
        "logpdf_copula_m": logpdf_copula_m,

        # Diagnostics over time
        "beta_m": beta_m,
        "copula_family_m": copula_family_m,
        "pred_mat_m": pred_mat_m,  # ADDED
        "y_pred_m": y_pred_m,            # ADDED

        # Timing
        "fit_times": fit_times,
        "update_times": update_times,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"month_{setting_name}_{mode}_{i_start}_{i_end}.pkl"

    with open(out_path, "wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)

    del result
    gc.collect()

    return str(out_path)


# result_*.pkl go next to the marginals they were fitted on, in this scheme's
# folder. 03_plots.py globs those folders and takes the scheme from the folder
# name, so the layout is what identifies a result.
TEMP_DIR = SCHEME_DIR / "tmp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

FINAL_DIR = SCHEME_DIR
FINAL_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# MAIN LOOP OVER SETTINGS x MODES
# =============================================================================
all_results = {}
summary_rows = []

start = time.time()

for setting_name, setting in SETTINGS.items():
    for mode in MODES:
        print(f"Running setting={setting_name}, mode={mode}")

        n_store = {}
        es = {}
        vs05 = {}
        vs10 = {}
        dss = {}
        logpdf = {}
        es_copula = {}
        vs05_copula = {}
        vs10_copula = {}
        dss_copula = {}
        logpdf_copula = {}

        # Diagnostics over time
        beta_store = {}
        copula_family_store = {}
        pred_mat_store = {}  # ADDED
        y_pred_store = {}       # ADDED

        # Timing
        all_fit_times = []
        all_update_times = []

        result_files = Parallel(n_jobs=12, backend="loky")(
            delayed(run_one_month)(
                f=f,
                freq=freq,
                i_start=i_s,
                i_end=i_e,
                setting_name=setting_name,
                setting=setting,
                mode=mode,
                out_dir=TEMP_DIR,
                seed = SEED,
            )
            for i_s, i_e in zip(month_starts, month_ends)
        )

        for result_file in result_files:
            with open(result_file, "rb") as fh:
                res = pickle.load(fh)

            n_store.update(res["n_m"])
            es.update(res["es_m"])
            vs05.update(res["vs05_m"])
            vs10.update(res["vs10_m"])
            dss.update(res["dss_m"])
            logpdf.update(res["logpdf_m"])
            es_copula.update(res["es_copula_m"])
            vs05_copula.update(res["vs05_copula_m"])
            vs10_copula.update(res["vs10_copula_m"])
            dss_copula.update(res["dss_copula_m"])
            logpdf_copula.update(res["logpdf_copula_m"])

            # Diagnostics over time
            beta_store.update(res["beta_m"])
            copula_family_store.update(res["copula_family_m"])
            pred_mat_store.update(res["pred_mat_m"])  # ADDED
            y_pred_store.update(res["y_pred_m"])            # ADDED

            # Timing
            all_fit_times.extend(res["fit_times"])
            all_update_times.extend(res["update_times"])

            del res
            try:
                os.remove(result_file)
            except OSError:
                pass
            gc.collect()

       
        means = {
            "es": np.nanmean(list(es.values())),
            "vs05": np.nanmean(list(vs05.values())),
            "vs10": np.nanmean(list(vs10.values())),
            "dss": np.nanmean(list(dss.values())),
            "logscore": -np.mean(np.log(np.maximum(list(logpdf.values()), 1e-300))),
            "es_copula": np.nanmean(list(es_copula.values())),
            "vs05_copula": np.nanmean(list(vs05_copula.values())),
            "vs10_copula": np.nanmean(list(vs10_copula.values())),
            "dss_copula": np.nanmean(list(dss_copula.values())),
            "logscore_copula": -np.mean(list(logpdf_copula.values())),
        }

        idx_sorted = sorted(es.keys())
        df_scores = pd.DataFrame({
            "i": idx_sorted,
            "n": [n_store[i] for i in idx_sorted],
            "es": [es[i] for i in idx_sorted],
            "vs05": [vs05[i] for i in idx_sorted],
            "vs10": [vs10[i] for i in idx_sorted],
            "dss": [dss[i] for i in idx_sorted],
            "logpdf": [logpdf[i] for i in idx_sorted],
            "es_copula": [es_copula[i] for i in idx_sorted],
            "vs05_copula": [vs05_copula[i] for i in idx_sorted],
            "vs10_copula": [vs10_copula[i] for i in idx_sorted],
            "dss_copula": [dss_copula[i] for i in idx_sorted],
            "logpdf_copula": [logpdf_copula[i] for i in idx_sorted],
        })

        final_result = {
            "setting_name": setting_name,
            "setting": setting,
            "mode": mode,
            "means": means,
            "df_scores": df_scores,

            # Diagnostics over time
            "beta_store": beta_store,
            "copula_family_store": copula_family_store,
            "pred_mat_store": pred_mat_store,  # ADDED
            "y_pred_store": y_pred_store,            # ADDED
            "fit_times": all_fit_times,      # ADD
            "update_times": all_update_times, # ADD
        }

        final_path = FINAL_DIR / f"result_{setting_name}_{mode}.pkl"
        with open(final_path, "wb") as fh:
            pickle.dump(final_result, fh, protocol=pickle.HIGHEST_PROTOCOL)

        all_results[(setting_name, mode)] = {
            "means": means,
            "result_file": str(final_path),
        }

        summary_rows.append({
            "setting_name": setting_name,
            "mode": mode,
            "method_tree_1": setting["method_tree_1"],
            "method_tree_2plus": setting["method_tree_2plus"],
            "fit_intercept": setting["fit_intercept"],
            **means,
        })

        del n_store, es, vs05, vs10, dss, logpdf
        del es_copula, vs05_copula, vs10_copula, dss_copula, logpdf_copula
        #del beta_store, copula_family_store, kendall_tau_store, y_pred_store
        del df_scores, final_result, result_files
        gc.collect()

stop = time.time()
print(f"Fitting time (all settings x modes): {stop - start:.4f} seconds")


