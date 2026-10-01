#%%
import os
import pickle

import numpy as np
from joblib import Parallel, delayed

from ondil.methods import ElasticNetPath

from online_copula_experiments.application.const_and_helper.constants import (
    DISTRIBUTIONS,
    EQUATIONS,
    INFORMATION_CRITERION,
    FIT_INTERCEPT,
    REG_INTERCEPT,
    DEBUG,
    SCHEME,
    SCHEME_DIR,
    freq,
    f,
)
from online_copula_experiments.application.const_and_helper.data import (
    N_TRAIN,
    N_TEST,
    NEW_MONTH,
    X_numpy,
    y_numpy,
)

# %% ###########################################################################
# MAIN STUDY - Run the model batch vs online
################################################################################

METHODS = [ElasticNetPath(alpha=0)]  # Don't run elasticnet here in this online setting
# loop over up to 14 different datasets
regions = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P"]

regions = ["Scotland", "Northern", "Midlands", "London", "Southern"]

# Every per-step store a region fit produces. Each is a dict keyed by
# (step, region_number, forget); collecting them by name keeps the region
# workers, the forget loop and the save block free of positional plumbing.
STORES = (
    "predictions",       # predicted distribution parameters of row n
    "theta_in_sample",   # parameters at the rows the model was trained on
    "theta_out_sample",  # parameters of row n (same object as predictions)
    "cdf",               # frozen predictive marginal of row n
    "u_in_sample",       # in-sample PIT, the pseudo-obs the copula is fitted on
    "u_out_sample",      # one-step-ahead PIT of row n, for calibration
    "y_actual",          # realized y of row n, paired with u_out_sample
    "theta_one_step",    # parameters before the update, for forecast-vs-actual
    "marginal_loglik",   # log f_j(y_n | x_n) under the predictive marginal
    "timings",
)

FORGET_GRID = [0]
# The 5 macro-region marginal fits are independent, so run them in parallel.
# Use the threading backend (not loky): the regions share the already-loaded
# design matrices, so a process-based backend would re-import data.py and reload
# the multi-GB Xy_data_macro.pkl in every worker. The per-region work is heavy
# ondil/numba/numpy that releases the GIL, so threads still parallelise well.
N_JOBS_REGIONS = min(len(regions), os.cpu_count() or 1)


def _fit_one_region(rn, r, forget_lr):
    import time
    from ondil.estimators import OnlineDistributionalRegression

    TO_SCALE = ~(X_numpy[r].columns.str.contains("binary")
                 | X_numpy[r].columns.str.contains("spline"))
    # Materialize this region's arrays once (previously re-created on every
    # timestep via X_numpy[r].to_numpy(), churning ~0.5 GB per iteration).
    X_r = X_numpy[r].to_numpy()
    y_r = y_numpy[r].to_numpy()

    out = {name: {} for name in STORES}
    models_r = {}

    for d, (d_name, dist) in enumerate(DISTRIBUTIONS.items()):
        for method in METHODS:
            for p in range(1, 2):

                for i, n in zip(
                    range(0, N_TEST, freq),
                    range(N_TRAIN, N_TRAIN + N_TEST, freq),
                    strict=True,
                ):
                    key = ("online", d_name, method, p, forget_lr, f, freq, r)
                    store_key = (i, rn, forget_lr)

                    # ===========================================================
                    # 1) BRING THE MODEL TO "TRAINED THROUGH n-1"
                    #    repeated_batch  : refit monthly, frozen within month
                    #    repeated_online : refit monthly, updated within month
                    #    online          : fit once, updated from then on
                    # ===========================================================
                    refit = i == 0 or (NEW_MONTH[i] == 1 and SCHEME != "online")
                    if refit:
                        models_r[key] = OnlineDistributionalRegression(
                            distribution=dist,
                            equation=EQUATIONS[d_name][p],
                            method=method,
                            fit_intercept=FIT_INTERCEPT,
                            regularize_intercept=REG_INTERCEPT,
                            forget=forget_lr,
                            debug=DEBUG,
                            verbose=0,
                            ic=INFORMATION_CRITERION,
                            scale_inputs=TO_SCALE,
                        )
                        start = time.time()
                        X_train = X_r[:n, :]       # rows 0..n-1
                        y_train = y_r[:n]
                        models_r[key].fit(X=X_train, y=y_train)
                        stop = time.time()
                    else:
                        start = time.time()
                        X_train = X_r[n - 1, :].reshape(1, -1)  # previous target row n-1
                        y_train = y_r[n - 1].reshape(1, -1)
                        # repeated_batch keeps the model frozen between refits;
                        # the row is still needed below for the in-sample PIT.
                        if SCHEME != "repeated_batch":
                            models_r[key].update(X=X_train, y=y_train)
                        stop = time.time()

                    out["timings"][store_key] = stop - start

                    # ===========================================================
                    # 2) ONE-STEP-AHEAD FORECAST OF ROW n
                    #    model is through n-1, row n not yet seen -> no look-ahead
                    # ===========================================================
                    X_test = X_r[n, :].reshape(1, -1)
                    y_test = y_r[n].reshape(1, -1)

                    theta_out = models_r[key].predict_distribution_parameters(X=X_test)
                    dist_out  = dist.scipy_dist(**dist.theta_to_scipy_params(theta_out))

                    out["predictions"][store_key]      = theta_out
                    out["theta_out_sample"][store_key] = theta_out
                    out["cdf"][store_key]              = dist_out                     # predictive marginal for row n
                    out["u_out_sample"][store_key]     = dist_out.cdf(y_test).ravel()  # out-of-sample PIT of row n
                    out["y_actual"][store_key]         = y_test.ravel()
                    out["theta_one_step"][store_key]   = theta_out
                    # The copula stage scores its own density alone. Summed over
                    # regions this is the marginal half of the Sklar
                    # decomposition, and adding it turns that into the log score
                    # of the full predictive density. Evaluated here, where the
                    # predictive marginal and the realization are both in hand.
                    out["marginal_loglik"][store_key] = float(
                        dist_out.logpdf(y_test).ravel()[0]
                    )

                    # ===========================================================
                    # 3) IN-SAMPLE PIT (the pseudo-obs the copula is fitted on)
                    #    batch  : PITs of all training rows 0..n-1
                    #    online : PIT of the just-ingested row n-1
                    # ===========================================================
                    theta_in = models_r[key].predict_distribution_parameters(X=X_train)
                    dist_in  = dist.scipy_dist(**dist.theta_to_scipy_params(theta_in))
                    out["theta_in_sample"][store_key] = theta_in
                    out["u_in_sample"][store_key]     = dist_in.cdf(y_train)

    return out


def run_one_forget(forget_lr):
    region_results = Parallel(n_jobs=N_JOBS_REGIONS, backend="threading")(
        delayed(_fit_one_region)(rn, r, forget_lr) for rn, r in enumerate(regions)
    )
    merged = {name: {} for name in STORES}
    for region_out in region_results:
        for name, store in region_out.items():
            merged[name].update(store)
    return merged


# --- run over the forget grid (single value); regions run in parallel inside ---
results = {name: {} for name in STORES}
for forget_lr in FORGET_GRID:
    for name, store in run_one_forget(forget_lr).items():
        results[name].update(store)


# Everything below is written to intermediate_application/<scheme>/, so the
# three schemes never share a file name and the copula stage reads back exactly
# the marginals that were estimated the same way.
SCHEME_DIR.mkdir(parents=True, exist_ok=True)
print(f"Updating scheme: {SCHEME}  ->  {SCHEME_DIR}")

PICKLE_OUTPUTS = {
    "cdf_all_macro_forget": "cdf",                    # predictive marginals, read by the copula stage
    "u_in_sample_macro_forget": "u_in_sample",        # pseudo-observations for vine copula fitting
    "u_out_sample_macro_forget": "u_out_sample",      # one-step-ahead PITs for calibration diagnostics
    "y_actual_macro_forget": "y_actual",              # realizations paired with the forecasts
    "theta_one_step_macro_forget": "theta_one_step",  # parameters for forecast-vs-actual plots
}
for stem, store in PICKLE_OUTPUTS.items():
    with open(SCHEME_DIR / f"{stem}.pkl", "wb") as pkl_file:
        pickle.dump(results[store], pkl_file)

# Marginal contribution to the log score: sum_j log f_j(y_ij | x_i) per step.
# 03_plots.py adds it to the copula log score to obtain the log score of the
# full predictive density, which is the object the ES, the VS and the DSS also
# grade. Each scheme has its own marginals, so this term is what makes the
# schemes comparable to one another.
_forget = FORGET_GRID[0]
_ll = results["marginal_loglik"]
_steps = sorted({step for (step, _rn, _fg) in _ll})
marginal_ll = np.array(
    [sum(_ll[(step, rn, _forget)] for rn in range(len(regions))) for step in _steps]
)
np.save(SCHEME_DIR / "marginal_ll.npy", marginal_ll)
print(
    f"marginal_ll.npy: n={len(marginal_ll)}  "
    f"mean sum_j log f_j = {marginal_ll.mean():.4f}  "
    f"-> adds {-marginal_ll.mean():.4f} to the negative log score"
)
