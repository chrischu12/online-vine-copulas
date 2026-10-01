#%%

import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist
from online_copula_experiments.application import BASE_PATH
from online_copula_experiments.application.const_and_helper.constants import (
    APP_DIR,
    SCHEMES,
)
from ondil.links import GaussianParameterToKendallsTau, ClaytonParameterToKendallsTau, GumbelParameterToKendallsTau

_param_link = {
    1:  GaussianParameterToKendallsTau(),
    2:  GaussianParameterToKendallsTau(),
    31: ClaytonParameterToKendallsTau(), 32: ClaytonParameterToKendallsTau(),
    33: ClaytonParameterToKendallsTau(), 34: ClaytonParameterToKendallsTau(),
    41: GumbelParameterToKendallsTau(),  42: GumbelParameterToKendallsTau(),
    43: GumbelParameterToKendallsTau(),  44: GumbelParameterToKendallsTau(),
}

# =============================================================================
# LOAD ONE FOLDER PER UPDATING SCHEME
# =============================================================================
# Every scheme-specific artefact of a run -- marginals, PITs, copula results,
# score cache -- lives in intermediate_application/<scheme>/. The folder name is
# therefore what identifies a result, and no file name has to be parsed and no
# mode has to be filtered: a scheme folder holds one complete forecasting model
# per copula specification, with both stages estimated the same way.
SCHEME_DIRS = {s: APP_DIR / s for s in SCHEMES if (APP_DIR / s).is_dir()}
if not SCHEME_DIRS:
    raise FileNotFoundError(
        f"No scheme folder {SCHEMES} under {APP_DIR}. Run 00_main.py first."
    )
for _s, _d in SCHEME_DIRS.items():
    print(f"  reading {_s:16s} {_d}")

# ---------------------------------------------------------------------------
# Marginal contribution to the log score
# ---------------------------------------------------------------------------
# The stored "logscore" grades the vine copula density alone. Adding the
# marginal log-likelihood turns it into the log score of the full predictive
# density, which is what the ES, the VS and the DSS also measure. Each scheme
# has its own marginals, so this term is what makes the schemes comparable.
# Written by 02_marginals.py alongside the marginals it was computed from.
MARGINAL_NLL = {}
MARGINAL_NLL_SERIES = {}
for _scheme, _dir in SCHEME_DIRS.items():
    _f = _dir / "marginal_ll.npy"
    if _f.exists():
        _series = -np.load(_f)  # negative log-likelihood per step
        MARGINAL_NLL_SERIES[_scheme] = _series
        MARGINAL_NLL[_scheme] = float(_series.mean())
    else:
        print(f"  WARNING: {_f} missing; total log score unavailable for {_scheme}")
print(f"Marginal negative log-likelihoods: "
      + ", ".join(f"{m}={v:.4f}" for m, v in sorted(MARGINAL_NLL.items())))

# With TABLES_ONLY set, read the slim scores_*.pkl written by 02_cache_scores.py
# instead of the full result_*.pkl. They carry the scores and the update times
# but none of the per-step stores, so every table can be produced while the
# figures cannot. The script exits before the figure sections in that case.
TABLES_ONLY = os.environ.get("TABLES_ONLY", "").lower() in ("1", "true", "yes")
_PREFIX = "scores_" if TABLES_ONLY else "result_"
if TABLES_ONLY:
    print("TABLES_ONLY: reading the score cache, figures will be skipped")

all_results = {}

for scheme, scheme_dir in SCHEME_DIRS.items():
    for pkl_file in sorted(scheme_dir.glob(f"{_PREFIX}*.pkl")):
        with open(pkl_file, "rb") as fh:
            final_result = pickle.load(fh)
        # The folder is authoritative for the scheme, so results written before
        # the schemes were renamed still key correctly.
        all_results[(final_result["setting_name"], scheme)] = final_result

# =============================================================================
# SUMMARY TABLE  (console only — not a paper object; sanity-check of all means)
# =============================================================================
rows = []
for (setting_name, mode), res in all_results.items():
    row = {
        "setting_name": setting_name,
        "mode": mode,
        **res["means"],
    }
    rows.append(row)

mode_order_map = {"online": 0, "repeated_online": 1, "repeated_batch": 2}

df_summary = (
    pd.DataFrame(rows)
    .assign(mode_order=lambda df: df["mode"].map(mode_order_map))
    .sort_values(["setting_name", "mode_order"])
    .drop(columns="mode_order")
    .reset_index(drop=True)
)

print(df_summary.to_string(index=False, float_format="%.6f"))

# =============================================================================
# HELPERS
# =============================================================================
def _to_scalar(x):
    if x is None:
        return np.nan

    while isinstance(x, list) and len(x) == 1:
        x = x[0]

    if isinstance(x, np.ndarray):
        if x.size == 0:
            return np.nan
        return float(np.ravel(x)[0])

    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    return np.nan


def dm_test_simple(d):
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]

    T = len(d)
    d_mean = np.mean(d)
    s_d = np.std(d, ddof=1)

    dm_stat = d_mean / (s_d / np.sqrt(T))
    p_value = 2 * (1 - t.cdf(np.abs(dm_stat), df=T - 1))

    return dm_stat, p_value, d_mean, T


def fmt_num(x):
    if pd.isna(x):
        return ""
    return f"{x:.4f}"


def write_tex_file(path, content):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def extract_beta_vec(beta_edge):
    try:
        if isinstance(beta_edge, list):
            beta_edge = beta_edge[0]
        beta_vec = np.asarray(beta_edge[0][0], dtype=float).ravel()
        return beta_vec
    except Exception:
        return np.array([], dtype=float)


def eta_to_tau(family_code, eta):
    """Convert eta to Kendall's tau via the family-specific link."""
    try:
        fc  = int(family_code)
        eta = float(eta)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(eta):
        return np.nan
    if fc == 0:                        # independence
        return 0.0
    elif fc in (1, 2):                 # Gaussian / Student-t  (FisherZ link)
        rho = np.tanh(eta / 2.0)
        return float((2.0 / np.pi) * np.arcsin(np.clip(rho, -1 + 1e-9, 1 - 1e-9)))
    elif fc in (31, 32, 33, 34):       # Clayton rotations  (Log link)
        theta = np.exp(eta)
        tau   = theta / (2.0 + theta)
        return float(-tau if fc in (32, 34) else tau)
    elif fc in (41, 42, 43, 44):       # Gumbel rotations  (GumbelLink: exp(eta)+1)
        theta = np.exp(eta) + 1.0
        tau   = 1.0 - 1.0 / theta
        return float(-tau if fc in (42, 44) else tau)
    else:
        return np.nan



# =============================================================================
# SPECIFICATION LABELS
# =============================================================================
spec_order = [
    ("gaussian_intercept", "M+GC"),
    ("all_intercept", "M+VC"),
    ("all_all", "M+CVC"),
]


mode_label = {
    "online": "Online",
    "repeated_online": "Repeated Online",
    "repeated_batch": "Repeated Batch",
}

# Restrict to the schemes actually present, so a partial run -- one scheme
# folder still missing -- produces the same objects without empty rows or
# missing-key errors.
_PRESENT_MODES = {mode for (_setting, mode) in all_results}
mode_order = [m for m in ["online", "repeated_online", "repeated_batch"] if m in _PRESENT_MODES]
print(f"Updating schemes found: {mode_order}")



# =============================================================================
# PAPER TABLE 4  (tab:forecast_eval)  ->  input/tab_forecast_eval.tex
# "Forecast evaluation of copula-based models under different updating schemes."
# =============================================================================
forecast_rows = []

for setting_key, setting_label in spec_order:
    for mode in mode_order:
        tmp = df_summary[
            (df_summary["setting_name"] == setting_key) &
            (df_summary["mode"] == mode)
        ]
        if len(tmp) == 0:
            continue

        r = tmp.iloc[0]
        forecast_rows.append({
            "Specification": setting_label,
            "Mode": mode_label[mode],
            "ES": r["es"],
            "VS05": r["vs05"],
            "VS10": r["vs10"],
            "DSS": r["dss"],
            "LogScore": r["logscore"],
            "LogScoreTotal": r["logscore"] + MARGINAL_NLL.get(mode, np.nan),
        })

df_forecast = pd.DataFrame(forecast_rows)

# Figures and tables are written to BASE_PATH/results so a server run is
# self-contained (copy them into the paper tree afterwards).
RESULTS_DIR = Path(BASE_PATH) / "results"
FIG_DIR = RESULTS_DIR / "figures"
TAB_DIR = RESULTS_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# Every row is a complete forecasting model, with both stages under the same
# scheme, so all four scores grade the same object and LS is the log score of
# the full predictive density rather than of the copula alone.
_cols_spec = "cccc"
_hdr = r"Model & Scheme & ES & VS$_{0.5}$ & DSS & LS \\"
metric_cols = ["ES", "VS05", "DSS", "LogScoreTotal"]

table1_lines = []
table1_lines.append(r"\begingroup")
table1_lines.append(r"\setlength{\tabcolsep}{9pt}")
table1_lines.append(r"\begin{tabular}{ll@{\hspace{1.0em}}" + _cols_spec + "}")
table1_lines.append(r"\toprule")
table1_lines.append(_hdr)
table1_lines.append(r"\midrule")
min_idx = {col: df_forecast[col].idxmin() for col in metric_cols}

n_modes = len(mode_order)

group_min_idx = {}
for g in range(len(spec_order)):
    group_idx = df_forecast.index[g * n_modes : (g + 1) * n_modes]
    for col in metric_cols:
        group_min_idx[(g, col)] = df_forecast.loc[group_idx, col].idxmin()

for i, (idx, r) in enumerate(df_forecast.iterrows()):
    if i > 0 and i % n_modes == 0:
        table1_lines.append(r"\addlinespace[6pt]")

    g = i // n_modes
    cells = []

    for col in metric_cols:
        s = fmt_num(r[col])

        if idx == min_idx[col]:
            s = r"\underline{" + s + r"}"

        if idx == group_min_idx[(g, col)]:
            s = r"\cellcolor{green!25}" + s

        cells.append(s)

    # The model label is printed once per block; the remaining rows of the block
    # leave the cell empty.
    _spec_cell = f"{r['Specification']} " if i % n_modes == 0 else ""
    table1_lines.append(
        f"{_spec_cell}& {r['Mode']} & "
        + " & ".join(cells) + r" \\"
    )

table1_lines.append(r"\bottomrule")
table1_lines.append(r"\end{tabular}")
table1_lines.append(r"\endgroup")

write_tex_file(TAB_DIR / f"tab_forecast_eval.tex", "\n".join(table1_lines))



# =============================================================================
# DM TEST HELPERS
# =============================================================================
def _dm_test_metric(d):
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    T = len(d)
    if T < 2:
        return np.nan, np.nan, np.nan, T
    d_mean = np.mean(d)
    s_d = np.std(d, ddof=1)
    if s_d == 0:
        return np.nan, np.nan, d_mean, T
    dm_stat = d_mean / (s_d / np.sqrt(T))
    p_val = 2 * (1 - t_dist.cdf(np.abs(dm_stat), df=T - 1))
    return dm_stat, p_val, d_mean, T


def _get_scores_df(res, mode=None):
    """Per-step scores. With `mode` given, the log score is the one of the full
    predictive density: the copula term plus that scheme's marginal
    log-likelihood, step by step. Needed whenever two rows are compared that do
    not share their marginal models."""
    df = res["df_scores"].sort_values("i").copy()
    df["logpdf"] = df["logpdf"].map(_to_scalar)
    df["logscore"] = -np.log(np.maximum(df["logpdf"], 1e-300))
    if mode is not None:
        series = MARGINAL_NLL_SERIES.get(mode)
        if series is not None:
            steps = df["i"].to_numpy()
            if len(series) > steps.max():
                df["logscore"] = df["logscore"].to_numpy() + series[steps]
    return df[["i", "es", "vs05", "vs10", "dss", "logscore"]]


def _dm_all_metrics(df_A, df_B):
    df = df_A.merge(df_B, on="i", suffixes=("_A", "_B"))
    return {
        m: _dm_test_metric((df[f"{m}_A"] - df[f"{m}_B"]).dropna().values)
        for m in ["es", "vs05", "vs10", "dss", "logscore"]
    }


def _fmt_dm(dm_stat, p_val):
    if np.isnan(dm_stat):
        return ""
    stars = ("***" if p_val < 0.01 else
             "**"  if p_val < 0.05 else
             "*"   if p_val < 0.10 else "")
    s = f"{dm_stat:.3f}"
    # Right-align on the last digit of the statistic: the number sets the cell
    # width, the significance stars hang to the right in a zero-width \rlap so
    # they never shift the alignment of the decimals across rows.
    if stars:
        return rf"${s}$\rlap{{$^{{{stars}}}$}}"
    return f"${s}$"


dm_metrics = ["es", "vs05", "dss", "logscore"]
dm_col_header = r"ES & VS$_{0.5}$ & DSS & LS"


# =============================================================================
# PAPER TABLE 6  (tab:dm_online_functionality)  ->  input/tab_dm_online_functionality.tex
# "Two-sided Diebold--Mariano tests comparing online and repeated-online
#  updating to repeated batch estimation across all scoring rules."
#  (online / repeated_online vs repeated_batch, per copula.)
# =============================================================================
online_dm_rows = []
for setting_key, setting_label in spec_order:
    res_batch = all_results.get((setting_key, "repeated_batch"))
    if res_batch is None:
        continue
    df_batch = _get_scores_df(res_batch)
    for mode_key, cmp_label in [("online",          "Online"),
                                  ("repeated_online", r"Rep.\ Online")]:
        res_A = all_results.get((setting_key, mode_key))
        if res_A is None:
            continue
        dm_res = _dm_all_metrics(_get_scores_df(res_A), df_batch)
        online_dm_rows.append({
            "Specification": setting_label,
            "Comparison": cmp_label,
            **dm_res,
        })

dm2_lines = []
dm2_lines.append(r"\begingroup")
dm2_lines.append(r"\setlength{\tabcolsep}{9pt}")  # horizontal spacing only
dm2_lines.append(r"\begin{tabular}{ll@{\hspace{1.0em}}rrrr}")
dm2_lines.append(r"\toprule")
dm2_lines.append(r"Model & Rep.\ Batch vs $\cdots$ & " + dm_col_header + r" \\")
dm2_lines.append(r"\midrule")

for i, row in enumerate(online_dm_rows):
    if i > 0 and i % 2 == 0:
        dm2_lines.append(r"\addlinespace[6pt]")
    cells = [_fmt_dm(*row[m][:2]) for m in dm_metrics]
    _spec_cell = f"{row['Specification']} " if i % 2 == 0 else ""
    dm2_lines.append(
        f"{_spec_cell}& {row['Comparison']} & "
        + " & ".join(cells) + r" \\"
    )

dm2_lines.append(r"\bottomrule")
dm2_lines.append(r"\end{tabular}")
dm2_lines.append(r"\endgroup")
write_tex_file(TAB_DIR / f"tab_dm_online_functionality.tex", "\n".join(dm2_lines))


# =============================================================================
# PAPER TABLE 5  (tab:dm_gaussian_baseline)  ->  input/tab_dm_gaussian_baseline.tex
# "Two-sided Diebold--Mariano tests: the vine copula specifications (VC, CVC),
#  in all updating schemes, versus the Gaussian copula repeated-batch benchmark
#  across all scoring rules."
# =============================================================================
gc_batch = all_results.get(("gaussian_intercept", "repeated_batch"))
df_gc_batch = _get_scores_df(gc_batch)

gauss_dm_rows = []
for setting_key, setting_label in spec_order:
    # Only benchmark the vine specifications (VC, CVC) against the GC repeated
    # batch baseline; skip the Gaussian-copula rows entirely.
    if setting_key == "gaussian_intercept":
        continue
    for mode_key in mode_order:
        res_other = all_results.get((setting_key, mode_key))
        if res_other is None:
            continue
        dm_res = _dm_all_metrics(_get_scores_df(res_other), df_gc_batch)
        gauss_dm_rows.append({
            "Specification": setting_label,
            "Mode": mode_label[mode_key],
            **dm_res,
        })

dm3_lines = []
dm3_lines.append(r"\begingroup")
dm3_lines.append(r"\setlength{\tabcolsep}{9pt}")  # horizontal spacing only
dm3_lines.append(r"\begin{tabular}{ll@{\hspace{1.0em}}rrrr}")
dm3_lines.append(r"\toprule")
dm3_lines.append(r"Model & Scheme & " + dm_col_header + r" \\")
dm3_lines.append(r"\midrule")

prev_spec = None
for row in gauss_dm_rows:
    _new_block = row["Specification"] != prev_spec
    if prev_spec is not None and _new_block:
        dm3_lines.append(r"\addlinespace[6pt]")
    prev_spec = row["Specification"]
    cells = [_fmt_dm(*row[m][:2]) for m in dm_metrics]
    _spec_cell = f"{row['Specification']} " if _new_block else ""
    dm3_lines.append(
        f"{_spec_cell}& {row['Mode']} & "
        + " & ".join(cells) + r" \\"
    )

dm3_lines.append(r"\bottomrule")
dm3_lines.append(r"\end{tabular}")
dm3_lines.append(r"\endgroup")
write_tex_file(TAB_DIR / f"tab_dm_gaussian_baseline.tex", "\n".join(dm3_lines))

# =============================================================================
# DM TESTS AGAINST THE BEST MODEL  ->  tab_dm_vs_best.tex
# Every competing specification and updating scheme against the conditional
# vine copula under repeated online updating, which is best on all four scores.
# A positive statistic means the competitor has the higher (worse) score.
# The log score is the one of the full predictive density, so rows with
# different marginal models remain comparable.
# =============================================================================
BEST_KEY = ("all_all", "repeated_online")  # CVC, repeated online
best_res = all_results.get(BEST_KEY)

if best_res is None:
    print("WARNING: CVC repeated online not available; skipping tab_dm_vs_best")
else:
    df_best = _get_scores_df(best_res, mode=BEST_KEY[1])

    best_dm_rows = []
    for setting_key, setting_label in spec_order:
        for mode in mode_order:
            if (setting_key, mode) == BEST_KEY:
                continue
            res_other = all_results.get((setting_key, mode))
            if res_other is None:
                continue
            dm_res = _dm_all_metrics(_get_scores_df(res_other, mode=mode), df_best)
            best_dm_rows.append({
                "Specification": setting_label,
                "Scheme": mode_label[mode],
                **dm_res,
            })

    dm4_lines = []
    dm4_lines.append(r"\begingroup")
    dm4_lines.append(r"\setlength{\tabcolsep}{9pt}")
    dm4_lines.append(r"\begin{tabular}{ll@{\hspace{1.0em}}rrrr}")
    dm4_lines.append(r"\toprule")
    dm4_lines.append(r"Model & Scheme & " + dm_col_header + r" \\")
    dm4_lines.append(r"\midrule")

    _prev_spec = None
    for row in best_dm_rows:
        _new_block = row["Specification"] != _prev_spec
        if _prev_spec is not None and _new_block:
            dm4_lines.append(r"\addlinespace[6pt]")
        _prev_spec = row["Specification"]
        cells = [_fmt_dm(*row[m][:2]) for m in dm_metrics]
        _spec_cell = f"{row['Specification']} " if _new_block else ""
        dm4_lines.append(
            f"{_spec_cell}& {row['Scheme']} & " + " & ".join(cells) + r" \\"
        )

    dm4_lines.append(r"\bottomrule")
    dm4_lines.append(r"\end{tabular}")
    dm4_lines.append(r"\endgroup")

    write_tex_file(TAB_DIR / f"tab_dm_vs_best.tex", "\n".join(dm4_lines))
    print(f"wrote tab_dm_vs_best.tex with {len(best_dm_rows)} comparisons")


def _dm_table(rows, first_header, second_header, path, n_group):
    """Render a DM table with two label columns and the four score columns."""
    out = [
        r"\begingroup",
        r"\setlength{\tabcolsep}{9pt}",
        r"\begin{tabular}{ll@{\hspace{1.0em}}rrrr}",
        r"\toprule",
        f"{first_header} & {second_header} & " + dm_col_header + r" \\",
        r"\midrule",
    ]
    for i, row in enumerate(rows):
        if i > 0 and i % n_group == 0:
            out.append(r"\addlinespace[6pt]")
        cells = [_fmt_dm(*row[m][:2]) for m in dm_metrics]
        _a_cell = f"{row['A']} " if i % n_group == 0 else ""
        out.append(f"{_a_cell}& {row['B']} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\endgroup"]
    write_tex_file(path, "\n".join(out))
    print(f"wrote {path.name} with {len(rows)} comparisons")


# =============================================================================
# DM TESTS WITHIN COPULA  ->  tab_dm_within_copula.tex
# Updating schemes compared against repeated online, holding the copula fixed.
# Isolates the updating scheme; the marginals differ across rows, so the log
# score is the one of the full predictive density.
# =============================================================================
within_copula_rows = []
for setting_key, setting_label in spec_order:
    res_bench = all_results.get((setting_key, "repeated_online"))
    if res_bench is None:
        continue
    df_bench = _get_scores_df(res_bench, mode="repeated_online")
    for mode in mode_order:
        if mode == "repeated_online":
            continue
        res_other = all_results.get((setting_key, mode))
        if res_other is None:
            continue
        within_copula_rows.append({
            "A": setting_label,
            "B": mode_label[mode],
            **_dm_all_metrics(_get_scores_df(res_other, mode=mode), df_bench),
        })

if within_copula_rows:
    _dm_table(
        within_copula_rows, "Model", r"Rep.\ Online vs $\cdots$",
        TAB_DIR / f"tab_dm_within_copula.tex",
        n_group=max(1, len(mode_order) - 1),
    )

# =============================================================================
# DM TESTS WITHIN UPDATING SCHEME  ->  tab_dm_within_scheme.tex
# Copula specifications compared against the CVC, holding the scheme fixed.
# Both rows of a block share their marginal models, so this isolates the
# dependence model.
# =============================================================================
within_scheme_rows = []
for mode in mode_order:
    res_bench = all_results.get(("all_all", mode))  # CVC
    if res_bench is None:
        continue
    df_bench = _get_scores_df(res_bench, mode=mode)
    for setting_key, setting_label in spec_order:
        if setting_key == "all_all":
            continue
        res_other = all_results.get((setting_key, mode))
        if res_other is None:
            continue
        within_scheme_rows.append({
            "A": mode_label[mode],
            "B": setting_label,
            **_dm_all_metrics(_get_scores_df(res_other, mode=mode), df_bench),
        })

if within_scheme_rows:
    _dm_table(
        within_scheme_rows, "Scheme", r"M+CVC vs $\cdots$",
        TAB_DIR / f"tab_dm_within_scheme.tex",
        n_group=max(1, len(spec_order) - 1),
    )

def _write_mcs_table(mcs_pvalues, model_labels, path):
    mcs_lines = [
        r"\begingroup",
        r"\setlength{\tabcolsep}{9pt}",
        r"\begin{tabular}{ll@{\hspace{1.0em}}rrrr}",
        r"\toprule",
        r"Model & Scheme & " + dm_col_header + r" \\",
        r"\midrule",
    ]
    _n_group = max(1, len(mode_order))
    for i, (name, (spec_lab, mode_lab)) in enumerate(model_labels.items()):
        if i > 0 and i % _n_group == 0:
            mcs_lines.append(r"\addlinespace[6pt]")
        spec_lab = f"{spec_lab} " if i % _n_group == 0 else ""
        cells = []
        for m in dm_metrics:
            p = float(mcs_pvalues[m].get(name, np.nan))
            s = "" if np.isnan(p) else f"{p:.3f}"
            # Members of the MCS are set in bold.
            if s and p >= MCS_SIZE:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        mcs_lines.append(f"{spec_lab}& {mode_lab} & " + " & ".join(cells) + r" \\")
    mcs_lines += [r"\bottomrule", r"\end{tabular}", r"\endgroup"]

    write_tex_file(path, "\n".join(mcs_lines))
    print(f"wrote {path.name}")



# =============================================================================
# PAPER TABLE 9  (tab:computation_times)  ->  input/tab_computation_times.tex
# "Computation times of the copula-based models under different updating
#  schemes." (Appendix.) Mean/total fit and update times for all three modes.
# =============================================================================


# =============================================================================
# MODEL CONFIDENCE SET  ->  tab_mcs.tex
# The DM tests above are indicative: their benchmark is the model that turned
# out best, so the comparison is not corrected for having selected it. The MCS
# of Hansen, Lunde and Nason (2011) avoids this. It returns the set of models
# that cannot be distinguished from the best one at a given confidence level,
# correcting for the multiplicity of the pairwise comparisons.
# =============================================================================
MCS_SIZE = 0.10          # models with p >= MCS_SIZE are in the 90% MCS
MCS_REPS = 5000          # above R's default of 1000, to steady the p-values

# The settings follow the R package MCS of Bernardi and Catania, so that the
# results can be reproduced with either implementation:
#   - moving block bootstrap, as in MCS:::GetIndices
#   - block length k = max_j ar(x_j)$order, floored at 3, as in MCSprocedure
#   - statistic "Tmax" of Hansen, Lunde and Nason, R's default, which is arch's
#     method "max" (arch defaults to "R", so this has to be set explicitly)
# R's ar() selects the order by AIC from Yule-Walker fits up to
# floor(10*log10(T)); ar_order_aic below reproduces that rule.
MCS_BOOTSTRAP = "moving block"


def ar_order_aic(x):
    """Order of an AR fit chosen by AIC over Yule-Walker estimates, as in R's ar()."""
    from statsmodels.regression.linear_model import yule_walker

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    n = len(x)
    order_max = min(int(np.floor(10 * np.log10(n))), n - 1)
    aics = [n * np.log(x.dot(x) / n)]
    for p in range(1, order_max + 1):
        _, sigma = yule_walker(x, order=p, method="mle")
        aics.append(n * np.log(sigma**2) + 2 * p)
    return int(np.argmin(aics))


def mcs_block_length(losses, min_k=3):
    """R's rule: the largest AR order across the loss series, floored at min_k."""
    return max(max(ar_order_aic(losses[c]) for c in losses.columns), min_k)


try:
    from arch.bootstrap import MCS
except ImportError:
    MCS = None
    print("WARNING: arch not installed; skipping the model confidence set")

if MCS is not None:
    # Per-step losses for every model, aligned on the step index.
    loss_panels = {m: {} for m in dm_metrics}
    model_labels = {}
    for setting_key, setting_label in spec_order:
        for mode in mode_order:
            res = all_results.get((setting_key, mode))
            if res is None:
                continue
            name = f"{setting_label} {mode_label[mode]}"
            model_labels[name] = (setting_label, mode_label[mode])
            df = _get_scores_df(res, mode=mode).set_index("i")
            for m in dm_metrics:
                loss_panels[m][name] = df[m]

    # Both statistics of Hansen, Lunde and Nason. "max" is T_max, the default of
    # the R package and the one reported in the main text: it compares each model
    # with the average of the surviving set, so pairs of near-identical models
    # cannot drive the elimination on their own. "R" is the range statistic T_R,
    # the maximum over all pairs, reported alongside it in the appendix. (Their
    # semi-quadratic T_SQ is a third statistic that neither package implements.)
    for _method, _suffix in (("max", ""), ("R", "_tr")):
        mcs_pvalues = {}
        for m in dm_metrics:
            losses = pd.DataFrame(loss_panels[m]).dropna()
            block = mcs_block_length(losses)
            mcs = MCS(losses, size=MCS_SIZE, reps=MCS_REPS, block_size=block,
                      bootstrap=MCS_BOOTSTRAP, method=_method, seed=20240803)
            mcs.compute()
            mcs_pvalues[m] = mcs.pvalues["Pvalue"]
            included = list(mcs.included)
            print(f"MCS[{_method:3s}] {m:9s} k={block:3d} included at "
                  f"{int((1 - MCS_SIZE) * 100)}%: {included}")

        _write_mcs_table(mcs_pvalues, model_labels, TAB_DIR / f"tab_mcs{_suffix}.tex")


time_rows = []
for setting_key, setting_label in spec_order:
    for mode in mode_order:
        key = (setting_key, mode)
        if key not in all_results:
            continue
        res = all_results[key]

        # handle key differences across modes
        fit_times    = [t for t in res.get("fit_times",    []) if np.isfinite(t)]
        update_times = [t for t in res.get("update_times", []) if np.isfinite(t)]
        if not update_times:
            update_times = [t for t in res.get("all_update_times", []) if np.isfinite(t)]

        # In the fully online scheme the first recorded step (i == 0) is the initial
        # full batch fit (fit_vinecopstructure(online=0)), stored as the first
        # element of update_times; count it as the fit time rather than as an
        # online update.
        if mode == "online" and len(fit_times) == 0 and len(update_times) > 0:
            fit_times    = [update_times[0]]
            update_times = update_times[1:]

        if len(fit_times) == 0 and len(update_times) == 0:
            continue

        time_rows.append({
            "Specification":     setting_label,
            "Mode":              mode_label[mode],
            "N fit":             len(fit_times)                      if fit_times    else np.nan,
            "Mean fit (s)":      np.mean(fit_times)                  if fit_times    else np.nan,
            "Total fit (s)":     np.sum(fit_times)                   if fit_times    else np.nan,
            "N update":          len(update_times)                   if update_times else np.nan,
            "Mean update (s)":   np.mean(update_times)               if update_times else np.nan,
            "Total update (s)":  np.sum(update_times)                if update_times else np.nan,
            "Total (s)":         np.sum(fit_times) + np.sum(update_times),
        })

df_times = pd.DataFrame(time_rows)
print(df_times.to_string(index=False, float_format="%.4f"))


# -----------------------------------------------------------------------------
# LaTeX table — computation times (appendix)
# -----------------------------------------------------------------------------
def _fmt_time(x, digits=2):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "--"
    return f"{x:.{digits}f}"


def _fmt_int(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "--"
    return f"{int(x)}"


time_tab_lines = []
time_tab_lines.append(r"\begin{tabular}{llrrrrrrr}")
time_tab_lines.append(r"\toprule")
time_tab_lines.append(
    r"& & \multicolumn{3}{c}{Fit} & \multicolumn{3}{c}{Update} & \\"
)
time_tab_lines.append(r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}")
time_tab_lines.append(
    r"Model & Scheme & $N$ & Mean (s) & Total (s) & "
    r"$N$ & Mean (s) & Total (s) & Total (s) \\"
)
time_tab_lines.append(r"\midrule")

n_modes = len(mode_order)
for i, row in df_times.iterrows():
    if i > 0 and i % n_modes == 0:
        time_tab_lines.append(r"\addlinespace[6pt]")
    time_tab_lines.append(
        f"{row['Specification']} & {row['Mode']} & "
        f"{_fmt_int(row['N fit'])} & "
        f"{_fmt_time(row['Mean fit (s)'])} & "
        f"{_fmt_time(row['Total fit (s)'])} & "
        f"{_fmt_int(row['N update'])} & "
        f"{_fmt_time(row['Mean update (s)'])} & "
        f"{_fmt_time(row['Total update (s)'])} & "
        f"{_fmt_time(row['Total (s)'])} \\\\"
    )

time_tab_lines.append(r"\bottomrule")
time_tab_lines.append(r"\end{tabular}")
write_tex_file(TAB_DIR / f"tab_computation_times.tex", "\n".join(time_tab_lines))








# =============================================================================
# FIGURES
# =============================================================================
# Everything below needs the per-step stores (beta_store, copula_family_store)
# that only the full result pickles carry. With TABLES_ONLY set the results are
# read from the lightweight score cache instead, so the figures are skipped.
if TABLES_ONLY:
    print("TABLES_ONLY set: tables written, skipping figures.")
    raise SystemExit(0)

# =============================================================================
# CVC BETA COEFFICIENTS OVER TIME — FIRST-TREE EDGES
# Shared data prep for the three beta figures below:
#   - beta_coefs.pdf          : full grid   -> NOT used in the paper (exploratory)
#   - beta_first4_coefs.pdf   : first 4 coefs-> fig:beta_first4_coefs (currently
#                                               COMMENTED OUT in main.tex)
#   - beta_zoom_edge0_1_coef1 : single zoom -> PAPER FIGURE 6 (fig:beta_zoom_online)
# =============================================================================

setting_name = "all_all"
modes = [m for m in ["repeated_online", "repeated_batch", "online"] if m in _PRESENT_MODES]

mode_style = {
    "online":         (":",  "C2"),
    "repeated_online": ("-", "C0"),
    "repeated_batch": ("--", "C1"),
}


first_tree_edges = [(0, 0), (0, 1), (0, 2), (0, 3)]
n_edges = len(first_tree_edges)

# ------------------------------------------------------------------
# Collect CVC beta data
# ------------------------------------------------------------------
all_beta_data = {}
for edge in first_tree_edges:
    rows_beta = []
    for mode in modes:
        key = (setting_name, mode)
        if key not in all_results:
            continue
        bs = all_results[key]["beta_store"]
        for i in sorted(bs.keys()):
            beta_vec = extract_beta_vec(bs[i][edge])
            for coef_idx, beta_val in enumerate(beta_vec):
                rows_beta.append({"mode": mode, "i": i,
                                   "coef_idx": coef_idx, "beta": beta_val})
    all_beta_data[edge] = pd.DataFrame(rows_beta)

n_coefs = max(
    (df["coef_idx"].nunique() for df in all_beta_data.values() if not df.empty),
    default=1,
)

# ------------------------------------------------------------------
# Serif fonts matching the LaTeX main text (Computer Modern)
# ------------------------------------------------------------------
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    # "Computer Modern Roman" is not a font name matplotlib knows; it silently
    # fell back to DejaVu Serif, which does not match the LaTeX text. cmr10 is
    # Computer Modern and ships with matplotlib.
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,   # cmr10 has no U+2212
})


def _edge_label(edge):
    # code edge (0, j) -> paper region pair (j+1, j+2)
    return f"$e = ({edge[1] + 1},{edge[1] + 2})$"


# ------------------------------------------------------------------
# beta_coefs.pdf — full grid, 4 columns x n_coefs rows
# NOT used in the paper (exploratory overview of all coefficients).
# ------------------------------------------------------------------
#fig1, axes1 = plt.subplots(n_coefs, n_edges, figsize=(8.27, 11.0), sharex="col", sharey="row")
#if n_coefs == 1:
#    axes1 = axes1[np.newaxis, :]

#for col_idx, edge in enumerate(first_tree_edges):
#    df_b = all_beta_data[edge]
#    coef_ids = sorted(df_b["coef_idx"].unique()) if not df_b.empty else []
#    axes1[0, col_idx].set_title(_edge_label(edge), fontsize=8, pad=3)
#    for row_idx in range(n_coefs):
#        ax = axes1[row_idx, col_idx]
#        if row_idx < len(coef_ids):
#            df_c = df_b[df_b["coef_idx"] == coef_ids[row_idx]]
#            for mode in modes:
#                df_m = df_c[df_c["mode"] == mode].sort_values("i")
#                if df_m.empty:
#                    continue
#                ls, col = mode_style[mode]
#                lbl = mode_label[mode] if (col_idx == 0 and row_idx == 0) else None
#                ax.plot(df_m["i"], df_m["beta"],
#                        linestyle=ls, color=col, linewidth=1.0, label=lbl)
#        ax.axhline(0, linestyle=":", linewidth=0.6, color="k")
#        ax.tick_params(labelsize=6)
#        if col_idx == 0:
#            ax.set_ylabel(f"$\\beta_{{e,{row_idx}}}$", fontsize=7)
#        if row_idx == n_coefs - 1:
#            ax.set_xlabel("$i$", fontsize=7)

#handles1, labels1 = axes1[0, 0].get_legend_handles_labels()
#fig1.legend(handles1, labels1, loc="upper center", ncol=3,
#            frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0))
#fig1.suptitle("CVC — coefficients over time, first-tree edges", fontsize=9, y=1.02)
#plt.tight_layout()
#fig1.savefig(FINAL_DIR / "beta_coefs.pdf", bbox_inches="tight")

#plt.show()

# ------------------------------------------------------------------
# beta_first4_coefs.pdf — first 4 beta coefficients only
# fig:beta_first4_coefs — currently COMMENTED OUT in main.tex (not in paper).
# ------------------------------------------------------------------
#_n_rows_sub = min(4, n_coefs)

#fig1b, axes1b = plt.subplots(_n_rows_sub, n_edges,
#                              figsize=(8.27, _n_rows_sub * 2.2),
#                              sharex="col", sharey="row")
#if _n_rows_sub == 1:
#    axes1b = axes1b[np.newaxis, :]###

#for col_idx, edge in enumerate(first_tree_edges):
#    df_b = all_beta_data[edge]
#    coef_ids = sorted(df_b["coef_idx"].unique()) if not df_b.empty else []
#    axes1b[0, col_idx].set_title(_edge_label(edge), fontsize=8, pad=3)
#    for row_idx in range(_n_rows_sub):
#        ax = axes1b[row_idx, col_idx]
#        if row_idx < len(coef_ids):
#            df_c = df_b[df_b["coef_idx"] == coef_ids[row_idx]]
#            for mode in modes:
#                df_m = df_c[df_c["mode"] == mode].sort_values("i")
#                if df_m.empty:
#                    continue
#                ls, col = mode_style[mode]
#                lbl = mode_label[mode] if (col_idx == 0 and row_idx == 0) else None
#                ax.plot(df_m["i"], df_m["beta"],
#                        linestyle=ls, color=col, linewidth=1.0, label=lbl)
#        ax.axhline(0, linestyle=":", linewidth=0.6, color="k")
#        ax.tick_params(labelsize=6)
#        if col_idx == 0:
#            ax.set_ylabel(f"$\\beta_{{e,{row_idx}}}$", fontsize=7)
#        if row_idx == _n_rows_sub - 1:
#            ax.set_xlabel("$i$", fontsize=7)

#handles1b, labels1b = axes1b[0, 0].get_legend_handles_labels()
#fig1b.legend(handles1b, labels1b, loc="upper center", ncol=3,
#             frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0))
#plt.tight_layout()
#fig1b.savefig(TEX_DIR / "beta_first4_coefs.pdf", bbox_inches="tight")
#plt.show()

# ------------------------------------------------------------------
# PAPER FIGURE 6 (fig:beta_zoom_online) -> input/beta_zoom_edge0_1_coef1.pdf
# "Estimated coefficient on the Northern--Midlands temperature spread
#  (first tree, CVC specification) over the test period, by updating scheme."
# Single-edge, single-coefficient zoom; default edge (0,1) coef 1 = Delta temp.
# ------------------------------------------------------------------
_zoom_edge    = (0, 1)   # ← change to zoom into a different edge
_zoom_coef    = 1        # ← change to zoom into a different beta index

_regions = ["Scotland", "Northern", "Midlands", "London", "Southern"]
_coef_names = {
    0: "intercept",
    1: r"$\Delta$temp",
    2: r"$\Delta$rain",
    3: r"$\Delta$irr",
    4: r"$\Delta$wsp$^{10}$",
    5: r"$\Delta$wsp$^{100}$",
}

_r1 = _regions[_zoom_edge[1]]
_r2 = _regions[_zoom_edge[1] + 1]
# paper region numbers: code edge (0, j) -> regions (j+1, j+2)
_r1_num = _zoom_edge[1] + 1
_r2_num = _zoom_edge[1] + 2
_coef_syms = {0: r"\mathrm{0}", 1: r"\mathrm{temp}", 2: r"\mathrm{rain}",
              3: r"\mathrm{irr}", 4: r"\mathrm{wsp10}", 5: r"\mathrm{wsp100}"}
_coef_sym  = _coef_syms.get(_zoom_coef, str(_zoom_coef))
_coef_label = _coef_names.get(_zoom_coef, f"$\\beta_{{{_zoom_coef}}}$")

fig1c, ax1c = plt.subplots(figsize=(6.5, 2.9))

df_b = all_beta_data[_zoom_edge]
coef_ids = sorted(df_b["coef_idx"].unique()) if not df_b.empty else []
if _zoom_coef < len(coef_ids):
    df_c = df_b[df_b["coef_idx"] == coef_ids[_zoom_coef]]
    for mode in modes:
        df_m = df_c[df_c["mode"] == mode].sort_values("i")
        if df_m.empty:
            continue
        ls, col = mode_style[mode]
        ax1c.plot(df_m["i"], df_m["beta"],
                  linestyle=ls, color=col, linewidth=1.0, label=mode_label[mode])

ax1c.axhline(0, linestyle=":", linewidth=0.6, color="k")
ax1c.set_xlabel("$i$", fontsize=10)
ax1c.set_ylabel(f"$\\widehat{{\\beta}}^{{{_coef_sym}}}_{{({_r1_num},{_r2_num}),i}}$", fontsize=10)
ax1c.set_title(f"{_r1} - {_r2},  $\\widehat{{\\beta}}^{{{_coef_sym}}}_{{({_r1_num},{_r2_num}),i}}$", fontsize=10)
ax1c.legend(frameon=False, fontsize=9)
ax1c.tick_params(labelsize=9)
plt.tight_layout()
_zoom_fname = f"beta_zoom_edge{'_'.join(map(str, _zoom_edge))}_coef{_zoom_coef}.pdf"
fig1c.savefig(FIG_DIR / _zoom_fname, bbox_inches="tight")
plt.show()


# =============================================================================
# COPULA FAMILY STORE  (console only — inspects selected families per edge;
# the family shares are also annotated in the tau figure titles below)
# =============================================================================
all_copula_family_stores = {
    (setting_name, mode): res["copula_family_store"]
    for (setting_name, mode), res in all_results.items()
}

rows = []

for (setting_name, mode), store in all_copula_family_stores.items():
    for i, C_i in store.items():
        for edge in zip(*np.where(~np.isnan(C_i))):
            rows.append({
                "setting_name": setting_name,
                "mode": mode,
                "i": i,
                "edge": edge,
                "copula_family": C_i[edge],
            })

df_copulas = pd.DataFrame(rows)

print(df_copulas.head())

print(df_copulas.query("setting_name == 'all_all'"))



# =============================================================================
# PAPER FIGURE 7 (fig:tau_all_edges) -> input/tau_all_edges.pdf (+ BASE_PATH copy)
# "Model-implied Kendall's tau over the test period for every pair-copula of the
#  fitted CVC vine, under the three updating schemes." Chosen family/share is
#  written into each subplot title.
# =============================================================================
_setting_tau = "all_all"
_edges_tau   = [(0, 0), (0, 1), (0, 2), (0, 3)]

fig_tau, axes_tau = plt.subplots(1, len(_edges_tau),
                                  figsize=(8.27, 2.5), sharey=True)

# Repeated Online and Repeated Batch nearly coincide; draw batch underneath as a
# solid line and Repeated Online on top with a dash pattern so both stay visible.
_tau_draw_order = [m for m in ["online", "repeated_batch", "repeated_online"] if m in _PRESENT_MODES]
_tau_emph = {
    "online":         dict(linewidth=0.8, zorder=1, alpha=0.9),
    "repeated_batch": dict(linewidth=1.6, zorder=2, alpha=0.9),
    "repeated_online": dict(linewidth=1.1, zorder=3, alpha=0.95, dashes=(4, 3)),
}
_family_names = {
    0:  "Indep.",
    1:  "Gaussian",  2: "Student-t",
    31: "Clayton",   32: r"Clayton$^{90}$",  33: r"Clayton$^{180}$",  34: r"Clayton$^{270}$",
    41: "Gumbel",    42: r"Gumbel$^{90}$",   43: r"Gumbel$^{180}$",   44: r"Gumbel$^{270}$",
}
_region_abbr = ["Sco", "Nor", "Mid", "Lon", "Sou"]


def _theta_to_tau(fc, theta):
    """Map a fitted copula parameter to Kendall's tau, signed for 90/270 rotations."""
    link = _param_link.get(fc)
    if link is None:
        return 0.0
    tau = float(link.link(np.array([theta]))[0])
    return -abs(tau) if fc in (32, 34, 42, 44) else tau


# detect all vine edges as the union of finite family-matrix positions over time
_ref_store = None
for mode in modes:
    key = (_setting_tau, mode)
    if key in all_results and all_results[key].get("copula_family_store"):
        _ref_store = all_results[key]["copula_family_store"]
        break

edge_mask = None
for _M in _ref_store.values():
    fm = np.isfinite(np.asarray(_M, dtype=float))
    edge_mask = fm if edge_mask is None else (edge_mask | fm)

edges_by_tree = {r: [c for c in range(edge_mask.shape[1]) if edge_mask[r, c]]
                 for r in range(edge_mask.shape[0])}
edges_by_tree = {r: cols for r, cols in edges_by_tree.items() if cols}
trees    = sorted(edges_by_tree.keys())
max_cols = max(len(cols) for cols in edges_by_tree.values())

fig_all, axes_all = plt.subplots(len(trees), max_cols,
                                 figsize=(8.27, 2.0 * len(trees)),
                                 sharex=True, squeeze=False)

for row_pos, r in enumerate(trees):
    cols = edges_by_tree[r]
    for col_pos in range(max_cols):
        ax = axes_all[row_pos, col_pos]
        if col_pos >= len(cols):
            ax.axis("off")
            continue
        c = cols[col_pos]
        edge = (r, c)
        chosen = []   # distinct copula families selected over the test period
        for mode in modes:
            key = (_setting_tau, mode)
            if key not in all_results:
                continue
            kts = all_results[key].get("pred_mat_store")
            cfs = all_results[key].get("copula_family_store")
            if kts is None or cfs is None:
                continue
            pts = []
            for i, pred_mat in sorted(kts.items()):
                theta = pred_mat[r, c]
                if not np.isfinite(theta):
                    continue
                fam_matrix = cfs[i] if isinstance(cfs, list) else cfs.get(
                    i, cfs[min(cfs, key=lambda k: abs(k - i))])
                fc = int(fam_matrix[edge])
                chosen.append(fc)
                pts.append((i, _theta_to_tau(fc, theta)))
            if not pts:
                continue
            xs, ys = zip(*pts)
            ls, col_ = mode_style[mode]
            ax.plot(xs, ys, linestyle=ls, color=col_, linewidth=0.8,
                    label=mode_label[mode] if (row_pos == 0 and col_pos == 0) else None)

        ax.axhline(0, linestyle=":", linewidth=0.6, color="k")

        # title: edge label (region pair on tree 1) + the chosen copula family/families
        if r == 0 and c + 1 < len(_region_abbr):
            edge_lbl = f"{_region_abbr[c]}–{_region_abbr[c + 1]}"
        else:
            edge_lbl = f"T{r + 1} e{c}"
        if chosen:
            _vals, _counts = np.unique(chosen, return_counts=True)
            _order = np.argsort(_counts)[::-1]   # most frequent first
            fam_lbl = " / ".join(
                f"{_family_names.get(int(_vals[k]), str(int(_vals[k])))} "
                f"({100 * _counts[k] / len(chosen):.0f}%)"
                for k in _order
            )
        else:
            fam_lbl = "n/a"
        edge_lbl = f"T1, e = ({c + 1},{c + 2})" if r == 0 else f"T{r + 1}, e{c}"
        ax.set_title(f"{edge_lbl}: {fam_lbl}", fontsize=7, pad=3)
        ax.tick_params(labelsize=6)
        if col_pos == 0:
            ax.set_ylabel(r"$\hat\tau$", fontsize=7)
        if row_pos == len(trees) - 1:
            ax.set_xlabel("$i$", fontsize=7)

_handles_all, _labels_all = axes_all[0, 0].get_legend_handles_labels()
plt.tight_layout()
fig_all.legend(_handles_all, _labels_all, loc="lower right",
               frameon=False, fontsize=8, bbox_to_anchor=(0.98, 0.06))
fig_all.savefig(FIG_DIR / f"tau_all_edges.pdf", bbox_inches="tight")
# (single copy written to FIG_DIR above)
plt.show()



