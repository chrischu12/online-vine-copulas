# -*- coding: utf-8 -*-
"""Figures and tables for the Simulations section of the paper.

Consolidates the simulation-side plotting/tabulating into a single script,
mirroring ``application/03_plots.py``. Each block below maps to one paper
object:

  * PAPER FIGURE 2 (fig:beta_estimates)          -> figures/beta_tree_estimates_merged.pdf
  * PAPER FIGURE 3 (fig:ols_vs_ridge_boxplots)   -> input/boxplot_batch_online.pdf
  * PAPER TABLE 2  (tab:time_ridge)              -> input/table_vine_time_ridge_avg.tex
    APPENDIX TABLE  (tab:time_ols)               -> input/table_vine_time_ols_avg.tex
  * PAPER FIGURE 8 (fig:break_vs_stable_gamma)   -> input/compare_break_vs_stable_score_vs_gamma.pdf

Data sources (Monte-Carlo results, all under BASE_PATH):
  * batch_vine_{betas,C,C_true}_{1000,5000}.pkl  (batch-estimation study)
  * all_runs_50.pkl        = Stable  (stationary DGP)
  * all_runs_50_break.pkl  = Break   (structural break in beta at obs 750)
"""

import pickle
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from online_copula_experiments.application import BASE_PATH

# ---------------------------------------------------------------------------
# Paths — simulation results are read from BASE_PATH/intermediate_simulation;
# figures and tables are written to BASE_PATH/results so a server run is
# self-contained (copy them into the paper tree afterwards).
# ---------------------------------------------------------------------------
BASE_PATH = Path(BASE_PATH)

SIM_DIR = BASE_PATH / "intermediate_simulation"

RESULTS_DIR = BASE_PATH / "results"
FIG_DIR = RESULTS_DIR / "figures"
TAB_DIR = RESULTS_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)


def data_path(fname):
    """All simulation Monte-Carlo inputs live in BASE_PATH/intermediate_simulation."""
    p = SIM_DIR / fname
    if not p.exists():
        raise FileNotFoundError(f"{fname} not found in {SIM_DIR}")
    return p

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------
SETTING_NAME = "batch_vine"          # prefix of the batch-study pickles
SAMPLE_SIZES = [1000, 5000]
N_RUNS_COMPARE = 50                  # first 50 run_ids of the forecasting study
KEEP_FREQ = 1

STABLE_FILE = "all_runs_50.pkl"       # stationary DGP (Figures 3 and 8)
BREAK_FILE = "all_runs_50_break.pkl"  # structural break (Figure 8)
SCENARIOS = [("Stable", STABLE_FILE), ("Break", BREAK_FILE)]

EST_NAMES = {0: "Batch", 1: "Online"}
REG_COLORS = {0: "#4C78A8", 1: "#E45756"}   # OLS blue, Ridge red
GAUSS_COLOR = "#BAB0AC"
SCEN_COLORS = {"Stable": "#4C78A8", "Break": "#E45756"}

GAMMA_LABEL = {0.0: r"$\gamma=0$", 1/400: r"$\gamma=1/400$", 1/200: r"$\gamma=1/200$",
               1/100: r"$\gamma=1/100$", 1/50: r"$\gamma=1/50$"}
GAMMA_ORDER = [0.0, 1/400, 1/200, 1/100, 1/50]   # paper column order

# ---------------------------------------------------------------------------
# Matplotlib settings — render all text/numbers in the paper font (LaTeX / CM).
# Set USE_LATEX = False to fall back to matplotlib mathtext if no LaTeX is
# installed (e.g. on a replication machine without a TeX distribution).
# ---------------------------------------------------------------------------
USE_LATEX = False

if USE_LATEX:
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    })
else:
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        # cmr10 is Computer Modern and ships with matplotlib; "Computer Modern
        # Roman" is not a name matplotlib resolves, so it used to fall through
        # to DejaVu Serif and the figure text did not match the LaTeX text.
        "font.serif": ["cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,   # cmr10 has no U+2212
    })

mpl.rcParams.update({
    "axes.labelsize": 14, "axes.titlesize": 15,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 12, "axes.linewidth": 0.9,
    "grid.linewidth": 0.6, "lines.linewidth": 1.2,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Shared loaders for the forecasting-study pickles
# ---------------------------------------------------------------------------
def load_scores(fname):
    """Pool the per-run score frames (df_scores_run) for one scenario."""
    with open(data_path(fname), "rb") as f:
        all_runs = pickle.load(f)
    frames = [
        all_runs[r]["df_scores_run"]
        for r in sorted(all_runs)
        if r < N_RUNS_COMPARE and len(all_runs[r]["df_scores_run"]) > 0
    ]
    df = pd.concat(frames, ignore_index=True)
    return df[df["freq"] == KEEP_FREQ].copy()


def load_timing(fname):
    """Pool the per-run vine timing frames (df_time_vine_run) for one scenario."""
    with open(data_path(fname), "rb") as f:
        all_runs = pickle.load(f)
    frames = [
        all_runs[r]["df_time_vine_run"]
        for r in sorted(all_runs)
        if r < N_RUNS_COMPARE
        and "df_time_vine_run" in all_runs[r]
        and len(all_runs[r]["df_time_vine_run"]) > 0
    ]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df[df["freq"] == KEEP_FREQ].copy()


# =============================================================================
# PAPER FIGURE 2 (fig:beta_estimates) -> figures/beta_tree_estimates_merged.pdf
# "Monte Carlo distribution of beta_j across trees for n=1000 and n=5000."
# =============================================================================
def _load_batch_study(sample_size):
    out = {}
    for tag in ("betas", "C", "C_true"):
        with open(data_path(f"{SETTING_NAME}_{tag}_{sample_size}.pkl"), "rb") as f:
            out[tag] = pickle.load(f)
    return out


def _assemble_test_array(betas_pred, Q=10):
    """Reproduce the (RUNS, J, Q) tensor: one randomly selected family per edge."""
    settings = list({key[0] for key in betas_pred})
    runs = list({key[1] for key in betas_pred})

    all_betas_hat = []
    for setting_name, run_idx in product(settings, runs):
        betas_hat = betas_pred[(setting_name, run_idx)]
        selected = []
        for row in betas_hat:
            row = np.asarray(row, dtype=object)
            valid_idx = np.where(~pd.isna(row))[0]
            selected.append(np.nan if len(valid_idx) == 0
                            else row[np.random.choice(valid_idx)])
        all_betas_hat.append(np.array(selected, dtype=object))

    RUNS = len(all_betas_hat)
    J = len(all_betas_hat[0]) - 1
    test = np.full((RUNS, J, Q), np.nan)
    for i in range(RUNS):
        r = all_betas_hat[i]
        for j in range(J):
            cell = r[j]
            if isinstance(cell, float) and np.isnan(cell):
                continue
            test[i, j, :] = cell[0][0].ravel()
    return test


def _draw_beta_panel(ax, test_arr, true_betas, title):
    _, n_trees, Q = test_arr.shape
    means = np.nanmean(test_arr, axis=0)
    q5 = np.nanpercentile(test_arr, 5, axis=0)
    q95 = np.nanpercentile(test_arr, 95, axis=0)

    x_base = np.arange(Q)
    group_width = 0.6
    offsets = (np.arange(n_trees) - (n_trees - 1) / 2) * (group_width / n_trees)
    colors = plt.cm.tab10(np.linspace(0, 1, n_trees))

    for t in range(n_trees):
        m, ql, qu = means[t], q5[t], q95[t]
        ax.errorbar(x_base + offsets[t], m, yerr=[m - ql, qu - m], fmt="o",
                    color=colors[t], capsize=3, markersize=4, linewidth=1,
                    label=f"Tree {t + 1}")

    line_width = group_width / 2
    for i, true_val in enumerate(true_betas):
        ax.plot([i - line_width, i + line_width], [true_val, true_val],
                color="black", linestyle=":", linewidth=1.5, alpha=0.8)

    ax.axhline(y=0, color="gray", linewidth=0.8, alpha=0.6)
    ax.set_xlim(-0.5, Q - 0.5)
    ax.set_ylim(-0.4, 0.4)
    ax.set_xticks(x_base)
    ax.set_xticklabels([f"{i + 1}" for i in range(Q)])
    ax.set_xlabel("Coefficient index")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    return n_trees, colors


Q = 10
_true_betas = np.array([0.25, 0.25, -0.25, 0, 0, 0.25, -0.25, -0.25, 0, 0])[:Q]

test_by_n = {}
for n in SAMPLE_SIZES:
    print(f"[Fig 2] loading batch study n = {n} ...")
    test_by_n[n] = _assemble_test_array(_load_batch_study(n)["betas"], Q=Q)

fig2, axes2 = plt.subplots(1, 2, figsize=(13.5, 5.8), sharey=True, squeeze=False)
_n_trees = _colors = None
for j, n in enumerate(SAMPLE_SIZES):
    _n_trees, _colors = _draw_beta_panel(axes2[0, j], test_by_n[n], _true_betas,
                                         title=rf"$n = {n}$")
axes2[0, 0].set_ylabel(r"$\beta_j$")

_beta_handles = [Line2D([0], [0], marker="o", color=_colors[t], linestyle="",
                        markersize=5, label=f"Tree {t + 1}") for t in range(_n_trees)]
_beta_handles.append(Line2D([0], [0], color="black", linestyle=":", linewidth=1.5,
                            label="True value"))
fig2.legend(handles=_beta_handles, loc="lower center", ncol=len(_beta_handles),
            frameon=False, bbox_to_anchor=(0.5, -0.02))
fig2.tight_layout(rect=[0, 0.06, 1, 1])
_out = FIG_DIR / "beta_tree_estimates_merged.pdf"
fig2.savefig(_out, bbox_inches="tight")
print(f"saved {_out}")


# =============================================================================
# PAPER FIGURE 3 (fig:ols_vs_ridge_boxplots) -> input/boxplot_batch_online.pdf
# "Distributions of log-scores across updating schemes, estimation methods, and
#  forgetting factors gamma." Uses the Stable (standard) DGP of Section 3.2.
# =============================================================================
def _fmt_gamma(val):
    return rf"$\gamma = {val:.4f}$"


def _draw_score_boxplot(ax, cell_df, title, forget_values):
    data, positions, colors, centers, labels = [], [], [], [], []
    for gi, fv in enumerate(forget_values):
        ols = cell_df.loc[(cell_df["regularization"] == 0) & (cell_df["forget"] == fv),
                          "model"].dropna().values
        ridge = cell_df.loc[(cell_df["regularization"] == 1) & (cell_df["forget"] == fv),
                            "model"].dropna().values
        data.append(ols);   positions.append(gi - 0.22); colors.append(REG_COLORS[0])
        data.append(ridge); positions.append(gi + 0.22); colors.append(REG_COLORS[1])
        centers.append(gi); labels.append(_fmt_gamma(fv))

    gi = len(forget_values)
    gvals = cell_df.drop_duplicates(subset="run_id")["gaussian"].dropna().values
    data.append(gvals); positions.append(gi); colors.append(GAUSS_COLOR)
    centers.append(gi); labels.append("Gaussian")

    bp = ax.boxplot(data, positions=positions, widths=0.38, patch_artist=True,
                    showfliers=True,
                    flierprops=dict(marker="o", markersize=3.0, markerfacecolor="none",
                                    markeredgecolor="black", markeredgewidth=0.6),
                    medianprops=dict(color="black", linewidth=1.2),
                    boxprops=dict(linewidth=0.85), whiskerprops=dict(linewidth=0.85),
                    capprops=dict(linewidth=0.85))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_edgecolor("black")

    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)


print("[Fig 3] loading Stable scores ...")
_box_df = load_scores(STABLE_FILE)
_clip_lo, _clip_hi = np.percentile(_box_df["model"].dropna().values, [2, 98])
_box_df = _box_df[(_box_df["model"] >= _clip_lo) & (_box_df["model"] <= _clip_hi)].copy()
print(f"[Fig 3] clipped model scores to [{_clip_lo:.4f}, {_clip_hi:.4f}]")
_box_forget = sorted(_box_df["forget"].dropna().unique())

fig3, axes3 = plt.subplots(1, 2, figsize=(13.5, 5.8), sharey=True, squeeze=False)
for j, est in enumerate([0, 1]):
    cell = _box_df[_box_df["estimation"] == est].copy()
    _draw_score_boxplot(axes3[0, j], cell, EST_NAMES[est], _box_forget)
axes3[0, 0].set_ylabel("Log score")

_box_handles = [Patch(facecolor=REG_COLORS[0], edgecolor="black", label="OLS"),
                Patch(facecolor=REG_COLORS[1], edgecolor="black", label="Ridge"),
                Patch(facecolor=GAUSS_COLOR, edgecolor="black", label="Gaussian")]
fig3.legend(handles=_box_handles, loc="lower center", ncol=3, frameon=False,
            bbox_to_anchor=(0.5, -0.02))
fig3.tight_layout(rect=[0, 0.06, 1, 1])
_out = FIG_DIR / "boxplot_batch_online.pdf"
fig3.savefig(_out, bbox_inches="tight")
print(f"saved {_out}")


# =============================================================================
# PAPER TABLE 2 (tab:time_ridge) + APPENDIX TABLE (tab:time_ols)
#   -> input/table_vine_time_ridge_avg.tex, input/table_vine_time_ols_avg.tex
# Average vine estimation time per update = sum(total_time) / sum(n_updates),
# per (estimation, gamma) cell. Both paper tables report the Stable scenario.
# =============================================================================
_timing_dfs = {name: load_timing(fname) for name, fname in SCENARIOS}


def _build_time_grid(scenario, reg):
    """Average vine time per update, rows = Batch/Online, cols = gamma."""
    t = _timing_dfs.get(scenario, pd.DataFrame())
    if t.empty:
        return None
    sub = t[t["regularization"] == reg]
    grid = pd.DataFrame(index=["Batch", "Online"], columns=GAMMA_ORDER, dtype=float)
    for est, lab in EST_NAMES.items():
        for g in GAMMA_ORDER:
            cell = sub[(sub["estimation"] == est) & (sub["forget"] == g)]
            tot_n = cell["n_updates"].sum()
            grid.loc[lab, g] = cell["total_time"].sum() / tot_n if tot_n > 0 else np.nan
    return grid


def _write_time_tex(scenario, reg, fname):
    grid = _build_time_grid(scenario, reg)
    if grid is None:
        print(f"[timing] scenario {scenario} unavailable; skipped {fname}")
        return
    headers = [GAMMA_LABEL[g] for g in GAMMA_ORDER]
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             " & " + " & ".join(headers) + r" \\", r"\midrule"]
    for row_lab in ["Batch", "Online"]:
        cells = [("" if pd.isna(grid.loc[row_lab, g]) else f"{grid.loc[row_lab, g]:.3f}")
                 for g in GAMMA_ORDER]
        lines.append(f"{row_lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out = TAB_DIR / fname
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {out}")


_write_time_tex("Stable", reg=1, fname="table_vine_time_ridge_avg.tex")
_write_time_tex("Stable", reg=0, fname="table_vine_time_ols_avg.tex")


# =============================================================================
# PAPER FIGURE 8 (fig:break_vs_stable_gamma)
#   -> input/compare_break_vs_stable_score_vs_gamma.pdf
# Mean predictive log score vs forgetting factor gamma, Stable vs Break, for
# batch and online estimation and for the OLS and ridge fits. Score-minimizing
# gamma ringed. (The full scenario comparison lived in compare_break_vs_stable.py;
# only this figure is kept for the paper.)
# =============================================================================
print("[Fig 8] loading Stable + Break scores ...")
_scen_dfs = {name: load_scores(fname) for name, fname in SCENARIOS}
_gamma_forget = sorted(pd.concat(_scen_dfs.values())["forget"].dropna().unique())
_reg_style = {0: ("-", "OLS"), 1: ("--", "Ridge")}

fig8, axes8 = plt.subplots(1, 2, figsize=(12, 5), sharey=True, squeeze=False)
for c, est in enumerate([0, 1]):
    ax = axes8[0, c]
    for scen_name, _ in SCENARIOS:
        df = _scen_dfs[scen_name]
        for reg, (ls, reg_lab) in _reg_style.items():
            sub = df[(df["estimation"] == est) & (df["regularization"] == reg)]
            means = sub.groupby("forget")["model"].mean().reindex(_gamma_forget)
            ax.plot(_gamma_forget, means.values, ls=ls, marker="o", markersize=4,
                    color=SCEN_COLORS[scen_name], label=f"{scen_name} --- {reg_lab}")
            gbest = means.idxmin()
            ax.scatter([gbest], [means.loc[gbest]], s=90, zorder=5,
                       facecolor="none", edgecolor=SCEN_COLORS[scen_name], linewidth=1.6)
    ax.set_title(EST_NAMES[est])
    ax.set_xlabel(r"$\gamma$")
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    if c == 0:
        ax.set_ylabel("Mean log score")
axes8[0, 0].legend(frameon=False, fontsize=10)
fig8.tight_layout()
_out = FIG_DIR / "compare_break_vs_stable_score_vs_gamma.pdf"
fig8.savefig(_out, bbox_inches="tight")
print(f"saved {_out}")

plt.show()
