"""Static configuration for the application study.

Distributions, model equations, estimation settings and the default
forget/frequency selection. This module loads **no data** — importing it is
cheap and side-effect free, so scripts can pull a single constant without
triggering the data pipeline in ``data.py``.
"""

import os
from pathlib import Path

import numpy as np
import ondil

from online_copula_experiments.application import BASE_PATH

np.set_printoptions(precision=3, suppress=True)

# Distributions -------------------------------------------------------------
DISTRIBUTIONS = {
    "Normal": ondil.distributions.Normal(
        loc_link=ondil.links.Identity(),
        scale_link=ondil.links.Log(),
    )
}

# Per-distribution equations: parameter p gets "all" covariates once p <= i,
# "intercept" only otherwise.
EQUATIONS = {
    dist_name: {
        i: {p: "all" if p <= i else "intercept" for p in range(dist.n_params)}
        for i in range(dist.n_params)
    }
    for dist_name, dist in DISTRIBUTIONS.items()
}

# Estimation settings -------------------------------------------------------
INFORMATION_CRITERION = "aic"
FIT_INTERCEPT = True
REG_INTERCEPT = False
DEBUG = False

# Forget-factor grid and default selection.
FORGETS = dict(enumerate([0]))
freq = 1
f = 0
forget = FORGETS[f]

# Updating scheme ------------------------------------------------------------
# One scheme drives both stages. The marginal models and the copula are always
# estimated the same way, so that every fitted object is a complete forecasting
# model rather than a mixture of two updating regimes. These are the three
# schemes of the paper, under the same names:
#
#   repeated_batch   refit at each month start, frozen within the month
#   repeated_online  refit at each month start, updated within the month
#   online           fit once on the training sample, updated from then on
#
# Set through the SCHEME environment variable, so a run needs no edits. Every
# scheme-specific output goes to intermediate_application/<scheme>/, which is
# also how 03_plots.py recovers the scheme of a result: from its folder.
#
# Pairing the two stages by scheme also dissolves what used to be a constraint.
# A copula batch fit consumes the whole history of in-sample PITs, which the
# marginal stage produces only at a step where it refits, so the copula can
# batch-fit exactly where the marginals do. Under matched schemes that holds by
# construction, and no invalid pairing is expressible any more.
SCHEMES = ("repeated_batch", "repeated_online", "online")
SCHEME = os.environ.get("SCHEME", "repeated_online")
if SCHEME not in SCHEMES:
    raise ValueError(f"SCHEME must be one of {SCHEMES}, got {SCHEME!r}.")

APP_DIR = Path(BASE_PATH) / "intermediate_application"
SCHEME_DIR = APP_DIR / SCHEME

# 02_vine_copula.py covers the two schemes that re-fit monthly,
# 02_vine_copula_full_online.py the one that never does; each skips the schemes
# that are not its own.
RUN_FULL_ONLINE_COPULA = SCHEME == "online"
