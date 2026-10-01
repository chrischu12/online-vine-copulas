"""Application-specific copula equation builders.

These functions live in the settings layer (not the library): they encode the
region-naming convention of this study and are passed to ``fit_vinecopstructure``
as callable ``equation`` arguments. The library evaluates them per edge.
"""

import re

import numpy as np
import pandas as pd


def edge_covariate_equation(X_cols, edge):
    """Equation selecting, for a given vine edge, the edge-specific covariate
    columns (region pair ``r1_r2_*`` / ``r2_r1_*``) plus all non-pair-specific
    columns. Returns an equation dict for copula parameter 0.
    """
    if isinstance(X_cols, pd.DataFrame):
        cols_str = X_cols.columns.astype(str).to_numpy()
    else:
        cols_str = np.asarray(X_cols, dtype=str)

    regions = ["1", "2", "3", "4", "5"]
    r1 = regions[int(edge[0])]
    r2 = regions[int(edge[1])]

    prefix_12 = f"{r1}_{r2}_"
    prefix_21 = f"{r2}_{r1}_"
    is_edge = np.char.startswith(cols_str, prefix_12) | np.char.startswith(cols_str, prefix_21)

    # any pair-specific prefix like "1_2_", "3_5_", etc.
    is_any_pair = np.array([bool(re.match(r"^\d+_\d+_", c)) for c in cols_str])

    idx = np.arange(len(cols_str))[is_edge | (~is_any_pair)]
    return {0: {0: idx}}
