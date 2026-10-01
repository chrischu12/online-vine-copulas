"""Application data and derived splits.

Importing this module reads the macro-region pickles under ``BASE_PATH`` and
exposes the design matrices, the train/val/test sizes and the copula scaling
mask. Keep this separate from ``constants.py`` so scripts that only need static
configuration never trigger the data load.
"""

import gc
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from online_copula_experiments.application import BASE_PATH

# Application intermediate results live under BASE_PATH/intermediate_application.
APP_DIR = BASE_PATH / "intermediate_application"

# Raw data ------------------------------------------------------------------
with open(APP_DIR / "Xy_data_macro.pkl", "rb") as fh:
    _data = pickle.load(fh)

with open(APP_DIR / "combined_Xy_data_copula_absspread.pkl", "rb") as fh:
    _data_copula = pickle.load(fh)

# Train / val / test splits -------------------------------------------------
END_TRAIN = pd.to_datetime(datetime(2016, 12, 31))
END_VAL = pd.to_datetime(datetime(2017, 12, 31))

_flag = np.where(
    _data["X"]["Scotland"]["Date"] <= END_TRAIN,
    "train",
    np.where(_data["X"]["Scotland"]["Date"] <= END_VAL, "val", "test"),
)
_data["X"]["Scotland"]["flag"] = _flag

_counts = _data["X"]["Scotland"]["flag"].value_counts().to_dict()
N_TRAIN = int(_counts.get("train", 0))
N_VAL = int(_counts.get("val", 0))
N_TRAIN = N_TRAIN + N_VAL  # validation is folded into the training window
N_TEST = int(_counts.get("test", 0))
N = N_TRAIN + N_VAL + N_TEST

# Boolean array marking month starts across the test window; triggers batch refit.
NEW_MONTH = (
    _data["X"]["Scotland"]["new_month"][N_TRAIN : N_TRAIN + N_TEST + N_VAL]
    .to_numpy()
    .astype(bool)
)

# Design matrices -----------------------------------------------------------
_DROP_PATTERNS = ["Date", "flag", "new_month", "region"]


def _drop_date_flag(df):
    return df.drop(columns=_DROP_PATTERNS, errors="ignore")


# Build the design matrices by popping each raw region as its copy is made, so the
# raw and derived copies are never both fully resident (keeps peak memory low).
X_numpy = {}
for _k in list(_data["X"].keys()):
    X_numpy[_k] = _drop_date_flag(_data["X"].pop(_k))
y_numpy = {}
for _k in list(_data["y"].keys()):
    y_numpy[_k] = _drop_date_flag(_data["y"].pop(_k))

X_numpy_copula = _data_copula["X"].loc[
    :, ~_data_copula["X"].columns.str.contains("|".join(_DROP_PATTERNS))
]
y_numpy_copula = _drop_date_flag(_data_copula["y"])

# Scaling masks -------------------------------------------------------------
# Marginal covariates: everything except binary/spline columns.
TO_SCALE = ~(
    X_numpy["Scotland"].columns.str.contains("binary")
    | X_numpy["Scotland"].columns.str.contains("spline")
)
# Copula covariates: spread and mean columns only.
TO_SCALE_COP = X_numpy_copula.columns.str.contains("spread") | X_numpy_copula.columns.str.contains("mean")

# Free the raw pickles: X_numpy / y_numpy / *_copula above are independent copies,
# so the originals (~2.8 GB) are no longer needed. Halves the resident footprint.
X_numpy_copula = X_numpy_copula.copy()
del _data, _data_copula
gc.collect()
