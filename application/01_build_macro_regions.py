#%%

"""
00_build_macro_regions.py

Aggregate the 14 GSP-group node datasets into 5 macro regions and write one
CSV per region (macro_<region>.csv). 

Input : GSPGroup_NetD_final.RData (the NodeData1 list of 14 GSP-group tables)
        placed in BASE_PATH.
Output: <OUT_DIR>/macro_<region>.csv  (5 files)

"""

import warnings
from pathlib import Path

import pandas as pd
import rdata

from online_copula_experiments.application import BASE_PATH

# --- paths ---------------------------------------------------------------
# Single data root: drop the original GSPGroup_NetD_final.RData into BASE_PATH;
# the aggregated macro_<region>.csv are written to the application intermediate
# results folder BASE_PATH/intermediate_application/macro_regions_out.
DATA_DIR = Path(BASE_PATH)
OUT_DIR = DATA_DIR / "intermediate_application" / "macro_regions_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the 14 GSP node data.frames straight from the RData. NodeData1 is an R
# list of data.tables, so we use `rdata` (pyreadr can't read R lists). Note: parsing the ~120 MB
# file takes ~1-2 minutes.
_RDATA_FILE = DATA_DIR / "GSPGroup_NetD_final.RData"
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # POSIXct/Date lack converters; handled in load_gsp
    _NODE_DATA = {
        str(k): v for k, v in rdata.read_rda(_RDATA_FILE)["NodeData1"].items()
    }

TIME_COL = "targetTime"

# Static / calendar variables, taken from the representative GSP
STATIC_VARS = ["Date", "t", "doy", "clock_hour", "dow_RpH", "School_Hol"]

# Covariates required from every GSP in a region
VARS_NEEDED = [
    "node_n",                          # dependent variable (net load)
    "node_n_sm_L1",                    # lagged net load
    "WindSpd100_weighted.mean_cell",
    "WindSpd10_weighted.mean_cell",
    "EMBEDDED_WIND_CAPACITY",
    "TP_weighted.mean_cell",
    "SSRD_mean_2_Cap",
    "x2Tsm_point",
    "x2T_weighted.mean_p_max_point",
    "n2ex",                            # system-wide (kept from rep GSP only)
]

# 14 GSP groups in NodeData1 order -> file letters (no I, no O)
GSP_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P"]

# 14 GSP (1-based, as in the R script) -> 5 macro regions
MACRO_MAP = {
    "Scotland": [1, 2],
    "Northern": [3, 4, 5, 6],
    "Midlands": [7, 8, 9, 10],
    "London":   [11],
    "Southern": [12, 13, 14],
}

# Aggregation rule per covariate (sum = additive, mean = regional average).
AGG = {
    "node_n": "sum",
    "node_n_sm_L1": "sum",
    "WindSpd100_weighted.mean_cell": "mean",
    "WindSpd10_weighted.mean_cell": "mean",
    "EMBEDDED_WIND_CAPACITY": "sum",
    "SSRD_mean_2_Cap": "sum",
    "TP_weighted.mean_cell": "mean",
    "x2T_weighted.mean_p_max_point": "mean",
    "x2Tsm_point": "mean",
}


def load_gsp(idx: int) -> pd.DataFrame:
    """Load one GSP node dataset by its 1-based NodeData1 index (from the RData).

    rdata returns POSIXct/Date columns as raw numerics (seconds / days since the
    epoch); convert the ones used downstream so the output matches the original
    R CSV export.
    """
    df = _NODE_DATA[GSP_LETTERS[idx - 1]].copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], unit="s", utc=True)  # POSIXct seconds
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], unit="D").dt.strftime("%Y-%m-%d")
    for c in df.columns:
        if str(df[c].dtype) == "category":
            df[c] = df[c].astype(str)
    return df


def make_macro_df(gsp_idx, macro_name, rep_idx) -> pd.DataFrame:
    # Representative GSP: static/calendar vars + n2ex
    rep = load_gsp(rep_idx)
    rep_keep = rep[[TIME_COL, *STATIC_VARS, "n2ex"]].copy()

    # Stack the region's GSPs (time + covariates)
    frames = []
    for i in gsp_idx:
        dt = load_gsp(i)[[TIME_COL, *VARS_NEEDED]].copy()
        dt["gsp_id"] = i
        frames.append(dt)
    stacked = pd.concat(frames, ignore_index=True)

    # Keep only timestamps present in ALL GSPs of the region
    n_gsp = len(gsp_idx)
    present = stacked.groupby(TIME_COL)["gsp_id"].nunique()
    keep_times = present.index[present == n_gsp]
    stacked = stacked[stacked[TIME_COL].isin(keep_times)]

    # Aggregate covariates by timestamp (n2ex is not aggregated here)
    macro = stacked.groupby(TIME_COL).agg(AGG).reset_index()

    # Attach static vars + n2ex from the representative GSP
    out = macro.merge(rep_keep, on=TIME_COL, how="left")

    # Column order: region, time, static, n2ex, then covariates in AGG order
    # (reproduces R setcolorder + summarise order exactly)
    out["macro_region"] = macro_name
    lead = ["macro_region", TIME_COL, *STATIC_VARS, "n2ex"]
    ordered = lead + [c for c in AGG]
    out = out[ordered]

    # Sort chronologically and emit ISO-8601 UTC timestamps (as R fwrite does)
    out[TIME_COL] = pd.to_datetime(out[TIME_COL], format="mixed", utc=True)
    out = out.sort_values(TIME_COL).reset_index(drop=True)
    out[TIME_COL] = out[TIME_COL].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return out


def main():
    for name, idx in MACRO_MAP.items():
        df = make_macro_df(idx, name, rep_idx=idx[0])
        # float_format="%.15g" reproduces data.table::fwrite numeric formatting
        df.to_csv(OUT_DIR / f"macro_{name}.csv", index=False, float_format="%.15g")
        print(f"wrote macro_{name}.csv  ({df.shape[0]} rows, {df.shape[1]} cols)")


if __name__ == "__main__":
    main()
