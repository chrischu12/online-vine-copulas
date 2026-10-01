# %%
from datetime import datetime

import numpy as np
import pandas as pd
import pickle

from online_copula_experiments.application import BASE_PATH

# Application intermediate results live under BASE_PATH/intermediate_application.
APP_DIR = BASE_PATH / "intermediate_application"
from online_copula_experiments.application.const_and_helper.design import DesignBuilder


# ------------------------------------------------------------------------------
# MAIN PROCESS  (marginal design matrices; one per macro region)
# ------------------------------------------------------------------------------
END_TRAIN = pd.to_datetime(datetime(2016, 12, 31))
END_VAL = pd.to_datetime(datetime(2017, 12, 31))
X_dict = {}
y_dict = {}
blocks_dict = {} # store block metadata for later prediction use


regions = ["Scotland","Northern","Midlands","London","Southern"]


for r in regions:
    print(f"\n===== Processing Region {r} =====")
    
    b = DesignBuilder()

    file_path = APP_DIR / "macro_regions_out" / f"macro_{r}.csv"
    df = pd.read_csv(file_path)
    
    df["Date"] = pd.to_datetime(df["Date"])
    conditions = [
        df["Date"] <= END_TRAIN,
        df["Date"] <= END_VAL
    ]
    choices = ["train", "val"]
    df["flag"] = np.select(conditions, choices, default="test")
    y_series = df["node_n"].copy()

    # ----------------- LINEAR TERMS -----------------
    b.add_block("t",   df["t"].to_numpy().reshape(-1, 1))
    b.add_block("t2",  (df["t"].to_numpy()**2).reshape(-1, 1))
    b.add_block("node_n_sm_L1", df["node_n_sm_L1"].to_numpy().reshape(-1, 1))
    b.add_block("WindSpd10", df["WindSpd10_weighted.mean_cell"].to_numpy().reshape(-1, 1))

    # ----------------- SPLINES -----------------
    b.spline_basis(df, "doy", 20, "spline(doy)")
    b.spline_basis(df, "clock_hour", 35, "spline(clock_hour)")
    b.spline_basis(df, "x2T_weighted.mean_p_max_point", 35, "spline(x2T)")
    b.spline_basis(df, "x2Tsm_point", 35, "spline(x2Tsm)")
    b.spline_basis(df, "SSRD_mean_2_Cap", 5, "spline(SSRD)")
    b.spline_basis(df, "n2ex", 10, "spline(n2ex)")
    b.spline_basis(df, "TP_weighted.mean_cell", 10, "spline(TP)")

    # ----------------- FACTORS -----------------
    B_dow = b.factor_basis(df, "dow_RpH", "binary_dow")
    B_sch = b.factor_basis(df, "School_Hol", "binary_School_Hol")

    # ----------------- FACTOR-SMOOTH INTERACTIONS -----------------
    B_hour = b.spline_basis(df, "clock_hour", 35, "dummy_s(clock)")  # second copy for interaction
    b.factor_smooth_interaction(B_hour, B_dow, "spline(clock_hour):dow")
    b.factor_smooth_interaction(B_hour, B_sch, "spline(clock_hour):School_Hol")

    # ----------------- "BY" WIND SMOOTH -----------------
    B_w100 = b.spline_basis(df, "WindSpd100_weighted.mean_cell", 20, "spline(WindSpd100)")
    cap = df["EMBEDDED_WIND_CAPACITY"].to_numpy().reshape(-1, 1)
    b.add_block("spline(WindSpd100):cap", B_w100 * cap)

    # ----------------- BUILD DESIGN MATRIX -----------------
    X_df = b.to_df()
    X_df["region"] = r
    X_df["Date"] = pd.to_datetime(df["Date"])
    X_df["new_month"] = X_df["Date"].dt.to_period("M").ne(X_df["Date"].shift().dt.to_period("M"))

    print(f"Region {r} -> X_df shape: {X_df.shape}, y length: {len(y_series)}")

    # Store
    

    X_dict[r] = X_df
    y_dict[r] = y_series
    blocks_dict[r] = dict(b.blocks)  # keep model structure for prediction

print("\nAll regions processed successfully!")

# Add region indicator column to each X_dict entry
with open(APP_DIR / "Xy_data_macro.pkl", "wb") as f:
    pickle.dump({"X": X_dict, "y": y_dict}, f, protocol=pickle.HIGHEST_PROTOCOL)
    # Rowbind all regions while preserving temporal order within each region

