#%%

import datetime
import pickle

import numpy as np
import pandas as pd

from online_copula_experiments.application import BASE_PATH

# Application intermediate results live under BASE_PATH/intermediate_application.
APP_DIR = BASE_PATH / "intermediate_application"
from online_copula_experiments.application.const_and_helper.design import DesignBuilder

#####################################################
# copula data  -  absolute spreads variant
# Spreads are |x_r - x_s| instead of x_r - x_s.
# Output: combined_Xy_data_copula_absspread.pkl
#####################################################
regions = ["Scotland","Northern","Midlands","London","Southern"]

X_dict = {}
y_dict = {}
blocks_dict = {} # store block metadata for later prediction use

for r in regions:
    print(f"\n===== Processing Region {r} =====")

    b = DesignBuilder()

    file_path = APP_DIR / "macro_regions_out" / f"macro_{r}.csv"
    df = pd.read_csv(file_path)

    y_series = df["node_n"].copy()

    # ----------------- LINEAR TERMS -----------------

    b.add_block("x2Twm", df["x2T_weighted.mean_p_max_point"].to_numpy().reshape(-1, 1))
    b.add_block("x2Tsm", df["x2Tsm_point"].to_numpy().reshape(-1, 1))
    b.add_block("WindSpd10", df["WindSpd10_weighted.mean_cell"].to_numpy().reshape(-1, 1))
    b.add_block("WindSpd100", df["WindSpd100_weighted.mean_cell"].to_numpy().reshape(-1, 1))
    b.add_block("sun", df["SSRD_mean_2_Cap"].to_numpy().reshape(-1, 1))
    b.add_block("rain", df["TP_weighted.mean_cell"].to_numpy().reshape(-1, 1))

    # ----------------- BUILD DESIGN MATRIX -----------------
    X_df = b.to_df()
    X_df["region"] = r

    # Store
    X_dict[r] = X_df
    y_dict[r] = y_series
    blocks_dict[r] = dict(b.blocks)  # keep model structure for prediction

print("\nAll regions processed successfully!")


# enforce region order if you want to be explicit
region_order = list(X_dict.keys())

out_df = None

for i in range(len(region_order) - 1):
    r1 = region_order[i]
    r2 = region_order[i + 1]

    df1 = X_dict[r1]
    df2 = X_dict[r2]

    # align on common index (safety)
    common_idx = df1.index.intersection(df2.index)
    df1 = df1.loc[common_idx]
    df2 = df2.loc[common_idx]

    pair = f"{i+1}_{i+2}"

    tmp = pd.DataFrame(index=common_idx)

    # absolute spreads: |x_r - x_s|
    tmp[f"{pair}_x2Twm_spread"] = np.abs(df1["x2Twm"] - df2["x2Twm"])
    #tmp[f"{pair}_x2Twm_mean"]   = 0.5 * (df1["x2Twm"] + df2["x2Twm"])
    tmp[f"{pair}_sun_spread"]   = np.abs(df1["sun"] - df2["sun"])
    tmp[f"{pair}_rain_spread"]  = np.abs(df1["rain"] - df2["rain"])
    tmp[f"{pair}_wind_spread_10"]  = np.abs(df1["WindSpd10"] - df2["WindSpd10"])
    tmp[f"{pair}_wind_spread_100"]  = np.abs(df1["WindSpd100"] - df2["WindSpd100"])

    # x2Tsm
    tmp[f"{pair}_x2Tsm_spread"] = np.abs(df1["x2Tsm"] - df2["x2Tsm"])

    # column-bind
    if out_df is None:
        out_df = tmp
    else:
        out_df = out_df.join(tmp, how="inner")

out_df.head()


########################################
# Builds a flexible block of covariates for all regions
########################################

b = DesignBuilder()
END_TRAIN = pd.to_datetime(datetime.datetime(2017, 12, 31))
file_path = APP_DIR / "macro_regions_out" / f"macro_{"Scotland"}.csv"
df = pd.read_csv(file_path)
df["Date"] = pd.to_datetime(df["Date"])
df["flag"] = np.where(df["Date"] <= END_TRAIN, "train", "test")

b.add_block("t",   df["t"].to_numpy().reshape(-1, 1))

X_df = b.to_df()
# ensure it's categorical with a fixed order
dow_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
df["dow_RpH"] = pd.Categorical(df["dow_RpH"], categories=dow_order, ordered=False)
dow_dummies = pd.get_dummies(df["dow_RpH"], prefix="dow", drop_first=True)
df["sin"] = np.sin(2*np.pi *df["clock_hour"] / 24)
df["cos"]  = np.cos(2*np.pi *df["clock_hour"] / 24)
X_df = pd.concat([X_df, dow_dummies], axis=1)

X_df["Date"] = pd.to_datetime(df["Date"])
X_df["new_month"] = X_df["Date"].dt.to_period("M").ne(X_df["Date"].shift().dt.to_period("M"))

X_df = X_df.drop(columns=["t"])

X_df = pd.concat([out_df, X_df], axis=1)

TO_SCALE = (X_df.columns.str.contains("spread") | X_df.columns.str.contains("mean"))


Y_wide = pd.concat([y_dict[r] for r in regions], axis=1)
Y_wide.columns = regions

y_combined = Y_wide

X_combined = X_df

# Save with distinct filename to distinguish from signed-spread version
with open(APP_DIR / "combined_Xy_data_copula_absspread.pkl", "wb") as f:
    pickle.dump({"X": X_combined, "y": y_combined}, f, protocol=pickle.HIGHEST_PROTOCOL)