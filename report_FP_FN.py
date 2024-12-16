# environment:
# placebo_api_local

import sys

sys.path.append("/Users/michael.simantov/Documents/mu-placebo-api")
sys.path.append("/Users/michael.simantov/Documents/mu-placebo-api/placebo_api")
sys.path.append("/Users/michael.simantov/Documents/mu-placebo-api/placebo_api/utils")

from typing import NamedTuple, Dict

# from placebo_api.utils import api_utils, date_utils
import api_utils, date_utils
from placebo.utils import snowflake_utils

# from placebo_api.utils.date_utils import LocalizedDateTime
from date_utils import LocalizedDateTime
import pandas as pd
import datetime
from tqdm import tqdm
import pickle
import requests
import io
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import os

import pickle

PURPOSE = "weekly_report"
# PURPOSE = "annual_report"

# run = "SPP"
# run = "MISO"
# run = "ERCOT"

# for the new plan of comparing version we need to find the proper run. This is the same one that we have to use in the file 'analyze_FP_FN.py' right after the comment "# the only change is the next line"
# Where to find the runs and schemes:  https://api1.marginalunit.com/pr-forecast/runs
# run = "SPP-1009"
# run = "miso-1008"
# run = "miso_1112"
# run = "miso_ng"
# run = "miso_1202"
# run = "miso"

# run = "ercot"
run = "ercot_prt_crcl"


def TP(row):
    if row["shadow_price"] > 0 and row["forecast_shadow_price"] > 0:
        return True
    return False


def FN(row):
    if row["shadow_price"] > 0 and row["forecast_shadow_price"] == 0:
        return True
    return False


def FP(row):
    if row["shadow_price"] == 0 and row["forecast_shadow_price"] > 0:
        return True
    return False


# Read the data from the file
with open(f"all_top_contingencies_{run}_NEW.pkl", "rb") as f:
    all_contingencies_OLD = pickle.load(f)
all_contingencies_OLD = all_contingencies_OLD.reset_index()

# fmt: off

OLD_TP = all_contingencies_OLD.loc[  (all_contingencies_OLD["shadow_price"] > 0) & (all_contingencies_OLD["forecast_shadow_price"] > 0)  ]
OLD_FP = all_contingencies_OLD.loc[  (all_contingencies_OLD["shadow_price"] == 0) & (all_contingencies_OLD["forecast_shadow_price"] > 0)  ]
OLD_FN = all_contingencies_OLD.loc[  (all_contingencies_OLD["shadow_price"] > 0) & (all_contingencies_OLD["forecast_shadow_price"] == 0)  ]

Precision = len(OLD_TP) / (len(OLD_TP) + len(OLD_FP))
Recall = len(OLD_TP) / (len(OLD_TP) + len(OLD_FN))

print(f"Precision: {Precision}, Recall: {Recall}")

all_contingencies_OLD["TP"] = all_contingencies_OLD.apply(TP, axis=1)
all_contingencies_OLD["FN"] = all_contingencies_OLD.apply(FN, axis=1)
all_contingencies_OLD["FP"] = all_contingencies_OLD.apply(FP, axis=1)

if PURPOSE == "weekly_report":

    ######weekly report########
    ###### FP ########

    FP_is_Dominant = (all_contingencies_OLD.groupby(["monitored_uid", "contingency_uid"]).agg({"shadow_price": "sum","forecast_shadow_price": "sum","TP": "sum","FN": "sum","FP": "sum",}).reset_index())
    FP_is_Dominant = FP_is_Dominant.rename(columns={"shadow_price": "Accumulated_shadow_price"})
    FP_is_Dominant = FP_is_Dominant.rename(columns={"forecast_shadow_price": "Accumulated_forecast_shadow_price"})
    FP_is_Dominant = FP_is_Dominant.sort_values(["FP","Accumulated_forecast_shadow_price"], ascending=False)[:1000]

    FN_is_Dominant = (all_contingencies_OLD.groupby(["monitored_uid", "contingency_uid"]).agg({"shadow_price": "sum","forecast_shadow_price": "sum","TP": "sum","FN": "sum","FP": "sum",}).reset_index())
    FN_is_Dominant = FN_is_Dominant.sort_values(["FN", "shadow_price"], ascending=False)[:1000]

else:
    ####### Forecast Version Comparison FP ########
    FP_is_Dominant = (all_contingencies_OLD.groupby(["monitored_uid", "contingency_uid"]).agg({"shadow_price": "sum","forecast_shadow_price": "sum","TP": "sum","FN": "sum","FP": "sum",}).reset_index())
    FP_is_Dominant = FP_is_Dominant.rename(columns={"shadow_price": "Accumulated_shadow_price"})
    FP_is_Dominant = FP_is_Dominant.rename(columns={"forecast_shadow_price": "Accumulated_forecast_shadow_price"})
    FP_is_Dominant = FP_is_Dominant[(FP_is_Dominant["FP"] > (FP_is_Dominant["TP"] + FP_is_Dominant["FN"]) * 3)& (FP_is_Dominant["Accumulated_forecast_shadow_price"]> FP_is_Dominant["Accumulated_shadow_price"] * 10)].sort_values("Accumulated_forecast_shadow_price", ascending=False)
    FP_is_Dominant["FP_shadow_price_percentage"] = (FP_is_Dominant["Accumulated_forecast_shadow_price"]/ FP_is_Dominant["Accumulated_forecast_shadow_price"].sum())
    # only keep cases where the FP_shadow_price_percentage is in the top 10%
    cutoff = np.percentile(FP_is_Dominant["FP_shadow_price_percentage"].values, 90)
    FP_is_Dominant = FP_is_Dominant[FP_is_Dominant["FP_shadow_price_percentage"] > cutoff]
    # only keep cases where the FP is more than 10
    FP_is_Dominant = FP_is_Dominant[FP_is_Dominant["FP"] > 10]
    FP_is_Dominant["FP_shadow_price_percentage"] = (  FP_is_Dominant["Accumulated_forecast_shadow_price"] / FP_is_Dominant["Accumulated_forecast_shadow_price"].sum()  )
    FP_is_Dominant = FP_is_Dominant.sort_values("FP_shadow_price_percentage", ascending=False)[:230]

    ####### FN ########
    FN_is_Dominant = (all_contingencies_OLD.groupby(["monitored_uid", "contingency_uid"]).agg({"shadow_price": "sum","forecast_shadow_price": "sum","TP": "sum","FN": "sum","FP": "sum",}).reset_index())
    FN_is_Dominant = FN_is_Dominant[(FN_is_Dominant["FN"] > (FN_is_Dominant["TP"] * 5))].sort_values("FN", ascending=False)
    FN_is_Dominant["TP_to_FN"] = (FN_is_Dominant["FN"]/ (1+FN_is_Dominant["TP"]))
    FN_is_Dominant = FN_is_Dominant.sort_values("TP_to_FN", ascending=False)[:30]



# plt.plot(FP_is_Dominant["FP_shadow_price_percentage"].values)

FP_is_Dominant.sort_values('Accumulated_forecast_shadow_price') / FP_is_Dominant['Accumulated_forecast_shadow_price'].sum()
print(18)

if False:
    # export FP_is_Dominant to an excel file
    # FP_is_Dominant.to_excel("SPP_constraints_to_work_on.xlsx")

    # with pd.ExcelWriter("Interesting_constraints.xlsx") as writer:
    #     FP_is_Dominant.to_excel(writer, sheet_name="README", index=False)

    # Load the existing workbook
    from openpyxl import load_workbook

    book = load_workbook("Interesting_constraints.xlsx")

    # Use ExcelWriter with mode='a' to append to the existing file
    with pd.ExcelWriter(
        "Interesting_constraints.xlsx",
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        writer.book = book
        FN_is_Dominant.to_excel(writer, sheet_name=f"{run}", index=False)


################


# Only_FP_exists = create_FP_data()

# def create_FP_data():
#     Only_FP_exists = (
#         OLD_FP[~OLD_FP.monitored_uid.isin(OLD_TP.monitored_uid)]
#         .groupby(["monitored_uid", "contingency_uid"])
#         .agg({"forecast_shadow_price": "sum", "shadow_price": "count"})
#         .sort_values("forecast_shadow_price", ascending=False)
#     )

#     # add column that shows the ratio between forecast_shadow_price and the sum of forecast_shadow_price for each monitored_uid
#     Only_FP_exists["ratio_of_FS_shadow_Price"] = (
#         Only_FP_exists["forecast_shadow_price"]
#         / Only_FP_exists["forecast_shadow_price"].sum()
#     )

#     Only_FP_exists = Only_FP_exists.sort_values(
#         "ratio_of_FS_shadow_Price", ascending=False
#     )

#     # Change the column names to be more descriptive
#     Only_FP_exists = Only_FP_exists.rename(
#         columns={"shadow_price": "number of days appearing"}
#     )
#     Only_FP_exists = Only_FP_exists.rename(
#         columns={"forecast_shadow_price": "Accumulated_forecast_shadow_price"}
#     )
#     Only_FP_exists = Only_FP_exists.rename(
#         columns={
#             "ratio_of_FS_shadow_Price": "ratio_of_FS_shadow_Price (=share out of the total shadow price in this worksheet) "
#         }
#     )

#     return Only_FP_exists
