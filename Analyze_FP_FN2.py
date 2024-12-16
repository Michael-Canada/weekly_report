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

WITH_IIR = False

run = "spp"
# run = "miso"
# run = "ercot"

run_for_actual_shadow_prices = f"{run}"


min_analysis_date = "2024-01-01"
max_analysis_date = "2024-12-01"

# To know which runs are available, paste this in the browser:
# https://api1.marginalunit.com/pr-forecast/runs
# The way to know that the backfill is done:
# more reliable version: curl -u $SELF -s https://api1.marginalunit.com/pr-forecast/spp-1009/as_ofs | grep 2023-01-02
# the only change is the next line. Also, change the Num_of_days_to_look_back to a range of dates.
# run = "spp-1009"
# run = "miso-1008"
# run = "miso_ng"
# run = "miso"
# run = "miso_1202"
# run = "ercot"
# run = "ercot_prt_crcl"
# run = "ercot_20240909"
run = "spp"


# run = "miso_1112"
# run_for_actual_shadow_prices = "miso"

# Use the following to compare versions of ERCOT
if False:
    run = "ercot"
    # run = "ercot_20240909"
    # run = "ercot_20240311"
    run_for_actual_shadow_prices = "ercot"

SHADOW_PRICE_CUTOFF = 0
MIN_REQUIRED_NUM_OF_HOURS_IN_CONTEGTION = 0

# list of hyperparameters
max_num_congestions_per_day = 1000  # 30
Num_of_days_to_look_back = 400  # 25
percent_of_ratings = [0.99, 0.95, 0.9, 0.85, 0.8]
# percent_of_ratings = [0.99, 0.8]


def process_actual(content_actual):

    df_bc_actual = pd.read_csv(content_actual)
    if len(df_bc_actual) == 0:
        return df_bc_actual
    df_bc_actual["period"] = df_bc_actual["period"].apply(
        datetime.datetime.fromisoformat
    )
    df_bc_actual["date"] = df_bc_actual["period"].apply(lambda x: x.date())

    return df_bc_actual


# The authentication for API is stored in the environment variable MU_API_AUTH
# tuple(os.environ["MU_API_AUTH"].split(":"))


def _get_auth():
    return tuple(os.environ["MU_API_AUTH"].split(":"))
    # return tuple(os.environ["SELF"].split(":"))


AUTH = _get_auth()
MISC_ROOT = "https://api1.marginalunit.com/misc-data"
CONSTRAINTDB_ROOT = "https://api1.marginalunit.com/constraintdb"


def _get_dfm(url):
    resp = requests.get(url, auth=AUTH)

    if resp.status_code != 200:
        print(f"skipping the following: {url}")
        return None

    # dfm = pd.read_csv(
    #     io.StringIO(
    #         resp.text
    #     )
    # )

    return io.StringIO(resp.text)


# the following line is to get the list of as_ofs
content = api_utils.fetch_from_placebo(
    endpoint=f"{run}/as_ofs",
)
as_ofs_str = [line.rstrip("\n") for line in content.readlines() if "as_of" not in line]


class AsOf(NamedTuple):
    as_of_str: str
    as_of: LocalizedDateTime


as_ofs = [
    AsOf(as_of_str=as_of_str, as_of=date_utils.localized_from_isoformat(as_of_str))
    for as_of_str in as_ofs_str
]

hour = 4
as_ofs_last_year = [
    as_of for as_of in as_ofs if as_of.as_of.year >= 2023 and as_of.as_of.hour == hour
]  ##### Michael
# as_ofs_20230320 = [
#     as_of for as_of in as_ofs if as_of.as_of_str == "2023-03-20T04:00:00-05:00"
# ]  ##### Michael

## get actual bc summary on a given as_of
# as_ofs_actual = set(as_ofs_last_year[:Num_of_days_to_look_back])
as_ofs_actual = as_ofs_last_year[:Num_of_days_to_look_back]

as_ofs_actual = [
    AsOf(
        as_of_str=as_of.as_of_str,
        as_of=date_utils.localized_from_isoformat(as_of.as_of_str),
    )
    for as_of in as_ofs_actual
]

## Actual
contents_actual = {}
contents_forecast = {}
contents_actual_old = {}

# the following works only if we look at Num_of_days_to_look_back that is small enough (around 50)
if False:
    date_first = as_ofs_actual[-1].as_of.date().strftime("%Y-%m-%d")
    date_last = (datetime.timedelta(days=1) + as_ofs_actual[0].as_of.date()).strftime(
        "%Y-%m-%d"
    )

    url = f"{CONSTRAINTDB_ROOT}/{run_for_actual_shadow_prices}/rt/binding_events?start_datetime={date_first}&end_datetime={date_last}"
    content = _get_dfm(url)
    if content is None:
        print(f"Error 293847238: no data")
        print(url)
        sdfkkdkjkfjlsdkfj
    df_actual_one = process_actual(content)
    contents_actual.update(
        {
            date_.strftime("%Y-%m-%d"): group
            for date_, group in df_actual_one.groupby(df_actual_one["date"])
        }
    )

for as_of in tqdm(as_ofs_actual[-1::-1]):
    date = as_of.as_of.date()  # zzz
    date_strp = datetime.datetime.strptime(date.strftime("%Y-%m-%d"), "%Y-%m-%d")
    # min_date = datetime.datetime.strptime("2023-09-01", "%Y-%m-%d")
    min_date = datetime.datetime.strptime(min_analysis_date, "%Y-%m-%d")
    max_date = datetime.datetime.strptime(max_analysis_date, "%Y-%m-%d")

    if date_strp < min_date or date_strp > max_date:
        continue

    url = f"{CONSTRAINTDB_ROOT}/{run_for_actual_shadow_prices}/rt/binding_events?start_datetime={date.strftime('%Y-%m-%d')}&end_datetime={(datetime.timedelta(days=1)+date).strftime('%Y-%m-%d')}"
    content = _get_dfm(url)
    if content is None:
        print(f"Error 293847238: no data")
        print(url)
        continue
    df_actual_one = process_actual(content)
    contents_actual_old[as_of.as_of_str] = df_actual_one
    # contents_actual2 = {date: group for date, group in df_actual_one.groupby("date")}
    contents_actual.update(
        {
            date_.strftime("%Y-%m-%d"): group
            for date_, group in df_actual_one.groupby(df_actual_one["date"])
        }
    )

    if WITH_IIR:
        url_FP = f"https://api1.marginalunit.com/pr-forecast/{run}/binding_constraint?as_of={date.strftime('%Y-%m-%d')}T04%3A00%3A00-05%3A00&resample=H&include_empty_timestamps=true&stream_uid=load_100_wind_100_sol_100_ng_100_iir"
        url_FP_winter = f"https://api1.marginalunit.com/pr-forecast/{run}/binding_constraint?as_of={date.strftime('%Y-%m-%d')}T05%3A00%3A00-05%3A00&resample=H&include_empty_timestamps=true&stream_uid=load_100_wind_100_sol_100_ng_100_iir"
        # url_FP = f"https://api1.marginalunit.com/pr-forecast/ercot/binding_constraint?as_of={date.strftime('%Y-%m-%d')}T04%3A00%3A00-05%3A00&resample=H&include_empty_timestamps=true&stream_uid=load_100_wind_100_sol_100_ng_100_iir"
    else:
        url_FP = f"https://api1.marginalunit.com/pr-forecast/{run}/binding_constraint?as_of={date.strftime('%Y-%m-%d')}T04%3A00%3A00-05%3A00&resample=H&include_empty_timestamps=true&stream_uid=load_100_wind_100_sol_100_ng_100"
        url_FP_winter = f"https://api1.marginalunit.com/pr-forecast/{run}/binding_constraint?as_of={date.strftime('%Y-%m-%d')}T05%3A00%3A00-05%3A00&resample=H&include_empty_timestamps=true&stream_uid=load_100_wind_100_sol_100_ng_100"

    content_FP = _get_dfm(url_FP)
    if content_FP is None:
        print(f"Error 92837498234: no data")
        print(url_FP)

    print(date)
    df_bc_actual_FP = pd.read_csv(content_FP)
    No_FP_data_was_found = False
    if len(df_bc_actual_FP) == 0:
        content_FP = _get_dfm(url_FP_winter)
        print("using Winter URL")
        df_bc_actual_FP = pd.read_csv(content_FP)
    if len(df_bc_actual_FP) == 0:
        print("No forecast contingencies were found")
        No_FP_data_was_found = True

    if not No_FP_data_was_found:
        date_later_1 = f"{(datetime.timedelta(days=1)+date).strftime('%Y-%m-%d')}"
        date_later_2 = f"{(datetime.timedelta(days=2)+date).strftime('%Y-%m-%d')}"
        # Forecast_RT_shadow_price_of_the_day = df_bc_actual_FP.query('timestamp > @date_later_1 and timestamp < @date_later_2').groupby(['contingency_uid','monitored_uid','timestamp']).agg({'shadow_price':sum})
        Forecast_RT_shadow_price_of_the_day = df_bc_actual_FP.query(
            "timestamp > @date_later_1 and timestamp < @date_later_2"
        )

        contents_forecast[date_later_1] = Forecast_RT_shadow_price_of_the_day

    # contents_actual is a dict, where the keys are dates. For each date it has all of the constraints that actually bound for that day. That is, their RT shadow price > 0
    # To see all of the contingencies:
    # contents_actual['2024-04-08T04:00:00-05:00'].contingency_uid.unique()

    # To choose one contingency and look at the actual shadow prices:
    # contents_actual['2024-04-08T04:00:00-05:00'].query('contingency_uid == "SBWDDBM5"').sort_values(['period'])

with open(f"bc_{run.upper()}_actual_as_of.pkl", "wb") as f:
    pickle.dump(contents_actual, f)


# the following function finds the top 10 congestions by aggregating over one day the shadow_price on all of the monitored_uid. It returns a dataframe with 3 columns: monitored_uid, contingency_uid, and the sum of the shadow_price for that monitored_uid.
def find_top_congestions(
    df_actual_one, df_forecast_one, num_of_congestions=max_num_congestions_per_day
):
    # top_congestions = (
    #     df_actual_one.groupby(["monitored_uid", "contingency_uid"])
    #     .agg({"shadow_price": "sum"})
    #     .sort_values("shadow_price", ascending=False)
    #     .head(num_of_congestions)
    # )
    top_congestions = (
        df_actual_one.groupby(["monitored_uid", "contingency_uid"])
        .agg({"shadow_price": "sum"})
        .merge(
            df_actual_one.groupby(["monitored_uid", "contingency_uid"])
            .size()
            .reset_index(name="count"),
            on=["monitored_uid", "contingency_uid"],
        )
        .sort_values("shadow_price", ascending=False)
        .head(num_of_congestions)
        .set_index(["monitored_uid", "contingency_uid"])
    )
    top_congestions = top_congestions[
        top_congestions["shadow_price"] > SHADOW_PRICE_CUTOFF
    ]
    top_congestions["forecast_shadow_price"] = 0

    forecast_data = (
        df_forecast_one.groupby(["monitored_uid", "contingency_uid"])
        .agg({"shadow_price": "sum"})
        .sort_values("shadow_price", ascending=False)
    )
    # forecast_data = (
    #     df_forecast_one.groupby(["monitored_uid", "contingency_uid"])
    #     .agg({"shadow_price": "sum"})
    #     .merge(
    #         df_forecast_one.groupby(["monitored_uid", "contingency_uid"])
    #         .size()
    #         .reset_index(name="num_hours"),
    #         on=["monitored_uid", "contingency_uid"],
    #     )
    #     .sort_values("shadow_price", ascending=False)
    #     .set_index(["monitored_uid", "contingency_uid"])
    # )
    forecast_data["forecast_shadow_price"] = forecast_data["shadow_price"]
    forecast_data["shadow_price"] = 0

    for monitored_uid, contingency_uid in top_congestions.index:
        if (
            top_congestions.loc[(monitored_uid, contingency_uid), "count"]
            < MIN_REQUIRED_NUM_OF_HOURS_IN_CONTEGTION
        ):
            # if the forecast is too short, then we don't want to consider this congestion
            top_congestions.drop((monitored_uid, contingency_uid), inplace=True)

    for monitored_uid, contingency_uid in forecast_data.index:
        if (monitored_uid, contingency_uid) in top_congestions.index:

            # Forecast_too_short = False
            Actual_too_short = False
            # if (
            #     forecast_data.loc[(monitored_uid, contingency_uid), "num_hours"]
            #     < MIN_REQUIRED_NUM_OF_HOURS_IN_CONTEGTION
            # ):
            #     Forecast_too_short = True

            if (
                top_congestions.loc[(monitored_uid, contingency_uid), "count"]
                < MIN_REQUIRED_NUM_OF_HOURS_IN_CONTEGTION
            ):
                Actual_too_short = True

            # if Forecast_too_short and Actual_too_short:
            if Actual_too_short:
                # if both forecast and actual congestions are too short, then we don't want to consider this congestion
                top_congestions.drop((monitored_uid, contingency_uid), inplace=True)
                # forecast_data.drop((monitored_uid, contingency_uid), inplace=True)
                continue

            # delete this row from forecast_data
            top_congestions.loc[
                (monitored_uid, contingency_uid), "forecast_shadow_price"
            ] = forecast_data.loc[
                (monitored_uid, contingency_uid), "forecast_shadow_price"
            ]
            forecast_data.drop((monitored_uid, contingency_uid), inplace=True)

        # else:  # we are here if (monitored_uid, contingency_uid) is NOT in top_congestions.index:
        #     if (
        #         top_congestions.loc[(monitored_uid, contingency_uid), "count"]
        #         < MIN_REQUIRED_NUM_OF_HOURS_IN_CONTEGTION
        #     ):
        #         # if the forecast is too short, then we don't want to consider this congestion
        #         top_congestions.drop((monitored_uid, contingency_uid), inplace=True)

    # concatenate the two dataframes
    top_congestions = pd.concat([top_congestions, forecast_data]).sort_values(
        "shadow_price", ascending=False
    )

    # remove raws from top_congestions that have forecast_shadow_price < SHADOW_PRICE_CUTOFF and shadow_price == 0
    top_congestions = top_congestions[
        (top_congestions["forecast_shadow_price"] > SHADOW_PRICE_CUTOFF)
        | (top_congestions["shadow_price"] >= SHADOW_PRICE_CUTOFF)
    ]
    return top_congestions


URL_ROOT = "https://api1.marginalunit.com/muse/api"


MUSE_DATA = {}
cnt_downloaded = 0
cnt_not_downloaded = 0
RT_bindings_NOT_caught_by_MUSE_and_FORECAST = []
collected_results = {}
RT_bindings_NOT_caught_by_MUSE_and_FORECAST = {}

all_top_contingencies = pd.DataFrame()
FP_total = FN_total = TP_total = 0
for as_of in tqdm(as_ofs_actual):

    this_date_ = as_of.as_of.date()  # zzz
    date_strp = datetime.datetime.strptime(this_date_.strftime("%Y-%m-%d"), "%Y-%m-%d")
    # min_date = datetime.datetime.strptime("2023-09-01", "%Y-%m-%d")
    min_date = datetime.datetime.strptime(min_analysis_date, "%Y-%m-%d")
    max_date = datetime.datetime.strptime(max_analysis_date, "%Y-%m-%d")

    if date_strp < min_date or date_strp > max_date:
        continue

    # early_date_for_MUSE_query = (
    #     (as_of.as_of - datetime.timedelta(days=Num_of_days_to_look_back + 1))
    #     .date()
    #     .strftime("%Y-%m-%d")
    # )
    from_date = as_of.as_of.date().strftime("%Y-%m-%d")
    to_date = datetime.datetime.now().strftime("%Y-%m-%d")
    today = from_date
    tomorrow_date = (
        (as_of.as_of + datetime.timedelta(days=1)).date().strftime("%Y-%m-%d")
    )

    if (
        datetime.datetime.now() - datetime.timedelta(days=1)
    ).date() <= as_of.as_of.date():
        continue

    try:
        top_contingencies = find_top_congestions(
            contents_actual[f"{tomorrow_date}"],
            contents_forecast[tomorrow_date],
        )
    except:
        print(f"Error 99398452385: no data for {tomorrow_date}")
        continue

    # add column of date to top_contingencies
    top_contingencies["date"] = tomorrow_date

    # add top_contingencies to all_top_contingencies
    all_top_contingencies = pd.concat([all_top_contingencies, top_contingencies])

    FP_daily = FN_daily = TP_daily = 0
    for row in top_contingencies.iterrows():
        shadow_price = row[1][0]
        hour_count = row[1][2]
        forecast_shadow_price = row[1][2]
        if shadow_price > 0 and forecast_shadow_price > 0:
            TP_daily += 1
            TP_total += 1
        elif shadow_price > 0 and forecast_shadow_price == 0:
            FN_daily += 1
            FN_total += 1
        elif shadow_price == 0 and forecast_shadow_price > 0:
            FP_daily += 1
            FP_total += 1

    collected_results[tomorrow_date] = {"FP": FP_daily, "FN": FN_daily, "TP": TP_daily}

confusion_metrix = {}
for k, v in collected_results.items():
    if (v["FN"] + v["FP"] + v["TP"]) == 0:
        continue
    confusion_metrix[k] = {
        "FP": v["FP"] / (v["FN"] + v["FP"] + v["TP"]),
        "FN": v["FN"] / (v["FN"] + v["FP"] + v["TP"]),
        "TP": v["TP"] / (v["FN"] + v["FP"] + v["TP"]),
    }
    if v["TP"] + v["FP"] == 0:
        confusion_metrix[k]["precision"] = 0
    else:
        confusion_metrix[k]["precision"] = v["TP"] / (v["TP"] + v["FP"])
    if v["TP"] + v["FN"] == 0:
        confusion_metrix[k]["recall"] = 0
    else:
        confusion_metrix[k]["recall"] = v["TP"] / (v["TP"] + v["FN"])

# deal with the Total:
confusion_metrix["Total"] = {
    "FP": FP_total / (FP_total + TP_total),
    "FN": FN_total / (FN_total + TP_total),
    "precision": TP_total / (TP_total + FP_total),
    "recall": TP_total / (TP_total + FN_total),
    "F1": 2 * TP_total / (2 * TP_total + FP_total + FN_total),
}

print(f"TP_total: {TP_total}, FP_total: {FP_total}, FN_total: {FN_total}")
print(confusion_metrix["Total"])
print(18)

# saving data about the contingencies (actual and forecast):
with open(f"all_top_contingencies_{run.upper()}_NEW.pkl", "wb") as f:
    # with open("all_top_contingencies_NEW.pkl", "wb") as f:
    pickle.dump(all_top_contingencies, f)

# with open("ERCOT_SHADOW_PRICE_CUTOFF_50_MIN_HOURS_2_NEW.pkl", "wb") as f:
#     pickle.dump(confusion_metrix, f)
