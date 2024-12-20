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
from datetime import datetime as dtdt

# list of hyperparameters
max_num_congestions_per_day = 30  # 30
Num_of_days_to_look_back = 35  # 9
percent_of_ratings = [0.99, 0.95, 0.9, 0.85, 0.8]
percent_of_ratings = [0.99, 0.9, 0.85, 0.8]
# percent_of_ratings = [0.99, 0.8]

# run = 'ercot'
# run = "miso-1008"
# run = "miso-1023"

# run = "miso"
# run = "miso-1023"
# run_for_actual_shadow_prices = "miso"

# run = "spp-1101"
# run = "spp"
# run_for_actual_shadow_prices = "spp"

run = "ercot"
run_for_actual_shadow_prices = "ercot"

run_name_for_file_saving_names = run_for_actual_shadow_prices.upper()

# min_analysis_date = "2024-11-05"
# max_analysis_date = "2024-11-06"
min_analysis_date = "2024-10-10"
max_analysis_date = "2024-11-10"

USE_SHADOW_PRICE_NOT_FLOW = True


class ConfusionMatrix:
    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TN = -1


counter = 0


def calc_collected_confusion_metrics(
    ConfusionMatrix: ConfusionMatrix,
    tomorrow_date,
    monitored_uid,
    contingency_uid,
    percent_of_rating,
    orig_rating,
    observed_RT_shadow_price,
    forecast_RT_shadow_price,
):

    # the following is a counter that grows by 1 for each new entry:
    global counter
    counter += 1

    total_hours = ConfusionMatrix.TP + ConfusionMatrix.FP + ConfusionMatrix.FN
    total_hours_squared = (
        ConfusionMatrix.TP**2 + ConfusionMatrix.FP**2 + ConfusionMatrix.FN**2
    )
    if not total_hours_squared == 0:
        # false_positive = ConfusionMatrix.FP * ConfusionMatrix.FP / total_hours_squared
        # false_negative = ConfusionMatrix.FN * ConfusionMatrix.FN / total_hours_squared
        # True_positive = ConfusionMatrix.TP * ConfusionMatrix.TP / total_hours_squared
        false_positive = ConfusionMatrix.FP / total_hours
        false_negative = ConfusionMatrix.FN / total_hours
        True_positive = ConfusionMatrix.TP / total_hours

        return True, (
            (
                counter,
                false_positive,
                false_negative,
                ConfusionMatrix.TP,
                ConfusionMatrix.FP,
                ConfusionMatrix.FN,
                tomorrow_date,
                monitored_uid,
                contingency_uid,
                total_hours,
                percent_of_rating,
                orig_rating,
                observed_RT_shadow_price,
                forecast_RT_shadow_price,
            )
        )

    return False, None


# def process_actual_reflow(content_actual):

#     df_bc_actual = pd.read_csv(content_actual)

#     if len(df_bc_actual) == 0:
#         return df_bc_actual
#     df_bc_actual["time_stamp"] = df_bc_actual["timestamp"].apply(
#         datetime.datetime.fromisoformat
#     )
#     df_bc_actual["date"] = df_bc_actual["time_stamp"].apply(lambda x: x.date())

#     return df_bc_actual


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


# # run = 'ercot'
# # run = "miso-1008"
# # run = "miso-1023"
# run = "miso"
# run_for_actual_shadow_prices = "miso"
# run_name_for_file_saving_names = run_for_actual_shadow_prices.upper()

# the following line is to get the list of as_ofs
content = api_utils.fetch_from_placebo(
    endpoint=f"{run}/as_ofs",
)
# min_date_format = datetime.datetime.strptime(min_analysis_date, "%Y-%m-%d")
# max_date_format = datetime.datetime.strptime(max_analysis_date, "%Y-%m-%d")
# as_ofs_str = [line.rstrip("\n") for line in content.readlines() if "as_of" not in line]
as_ofs_str = [
    line.rstrip("\n")
    for line in content.readlines()
    if "as_of" not in line
    and line.rstrip("\n")[:10] >= min_analysis_date
    and line.rstrip("\n")[:10] <= max_analysis_date
]


class AsOf(NamedTuple):
    as_of_str: str
    as_of: LocalizedDateTime


as_ofs = [
    AsOf(as_of_str=as_of_str, as_of=date_utils.localized_from_isoformat(as_of_str))
    for as_of_str in as_ofs_str
]

hour = 4
as_ofs_last_year = [
    as_of for as_of in as_ofs if as_of.as_of.year >= 2024 and as_of.as_of.hour == hour
]  ##### Michael
as_ofs_20230320 = [
    as_of for as_of in as_ofs if as_of.as_of_str == "2023-03-20T04:00:00-05:00"
]  ##### Michael

## get actual bc summary on a given as_of
as_ofs_actual = set(as_ofs_last_year[:Num_of_days_to_look_back])
as_ofs_actual = [
    AsOf(
        as_of_str=as_of.as_of_str,
        as_of=date_utils.localized_from_isoformat(as_of.as_of_str),
    )
    for as_of in as_ofs_actual
]

# url = f"{CONSTRAINTDB_ROOT}/ercot/rt/binding_events?start_datetime={date.strftime('%Y-%m-%d')}&end_datetime={(datetime.timedelta(days=1)+date).strftime('%Y-%m-%d')}"
# url = 'https://api1.marginalunit.com/constraintdb/ercot/rt/binding_events?start_datetime=2024-04-06&end_datetime=2024-04-07'
# url_for_MUSE_and_forecast = 'https://api1.marginalunit.com/constraintdb/ercot/binding_constraints/timeseries?market=rt&monitored_uid=LPLNW_LPLMD_1,LPLNW,115.0,LPLMD,115.0&contingency_uid=SBWDDBM5&start_datetime=2024-04-06T11:00:00-05:00&end_datetime=2024-04-09T11:28:00-05:00&frequency=1h&raw_results=false&columns=shadow_price&bc_event_type=all'

# content = _get_dfm(url)
# df_actual_one = process_actual(content)

## Actual
contents_actual = {}
contents_forecast = {}


for as_of in tqdm(as_ofs_actual):
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
    contents_actual[as_of.as_of_str] = df_actual_one

    url_FP = f"https://api1.marginalunit.com/pr-forecast/{run}/binding_constraint?as_of={date.strftime('%Y-%m-%d')}T04%3A00%3A00-05%3A00&resample=H&include_empty_timestamps=true&stream_uid=load_100_wind_100_sol_100_ng_100_iir"
    # url_FP = f"https://api1.marginalunit.com/pr-forecast/ercot/binding_constraint?as_of={date.strftime('%Y-%m-%d')}T04%3A00%3A00-05%3A00&resample=H&include_empty_timestamps=true&stream_uid=load_100_wind_100_sol_100_ng_100_iir"

    content_FP = _get_dfm(url_FP)
    if content_FP is None:
        print(f"Error 92837498234: no data")
        print(url_FP)
        continue

    df_bc_actual_FP = pd.read_csv(content_FP)
    No_FP_data_was_found = False
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

        # print(18)

    # contents_actual is a dict, where the keys are dates. For each date it has all of the constraints that actually bound for that day. That is, their RT shadow price > 0
    # To see all of the contingencies:
    # contents_actual['2024-04-08T04:00:00-05:00'].contingency_uid.unique()

    # To choose one contingency and look at the actual shadow prices:
    # contents_actual['2024-04-08T04:00:00-05:00'].query('contingency_uid == "SBWDDBM5"').sort_values(['period'])

# with open(f"bc_{run_name_for_file_saving_names}_actual_as_of.pkl", "wb") as f:
#     pickle.dump(contents_actual, f)


def _get_dataframe(muse_path, method=requests.get):
    url = URL_ROOT + muse_path
    # print(url)
    resp = method(url, auth=AUTH)
    if resp.status_code != 200:
        print(resp.text)

    try:
        resp.raise_for_status()
    except:
        print(resp.text)
        print("failed URL:", muse_path)
        return None
    return pd.read_csv(io.StringIO(resp.text))


# the following function finds the top 10 congestions by aggregating over one day the shadow_price on all of the monitored_uid. It returns a dataframe with 3 columns: monitored_uid, contingency_uid, and the sum of the shadow_price for that monitored_uid.
def find_top_congestions(
    df_actual_one, df_forecast_one, num_of_congestions=max_num_congestions_per_day
):
    top_congestions = (
        df_actual_one.groupby(["monitored_uid", "contingency_uid"])
        .agg({"shadow_price": "sum"})
        .sort_values("shadow_price", ascending=False)
        .head(num_of_congestions)
    )
    top_congestions = top_congestions[top_congestions["shadow_price"] > 50]
    # top_congestions["forecast_shadow_price"] = 0

    forecast_data = (
        df_forecast_one.groupby(["monitored_uid", "contingency_uid"])
        .agg({"shadow_price": "sum"})
        .sort_values("shadow_price", ascending=False)
    )
    forecast_data["forecast_shadow_price"] = forecast_data["shadow_price"]
    forecast_data["shadow_price"] = 0

    merged_df = top_congestions.merge(
        forecast_data[["forecast_shadow_price"]],
        left_index=True,
        right_index=True,
        how="outer",
    )
    merged_df["shadow_price"] = merged_df["shadow_price"].fillna(0)
    merged_df["forecast_shadow_price"] = merged_df["forecast_shadow_price"].fillna(0)

    return merged_df

    # for monitored_uid, contingency_uid in forecast_data.index:
    #     if (monitored_uid, contingency_uid) in top_congestions.index:
    #         # delete this row from forecast_data
    #         top_congestions.loc[
    #             (monitored_uid, contingency_uid), "forecast_shadow_price"
    #         ] = forecast_data.loc[
    #             (monitored_uid, contingency_uid), "forecast_shadow_price"
    #         ]
    #         forecast_data.drop((monitored_uid, contingency_uid), inplace=True)

    # # concatenate the two dataframes
    # top_congestions = pd.concat([top_congestions, forecast_data]).sort_values(
    #     "shadow_price", ascending=False
    # )

    # congestions_with_interest = top_congestions[
    #     (top_congestions["forecast_shadow_price"] > 50)
    #     | (top_congestions["shadow_price"] > 0)
    # ]
    # return congestions_with_interest

    # return top_congestions  # ZZZ change 1106


URL_ROOT = "https://api1.marginalunit.com/muse/api"

conf_matrix = ConfusionMatrix()

MUSE_DATA = {}
cnt_downloaded = 0
cnt_not_downloaded = 0
RT_bindings_NOT_caught_by_MUSE_and_FORECAST = []
collected_results = {}
RT_bindings_NOT_caught_by_MUSE_and_FORECAST = {}

for as_of in tqdm(as_ofs_actual):
    early_date_for_MUSE_query = (
        (as_of.as_of - datetime.timedelta(days=Num_of_days_to_look_back + 1))
        .date()
        .strftime("%Y-%m-%d")
    )
    from_date = as_of.as_of.date().strftime("%Y-%m-%d")
    to_date = datetime.datetime.now().strftime("%Y-%m-%d")
    today = from_date
    tomorrow_date = (
        (as_of.as_of + datetime.timedelta(days=1)).date().strftime("%Y-%m-%d")
    )
    # if not tomorrow_date == '2024-06-18':
    #     continue

    if (
        datetime.datetime.now() - datetime.timedelta(days=1)
    ).date() <= as_of.as_of.date():
        continue

    try:
        top_contingencies = find_top_congestions(
            contents_actual[f"{tomorrow_date}T04:00:00-05:00"],
            contents_forecast[tomorrow_date],
        )
    except:
        print(f"ERROR: 23409230948")
        continue
        # top_contingencies = find_top_congestions(contents_actual[f'{tomorrow_date}T04:00:00-05:00'], contents_forecast[tomorrow_date])

    # collected_results[tomorrow_date] = collectedConfusionMatrix()
    collected_results[tomorrow_date] = []
    RT_bindings_NOT_caught_by_MUSE_and_FORECAST[tomorrow_date] = []

    print(f"working on {tomorrow_date}")
    for monitored_uid, contingency_uid in top_contingencies.index:
        # if not (contingency_uid == 'DNORSD85' and tomorrow_date == '2024-05-01'):
        #     continue

        # if not 'SSHKCRI8' == contingency_uid:
        #     continue
        print(f"working on {monitored_uid} and {contingency_uid}")
        conf_matrix = ConfusionMatrix()
        constraint_uid = f"{monitored_uid} ;;; {contingency_uid}"

        # save run time by downloading the data only once (i.e., if we have already ran this API call, then we don't need to run it again)
        # if not constraint_uid in MUSE_DATA:  # zzz

        #     # Use MUSE as truth:
        #     # url = f"/{run_for_actual_shadow_prices}/constraint_flow.csv?uid={constraint_uid}&from_date={early_date_for_MUSE_query}&to_date={to_date}&resample_rate=1h"
        #     # MUSE_constraint_flow = _get_dataframe(url)
        #     # if MUSE_constraint_flow is None:
        #     #     print(f"Error: no data for {constraint_uid}")
        #     #     print(url)
        #     #     continue

        #     # Use reflow as truth
        #     df_contingencies = _get_dfm(
        #         f"https://api1.marginalunit.com/reflow/{run}-se/constraint/flow?monitored_uid={monitored_uid}&contingency_uid={contingency_uid}"
        #     )
        #     if df_contingencies is None:
        #         print(
        #             f"Error: no data for the constraint {constraint_uid} with contingency_uid {monitored_uid}"
        #         )
        #         print(url)
        #         continue
        #     MUSE_constraint_flow = process_actual_reflow(df_contingencies)
        #     if MUSE_constraint_flow is None or len(MUSE_constraint_flow) == 0:
        #         print(f"Error: no data for {constraint_uid}")
        #         print(url)
        #         continue

        #     MUSE_DATA[constraint_uid] = MUSE_constraint_flow
        #     cnt_downloaded += 1
        # else:
        #     MUSE_constraint_flow = MUSE_DATA[constraint_uid]
        #     cnt_not_downloaded += 1

        # print(18)
        # try:
        #     MUSE_constraint_flow_tomorrow = (
        #         MUSE_constraint_flow[
        #             # MUSE_constraint_flow["time_stamp"].apply(lambda x: x[:10])
        #             # .strftime("%Y-%m-%d")
        #             # MUSE_constraint_flow["time_stamp"].dt.date
        #             MUSE_constraint_flow["time_stamp"].apply(lambda x: x.date)
        #             == dtdt.strptime(tomorrow_date, "%Y-%m-%d").date()
        #         ]
        #         .sort_values("time_stamp")
        #         .reset_index(drop=True)
        #     )
        # except:
        #     print(18)
        #     continue
        #########
        # extraploate MUSE_constraint_flow_tomorrow so that it has 24 hours
        # if not len(MUSE_constraint_flow_tomorrow) == 24:
        #     # Assuming 'timestamp_column_name' is the name of your timestamp column
        #     timestamp_column_name = "time_stamp"

        # # Convert the timestamp column to datetime format
        # MUSE_constraint_flow_tomorrow[timestamp_column_name] = pd.to_datetime(
        #     MUSE_constraint_flow_tomorrow[timestamp_column_name]
        # )

        # # Determine the minimum and maximum dates
        # min_date = MUSE_constraint_flow_tomorrow[
        #     timestamp_column_name
        # ].dt.date.min()
        # max_date = MUSE_constraint_flow_tomorrow[
        #     timestamp_column_name
        # ].dt.date.max()

        # Generate a complete datetime range covering every hour between the min and max dates
        # full_datetime_range = pd.date_range(
        #     start=min_date,
        #     end=max_date + pd.Timedelta(days=1),
        #     freq="H",
        #     closed="left",
        # )

        # Create a DataFrame from this complete datetime range
        # df_full_range = pd.DataFrame(
        #     full_datetime_range, columns=[timestamp_column_name]
        # )

        # Convert the 'timestamp' column to datetime with timezone awareness
        # MUSE_constraint_flow_tomorrow["timestamp"] = pd.to_datetime(
        #     MUSE_constraint_flow_tomorrow["timestamp"]
        # )

        # Remove timezone information from MUSE_constraint_flow_tomorrow's timestamp
        # MUSE_constraint_flow_tomorrow["timestamp"] = MUSE_constraint_flow_tomorrow[
        #     "timestamp"
        # ].dt.tz_localize(None)

        # Merge the new DataFrame with the original, ensuring all hours are included
        # MUSE_constraint_flow_filled = pd.merge(
        #     df_full_range,
        #     MUSE_constraint_flow_tomorrow,
        #     left_on="time_stamp",
        #     right_on="timestamp",
        #     how="left",
        # )

        # ORIG:
        # # Merge the new DataFrame with the original, ensuring all hours are included
        # MUSE_constraint_flow_filled = pd.merge(
        #     df_full_range,
        #     MUSE_constraint_flow_tomorrow,
        #     on=timestamp_column_name,
        #     how="left",
        # )

        # Fill missing 'constraint_flow' values with 0 and forward fill 'rating' values
        # MUSE_constraint_flow_filled["constraint_flow"] = (
        #     MUSE_constraint_flow_filled["constraint_flow"].fillna(method="ffill")
        # )

        # # ORIG:
        # MUSE_constraint_flow_filled["constraint_flow"] = (
        #     MUSE_constraint_flow_filled["constraint_flow"].fillna(0)
        # )

        # the following line fills up the rating values that are missing with the previous value
        # MUSE_constraint_flow_filled["rating"] = MUSE_constraint_flow_filled[
        #     "rating"
        # ].ffill()

        # MUSE_constraint_flow_tomorrow = MUSE_constraint_flow_filled
        # DIE
        # continue

        #########

        # MUSE_constraint_flow['time_stamp'] = pd.to_datetime(MUSE_constraint_flow['time_stamp'],utc=True).dt.tz_convert('US/Central')

        # find historical forecast #zzz
        # url = f'https://api1.marginalunit.com/pr-forecast/ercot/constraint/lookahead_timeseries?monitored_uid={monitored_uid}&contingency_uid={contingency_uid}&stream_uid=load_100_wind_100_sol_100_ng_100_iir&from_datetime={from_date}'  #works but not what I need. This is foreacast for today's hours, which was created today in the morning
        url = f"https://api1.marginalunit.com/pr-forecast/{run}/constraint?as_of={from_date}T04:00:00-05:00&as_of_match=nearest_before&stream_uid=load_100_wind_100_sol_100_ng_100_iir&monitored_uid={monitored_uid}&contingency_uid={contingency_uid}"  # forecast future days
        content = _get_dfm(url)
        if content is None:
            print(f"Error 238476238476: no data for {constraint_uid}")
            print(url)
            continue
        # content_orig = content
        FORECAST_constraint_flow = pd.read_csv(content)
        FORECAST_constraint_flow["timestamp"] = pd.to_datetime(
            FORECAST_constraint_flow["timestamp"].apply(lambda x: x[:-6])
        ).apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        FORECAST_constraint_flow_tomorrow = (
            FORECAST_constraint_flow[
                FORECAST_constraint_flow["timestamp"].apply(lambda x: x[:10])
                == tomorrow_date
            ][["timestamp", "rating", "constraint_flow"]]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        if (
            len(FORECAST_constraint_flow_tomorrow)
            == 0
            # or len(MUSE_constraint_flow_tomorrow) == 0
        ):
            continue  # no data for this day
            print(
                f"Issue:  No data for FORECAST or MUSE for this date {tomorrow_date}. 9839845937487574"
            )

        if not len(FORECAST_constraint_flow_tomorrow) == 24:
            # Assuming 'timestamp_column_name' is the name of your timestamp column
            timestamp_column_name = "timestamp"

            # Convert the timestamp column to datetime format
            FORECAST_constraint_flow_tomorrow[timestamp_column_name] = pd.to_datetime(
                FORECAST_constraint_flow_tomorrow[timestamp_column_name]
            )

            # Determine the minimum and maximum dates
            min_date = FORECAST_constraint_flow_tomorrow[
                timestamp_column_name
            ].dt.date.min()
            max_date = FORECAST_constraint_flow_tomorrow[
                timestamp_column_name
            ].dt.date.max()

            # Generate a complete datetime range covering every hour between the min and max dates
            full_datetime_range = pd.date_range(
                start=min_date,
                end=max_date + pd.Timedelta(days=1),
                freq="H",
                closed="left",
            )

            # Create a DataFrame from this complete datetime range
            df_full_range = pd.DataFrame(
                full_datetime_range, columns=[timestamp_column_name]
            )

            # Merge the new DataFrame with the original, ensuring all hours are included
            FORECAST_constraint_flow_filled = pd.merge(
                df_full_range,
                FORECAST_constraint_flow_tomorrow,
                on=timestamp_column_name,
                how="left",
            )

            # Fill missing 'constraint_flow' values with 0 and forward fill 'rating' values
            FORECAST_constraint_flow_filled["constraint_flow"] = (
                FORECAST_constraint_flow_filled["constraint_flow"].fillna(0)
            )

            # the following line fills up the rating values that are missing with the previous value
            FORECAST_constraint_flow_filled["rating"] = FORECAST_constraint_flow_filled[
                "rating"
            ].ffill()

            FORECAST_constraint_flow_tomorrow = FORECAST_constraint_flow_filled
            # DIE
            # continue

        ############ Create a DataFrame for RT shadow prices with all hours of the day
        try:
            RT_data_for_this_date = (
                contents_actual[f"{tomorrow_date}T04:00:00-05:00"]
                .query(
                    "monitored_uid == @monitored_uid and contingency_uid == @contingency_uid"
                )
                .sort_values("period")
                .reset_index(drop=True)
            )
        except:
            continue

        if (
            len(RT_data_for_this_date) == 0
        ):  # this is a FP case, for which we don't have RT data (by definition)
            # add a row with all zeros
            new_row = pd.DataFrame(
                {
                    "period": [pd.to_datetime(tomorrow_date)],
                    "monitored_uid": [monitored_uid],
                    "contingency_uid": [contingency_uid],
                    "shadow_price": [0],
                    "date": [tomorrow_date],
                }
            )

            RT_data_for_this_date = RT_data_for_this_date.append(
                new_row, ignore_index=True
            )

        # Convert 'period' to datetime
        RT_data_for_this_date["period"] = pd.to_datetime(
            RT_data_for_this_date["period"]
        )
        # RT_data_for_this_date['period'] = RT_data_for_this_date['period'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Get the date
        date = RT_data_for_this_date["period"].dt.date[0]

        # Set start_date and end_date to cover the full day
        start_date = pd.Timestamp(date)
        end_date = start_date + pd.Timedelta(days=1, hours=-1)

        # Generate all hourly timestamps between start_date and end_date
        all_hours = pd.date_range(start_date, end_date, freq="H")

        # Create a DataFrame from all_hours
        all_hours_df = pd.DataFrame(all_hours, columns=["period"])

        # Convert 'period' in RT_data_for_this_date to datetime64[ns]
        # RT_data_for_this_date['period'] = pd.to_datetime(RT_data_for_this_date['period'])
        RT_data_for_this_date["period"] = RT_data_for_this_date["period"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        RT_data_for_this_date["period"] = pd.to_datetime(
            RT_data_for_this_date["period"]
        )

        # Merge the new DataFrame with the original one
        RT_data_for_this_date = pd.merge(
            all_hours_df, RT_data_for_this_date, on="period", how="left"
        ).sort_values("period")
        RT_data_for_this_date[RT_data_for_this_date["shadow_price"].isnull()] = 0
        ############

        for percent_of_rating in percent_of_ratings:

            # Michael replaces this with the following: 2024-06-14
            # # if no constraint flow is given then skip
            # if (all(FORECAST_constraint_flow_tomorrow['constraint_flow'] <= 10) is False) and (all(FORECAST_constraint_flow_tomorrow['constraint_flow'] > 10) is False):   #this is not a number!
            #     break
            # if no rating is given then skip:
            # if (all(FORECAST_constraint_flow_tomorrow['rating'] <= 10) is False) and (all(FORECAST_constraint_flow_tomorrow['rating'] > 10) is False):   #this is not a number!
            #     break

            # identify if the constraint_flow column in FORECAST_constraint_flow_tomorrow contains any non-numeric values
            nan_constraint_flow = FORECAST_constraint_flow_tomorrow[
                "constraint_flow"
            ].isnull()
            nan_rating = FORECAST_constraint_flow_tomorrow["rating"].isnull()

            # Print rows with NaN in 'constraint_flow'
            if nan_constraint_flow.any():
                print(
                    "Rows with NaN were found in 'constraint_flow'.   883847928387483"
                )
                break

            # Print rows with NaN in 'rating'
            if nan_rating.any():
                print("Rows with NaN were found in 'rating'.   8777347238847737")
                break

            if (
                len(
                    contents_forecast[tomorrow_date].query(
                        "monitored_uid == @monitored_uid and contingency_uid == @contingency_uid"
                    )
                )
                > 0
            ):
                Forecast_RT_shadow_prices = contents_forecast[tomorrow_date].query(
                    "monitored_uid == @monitored_uid and contingency_uid == @contingency_uid"
                )
                # Forecast_RT_shadow_prices represents 24 hours of a day. However, in the timestamp there may be only some of the 24 hours. Wherever Forecast_RT_shadow_prices' timestamp does not exist we have to add shadow price = 0:
                # Forecast_RT_shadow_prices['timestamp'] = pd.to_datetime(For   ecast_RT_shadow_prices['timestamp'])
                Forecast_RT_shadow_prices.loc[:, "timestamp"] = pd.to_datetime(
                    Forecast_RT_shadow_prices["timestamp"]
                )
                # Forecast_RT_shadow_prices['timestamp'] = Forecast_RT_shadow_prices['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                Forecast_RT_shadow_prices.loc[:, "timestamp"] = (
                    Forecast_RT_shadow_prices["timestamp"].dt.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )
                # Forecast_RT_shadow_prices['timestamp'] = pd.to_datetime(Forecast_RT_shadow_prices['timestamp'])
                Forecast_RT_shadow_prices.loc[:, "timestamp"] = pd.to_datetime(
                    Forecast_RT_shadow_prices["timestamp"]
                )
                Forecast_RT_shadow_prices = pd.merge(
                    all_hours_df,
                    Forecast_RT_shadow_prices,
                    left_on="period",
                    right_on="timestamp",
                    how="left",
                ).sort_values("period")
                Forecast_RT_shadow_prices[
                    Forecast_RT_shadow_prices["shadow_price"].isnull()
                ] = 0
                Forecast_RT_shadow_prices_array = (
                    Forecast_RT_shadow_prices.shadow_price.values
                )
            else:
                Forecast_RT_shadow_prices_array = np.zeros(24)

            # Compare prices and ratings
            used_rating = (
                percent_of_rating * FORECAST_constraint_flow_tomorrow["rating"]
            )
            # conf_matrix.FN = (
            #     ((MUSE_constraint_flow_tomorrow["constraint_flow"] >= used_rating))
            #     & (FORECAST_constraint_flow_tomorrow["constraint_flow"] <= used_rating)
            # ).sum()
            # conf_matrix.TP = (
            #     (
            #         (MUSE_constraint_flow_tomorrow["constraint_flow"] >= used_rating)
            #         | (RT_data_for_this_date["shadow_price"] > 0)
            #     )
            #     & (FORECAST_constraint_flow_tomorrow["constraint_flow"] >= used_rating)
            # ).sum()
            # conf_matrix.FP = (
            #     (
            #         (MUSE_constraint_flow_tomorrow["constraint_flow"] <= used_rating)
            #         & (RT_data_for_this_date["shadow_price"] == 0)
            #     )
            #     & (FORECAST_constraint_flow_tomorrow["constraint_flow"] >= used_rating)
            # ).sum()

            if USE_SHADOW_PRICE_NOT_FLOW:
                conf_matrix.FN = (
                    (RT_data_for_this_date["shadow_price"] > 0)
                    & (
                        FORECAST_constraint_flow_tomorrow["constraint_flow"]
                        < used_rating
                    )
                    & (Forecast_RT_shadow_prices_array == 0)
                ).sum()
                ### conf_matrix.FN += (  (RT_data_for_this_date['shadow_price'] == 0)  & (FORECAST_constraint_flow_tomorrow['constraint_flow'] >= used_rating) | (Forecast_RT_shadow_prices_array > 0)).sum()
                conf_matrix.TP = (
                    (RT_data_for_this_date["shadow_price"] > 0)
                    & (
                        (
                            FORECAST_constraint_flow_tomorrow["constraint_flow"]
                            >= used_rating
                        )
                        | (Forecast_RT_shadow_prices_array > 0)
                    )
                ).sum()
                conf_matrix.FP = (
                    ((RT_data_for_this_date["shadow_price"] == 0))
                    & (
                        (
                            FORECAST_constraint_flow_tomorrow["constraint_flow"]
                            >= FORECAST_constraint_flow_tomorrow["rating"]
                        )
                        | (Forecast_RT_shadow_prices_array > 0)
                    )
                ).sum()

                # find the shadow price based on top_contingencies[(monitored_uid, contingency_uid)]:
            obsered_RT_shadow_price = top_contingencies.loc[
                (monitored_uid, contingency_uid)
            ].shadow_price
            forecast_RT_shadow_price = top_contingencies.loc[
                (monitored_uid, contingency_uid)
            ].forecast_shadow_price

            ans, constraint_data = calc_collected_confusion_metrics(
                conf_matrix,
                tomorrow_date,
                monitored_uid,
                contingency_uid,
                percent_of_rating,
                percent_of_rating,
                obsered_RT_shadow_price,
                forecast_RT_shadow_price,
            )
            if constraint_data is None:
                continue
                # print(18)
                # ans, constraint_data = calc_collected_confusion_metrics(conf_matrix, tomorrow_date, monitored_uid, contingency_uid, percent_of_rating, percent_of_rating, obsered_RT_shadow_price, forecast_RT_shadow_price)
            else:
                collected_results[tomorrow_date].append(constraint_data)

            # #if this case had RT shadow price > 0 but the forecast was under the used_rating then this is also a FN case
            # if conf_matrix.FN + conf_matrix.TP + conf_matrix.FP == 0:
            #     RT_bindings_NOT_caught_by_MUSE_and_FORECAST[tomorrow_date].append((tomorrow_date, monitored_uid, contingency_uid, percent_of_rating, percent_of_rating, obsered_RT_shadow_price))
            #     conf_matrix.FN += ((  (MUSE_constraint_flow_tomorrow['constraint_flow'] <= used_rating) & (RT_data_for_this_date['shadow_price'] > 0)  ) & (FORECAST_constraint_flow_tomorrow['constraint_flow'] <= used_rating)).sum()

            # else:
            #     ans, constraint_data = calc_collected_confusion_metrics(conf_matrix, tomorrow_date, monitored_uid, contingency_uid, percent_of_rating, percent_of_rating, obsered_RT_shadow_price, forecast_RT_shadow_price)
            #     collected_results[tomorrow_date].append(constraint_data)

print("23847623846 ARRIAVED HERE")
data_to_save = {
    "collected_results": collected_results,
    "RT_bindings_NOT_caught_by_MUSE_and_FORECAST": RT_bindings_NOT_caught_by_MUSE_and_FORECAST,
    "percent_of_ratings": percent_of_ratings,
}

print(f"size of collected_results: {len(collected_results)}")

today_date = datetime.datetime.now().strftime("%Y-%m-%d")
with open(
    f"weekly_report_saved_data_{run_name_for_file_saving_names}_as_of_{today_date}.pkl",
    "wb",
) as f:
    pickle.dump(data_to_save, f)
    f.flush()
    os.fsync(f.fileno())

# today_date = datetime.datetime.now().strftime('%Y-%m-%d')
# data_to_save = pickle.load(open(f'weekly_report_as_of_{today_date}.pkl', 'rb'))
# collected_results = data_to_save['collected_results']
# RT_bindings_NOT_caught_by_MUSE_and_FORECAST = data_to_save['RT_bindings_NOT_caught_by_MUSE_and_FORECAST']

if False:
    # The code below aggregates alf the results for the entire week.
    daily_confusion_FP = []
    daily_confusion_FN = []
    daily_confusion_TP = []
    num_of_points = []
    # original_points = []  # Initialize variables to store original points
    point_ind = []  # Initialize variables to store the index of the merged point
    # original_pnts = []
    radius_for_points = []
    titles = [
        "index",
        "FP_for_chart",
        "FN_for_chart",
        "TP",
        "FP",
        "FN",
        "Date",
        "Monitored_uid",
        "Contingency_uid",
        "Num Hours",
        "Rating",
        "observed_RT_shadow_price",
    ]
    weekly_info = pd.DataFrame(columns=titles)

    # At this point we have to save. Any graphing will be done based on the saved data to the DataFrame
    # options to give the user:
    # 1. Choose a range of dates
    # 2. For some dates there are well less than 10 points. Check why
    # 3. Choose a specific monitored_uid and contingency_uid
    # 4. Choose a specific rating
    # 5. Choose to show only data with RT shadow price > Threshold
    # 6. Choose to show only data with number of hours less (or more) than a threshold

    i = 0
    for date_of_forecast in collected_results:
        for (
            counter,
            false_positive,
            false_negative,
            TP,
            FP,
            FN,
            tomorrow_date,
            monitored_uid,
            contingency_uid,
            num_points,
            rating,
            orig_rating,
            observed_RT_shadow_price,
        ) in collected_results[date_of_forecast]:
            daily_confusion_FP.append(false_positive)
            daily_confusion_FN.append(false_negative)
            daily_confusion_TP.append(num_points - false_negative - false_positive)
            radius_for_points.append(0)
            num_of_points.append(num_points)
            # original_points.append([(false_positive, false_negative)])  # Initialize original points for each merged point
            point_ind.append([i])

            new_row = {
                "index": i,
                "FP_for_chart": false_positive,
                "FN_for_chart": false_negative,
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "Date": tomorrow_date,
                "Monitored_uid": monitored_uid,
                "Contingency_uid": contingency_uid,
                "Num Hours": num_points,
                "Rating": rating,
                "orig_rating": rating,
                "observed_RT_shadow_price": observed_RT_shadow_price,
            }
            weekly_info.loc[len(weekly_info)] = new_row
            i += 1

    def on_click(
        event,
        daily_confusion_FP=daily_confusion_FP,
        daily_confusion_FN=daily_confusion_FN,
        num_of_points=num_of_points,
        radius_for_points=radius_for_points,
    ):
        # radius_for_points=0.05
        if event.button == 1:  # Left mouse button
            for i in range(len(daily_confusion_FP)):
                circle = Circle(
                    (np.sqrt(daily_confusion_FP[i]), np.sqrt(daily_confusion_FN[i])),
                    radius=radius_for_points[i],
                )
                if circle.contains_point((event.xdata, event.ydata)):
                    print(" ")
                    num_of_days = len(weekly_info.query("index in @point_ind[@i]"))
                    print(f" Number of days: {num_of_days}")
                    print(
                        f" Total (and average) number of hours: {num_of_points[i]}, ({num_of_points[i] / num_of_days})"
                    )
                    total_RT_shadow_price = weekly_info.query(
                        "index in @point_ind[@i]"
                    ).observed_RT_shadow_price.sum()
                    print(
                        f" Total (and average) RT shadow price: {total_RT_shadow_price} ({total_RT_shadow_price / num_of_days})"
                    )
                    print(f" False Positive: {daily_confusion_FP[i]}")
                    print(f" False Negative: {daily_confusion_FN[i]}")

                    print(weekly_info.query("index in @point_ind[@i]"))
                    print(" ")
                    break

    def weight_on_number_of_hours_in_the_blob(i):
        return 100 * num_of_points[i] / sum(num_of_points)

    def weight_on_average_num_of_active_hours_per_day(i):
        total = 0
        for j in range(len(daily_confusion_FP)):
            num_of_days = len(weekly_info.query("index in @point_ind[@j]"))
            total += num_of_points[j] / num_of_days
        num_of_days = len(weekly_info.query("index in @point_ind[@i]"))
        return 100 * num_of_points[i] / num_of_days / total

    def weight_on_num_of_days_in_the_blob(i):
        num_of_days = len(weekly_info.query("index in @point_ind[@i]"))
        total_days = len(weekly_info)
        return 100 * num_of_days / total_days

    def weight_on_total_RT_shadow_price(i):
        total_shadow_price = weekly_info.observed_RT_shadow_price.sum()
        RT_shadow_price_of_this_blob = weekly_info.query(
            "index in @point_ind[@i]"
        ).observed_RT_shadow_price.sum()
        return 100 * RT_shadow_price_of_this_blob / total_shadow_price

    def weight_on_average_RT_shadow_prices_per_day(i):
        total = 0
        for j in range(len(daily_confusion_FP)):
            RT_shadow_price_of_this_blob = weekly_info.query(
                "index in @point_ind[@j]"
            ).observed_RT_shadow_price.sum()
            num_of_days = len(weekly_info.query("index in @point_ind[@j]"))
            total += RT_shadow_price_of_this_blob / num_of_days
        RT_shadow_price_of_this_blob = weekly_info.query(
            "index in @point_ind[@i]"
        ).observed_RT_shadow_price.sum()
        num_of_days = len(weekly_info.query("index in @point_ind[@i]"))
        return 100 * RT_shadow_price_of_this_blob / total / num_of_days

    Points_unified = []

    def create_plot(
        Radius_to_unify_points=0.05,
        chosen_function=weight_on_average_num_of_active_hours_per_day,
    ):

        if False:
            # Create a histogram of daily_confusion_TP
            bins = np.linspace(0, 1, 10)  # Adjust the number of bins as needed
            hist, _ = np.histogram(daily_confusion_TP, bins=bins, weights=num_of_points)

            # Plot histogram
            plt.bar(
                bins[:-1], hist / sum(num_of_points), width=np.diff(bins), align="edge"
            )
            plt.xlabel("True Positive")
            plt.ylabel("Percent")
            plt.title(f"Histogram of True Positive, rating = {percent_of_rating}")

            plt.figure()

        # some of the points are very close to each other, so we need to unify them into one big point:
        for i in range(len(daily_confusion_FP)):
            if i in Points_unified:
                continue
            for j in range(len(daily_confusion_FP) - 1, i, -1):
                if j in Points_unified:
                    continue
                if (
                    abs(daily_confusion_FP[i] - daily_confusion_FP[j])
                    < Radius_to_unify_points
                    and abs(daily_confusion_FN[i] - daily_confusion_FN[j])
                    < Radius_to_unify_points
                ):
                    # original_points[i].append((daily_confusion_FP[j], daily_confusion_FN[j])) # Add original point to the corresponding merged point
                    point_ind[i].append(j)
                    daily_confusion_FP[i] = (
                        daily_confusion_FP[i] * num_of_points[i]
                        + daily_confusion_FP[j] * num_of_points[j]
                    ) / (num_of_points[i] + num_of_points[j])
                    daily_confusion_FN[i] = (
                        daily_confusion_FN[i] * num_of_points[i]
                        + daily_confusion_FN[j] * num_of_points[j]
                    ) / (num_of_points[i] + num_of_points[j])
                    num_of_points[i] = num_of_points[i] + num_of_points[j]
                    Points_unified.append(j)

        # Eliminate points that were unified
        for i in range(len(daily_confusion_FP), -1, -1):
            if i in Points_unified:
                daily_confusion_FP.pop(i)
                daily_confusion_FN.pop(i)
                num_of_points.pop(i)
                # original_points.pop(i)
                point_ind.pop(i)
                radius_for_points.pop(i)

        # create contours for equal TP
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")  # Set aspect ratio to be equal
        theta = np.linspace(
            0, 2 * np.pi / 4, 100
        )  # Define angles for the circular grid
        r = np.arange(0, 1.1, 0.2)  # Define radii for the circular grid
        for radius in r:
            x_circle = radius * np.cos(theta)
            y_circle = radius * np.sin(theta)
            ax.plot(
                x_circle,
                y_circle,
                color="red",
                linestyle="dashed",
                alpha=0.2,
                linewidth=0.7,
            )
            ax.text(
                radius * np.cos(np.pi / 4),
                radius * np.sin(np.pi / 4),
                f"{100-100*radius**2:.0f}%",
                style="italic",
                color="red",
                fontsize=8,
                ha="center",
                va="center",
            )

        values = []
        # circle each blob using a radius that is proportional to the number of points in the blob
        for i in range(len(daily_confusion_FP)):
            value = chosen_function(i)
            values.append(value)

        # The number of points of each blob and the curcle around it are proportional to the number of points in the blob
        text_size = np.interp(values, (min(values), max(values)), (10, 30))
        radius_range = np.interp(values, (min(values), max(values)), (0.03, 0.08))

        for i in range(len(daily_confusion_FP)):
            ax.text(
                np.sqrt(daily_confusion_FP[i]),
                np.sqrt(daily_confusion_FN[i]),
                f"{values[i]:.0f}",
                fontsize=text_size[i],
                ha="center",
                va="center",
            )

        for i in range(len(daily_confusion_FP)):
            # add a circle for each point
            radius_for_points[i] = radius_range[i]
            circle = Circle(
                (np.sqrt(daily_confusion_FP[i]), np.sqrt(daily_confusion_FN[i])),
                radius=radius_for_points[i],
                fill=False,
                color="black",
            )
            ax.add_patch(circle)

        plt.connect("button_press_event", on_click)
        plt.xlabel("False Positive")
        plt.ylabel("False Negative")
        plt.title(f"False Positive vs False Negative, {to_date}")

        # add grid to the image. The grid should be radial, and reflects the fact that the closer to the center, the better the results
        # plt.xticks([xi/10 for xi in range(0,10,2)], [int(10*np.sqrt(xi/10))/10 for xi in range(0,10,2)])
        plt.grid(True)

        plt.show()

        return daily_confusion_FP, daily_confusion_FN, radius_for_points

    # daily_confusion_FP, daily_confusion_FN, radius_for_points = create_plot(daily_confusion_FP, daily_confusion_FN, num_of_points)
    daily_confusion_FP, daily_confusion_FN, radius_for_points = create_plot(
        Radius_to_unify_points=0.05
    )

    print(f"cnt_NO = {cnt_downloaded}, cnt_YES = {cnt_not_downloaded}")
    print(18)


if False:  # Michael comments out because it is not relevant

    if False:

        def _get_snowflake_auth():
            return tuple(os.environ["SNOWFLAKE_AUTH"].split(":"))

        import snowflake.connector as snowflake_connector
        from snowflake.connector.connection import SnowflakeConnection

        SNOWFLAKE_ACCOUNT = "enverus_pr.us-east-1"

        def _get_cnx() -> SnowflakeConnection:
            user, password = tuple(_get_snowflake_auth())

            ctx = snowflake_connector.connect(
                user=user,
                password=password,
                account=SNOWFLAKE_ACCOUNT,
                insecure_mode=os.getenv("SNOWFLAKE_INSECURE_MODE") is not None,
            )

            return ctx

        cnx = _get_cnx()

        run = "20230320"
        run = "20230408"
        query_binding_constraints_performance = """
        SELECT
                stream_uid, timestamp, monitored_uid, contingency_uid, shadow_price,
                from_station, from_kv, to_station, to_kv, as_of
                FROM "pr-forecast"."20230320T1900Z"."binding_constraint"
                WHERE ABS(shadow_price) > 0
                AND HOUR(as_of) = 4
                AND stream_uid = 'load_100_wind_100_sol_100_ng_100_iir'
        """

        df_binding_constraints_performance = snowflake_utils.execute_and_fetch(
            cnx, query_binding_constraints_performance
        )

        bc_forecast = {}
        as_ofs = [
            AsOf(as_of_str=as_of.isoformat(), as_of=as_of)
            for as_of in df_binding_constraints_performance["as_of"].unique()
        ]
        for as_of in as_ofs:
            df_bc_forecast = df_binding_constraints_performance[
                df_binding_constraints_performance["as_of"] == as_of.as_of
            ]
            bc_forecast[as_of.as_of_str] = df_bc_forecast

        with open(f"bc_forecast_as_of_{run}.pkl", "wb") as f:
            pickle.dump(bc_forecast, f)

    def process_forecast_snowflake(df_bc_forecast):

        if len(df_bc_forecast) == 0:
            return df_bc_forecast
        df_bc_forecast["date"] = df_bc_forecast["timestamp"].apply(lambda x: x.date())
        df_bc_forecast["lookahead_days"] = (
            df_bc_forecast["timestamp"].dt.date - df_bc_forecast["as_of"].dt.date
        ).dt.days
        return df_bc_forecast

    def create_bc_dict(df_bc):
        return {
            (row.monitored_uid, row.date): row.shadow_price
            for row in df_bc.itertuples()
        }

    def process_forecast(content_forecast, as_of_str):
        df_bc_forecast = pd.read_csv(content_forecast)
        if len(df_bc_forecast) == 0:
            return df_bc_forecast
        df_bc_forecast["timestamp"] = df_bc_forecast["timestamp"].apply(
            datetime.datetime.fromisoformat
        )
        df_bc_forecast["date"] = df_bc_forecast["timestamp"].apply(lambda x: x.date())
        df_bc_forecast["as_of"] = datetime.datetime.fromisoformat(as_of_str)
        df_bc_forecast["lookahead_days"] = df_bc_forecast.apply(
            lambda x: x.timestamp.date() - x.as_of.date(), axis=1
        ).dt.days
        return df_bc_forecast

    ## get data from placebo api
    contents = dict()
    i = 0
    ## Forecast
    # the following loop runs through every day in the past, starting from today and backwards:
    for as_of in tqdm(as_ofs_last_year[:6]):
        i += 1
        content = api_utils.fetch_from_placebo(
            endpoint=f"{run}/binding_constraint",
            query_params={
                "as_of": str(as_of.as_of_str),
            },
        )
        df_bc_forecast = process_forecast(content, as_of.as_of_str)
        if len(df_bc_forecast) > 0:
            # print(as_of.as_of_str)
            contents[as_of.as_of_str] = df_bc_forecast

        # the following is an output identical to Mosaic:
        # print(contents['2024-04-08T04:00:00-05:00'].query('stream_uid=="load_100_wind_100_sol_100_ng_100_iir" and date == datetime.date(2024, 4, 8)').sort_values('shadow_price')[['timestamp','monitored_label','contingency_uid','shadow_price']][-13:])

        # if i % 100 == 0:
        #     time.sleep(30)
    with open(f"bc_forecast_as_of_{run}.pkl", "wb") as f:
        pickle.dump(contents, f)

    class BindingConstraintMetric(NamedTuple):
        as_of: LocalizedDateTime
        lookahead_days: int
        stream_uid: str
        confusion_matrix: ConfusionMatrix
        binding_constraint_actual_dict: Dict
        binding_constraint_forecast_dict: Dict

    # get the items if it is in the actual dict as well as the forecast dict
    def calc_confusion_metrics(forecast_dict, actual_dict):
        true_positive = {
            key: forecast_dict[key] for key in forecast_dict.keys() & actual_dict.keys()
        }
        false_positive = {
            key: forecast_dict[key] for key in forecast_dict.keys() - actual_dict.keys()
        }
        false_negative = {
            key: actual_dict[key] for key in actual_dict.keys() - forecast_dict.keys()
        }
        return ConfusionMatrix(
            true_positive=len(true_positive),
            false_positive=len(false_positive),
            false_negative=len(false_negative),
            true_negative=-1,
        )

    # MUSE_constraint_flow['time_stamp'].apply(lambda x: x.date()) == tomorrow_date]

    # find the forecast for tommoorrow:
    # MUSE_constraint_flow_relevant = MUSE_constraint_flow[MUSE_constraint_flow['time_stamp'].apply(lambda x: x.date()) == tomorrow_date]
    # MUSE_constraint_flow['time_stamp'].apply(lambda x: x.date()) == tomorrow_date
    # for line in MUSE_constraint_flow:
    #     if MUSE_constraint_flow.iloc[0].time_stamp == FORECAST_constraint_flow.iloc[0].timestamp[:-6]:
    #         # if the forecast if for tomorrow:
    #         if MUSE_constraint_flow['time_stamp'].apply(lambda x: x.date()) == tomorrow_date
    #         # and

    # for line in MUSE_constraint_flow:

    #         MUSE_constraint_flow.iloc[0].time_stamp == FORECAST_constraint_flow.iloc[0].timestamp[:-6]

    # the following finds the forecast for the same day as the actual

    #         for line in
    #         FORECAST_constraint_flow['timestamp'] = pd.to_datetime(FORECAST_constraint_flow['timestamp'],utc=True).dt.tz_convert('US/Central')
    #         for line in

    #         pd.to_datetime(FORECAST_constraint_flow['timestamp'],utc=True).dt.tz_convert('US/Central')

    #         relevant_FORECAST_data = FORECAST_constraint_flow[FORECAST_constraint_flow['timestamp'].apply(date)) == tomorrow_date]
    #         for
    #         = date_utils.localized_from_isoformat(FORECAST_constraint_flow['timestamp']).date()
    # date_utils.localized_from_isoformat(key).date()
    # 2024-04-09 15:00:00-05:00

    #         rating =

    top_contingencies = find_top_congestions(
        contents_actual["2024-04-08T04:00:00-05:00"]
    )
    for monitored_uid, contingency_uid in top_contingencies.index:
        print(monitored_uid, contingency_uid)

        # # all the constraints we monitor at a specific date and time:
        # ddd = _get_dataframe("/ercot/constraint_flows.csv?to_date=2024-04-09T17%3A15%3A00.000")

        # start_datetime={date.strftime('%Y-%m-%d')}&end_datetime={(datetime.timedelta(days=1)+date).strftime('%Y-%m-%d')}"
        # date_utils.localized_from_isoformat(key).date() > (datetime.datetime.now() - datetime.timedelta(days=3)).date()
        # one constraint:
        # OLD
        # constraint_uid = "101T158_1,ZORN,138.0,POOLRO,138.0 ;;; SMENLYT8"
        # from_date = "2024-04-01T00%3A00%3A00.000"
        # to_date = "2024-04-08T00%3A00%3A00.000"

        constraint_uid = f"{monitored_uid} ;;; {contingency_uid}"
        from_date = "2024-04-05"
        # to_date = "2024-04-08"
        to_date = datetime.datetime.now().strftime("%Y-%m-%d")
        MUSE_constraint_flow = _get_dataframe(
            f"/ercot/constraint_flow.csv?uid={constraint_uid}&from_date={from_date}&to_date={to_date}&resample_rate=1h"
        )[["time_stamp", "constraint_flow"]]

        # find historical forecast
        # url = f'https://api1.marginalunit.com/pr-forecast/ercot/constraint/lookahead_timeseries?monitored_uid={monitored_uid}&contingency_uid={contingency_uid}&stream_uid=load_100_wind_100_sol_100_ng_100_iir&from_datetime={from_date}'  #works but not what I need. This is foreacast for today's hours, which was created today in the morning
        url = f"https://api1.marginalunit.com/pr-forecast/ercot/constraint?as_of={from_date}&as_of_match=nearest_before&stream_uid=load_100_wind_100_sol_100_ng_100_iir&monitored_uid=BURNS_RIOHONDO_1,RIOHONDO,138.0,MV_BURNS,138.0&contingency_uid=XNED89"  # forecast future days
        content = _get_dfm(url)
        FORECAST_constraint_flow = pd.read_csv(content)

    contents_actual = {}
    for as_of in tqdm(as_ofs_actual):
        date = as_of.as_of.date()
        # "/<run_name>/constraint/lookahead_timeseries"
        url = "https://api1.marginalunit.com/pr-forecast/ercot/constraint/lookahead_timeseries?monitored_uid=BURNS_RIOHONDO_1,RIOHONDO,138.0,MV_BURNS,138.0&contingency_uid=XNED89&stream_uid=load_100_wind_100_sol_100_ng_100_iir&lookahead_days=1&from_datetime=2024-04-07+10:00:00-05:00"  # historical forecast
        # url = 'https://api1.marginalunit.com/pr-forecast/ercot/constraint?as_of=2024-04-10T04:00:00-05:00&as_of_match=nearest_before&stream_uid=load_100_wind_100_sol_100_ng_100_iir&monitored_uid=BURNS_RIOHONDO_1,RIOHONDO,138.0,MV_BURNS,138.0&contingency_uid=XNED89'  #forecast future days

        content = _get_dfm(url)
        df_bc_actual = pd.read_csv(content)

        if False:  # Barrry method:
            # url from Barry:
            url = "https://mosaic.enverus.com/mu-muse2-data-access/app/constraint-flow-time-series"
            body = {
                "run": "ERCOT",
                "constraintUid": "KLNSW 138KV - HHSTH 138KV (630__B)",
                "startTime": "2024-04-07T13:00:00-05:00",
                "endTime": "2024-04-10T13:24:00-05:00",
                "resampleRate": "1h",
                "staticDecomp": False,
            }

            resp = requests.get(url, auth=AUTH, json=body)

            if resp.status_code != 200:
                print(resp.text)
                resp.raise_for_status()

    # Michael replaces one line with another:
    # run = '20230320'
    run = "ercot"

    ## calculate the metrics
    # change the run and lookahead days for different metrics

    lookahead_days = 1

    contents_forecast = pickle.load(open(f"bc_forecast_as_of_{run}.pkl", "rb"))
    contents_actual = pickle.load(open("bc_actual_as_of.pkl", "rb"))
    as_ofs_actual = [
        AsOf(as_of_str=as_of_str, as_of=date_utils.localized_from_isoformat(as_of_str))
        for as_of_str in contents_actual.keys()
    ]

    bc_metrics = []

    for key, content_forecast in tqdm(contents_forecast.items()):
        # We only want to analyze the last 3 days
        if (
            date_utils.localized_from_isoformat(key).date()
            > (datetime.datetime.now() - datetime.timedelta(days=3)).date()
        ):
            continue
        as_of = AsOf(as_of_str=key, as_of=date_utils.localized_from_isoformat(key))
        if (datetime.timedelta(days=lookahead_days) + as_of.as_of).date() not in [
            as_of_actual.as_of.date() for as_of_actual in as_ofs_actual
        ]:
            continue
        df_bc_forecast = contents_forecast[as_of.as_of_str]
        df_bc_forecast["as_of"] = as_of.as_of
        df_bc_forecast["timestamp"] = pd.to_datetime(
            df_bc_forecast["timestamp"], utc=True
        ).dt.tz_convert("US/Central")

        df_bc_forecast["date"] = df_bc_forecast["timestamp"].apply(lambda x: x.date())
        df_bc_forecast["lookahead_days"] = (
            df_bc_forecast["timestamp"].dt.date - df_bc_forecast["as_of"].dt.date
        ).dt.days
        df_bc_forecast = df_bc_forecast[
            df_bc_forecast["stream_uid"] == "load_100_wind_100_sol_100_ng_100_iir"
        ].copy()
        forecast_dates = (
            df_bc_forecast[["date", "lookahead_days"]]
            .drop_duplicates()
            .to_dict("records")
        )
        forecast_data = [
            forecast_data
            for forecast_data in forecast_dates
            if forecast_data["lookahead_days"] == lookahead_days
        ]
        if len(forecast_data) == 0:
            continue
        forecast_data = forecast_data[0]
        as_of_actual = [
            as_of_actual
            for as_of_actual in as_ofs_actual
            if as_of_actual.as_of.date()
            == (datetime.timedelta(days=lookahead_days) + as_of.as_of).date()
        ][0]
        df_actual_one = contents_actual[as_of_actual.as_of_str]
        if len(df_actual_one) == 0:
            continue
        date = forecast_data["date"].isoformat()
        actual_dict = create_bc_dict(df_actual_one)
        forecast_dict = create_bc_dict(df_bc_forecast)

        # Michael: to do
        # define various criteria for success
        confusion_matrix = calc_confusion_metrics(forecast_dict, actual_dict)
        bc_metrics.append(
            BindingConstraintMetric(
                as_of=date_utils.localized_from_isoformat(key),
                lookahead_days=lookahead_days,
                stream_uid="load_100_wind_100_sol_100_ng_100_iir",
                confusion_matrix=confusion_matrix,
                binding_constraint_actual_dict=actual_dict,
                binding_constraint_forecast_dict=forecast_dict,
            )
        )
        # except:
        #     print(key)
        #     continue

    ## calculate other classification metrics
    df_bc_metrics = pd.DataFrame(bc_metrics)
    df_bc_metrics["confusion_matrix"] = df_bc_metrics["confusion_matrix"].apply(
        lambda x: x._asdict()
    )
    df_bc_metrics = df_bc_metrics.join(
        pd.json_normalize(df_bc_metrics["confusion_matrix"])
    )
    df_bc_metrics.loc[
        df_bc_metrics["lookahead_days"] == lookahead_days,
        ["true_positive", "false_positive", "false_negative"],
    ].describe()

    df_bc_metrics["total_actual"] = df_bc_metrics[
        "binding_constraint_actual_dict"
    ].apply(lambda x: len(x))
    df_bc_metrics["total_forecast"] = df_bc_metrics[
        "binding_constraint_forecast_dict"
    ].apply(lambda x: len(x))

    df_bc_metrics["mape"] = (
        df_bc_metrics["total_actual"] - df_bc_metrics["total_forecast"]
    ).abs() / df_bc_metrics["total_actual"]
    df_bc_metrics["under(-)/over(+)-forecast"] = (
        df_bc_metrics["total_forecast"] - df_bc_metrics["total_actual"]
    )
    df_bc_metrics["under(-)/over(+)-forecast_pct"] = (
        df_bc_metrics["under(-)/over(+)-forecast"] / df_bc_metrics["total_actual"]
    )
    df_bc_metrics["precision"] = (
        df_bc_metrics["true_positive"] / df_bc_metrics["total_forecast"]
    )
    df_bc_metrics["recall/hit_rate"] = (
        df_bc_metrics["true_positive"] / df_bc_metrics["total_actual"]
    )
    df_bc_metrics["f1"] = (
        2
        * df_bc_metrics["precision"]
        * df_bc_metrics["recall/hit_rate"]
        / (df_bc_metrics["precision"] + df_bc_metrics["recall/hit_rate"])
    )
    df_bc_metrics["false_discovery_rate/false_alarm"] = (
        df_bc_metrics["false_positive"] / df_bc_metrics["total_forecast"]
    )
    df_bc_metrics["false_negative_rate/miss_rate"] = (
        df_bc_metrics["false_negative"] / df_bc_metrics["total_actual"]
    )

    # save the metrics for visuals and comparisons
    df_bc_metrics.loc[df_bc_metrics["lookahead_days"] == lookahead_days, :].to_csv(
        "bc_metrics_no_contigency_ercot_{}_day_{}.csv".format(run, lookahead_days)
    )
