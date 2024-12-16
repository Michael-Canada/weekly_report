# environment:
# placebo_api_local

import sys
import bisect

sys.path.append("/Users/michael.simantov/Documents/mu-placebo-api")
sys.path.append("/Users/michael.simantov/Documents/mu-placebo-api/placebo_api")
sys.path.append("/Users/michael.simantov/Documents/mu-placebo-api/placebo_api/utils")

# from typing import NamedTuple, Dict
# from placebo_api.utils import api_utils, date_utils
# import api_utils, date_utils
# from placebo.utils import snowflake_utils
# from placebo_api.utils.date_utils import LocalizedDateTime
# from date_utils import LocalizedDateTime
import pandas as pd
import datetime
from tqdm import tqdm
import pickle

# import requests
# import io
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import os
from scipy.interpolate import griddata
import copy
from create_confusion_matirx import plot_confusion_matrix
from create_FN_FP_RT_shadow_price import plot_FN_FP_RT_shadow_price
from scipy.stats import norm

# plt.ion()
run = "SPP"
# run = "MISO"
# run = "ERCOT"
# run = "MISO_1112"
# run = "ercot_prt_crcl"
# run = "MISO_1213"
# run = "miso_ng"
# run = "SPP_1210"
# run = "SPP_1212"

# list of hyperparameters
subplotting = False
to_date = datetime.datetime.now().strftime("%Y-%m-%d")


today_date = datetime.datetime.now().strftime("%Y-%m-%d")
# today_date = '2024-05-07'
# today_date = '2024-05-17'
# today_date = '2024-05-30'
# today_date = '2024-07-10'
# today_date = '2024-07-10'
data_to_save = pickle.load(
    open(f"weekly_report_saved_data_{run.upper()}_as_of_{today_date}.pkl", "rb")
)
collected_results_orig = data_to_save["collected_results"]
percent_of_ratings = data_to_save["percent_of_ratings"]
RT_bindings_NOT_caught_by_MUSE_and_FORECAST_orig = data_to_save[
    "RT_bindings_NOT_caught_by_MUSE_and_FORECAST"
]

collected_results = {}
RT_bindings_NOT_caught_by_MUSE_and_FORECAST = {}
# keep only the dates that are after 2024-04-12
for date in collected_results_orig:
    # choose the right week:
    # if date > '2024-03-27' and date <= '2024-04-05':
    # if date > '2024-04-05' and date <= '2024-04-12':
    # if date > '2024-04-12' and date <= '2024-04-19':
    # if date > '2024-04-19' and date <= '2024-04-26':
    # if date > '2024-04-26' and date <= '2024-05-03':
    # if date > "2024-10-10" and date <= "2024-10-18":
    # if date > "2024-11-08" and date <= "2024-11-15":
    if date > "2024-12-06" and date <= "2024-12-13":

        if collected_results_orig[date]:
            collected_results[date] = collected_results_orig[date]

for date in RT_bindings_NOT_caught_by_MUSE_and_FORECAST_orig:
    # choose the right week:
    # if date > '2024-03-27' and date <= '2024-04-05':
    # if date > '2024-04-05' and date <= '2024-04-12':
    # if date > '2024-04-12' and date <= '2024-04-19':
    # if date > '2024-04-19' and date <= '2024-04-26':
    # if date > "2024-09-19" and date <= "2024-09-26":
    # if date > "2024-09-26" and date <= "2024-10-10":
    # if date > "2024-10-10" and date <= "2024-10-18":
    # if date > "2024-11-08" and date <= "2024-11-15":
    if date > "2024-12-06" and date <= "2024-12-13":

        RT_bindings_NOT_caught_by_MUSE_and_FORECAST[date] = (
            RT_bindings_NOT_caught_by_MUSE_and_FORECAST_orig[date]
        )


print(" ")
print(
    "_________________________________________________________________________________"
)
print(f"the following rating percents are available: {percent_of_ratings}")
print(" ")
print(" PROBLEMATIC CONSTRAINTS")
# print the constraints that are not binding for any percent of ratings
collection_of_bad_constraints = {}
unidentified_constraint_counter = 0
unidentified_RT_shadow_price = 0
all_dates = sorted([k for k in RT_bindings_NOT_caught_by_MUSE_and_FORECAST])
for this_date in all_dates:
    problematic_constraints = {}
    for one_case in RT_bindings_NOT_caught_by_MUSE_and_FORECAST[this_date]:
        monitored_uid = one_case[1]
        contingency_uid = one_case[2]
        rating = one_case[4]
        RT_shadow_price = one_case[5]
        this_constraint = tuple((contingency_uid, monitored_uid))
        if not this_constraint in problematic_constraints:
            problematic_constraints[this_constraint] = [(rating, RT_shadow_price)]
        else:
            problematic_constraints[this_constraint].append(rating)

    for this_constraint in problematic_constraints:
        if len(problematic_constraints[this_constraint]) == len(percent_of_ratings):
            # print(f'{this_date}:  {this_constraint}, RT shadow price ${problematic_constraints[this_constraint][0][1]} is problematic for all ratings')
            unidentified_constraint_counter += 1
            unidentified_RT_shadow_price += problematic_constraints[this_constraint][0][
                1
            ]

            RT_shadow = problematic_constraints[this_constraint][0][1]
            monitored_uid = this_constraint[1]
            if not this_constraint[0] in collection_of_bad_constraints:
                collection_of_bad_constraints[this_constraint[0]] = [
                    (this_date, monitored_uid, RT_shadow)
                ]
            else:
                collection_of_bad_constraints[this_constraint[0]].append(
                    (this_date, monitored_uid, RT_shadow)
                )

for this_constraint in collection_of_bad_constraints:
    total_RT = 0
    for this_date, monitored_uid, RT_shadow in collection_of_bad_constraints[
        this_constraint
    ]:
        total_RT += RT_shadow
    print(f"Contingency: {this_constraint}, total RT shadow price ${total_RT:.2f}")
    for this_date, monitored_uid, RT_shadow in collection_of_bad_constraints[
        this_constraint
    ]:
        print(f"    {this_date}:  {monitored_uid}, RT shadow price ${RT_shadow:.2f}")
    print(" ")


# The code below aggregates all the results for the entire week.
Counters = []
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
    "orig_rating",
    "observed_RT_shadow_price",
    "forecast_RT_shadow_price",
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


# i=0
# for date_of_forecast in collected_results:
#     for false_positive, false_negative, TP, FP, FN, tomorrow_date, monitored_uid, contingency_uid, num_points, rating, observed_RT_shadow_price, forecast_RT_shadow_price in collected_results[date_of_forecast]:
#         new_row = {'index':i, 'FP_for_chart': false_positive, 'FN_for_chart': false_negative, 'TP': TP, 'FP': FP, 'FN': FN, 'Date': tomorrow_date, 'Monitored_uid': monitored_uid, 'Contingency_uid': contingency_uid, 'Num Hours': num_points, 'Rating': rating, 'observed_RT_shadow_price':observed_RT_shadow_price, forecast_RT_shadow_price:forecast_RT_shadow_price}
#         weekly_info.loc[len(weekly_info)] = new_row
#         i += 1

identified_constraint_counter = 0
identified_RT_shadow_price = 0
# the following for loop is to choose the best rating for each combination of monitored_uid and contingency_uid:
counter = 1000000
for date_of_forecast in collected_results:
    collected_results_summary = [l for l in collected_results[date_of_forecast]]
    identified_constraint_counter += len(collected_results_summary)
    identified_RT_shadow_price += np.sum([l[12] for l in collected_results_summary])

    # convert collected_results_summary to a DataFrame
    collected_results_summary = pd.DataFrame(collected_results_summary)
    collected_results_summary.columns = [
        "counter",
        "false_positive",
        "false_negative",
        "TP",
        "FP",
        "FN",
        "tomorrow_date",
        "monitored_uid",
        "contingency_uid",
        "num_points",
        "rating",
        "orig_rating",
        "observed_RT_shadow_price",
        "forecast_RT_shadow_price",
    ]

    # for every combination of monitored_uid and contingency_uid, keep the row with the lowest sum of false_positive + false_negative. Add collected_results_summary['rating']/1000 so that in case we have a tie, the one with the highest rating will be chosen.
    collected_results_summary["sum_FP_FN"] = (
        collected_results_summary["false_positive"]
        + collected_results_summary["false_negative"]
        - collected_results_summary["rating"] / 1000
    )
    collected_results_summary = collected_results_summary.sort_values(by="sum_FP_FN")
    collected_results_summary = collected_results_summary.drop_duplicates(
        subset=["monitored_uid", "contingency_uid"], keep="first"
    )
    # collected_results[date_of_forecast] = collected_results_summary[['counter', 'false_positive', 'false_negative', 'TP', 'FP', 'FN', 'tomorrow_date', 'monitored_uid', 'contingency_uid', 'num_points', 'rating', 'orig_rating', 'observed_RT_shadow_price', 'forecast_RT_shadow_price']].values.tolist()

    # change the rating at each row to become -1:
    collected_results_summary["rating"] = -1

    # convert the DataFrame back to a dictionary and add this dictionary to collected_results
    for i in collected_results_summary[
        [
            "counter",
            "false_positive",
            "false_negative",
            "TP",
            "FP",
            "FN",
            "tomorrow_date",
            "monitored_uid",
            "contingency_uid",
            "num_points",
            "rating",
            "orig_rating",
            "observed_RT_shadow_price",
            "forecast_RT_shadow_price",
        ]
    ].values.tolist():
        i[0] = counter
        counter += 1
        collected_results[date_of_forecast].append(tuple(i))

percent_of_ratings.append(-1)
if False:
    print(" ")
    print(
        f"OBSOLETE: this is not calculated anymore. Total RT shadow price for which both MUSE and Forecast did not reach {100 * min(percent_of_ratings[:-1])}% Rating: ${unidentified_RT_shadow_price:,.2f}"
    )
    print(
        f"OBSOLETE: this is not calculated anymore. Percent of constraints both MUSE and Forecast did not reach {100 * min(percent_of_ratings[:-1])}% Rating: {(100 * unidentified_constraint_counter / (identified_constraint_counter + unidentified_constraint_counter)):,.1f}%"
    )
    print(
        f"OBSOLETE: this is not calculated anymore. The RT shadow price associated with it: {(100 * unidentified_RT_shadow_price / (identified_RT_shadow_price + unidentified_RT_shadow_price)):,.1f}%"
    )
print(" ")

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
        forecast_RT_shadow_price,
    ) in collected_results[date_of_forecast]:
        new_row = {
            "index": counter,
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
            "orig_rating": orig_rating,
            "observed_RT_shadow_price": observed_RT_shadow_price,
            "forecast_RT_shadow_price": forecast_RT_shadow_price,
        }
        weekly_info.loc[len(weekly_info)] = new_row

# report on false positive and false negative
num_FP = len(
    weekly_info.query("Rating == -1 and FP_for_chart  == 1 and FN_for_chart == 0")
)
num_FN = len(
    weekly_info.query("Rating == -1 and FP_for_chart  == 0 and FN_for_chart == 1")
)
num_total_cases = len(weekly_info.query("Rating == -1"))
num_TP = num_total_cases - num_FP - num_FN
print("MUSE fails on these PROBLEMATIC CONSTRAINTS:")
print(
    f"False Positive Cases where RT shadow prices = 0 but the Forecast is above the Rating: ({num_FP} in total)"
)
print(
    f'Total "made-up" Forecast RT shadow price in FP days: ${weekly_info.query("Rating == -1 and forecast_RT_shadow_price > 0 and observed_RT_shadow_price == 0").forecast_RT_shadow_price.sum():,.2f}'
)
print(weekly_info.query("Rating == -1 and FP_for_chart  == 1 and FN_for_chart == 0"))
print(" ")
print("MUSE fails on these PROBLEMATIC CONSTRAINTS:")
print(
    f"False Negative Cases, where actual RT shadow price > 0 but Forecast is less than {100 * min(percent_of_ratings[:-1]):.0f}% of the Rating: ({num_FN} in total)"
)
print(
    f'Total missed RT shadow price in FN days: ${weekly_info.query("Rating == -1 and FP_for_chart  == 0 and FN_for_chart == 1").observed_RT_shadow_price.sum():,.2f}'
)
print(weekly_info.query("Rating == -1 and FP_for_chart  == 0 and FN_for_chart == 1"))

print(" ")

plot_FN_FP_RT_shadow_price(
    weekly_info.sort_values("forecast_RT_shadow_price", ascending=False)
    .query("Rating == -1")["forecast_RT_shadow_price"]
    .values,
    weekly_info.sort_values("forecast_RT_shadow_price", ascending=False)
    .query("Rating == -1")["observed_RT_shadow_price"]
    .values,
)

plot_FN_FP_RT_shadow_price(
    weekly_info.sort_values("observed_RT_shadow_price", ascending=False)
    .query("Rating == -1")["forecast_RT_shadow_price"]
    .values,
    weekly_info.sort_values("observed_RT_shadow_price", ascending=False)
    .query("Rating == -1")["observed_RT_shadow_price"]
    .values,
)

# find the 10 rows in weekly_info where the rating is -1 and the difference between forecast_RT_shadow_price and observed_RT_shadow_price is maximal
weekly_info["difference"] = (
    weekly_info.query("Rating == -1")["forecast_RT_shadow_price"]
    - weekly_info.query("Rating == -1")["observed_RT_shadow_price"]
)
max_forecast_minus_observed_RT_shadow_price = weekly_info.sort_values(
    "difference", ascending=False
).query("difference > 1000 and observed_RT_shadow_price < 1")
min_forecast_minus_observed_RT_shadow_price = weekly_info.sort_values(
    "difference"
).query("difference < -3000 and forecast_RT_shadow_price < 1")
weekly_info = weekly_info.drop("difference", axis=1)
print(" ")
print(
    f"Rows with high FORECAST RT shadow price but low ACTUAL RT shadow price: (sum: ${max_forecast_minus_observed_RT_shadow_price.forecast_RT_shadow_price.sum():,.0f})"
)
print(max_forecast_minus_observed_RT_shadow_price)
print(" ")
print(f"Rows with high ACTUAL RT shadow price but low FORECAST RT shadow price:")
print(min_forecast_minus_observed_RT_shadow_price)
print(" ")
print(
    f'Total observed_RT_shadow_price: ${weekly_info.query("Rating == -1").observed_RT_shadow_price.sum():,.2f}'
)
# passed_on_RT_shadow_price = weekly_info.query("Rating == -1 and observed_RT_shadow_price > 0 and forecast_RT_shadow_price == 0").observed_RT_shadow_price.sum()  # do not use it. It is good only for 100% rating, but not for cases of X% of Rating
passed_on_RT_shadow_price = weekly_info.query(
    "Rating == -1 and FP_for_chart  == 0 and FN_for_chart == 1"
).observed_RT_shadow_price.sum()
print(
    f'--> Total observed_RT_shadow_price in days that were passed on by the forecast (=false negative events): ${passed_on_RT_shadow_price:,.2f} out of ${weekly_info.query("Rating == -1").observed_RT_shadow_price.sum():,.2f} ({(100 * passed_on_RT_shadow_price / weekly_info.query("Rating == -1").observed_RT_shadow_price.sum()):.0f}%)'
)
print(" ")
print(
    f'Total forecast_RT_shadow_price: ${weekly_info.query("Rating == -1").forecast_RT_shadow_price.sum():,.2f}'
)
made_up_shadow_price = weekly_info.query(
    "Rating == -1 and forecast_RT_shadow_price > 0 and observed_RT_shadow_price == 0"
).forecast_RT_shadow_price.sum()
print(
    f'--> Total forecast_RT_shadow_price in days that the Forecast predicted congestion, but actual RT shadow price was 0 the whole day (=false positive events): ${made_up_shadow_price:,.2f} out of ${weekly_info.query("Rating == -1").forecast_RT_shadow_price.sum():,.2f} ({(100 * made_up_shadow_price / weekly_info.query("Rating == -1").forecast_RT_shadow_price.sum()):.0f}%)'
)
print(" ")
diff_10PM_4AM = (
    made_up_shadow_price
    - weekly_info.query(
        "Rating == -1 and FP_for_chart  == 1 and FN_for_chart == 0"
    ).forecast_RT_shadow_price.sum()
)
print(
    f"--> Measure of instability: This is the $ ammount that the forecast for RT shadow price changed between 4AM and 10PM the day before: ${diff_10PM_4AM:,.2f}"
)
print(" ")

print("Confusion_matrix:")
print(
    f"FP: {100 * num_FP/num_total_cases:.1f}, FN: {100 * num_FN/num_total_cases:.1f}, TP: {100 * num_TP/num_total_cases:.1f}"
)
print(" ")

if False:

    # Example rates
    TP_rate = 100 * num_TP / num_total_cases
    FP_rate = 100 * num_FP / num_total_cases
    FN_rate = 100 * num_FN / num_total_cases

    TP_rate = 53
    FP_rate = 36 / TP_rate
    FN_rate = 11 / TP_rate
    TP_rate = 1

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(30, 3))

    # Define the positions and heights of the bars
    bar_height = 0.01
    y1 = 1
    y2 = 0.989

    # Plot the bars
    ax.barh(y1, TP_rate + FP_rate, height=bar_height, color="green", label="TP Rate")
    ax.barh(y1, FP_rate, height=bar_height, color="blue", label="FP Rate")
    ax.barh(
        y2, FP_rate + TP_rate + FN_rate, height=bar_height, color="red", label="FN Rate"
    )
    ax.barh(y2, FP_rate + TP_rate, height=bar_height, color="green")
    ax.barh(y2, FP_rate, height=bar_height, color="white", label="FN Rate")

    # Add labels and title
    ax.set_xlabel("Rate")
    ax.set_yticks([y1, y2])
    ax.set_yticklabels(["Forecast", "Actual"], fontsize=20)
    plt.axis("off")

    ax.set_title("Forecast Stats Analysis", fontsize=20)
    # Add legend
    ax.legend(loc="upper right")

######################


percent_events_ignored = 100 * num_FN / (num_TP + num_FN)
percent_events_detected = 100 * num_TP / (num_TP + num_FN)
print(
    f'--> Ignoring "made-up" forecast RT shadow price, the % of real daily events that were ignored by Forecast is {percent_events_ignored:.0f}%'
)
print(
    f'--> Ignoring "made-up" forecast RT shadow price, the % of real daily events that were detected by Forecast is {percent_events_detected:.0f}%'
)
print(" ")
plot_confusion_matrix(
    num_TP / num_total_cases, num_FP / num_total_cases, num_FN / num_total_cases
)


def save_historica_printout_results(df_historic_actual, df_historic_FP):
    historic_printout_results = {}
    historic_printout_results["actual"] = df_historic_actual.values.tolist()
    historic_printout_results["FP"] = df_historic_FP.values.tolist()

    # Save the updated dictionary to a new file
    new_file_name = "histogram_printout_collection_temp.pkl"
    with open(new_file_name, "wb") as file:
        pickle.dump(historic_printout_results, file)
        file.flush()
        os.fsync(file.fileno())
    # Replace the original file with the new file
    original_file_name = f"histogram_printout_collection_{run}_NEW.pkl"
    os.replace(new_file_name, original_file_name)

    print("Data saved successfully.")

    return


def num_of_days_in_the_blob(i):
    return len(weekly_info.query("index in @point_ind[@i]"))


# bins = [0, .5, .9, 1]  # for the histogram: Number of bins as needed
bins = [0, 0.1, 0.5, 1]  # for the histogram: Number of bins as needed


def weight_on_num_of_days_in_the_blob(i, percent_of_rating):
    title = "weight: num of days in the blob"
    total_days = 0
    for j in range(len(daily_confusion_FP)):
        num_of_days = len(weekly_info.query("index in @point_ind[@j]"))
        total_days += num_of_days
    num_of_days = len(weekly_info.query("index in @point_ind[@i]"))
    # calculation for histogram
    hist_num_days, _ = np.histogram(
        [
            row["TP_for_chart"]
            for index, row in weekly_info.query(
                "Rating == @percent_of_rating"
            ).iterrows()
        ],
        bins=bins,
    )
    hist_total_days = len(weekly_info.query("Rating == @percent_of_rating"))
    return (
        (100 * num_of_days / total_days),
        (100 * hist_num_days / hist_total_days),
        title,
    )


def weight_on_number_of_hours_in_the_blob(i, percent_of_rating):
    title = "weight: number of hours in the blob"
    hist_num_hours, _ = np.histogram(
        [
            row["TP_for_chart"]
            for index, row in weekly_info.query(
                "Rating == @percent_of_rating"
            ).iterrows()
        ],
        bins=bins,
        weights=weekly_info.query("Rating == @percent_of_rating")["Num Hours"],
    )
    hist_total_hours = weekly_info.query("Rating == @percent_of_rating")[
        "Num Hours"
    ].sum()
    return (
        (100 * num_of_points[i] / sum(num_of_points)),
        (100 * hist_num_hours / hist_total_hours),
        title,
    )


# def weight_on_average_num_of_active_hours_per_day(i):
#     title = 'weight: average num of active hours per day'
#     total = 0
#     for j in range(len(daily_confusion_FP)):
#         num_of_days = len(weekly_info.query('index in @point_ind[@j]'))
#         total += num_of_points[j] / num_of_days
#     num_of_days = len(weekly_info.query('index in @point_ind[@i]'))
#     return (100 * num_of_points[i] / num_of_days / total), title
def weight_on_num_of_active_hours_per_day(i, percent_of_rating):
    title = "weight: num of active hours per day"
    # total = 0
    # for j in range(len(daily_confusion_FP)):
    #     num_of_days = len(weekly_info.query('index in @point_ind[@j]'))
    #     total += num_of_points[j] / num_of_days
    num_of_days = len(weekly_info.query("index in @point_ind[@i]"))
    return (num_of_points[i] / num_of_days), None, title


def weight_on_total_RT_shadow_price(i, percent_of_rating):
    title = "weight: total RT shadow price"
    total_shadow_price = 0
    for j in range(len(daily_confusion_FP)):
        RT_shadow_price_of_this_blob = weekly_info.query(
            "index in @point_ind[@j]"
        ).observed_RT_shadow_price.sum()
        total_shadow_price += RT_shadow_price_of_this_blob
    # total_shadow_price = weekly_info.observed_RT_shadow_price.sum()
    RT_shadow_price_of_this_blob = weekly_info.query(
        "index in @point_ind[@i]"
    ).observed_RT_shadow_price.sum()
    # histogram calculations:
    hist__RT_shadow_price, _ = np.histogram(
        [
            row["TP_for_chart"]
            for index, row in weekly_info.query(
                "Rating == @percent_of_rating"
            ).iterrows()
        ],
        bins=bins,
        weights=weekly_info.query("Rating == @percent_of_rating")[
            "observed_RT_shadow_price"
        ],
    )
    hist_total_RT_shadow_price = hist__RT_shadow_price.sum()
    return (
        (100 * RT_shadow_price_of_this_blob / total_shadow_price),
        (100 * hist__RT_shadow_price / hist_total_RT_shadow_price),
        title,
    )


def weight_on_average_RT_shadow_prices_per_day(i, percent_of_rating):
    title = "weight: average RT shadow prices per day"
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
    # histogram calculations:
    hist_total_RT_shadow_price, _ = np.histogram(
        [
            row["TP_for_chart"]
            for index, row in weekly_info.query(
                "Rating == @percent_of_rating"
            ).iterrows()
        ],
        bins=bins,
        weights=weekly_info.query("Rating == @percent_of_rating")[
            "observed_RT_shadow_price"
        ],
    )
    hist_num_days, _ = np.histogram(
        [
            row["TP_for_chart"]
            for index, row in weekly_info.query(
                "Rating == @percent_of_rating"
            ).iterrows()
        ],
        bins=bins,
    )
    hist_total_shadow_price = (hist_total_RT_shadow_price / hist_num_days).sum()
    return (
        (100 * RT_shadow_price_of_this_blob / total / num_of_days),
        (100 * hist_total_RT_shadow_price / hist_num_days / hist_total_shadow_price),
        title,
    )


collection_of_criteria = [
    weight_on_num_of_days_in_the_blob,
    weight_on_number_of_hours_in_the_blob,
    weight_on_num_of_active_hours_per_day,
    weight_on_total_RT_shadow_price,
    weight_on_average_RT_shadow_prices_per_day,
]


# def on_click(event, daily_confusion_FP=daily_confusion_FP, daily_confusion_FN=daily_confusion_FN, num_of_points=num_of_points, radius_for_points=radius_for_points):
def on_click(event):
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
                    f" Total (and average) number of hours: {num_of_points[i]}, ({num_of_points[i] / num_of_days}:.2f)"
                )
                total_RT_shadow_price = weekly_info.query(
                    "index in @point_ind[@i]"
                ).observed_RT_shadow_price.sum()
                print(
                    f" Total (and average) RT shadow price: {total_RT_shadow_price} ({total_RT_shadow_price / num_of_days}:.2f)"
                )
                print(f" False Positive: {daily_confusion_FP[i]}")
                print(f" False Negative: {daily_confusion_FN[i]}")

                print(weekly_info.query("index in @point_ind[@i]"))
                # print(weekly_info.query('index in @point_ind[@i]')[['TP_for_chart', 'Monitored_uid', 'Contingency_uid', 'Num Hours', 'observed_RT_shadow_price']])
                # print(weekly_info.query('index in @point_ind[@i]')[['TP_for_chart', 'Monitored_uid', 'Contingency_uid', 'Date', 'forecast_RT_shadow_price']])
                print(" ")
                break


weekly_info["TP_for_chart"] = 1 - (
    weekly_info["FP_for_chart"] + weekly_info["FN_for_chart"]
)
# new_row = {'title':title, 'Rating': percent_of_rating, 'bin':this_bin, 'value':this_hist_value, 'date_begining':date_begining, 'date_ending':date_ending}
# printout_collection = {}
# printout_collection['title'] = []
# printout_collection['Rating'] = []
# printout_collection['bin'] = []
# printout_collection['value'] = []
# printout_collection['date_begining'] = []
# printout_collection['date_ending'] = []

# Load the dictionary from the file
try:
    with open(f"histogram_printout_collection_{run}_NEW.pkl", "rb") as file:
        all_historic_printout_results = pickle.load(file)
        past_histogram_printout_collection = all_historic_printout_results["actual"]
        FP_historic_collection = all_historic_printout_results["FP"]
except FileNotFoundError:
    past_histogram_printout_collection = []
    FP_historic_collection = []


titles = ["title", "Rating", "bin", "value", "date_begining", "date_ending"]
titles_FP = ["title", "value", "date_begining", "date_ending"]
histogram_collection_of_data = []
# historic_printout_results = pd.DataFrame(columns=titles)
colors = ["green", "orange", "red"]


def create_plot(Radius_to_unify_points):
    FP_stats = {}

    # return the index of the blob that belongs to the FP
    def find_FP_cluster():
        ind = 0
        for FP, FN, TP in zip(
            daily_confusion_FP, daily_confusion_FN, daily_confusion_TP
        ):
            if TP == 0 and FP > 0 and FN == 0:
                return ind
            ind += 1

    def calculate_radius(i):
        return np.interp(daily_confusion_TP, (0, 1), (0.03, 0.2))[i]

    len_percent_of_ratings = len(percent_of_ratings)
    if subplotting:
        fig, axs = plt.subplots(
            len_percent_of_ratings, 5, figsize=(30, 6 * len_percent_of_ratings)
        )

    for ind_rating, percent_of_rating in enumerate(percent_of_ratings):
        # weekly_info = weekly_info.query('Rating == @percent_of_rating')
        # weekly_info = weekly_info.query('observed_RT_shadow_price > 0')
        # weekly_info = weekly_info.query('Num Hours > 10')
        # weekly_info = weekly_info.query('Date == @to_date')

        if not subplotting:
            fig, axs = plt.subplots(2, 5, figsize=(30, 12))

        Counters.clear()
        daily_confusion_FP.clear()
        daily_confusion_FN.clear()
        daily_confusion_TP.clear()
        num_of_points.clear()
        # original_points = []  # Initialize variables to store original points
        point_ind.clear()  # Initialize variables to store the index of the merged point
        # original_pnts = []
        radius_for_points.clear()

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
                forecast_RT_shadow_price,
            ) in collected_results[date_of_forecast]:
                # if contingency_uid == 'MFOWLOB5' and date_of_forecast == '2024-05-01':
                #     print(18)
                if rating != percent_of_rating:
                    continue
                Counters.append(counter)
                daily_confusion_FP.append(false_positive)
                daily_confusion_FN.append(false_negative)
                daily_confusion_TP.append(1 - false_negative - false_positive)
                radius_for_points.append(0)
                num_of_points.append(num_points)
                # original_points.append([(false_positive, false_negative)])  # Initialize original points for each merged point
                point_ind.append([])

        daily_confusion_FP_orig = daily_confusion_FP.copy()
        daily_confusion_FN_orig = daily_confusion_FN.copy()
        daily_confusion_TP_orig = daily_confusion_TP.copy()
        num_of_points_orig = num_of_points.copy()

        # daily_confusion_FP    # daily_confusion_FN     # num_of_points    # # original_points    # point_ind    # radius_for_points    #
        Points_unified = []
        # some of the points are very close to each other, so we need to unify them into one big point:
        for i, cnt_i in enumerate(Counters):
            if cnt_i in Points_unified:
                continue
            point_ind[i].append(cnt_i)
            # for j in range(len(daily_confusion_FP)-1,i,-1):
            j = len(Counters)
            for cnt_j in Counters[-1:i:-1]:
                j -= 1
                if cnt_j in Points_unified:
                    continue
                Radius_to_unify_points = calculate_radius(i)
                if (
                    abs(daily_confusion_FP[i] - daily_confusion_FP[j])
                    < Radius_to_unify_points
                    and abs(daily_confusion_FN[i] - daily_confusion_FN[j])
                    < Radius_to_unify_points
                ):
                    # original_points[i].append((daily_confusion_FP[j], daily_confusion_FN[j])) # Add original point to the corresponding merged point
                    point_ind[i].append(cnt_j)
                    daily_confusion_FP[i] = (
                        daily_confusion_FP[i] * num_of_points[i]
                        + daily_confusion_FP[j] * num_of_points[j]
                    ) / (num_of_points[i] + num_of_points[j])
                    daily_confusion_FN[i] = (
                        daily_confusion_FN[i] * num_of_points[i]
                        + daily_confusion_FN[j] * num_of_points[j]
                    ) / (num_of_points[i] + num_of_points[j])
                    daily_confusion_TP[i] = (
                        daily_confusion_TP[i] * num_of_points[i]
                        + daily_confusion_TP[j] * num_of_points[j]
                    ) / (num_of_points[i] + num_of_points[j])
                    num_of_points[i] = num_of_points[i] + num_of_points[j]
                    Points_unified.append(cnt_j)

        # Eliminate points that were unified
        i = len(Counters) - 1
        # for i in range(len(daily_confusion_FP),-1,-1):
        for cnt_i in Counters[-1::-1]:
            if cnt_i in Points_unified:
                daily_confusion_FP.pop(i)
                daily_confusion_FN.pop(i)
                daily_confusion_TP.pop(i)
                num_of_points.pop(i)
                # original_points.pop(i)
                point_ind.pop(i)
                radius_for_points.pop(i)
            i -= 1

        num_days_in_each_blob = []
        for i in range(len(daily_confusion_FP)):
            num_days_in_each_blob.append(num_of_days_in_the_blob(i))

        for ind, chosen_function in enumerate(collection_of_criteria):

            # create contours for equal TP
            # ax = plt.gca()
            # if subplotting == True:
            ax = axs[0, ind]
            # else:
            #     ax = plt.gca()

            historic_means_FP = []
            historic_stds_FP = []
            current_histogram_value_FP = []

            ax.set_aspect("equal", adjustable="box")  # Set aspect ratio to be equal
            theta = np.linspace(
                0, 2 * np.pi / 4, 100
            )  # Define angles for the circular grid
            # r = np.arange(0, 1.1, .2)  # Define radii for the circular grid
            r = [np.sqrt(r) for r in [0.5, 0.9, 1]]
            for this_ind, radius in enumerate(r):
                x_circle = radius * np.cos(theta)
                y_circle = radius * np.sin(theta)
                ax.plot(
                    x_circle,
                    y_circle,
                    color=colors[this_ind],
                    linestyle="dashed",
                    alpha=0.7,
                    linewidth=3.7,
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
            values_for_histogram = []
            # circle each blob using a radius that is proportional to the number of points in the blob
            for i in range(len(daily_confusion_FP)):
                value, value_histogram, title = chosen_function(i, percent_of_rating)
                values.append(value)
                values_for_histogram.append(value_histogram)

            # The number of points of each blob and the curcle around it are proportional to the number of points in the blob
            text_size = np.interp(
                values, (min(values), (max(values) + 100) / 2), (10, 30)
            )
            radius_range = np.interp(
                values, (min(values), (max(values) + 100) / 2), (0.03, 0.08)
            )
            # text_size = np.interp(values, (min(values), (max(values) + 100)/2), (5, 15))
            # radius_range = np.interp(values, (min(values), (max(values) + 100)/2), (.02, .04))

            # The FP blob's circle should be black:
            ind_of_FP_blob = find_FP_cluster()

            for i in range(len(daily_confusion_FP)):
                if daily_confusion_TP[i] >= 0.5:
                    ax.text(
                        np.sqrt(daily_confusion_FP[i]),
                        np.sqrt(daily_confusion_FN[i]),
                        f"{values[i]:.0f}",
                        fontsize=text_size[i],
                        ha="center",
                        va="center",
                        color=colors[0],
                    )
                elif daily_confusion_TP[i] > 0.1:
                    ax.text(
                        np.sqrt(daily_confusion_FP[i]),
                        np.sqrt(daily_confusion_FN[i]),
                        f"{values[i]:.0f}",
                        fontsize=text_size[i],
                        ha="center",
                        va="center",
                        color=colors[1],
                    )
                else:
                    ax.text(
                        np.sqrt(daily_confusion_FP[i]),
                        np.sqrt(daily_confusion_FN[i]),
                        f"{values[i]:.0f}",
                        fontsize=text_size[i],
                        ha="center",
                        va="center",
                        color=colors[2],
                    )
                if ind_of_FP_blob == i:
                    ax.text(
                        np.sqrt(daily_confusion_FP[i]),
                        np.sqrt(daily_confusion_FN[i]),
                        f"{values[i]:.0f}",
                        fontsize=text_size[i],
                        ha="center",
                        va="center",
                        color="black",
                    )

            for i in range(len(daily_confusion_FP)):
                # add a circle for each point
                radius_for_points[i] = radius_range[i]
                if daily_confusion_TP[i] >= 0.5:
                    circle = Circle(
                        (
                            np.sqrt(daily_confusion_FP[i]),
                            np.sqrt(daily_confusion_FN[i]),
                        ),
                        radius=radius_for_points[i],
                        fill=False,
                        color=colors[0],
                    )
                elif daily_confusion_TP[i] > 0.1:
                    circle = Circle(
                        (
                            np.sqrt(daily_confusion_FP[i]),
                            np.sqrt(daily_confusion_FN[i]),
                        ),
                        radius=radius_for_points[i],
                        fill=False,
                        color=colors[1],
                    )
                else:
                    circle = Circle(
                        (
                            np.sqrt(daily_confusion_FP[i]),
                            np.sqrt(daily_confusion_FN[i]),
                        ),
                        radius=radius_for_points[i],
                        fill=False,
                        color=colors[2],
                    )
                if ind_of_FP_blob == i:
                    circle = Circle(
                        (
                            np.sqrt(daily_confusion_FP[i]),
                            np.sqrt(daily_confusion_FN[i]),
                        ),
                        radius=radius_for_points[i],
                        fill=False,
                        color="black",
                    )

                ax.add_patch(circle)

            if percent_of_rating == -1:
                plt.connect("button_press_event", on_click)

            ax.set_xticks([np.sqrt(0.1), np.sqrt(0.5), np.sqrt(0.9), 1])
            ax.set_yticks([np.sqrt(0.1), np.sqrt(0.5), np.sqrt(0.9), 1])
            ax.set_xticklabels([f"{.1:.2f}", f"{.5:.2f}", f"{.9:.1f}  ", f"{1:.0f}"])
            ax.set_yticklabels([f"{.1:.2f}", f"{.5:.2f}", f"{.9:.1f}  ", f"{1:.0f}"])

            ax.set_xlabel("False Positive")
            ax.set_ylabel("False Negative")
            if percent_of_rating == -1:
                ax.set_title(f"{title}: \n rating: BEST")
            else:
                ax.set_title(f"{title}: \n rating: {int(100*percent_of_rating)}%")

            # add grid to the image. The grid should be radial, and reflects the fact that the closer to the center, the better the results
            ax.grid(True)

            # Create a histogram based on TP
            # if True:
            ax = axs[1, ind]
            # ind, chosen_function in enumerate(collection_of_criteria)
            if chosen_function == weight_on_num_of_active_hours_per_day:

                hist_total_hours, _ = np.histogram(
                    [
                        row["TP_for_chart"]
                        for index, row in weekly_info.query(
                            "Rating == @percent_of_rating"
                        ).iterrows()
                    ],
                    bins=bins,
                    weights=weekly_info.query("Rating == @percent_of_rating")[
                        "Num Hours"
                    ],
                )
                hist_num_days, _ = np.histogram(
                    [
                        row["TP_for_chart"]
                        for index, row in weekly_info.query(
                            "Rating == @percent_of_rating"
                        ).iterrows()
                    ],
                    bins=bins,
                )
                for ind, h in enumerate(hist_num_days):
                    if h == 0:
                        hist_num_days[ind] = 1
                hist = hist_total_hours / hist_num_days

                ax.bar(
                    [1 - bin for bin in bins][-1::-1][:-1],
                    hist[-1::-1],
                    width=np.abs(np.diff([1 - bin for bin in bins][-1::-1])),
                    align="edge",
                    edgecolor="black",
                    color=["green", "orange", "red"],
                    alpha=0.5,
                )

            else:
                hist = values_for_histogram[0]

                if False:
                    container = ax.bar(
                        [1 - bin for bin in bins][-1::-1][:-1],
                        hist[-1::-1],
                        width=np.abs(np.diff([1 - bin for bin in bins][-1::-1])),
                        align="edge",
                        edgecolor="black",
                        color=["green", "orange", "red"],
                        alpha=0.5,
                    )
                else:
                    total_height = hist[-1::-1]
                    ind_of_FP_blob = find_FP_cluster()
                    percent_of_FP_out_of_all_observations = values[
                        ind_of_FP_blob
                    ] / sum(values)
                    bottom_height = (
                        total_height
                        * np.array([1, 1, 1])
                        / (1 - percent_of_FP_out_of_all_observations)
                    )
                    FP_plus_FN = bottom_height[-1]
                    bottom_height[-1] = 100 - sum(bottom_height[:-1])
                    FP_percentage = FP_plus_FN - bottom_height[-1]
                    top_height = [0, 0, FP_percentage]

                    container = ax.bar(
                        [1 - bin for bin in bins][-1::-1][:-1],
                        bottom_height,
                        width=np.abs(np.diff([1 - bin for bin in bins][-1::-1])),
                        align="edge",
                        edgecolor="black",
                        color=["green", "orange", "red"],
                        alpha=0.5,
                    )

                    ax.bar(
                        [1 - bin + 0.1 for bin in bins][-1::-1][:-1],
                        top_height,
                        width=np.abs(np.diff([1 - bin for bin in bins][-1::-1])),
                        align="edge",
                        edgecolor="black",
                        color=["green", "orange", "black"],
                        alpha=0.5,
                    )

            if not chosen_function == weight_on_num_of_active_hours_per_day:
                ax.set_xticks([0.25, 0.7, 0.95, 1.05])
                ax.set_xticklabels(
                    ["TP > 0.5", "0.5-0.9", "< 0.1", " FP"], rotation=-45, fontsize=16
                )
            else:
                ax.set_xticks([0.25, 0.7, 0.95])
                ax.set_xticklabels(
                    ["TP > 0.5", "0.5-0.9", "< 0.1"], rotation=-45, fontsize=16
                )
            ax.set_xlabel("True Positive")
            ax.set_ylabel("Percent")
            ax.set_title(f"Histogram of True Positive, rating = {percent_of_rating}")

            # save data for printout
            date_begining = weekly_info.Date.min()
            date_ending = weekly_info.Date.max()

            # do not delete: This organizes the data in a pandas DataFrame
            # for this_bin, this_hist_value in zip(bins, hist):
            #     new_row = {'title':title, 'Rating': percent_of_rating, 'bin':this_bin, 'value':this_hist_value, 'date_begining':date_begining, 'date_ending':date_ending}
            #     historic_printout_results.loc[len(historic_printout_results)] = new_row

            df_historic_actual = pd.DataFrame(
                past_histogram_printout_collection, columns=titles
            )

            # remove duplicates in df_historic_actual
            df_historic_actual = df_historic_actual.drop_duplicates(
                subset=[
                    "title",
                    "Rating",
                    "bin",
                    "value",
                    "date_begining",
                    "date_ending",
                ]
            )

            # find the percentile of each bin and put it on the histogram
            locs_to_write_percentiles = [0.25, 0.7, 0.95]
            ind = 0
            historic_means = []
            historic_stds = []
            current_histogram_value = []
            for this_bin, this_hist_value in zip(
                [1 - bin for bin in bins][-1::-1], hist[-1::-1]
            ):

                if not chosen_function == weight_on_num_of_active_hours_per_day:
                    if ind < 2:
                        this_hist_value /= 1 - percent_of_FP_out_of_all_observations
                    else:
                        this_hist_value = 100 - sum(bottom_height[:-1])

                relevant_values = df_historic_actual.query(
                    "title == @title and Rating == @percent_of_rating and bin == @this_bin"
                )["value"]
                # convert relevant_values to np.array
                relevant_values = np.array(sorted(relevant_values))

                # find the statistics of the relevant_values
                historic_means.append(np.mean(relevant_values))
                historic_stds.append(np.std(relevant_values))
                current_histogram_value.append(this_hist_value)

                # find the percentile of this_hist_value in the relevant_values by assuming a normal distribution
                quantile = norm.cdf(
                    this_hist_value,
                    loc=np.mean(relevant_values),
                    scale=np.std(relevant_values),
                )
                # quantile = bisect.bisect(relevant_values, this_hist_value) / (len(relevant_values) + 1) + 1 / 2 / (len(relevant_values) + 1)

                # put the quantile on the histogram, in the middle of the bin
                if quantile < 0.1 or quantile > 0.9:
                    ax.text(
                        locs_to_write_percentiles[ind],
                        this_hist_value,
                        f"{100*quantile:.0f}%",
                        fontsize=24,
                        ha="center",
                        va="bottom",
                        color="red",
                    )
                else:
                    ax.text(
                        locs_to_write_percentiles[ind],
                        this_hist_value,
                        f"{100*quantile:.0f}%",
                        fontsize=12,
                        ha="center",
                        va="bottom",
                        color="black",
                    )
                ind += 1

                past_histogram_printout_collection.append(
                    [
                        title,
                        percent_of_rating,
                        this_bin,
                        this_hist_value,
                        date_begining,
                        date_ending,
                    ]
                )

            # find the percentile of each bin
            # Calculate yerr values. This is the difference between the current histogram value and the mean of the historic values
            yerr = [[], []]
            for mean, std, hist_value in zip(
                historic_means, historic_stds, current_histogram_value
            ):
                yerr[0].append(max(0, hist_value - mean))
                yerr[1].append(max(0, -hist_value + mean))

            # Add error bars
            # Calculate bin midpoints
            bin_midpoints = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

            # # Add error bars
            if chosen_function == weight_on_num_of_active_hours_per_day:
                ax.errorbar(
                    [1 - bin for bin in bin_midpoints][-1::-1],
                    hist[-1::-1],
                    yerr=yerr,
                    fmt="none",
                    capsize=5,
                    color="black",
                )
            else:
                ax.errorbar(
                    [1 - bin for bin in bin_midpoints][-1::-1],
                    bottom_height,
                    yerr=yerr,
                    fmt="none",
                    capsize=5,
                    color="black",
                )

            if (
                title
                in [
                    "weight: num of days in the blob",
                    "weight: number of hours in the blob",
                ]
                and percent_of_rating == -1
            ):

                # find the historic stats for the FP
                df_historic_FP = pd.DataFrame(FP_historic_collection, columns=titles_FP)
                relevant_values_FP = df_historic_FP.query("title == @title")["value"]

                # convert relevant_values_FP to np.array
                relevant_values_FP = np.array(sorted(relevant_values_FP))

                # remove dupolicates from df_historic_FP
                df_historic_FP = pd.DataFrame(FP_historic_collection, columns=titles_FP)
                df_historic_FP = df_historic_FP.drop_duplicates(
                    subset=["value", "date_begining", "date_ending"]
                )

                # find the percentile of this_hist_value in the relevant_values_FP by assuming a normal distribution
                FP_percentage_normalized = np.round(
                    100 * FP_percentage / (FP_percentage + 100)
                )
                quantile_FP = norm.cdf(
                    FP_percentage_normalized,
                    loc=np.mean(relevant_values_FP),
                    scale=np.std(relevant_values_FP),
                )
                FP_historic_collection.append(
                    [
                        title,
                        np.round(FP_percentage_normalized * 100) / 100,
                        date_begining,
                        date_ending,
                    ]
                )

                if (
                    title == "weight: num of days in the blob"
                    and percent_of_rating == -1
                ):
                    FP_stats["weight: num of days in the blob"] = {
                        "value": FP_percentage_normalized,
                        "quantile": np.round(100 * quantile_FP),
                        "Mean": np.round(np.mean(relevant_values_FP)),
                    }
                elif (
                    title == "weight: number of hours in the blob"
                    and percent_of_rating == -1
                ):
                    FP_stats["weight: number of hours in the blob"] = {
                        "value": FP_percentage_normalized,
                        "quantile": np.round(100 * quantile_FP),
                        "Mean": np.round(np.mean(relevant_values_FP)),
                    }

    # add the latest data to the historic_printout_results
    df_historic_actual = pd.DataFrame(
        past_histogram_printout_collection, columns=titles
    )
    df_historic_FP = pd.DataFrame(FP_historic_collection, columns=titles_FP)

    # remove duplicates in df_historic_actual and df_historic_FP
    df_historic_actual = df_historic_actual.drop_duplicates(
        subset=["title", "Rating", "bin", "value", "date_begining", "date_ending"]
    )
    df_historic_FP = df_historic_FP.drop_duplicates(
        subset=["title", "value", "date_begining", "date_ending"]
    )

    print(
        f'--> False Positive % for number of days: {FP_stats["weight: num of days in the blob"]["value"]:.0f}%  |   {FP_stats["weight: num of days in the blob"]["quantile"]:.0f}-th quantile of the historic data  |  Historic mean: {FP_stats["weight: num of days in the blob"]["Mean"]:.0f}'
    )
    print(
        f'--> False Positive % for number of hours: {FP_stats["weight: number of hours in the blob"]["value"]:.0f}%  |  {FP_stats["weight: number of hours in the blob"]["quantile"]:.0f}-th quantile of the historic data  |  Historic mean: {FP_stats["weight: number of hours in the blob"]["Mean"]:.0f}'
    )
    # save_historica_printout_results(df_historic_actual, df_historic_FP)

    ############

    ###############
    # # Given points
    # # X = [.2, .435, .56, .432, .76, .345]
    # # Y = [.4, .23, .4534, .48, .96, .452]
    # # values = [4, 6, 5, 7, 4.3, 2.9]

    # # Define grid size and generate meshgrid
    # grid_x, grid_y = np.mgrid[min(daily_confusion_FP):max(daily_confusion_FP):100j, min(daily_confusion_FN):max(daily_confusion_FN):100j]

    # # Interpolate values on the grid
    # grid_z = griddata((daily_confusion_FP, daily_confusion_FN), values, (grid_x, grid_y), method='cubic')

    # # Create heatmap
    # # ax.imshow(grid_z.T, extent=(min(daily_confusion_FP), max(daily_confusion_FP), min(daily_confusion_FN), max(daily_confusion_FN)), origin='lower', cmap='viridis')
    # ax.imshow(grid_z.T, extent=(min(daily_confusion_FP), max(daily_confusion_FP), min(daily_confusion_FN), max(daily_confusion_FN)), origin='lower', cmap='plasma', vmin=-4, vmax=10)
    # # plt.colorbar(label='Value')
    ###############

    # plt.show()

    # return daily_confusion_FP, daily_confusion_FN, radius_for_points
    return


create_plot(Radius_to_unify_points=0.15)

plt.show()
