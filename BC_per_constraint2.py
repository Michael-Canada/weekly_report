#!/usr/bin/env python3
"""
BC_per_constraint.py - Binding Constraint Analysis Script

This script analyzes binding constraints by comparing actual vs forecast shadow prices
across different ISOs (ERCOT, MISO, PJM, SPP) and calculates confusion metrics.

Environment: placebo_api_local
"""

# =============================================================================
# IMPORTS
# =============================================================================
import datetime
import io
import os
import pickle
import sys
from typing import Dict, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.patches import Circle
from tqdm import tqdm

# Add system paths for placebo modules
sys.path.append("/Users/michael.simantov/Documents/mu-placebo-api")
sys.path.append("/Users/michael.simantov/Documents/mu-placebo-api/placebo_api")
sys.path.append("/Users/michael.simantov/Documents/mu-placebo-api/placebo_api/utils")

import api_utils, date_utils
from date_utils import LocalizedDateTime
from placebo.utils import snowflake_utils

# =============================================================================
# CONFIGURATION AND SETTINGS
# =============================================================================

# Analysis flags
USE_SNOWFLAKE = True
WITH_IIR = True
GROUP_BY_BRANCH = False  # If True: group by monitored_uid only, if False: group by (monitored_uid, contingency_uid)

if not WITH_IIR:
    USE_SNOWFLAKE = True

# ISO Configuration
# ISO_name = "ercot"  # Options: "spp", "ercot", "pjm", "miso"
# ISO_name = "pjm"  # Options: "spp", "ercot", "pjm", "miso"
ISO_name = "miso"


# Schema mappings for different ISOs
SCHEMA_CONFIG = {
    "miso": {
        "multi_bids": "miso_20250513T1101Z",
        "default": "miso_20241126T0106Z"  # Jacob
    },
    "ercot": {
        "env": "ercot_20250421T1301Z", 
        "default": "20240909T1135Z"  # Jacob
    },
    "spp": {
        "default": "spp_20250522T1501Z"  # Jacob
    },
    "pjm": {
        # "default": "pjm_20250623T1002Z"  # Jacob
        "default": "pjm_20250930T1002Z"
        # "default": "pjm_20250623T1002Z"
    }
}

# Get schema for current ISO
if ISO_name in SCHEMA_CONFIG:
    schema = SCHEMA_CONFIG[ISO_name]["default"]
else:
    raise ValueError(f"Unsupported ISO: {ISO_name}")

# Date range for analysis
min_analysis_date = "2025-10-10"
max_analysis_date = "2025-10-30"

# Run configuration - maps to API endpoints
# Available runs can be found at: https://api1.marginalunit.com/pr-forecast/runs
RUN_OPTIONS = [
    "ercot_env", "miso", "spp", "pjm", "ercot_20240909", "ercot",
]
# run = "ercot_env"  # Current active run
# run = "pjm_20250930"
run = "miso"
# run = "pjm"
# run = "ercot"


# Analysis parameters
SHADOW_PRICE_CUTOFF = 0
MIN_REQUIRED_NUM_OF_HOURS_IN_CONTEGTION = 0
max_num_congestions_per_day = 1000
Num_of_days_to_look_back = 400  # Reduced from 400 to match analysis window

# Timezone mappings for different ISOs
ISO_TO_TZ = {
    "spp": "Us/Central",
    "miso": "Etc/GMT+5", 
    "ercot": "US/Central",
    "pjm": "US/Eastern",
}


# =============================================================================
# DATA CLASSES AND AUTHENTICATION
# =============================================================================

class AsOf(NamedTuple):
    """Container for as_of timestamp information."""
    as_of_str: str
    as_of: LocalizedDateTime


# =============================================================================
# API CONFIGURATION AND AUTHENTICATION
# =============================================================================

def _get_auth():
    """Get authentication credentials from environment variables."""
    return tuple(os.environ["MU_API_AUTH"].split(":"))


# API endpoints
AUTH = _get_auth()
MISC_ROOT = "https://api1.marginalunit.com/misc-data"
CONSTRAINTDB_ROOT = "https://api1.marginalunit.com/constraintdb"
URL_ROOT = "https://api1.marginalunit.com/muse/api"


def _get_dfm(url):
    """
    Fetch data from URL and return as StringIO for pandas processing.
    
    Args:
        url (str): URL to fetch data from
        
    Returns:
        io.StringIO or None: StringIO object with CSV data or None if error
    """
    resp = requests.get(url, auth=AUTH)
    
    if resp.status_code != 200:
        print(f"skipping the following: {url}")
        return None
    
    return io.StringIO(resp.text)


# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def process_actual(content_actual):
    """
    Process actual shadow price data from CSV content.
    
    Args:
        content_actual: CSV content as StringIO
        
    Returns:
        pd.DataFrame: Processed DataFrame with period and date columns
    """
    df_bc_actual = pd.read_csv(content_actual)
    if len(df_bc_actual) == 0:
        return df_bc_actual
    
    df_bc_actual["period"] = df_bc_actual["period"].apply(
        datetime.datetime.fromisoformat
    )
    df_bc_actual["date"] = df_bc_actual["period"].apply(lambda x: x.date())
    
    return df_bc_actual


# =============================================================================
# DATA COLLECTION AND INITIALIZATION
# =============================================================================

def collect_as_ofs():
    """
    Collect as_of timestamps for the analysis period.
    
    Returns:
        tuple: (as_ofs_actual, as_ofs_last_year)
    """
    # Get list of as_ofs from API
    content = api_utils.fetch_from_placebo(endpoint=f"{run}/as_ofs")
    as_ofs_str = [
        line.rstrip("\n") 
        for line in content.readlines() 
        if "as_of" not in line
    ]
    
    as_ofs = [
        AsOf(as_of_str=as_of_str, as_of=date_utils.localized_from_isoformat(as_of_str))
        for as_of_str in as_ofs_str
    ]
    
    hour = 4
    as_ofs_last_year = [
        as_of for as_of in as_ofs 
        if as_of.as_of.year >= 2023 and as_of.as_of.hour == hour
    ]
    
    return as_ofs, as_ofs_last_year


# Initialize data storage containers (will be populated in main)
contents_actual = {}
contents_forecast = {}
contents_actual_old = {}
as_ofs_actual = []

def collect_actual_and_forecast_data():
    """
    Collect actual and forecast shadow price data for all analysis dates.
    
    Returns:
        tuple: Updated contents_actual and contents_forecast dictionaries
    """

    for as_of in tqdm(as_ofs_actual[-1::-1]):
        date = as_of.as_of.date()
        date_strp = datetime.datetime.strptime(date.strftime("%Y-%m-%d"), "%Y-%m-%d")
        min_date = datetime.datetime.strptime(min_analysis_date, "%Y-%m-%d")
        max_date = datetime.datetime.strptime(max_analysis_date, "%Y-%m-%d")

        # Skip dates outside analysis range
        if date_strp < min_date or date_strp > max_date:
            continue

        # Collect actual binding events data
        url = f"{CONSTRAINTDB_ROOT}/{ISO_name}/rt/binding_events?start_datetime={date.strftime('%Y-%m-%d')}&end_datetime={(datetime.timedelta(days=1)+date).strftime('%Y-%m-%d')}"
        content = _get_dfm(url)
        if content is None:
            print(f"Error 293847238: no data for {url}")
            continue
        
        df_actual_one = process_actual(content)
        contents_actual_old[as_of.as_of_str] = df_actual_one
        contents_actual.update({
            date_.strftime("%Y-%m-%d"): group
            for date_, group in df_actual_one.groupby(df_actual_one["date"])
        })

        # Collect forecast data based on data source
        if USE_SNOWFLAKE:
            # Use Snowflake data source
            stream_uid = 'load_100_wind_100_sol_100_ng_100_iir' if WITH_IIR else 'load_100_wind_100_sol_100_ng_100'
            
            query = f"""
            select stream_uid, timestamp, monitored_uid, contingency_uid, shadow_price, 
                   from_station, from_kv, to_station, to_kv
            from "binding_constraint" 
            where stream_uid = '{stream_uid}'
            AND HOUR(as_of) = 4 
            and DATE(as_of) = '{date.strftime("%Y-%m-%d")}'
            and shadow_price > 0 
            order by shadow_price
            """
            
            cnx_string = f'snowflake://MICHAEL.SIMANTOV@DRILLINGINFO.COM@enverus_pr.us-east-1/"pr-forecast"/"{schema}"?privatekeyfile=./rsa_key.p8'
            cnx_params = snowflake_utils.ConnectionParams.from_cnx_string(cnx_string)
            cnx = cnx_params.build()
            
            df_bc_actual_FP = snowflake_utils.execute_and_fetch(cnx, query)
            
            if len(df_bc_actual_FP) > 0:
                df_bc_actual_FP.timestamp = df_bc_actual_FP.timestamp.dt.tz_convert(ISO_TO_TZ[ISO_name])
                df_bc_actual_FP["as_of"] = df_bc_actual_FP.timestamp.dt.tz_convert(ISO_TO_TZ[ISO_name])
                df_bc_actual_FP["timestamp"] = df_bc_actual_FP["timestamp"].astype(str)
        
        else:
            # Use API data source
            url_FP = f"https://api1.marginalunit.com/pr-forecast/{run}/binding_constraint?as_of={date.strftime('%Y-%m-%d')}T04%3A00%3A00-05%3A00&resample=H&include_empty_timestamps=true&stream_uid=load_100_wind_100_sol_100_ng_100_iir"
            url_FP_winter = f"https://api1.marginalunit.com/pr-forecast/{run}/binding_constraint?as_of={date.strftime('%Y-%m-%d')}T05%3A00%3A00-05%3A00&resample=H&include_empty_timestamps=true&stream_uid=load_100_wind_100_sol_100_ng_100_iir"
            
            content_FP = _get_dfm(url_FP)
            if content_FP is None:
                print(f"Error 92837498234: no data for {url_FP}")
            
            print(date)
            df_bc_actual_FP = pd.read_csv(content_FP) if content_FP else pd.DataFrame()
            
            # Try winter URL if no data found
            if len(df_bc_actual_FP) == 0:
                content_FP = _get_dfm(url_FP_winter)
                if content_FP is None:
                    print(f"Error 738475839485798: no data in Winter URL {url_FP_winter}")
                    df_bc_actual_FP = pd.DataFrame()
                else:
                    print("using Winter URL")
                    df_bc_actual_FP = pd.read_csv(content_FP)

        # Process forecast data if available
        No_FP_data_was_found = len(df_bc_actual_FP) == 0
        
        if not No_FP_data_was_found:
            date_later_1 = f"{(datetime.timedelta(days=1)+date).strftime('%Y-%m-%d')}"
            date_later_2 = f"{(datetime.timedelta(days=2)+date).strftime('%Y-%m-%d')}"
            Forecast_RT_shadow_price_of_the_day = df_bc_actual_FP.query(
                "timestamp > @date_later_1 and timestamp < @date_later_2"
            )
            contents_forecast[date_later_1] = Forecast_RT_shadow_price_of_the_day
    
    return contents_actual, contents_forecast

# Data collection will be moved to main execution block


# =============================================================================
# CONGESTION ANALYSIS FUNCTIONS  
# =============================================================================

def find_top_congestions(df_actual_one, df_forecast_one, num_of_congestions=None):
    """
    Find top congestions by aggregating shadow prices and merging actual vs forecast data.
    
    Args:
        df_actual_one (pd.DataFrame): Actual shadow price data
        df_forecast_one (pd.DataFrame): Forecast shadow price data  
        num_of_congestions (int): Maximum number of congestions to return
        
    Returns:
        pd.DataFrame: Top congestions with actual and forecast shadow prices
    """
    if num_of_congestions is None:
        num_of_congestions = max_num_congestions_per_day
        
    # Process actual congestion data
    top_congestions = (
        df_actual_one.groupby(["monitored_uid", "contingency_uid"])
        .agg({"shadow_price": "sum"})
        .sort_values("shadow_price", ascending=False)
        .head(num_of_congestions)
    )

    # Process forecast congestion data
    forecast_data = (
        df_forecast_one.groupby(["monitored_uid", "contingency_uid"])
        .agg({"shadow_price": "sum"})
        .sort_values("shadow_price", ascending=False)
    )
    forecast_data["forecast_shadow_price"] = forecast_data["shadow_price"]
    forecast_data["shadow_price"] = 0

    # Merge actual and forecast data
    merged_df = top_congestions.merge(
        forecast_data[["forecast_shadow_price"]],
        left_index=True,
        right_index=True,
        how="outer",
    )
    merged_df["shadow_price"] = merged_df["shadow_price"].fillna(0)
    merged_df["forecast_shadow_price"] = merged_df["forecast_shadow_price"].fillna(0)

    # Filter for congestions of interest
    congestions_with_interest = merged_df[
        (merged_df["forecast_shadow_price"] > 10) | 
        (merged_df["shadow_price"] > 10) | 
        ((merged_df["forecast_shadow_price"] > 0) & (merged_df["shadow_price"] > 0))
    ]

    return congestions_with_interest


def find_top_congestions_legacy(df_actual_one, df_forecast_one, num_of_congestions=None):
    """
    Legacy version of congestion finding function (before Jacob's modifications).
    
    Args:
        df_actual_one (pd.DataFrame): Actual shadow price data
        df_forecast_one (pd.DataFrame): Forecast shadow price data
        num_of_congestions (int): Maximum number of congestions to return
        
    Returns:
        pd.DataFrame: Top congestions with actual and forecast shadow prices
    """
    if num_of_congestions is None:
        num_of_congestions = max_num_congestions_per_day
        
    top_congestions = (
        df_actual_one.groupby(["monitored_uid", "contingency_uid"])
        .agg({"shadow_price": "sum"})
        .sort_values("shadow_price", ascending=False)
        .head(num_of_congestions)
    )
    top_congestions = top_congestions[top_congestions["shadow_price"] > 10]

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

    congestions_with_interest = merged_df[
        (merged_df["forecast_shadow_price"] > 10) | (merged_df["shadow_price"] > 0)
    ]

    return congestions_with_interest


def find_top_congestions_with_filtering(df_actual_one, df_forecast_one, num_of_congestions=None):
    """
    Enhanced congestion finding with hour count filtering and detailed processing.
    
    This function finds top congestions by aggregating shadow prices and applies 
    filtering based on minimum required hours and shadow price cutoffs.
    
    Args:
        df_actual_one (pd.DataFrame): Actual shadow price data
        df_forecast_one (pd.DataFrame): Forecast shadow price data  
        num_of_congestions (int): Maximum number of congestions to return
        
    Returns:
        pd.DataFrame: Filtered top congestions with actual and forecast shadow prices
    """
    if num_of_congestions is None:
        num_of_congestions = max_num_congestions_per_day
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


# =============================================================================
# MAIN ANALYSIS EXECUTION
# =============================================================================

def run_analysis():
    """
    Execute the main binding constraint analysis workflow.
    
    Returns:
        tuple: (all_top_contingencies, confusion_matrix, collected_results)
    """
    # Initialize analysis variables
    MUSE_DATA = {}
    cnt_downloaded = 0
    cnt_not_downloaded = 0
    RT_bindings_NOT_caught_by_MUSE_and_FORECAST = []
    collected_results = {}
    RT_bindings_NOT_caught_by_MUSE_and_FORECAST = {}

    all_top_contingencies = pd.DataFrame()
    FP_total = FN_total = TP_total = 0
    # Main analysis loop
    for as_of in tqdm(as_ofs_actual):
        this_date_ = as_of.as_of.date()
        date_strp = datetime.datetime.strptime(this_date_.strftime("%Y-%m-%d"), "%Y-%m-%d")
        min_date = datetime.datetime.strptime(min_analysis_date, "%Y-%m-%d")
        max_date = datetime.datetime.strptime(max_analysis_date, "%Y-%m-%d")

        # Skip dates outside analysis range
        if date_strp < min_date or date_strp > max_date:
            continue

        from_date = as_of.as_of.date().strftime("%Y-%m-%d")
        to_date = datetime.datetime.now().strftime("%Y-%m-%d")
        today = from_date
        tomorrow_date = (
            (as_of.as_of + datetime.timedelta(days=1)).date().strftime("%Y-%m-%d")
        )

        # Skip if too recent (no actual data available yet)
        if (datetime.datetime.now() - datetime.timedelta(days=1)).date() <= as_of.as_of.date():
            continue

        # Process congestions for this date
        try:
            top_contingencies = find_top_congestions(
                contents_actual[f"{tomorrow_date}"],
                contents_forecast[tomorrow_date],
            )
        except:
            print(f"Error 99398452385: no data for {tomorrow_date}")
            continue

        # Add date column and append to results
        top_contingencies["date"] = tomorrow_date
        all_top_contingencies = pd.concat([all_top_contingencies, top_contingencies])

        # Calculate confusion matrix metrics for this day
        FP_daily = FN_daily = TP_daily = 0
        for row in top_contingencies.iterrows():
            shadow_price = row[1][0]
            forecast_shadow_price = row[1][1]
            
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

    # Calculate confusion matrix metrics
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

    # Calculate total confusion matrix (matching Test_Jacob.py implementation)
    confusion_metrix["Total"] = {
        "FP": FP_total / (FP_total + TP_total),
        "FN": FN_total / (FN_total + TP_total),
        "precision": TP_total / (TP_total + FP_total),
        "recall": TP_total / (TP_total + FN_total),
        "F1": 2 * TP_total / (2 * TP_total + FP_total + FN_total),
    }

    print(f"TP_total: {TP_total}, FP_total: {FP_total}, FN_total: {FN_total}")
    print(confusion_metrix["Total"])

    return all_top_contingencies, confusion_metrix, collected_results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """Main execution block."""
    try:
        # Initialize timestamp collections
        print("Collecting as_of timestamps...")
        as_ofs, as_ofs_last_year = collect_as_ofs()
        
        # Select subset for analysis  
        as_ofs_actual = as_ofs_last_year[:Num_of_days_to_look_back]
        as_ofs_actual = [
            AsOf(
                as_of_str=as_of.as_of_str,
                as_of=date_utils.localized_from_isoformat(as_of.as_of_str),
            )
            for as_of in as_ofs_actual
        ]
        
        # Execute data collection
        print("Collecting actual and forecast data...")
        contents_actual, contents_forecast = collect_actual_and_forecast_data()

        # Save collected data
        print("Saving collected data...")
        with open(f"bc_{run.upper()}_actual_as_of_test.pkl", "wb") as f:
            pickle.dump(contents_actual, f)
        
        # Run the main analysis
        print("Running analysis...")
        all_top_contingencies, confusion_metrix, collected_results = run_analysis()

        all_top_contingencies['TP'] = (all_top_contingencies['shadow_price'] > 0) & (all_top_contingencies['forecast_shadow_price'] > 0)
        all_top_contingencies['FP'] = (all_top_contingencies['shadow_price'] == 0) & (all_top_contingencies['forecast_shadow_price'] > 0)
        all_top_contingencies['FN'] = (all_top_contingencies['shadow_price'] > 0) & (all_top_contingencies['forecast_shadow_price'] == 0)

        # Calculate total confusion matrix per monitored_uid and contingency_uid
        confusion_matrix_per_uid = {}
        
        # Group by branch (monitored_uid) or by individual constraints (monitored_uid, contingency_uid)
        if GROUP_BY_BRANCH:
            # Group by monitored_uid only (branch level)
            groupby_columns = ['monitored_uid']
            for monitored_uid, group in all_top_contingencies.groupby(groupby_columns):
                TP = group['TP'].sum()
                FP = group['FP'].sum()
                FN = group['FN'].sum()
                confusion_matrix_per_uid[monitored_uid] = {
                    'TP': TP,
                    'FP': FP,
                    'FN': FN,
                    'precision': TP / (TP + FP) if (TP + FP) > 0 else 0,
                    'recall': TP / (TP + FN) if (TP + FN) > 0 else 0,
                    'F1': 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
                }
        else:
            # Group by (monitored_uid, contingency_uid) - individual constraint level
            groupby_columns = ['monitored_uid', 'contingency_uid']
            for (monitored_uid, contingency_uid), group in all_top_contingencies.groupby(groupby_columns):
                TP = group['TP'].sum()
                FP = group['FP'].sum()
                FN = group['FN'].sum()
                confusion_matrix_per_uid[(monitored_uid, contingency_uid)] = {
                    'TP': TP,
                    'FP': FP,
                    'FN': FN,
                    'precision': TP / (TP + FP) if (TP + FP) > 0 else 0,
                    'recall': TP / (TP + FN) if (TP + FN) > 0 else 0,
                    'F1': 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
                }   

        today_date = datetime.datetime.today().strftime("%Y-%m-%d")

        # save all_top_contingencies in a csv file
        all_top_contingencies.to_csv(f"all_top_contingencies_{run.upper()}_{today_date}.csv")

        # create a pd.DataFrame from confusion_matrix_per_uid and save as csv
        df_confusion_matrix_per_uid = pd.DataFrame.from_dict(confusion_matrix_per_uid, orient='index')
        
        # Set filename based on grouping method
        grouping_suffix = "per_branch" if GROUP_BY_BRANCH else "per_constraint"
        df_confusion_matrix_per_uid.to_csv(f"confusion_matrix_{grouping_suffix}_{run.upper()}_{today_date}.csv")

        # Save results
        # with open(f"all_top_contingencies_{run.upper()}_{today_date}.pkl", "wb") as f:
        #     pickle.dump(all_top_contingencies, f)
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise
