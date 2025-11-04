import pandas as pd
import matplotlib.pyplot as plt
import os
import io
import requests
import json
from datetime import datetime
import os
import numpy as np
import scipy.interpolate as interp
import scipy.optimize as opt
import time

plt.ion()

from placebo.utils import snowflake_utils

# from sklearn.metrics import r2_score

SAVE_DATA = False
USE_PERCENTILE = False
DELETE_CONTINGENCY = False

FROM_DATETIME = "2025-10-10T00:00:00-05:00"
TO_DATETIME = "2025-10-30T00:00:00-05:00"

# ISO = "spp"
# ISO = "miso"
# ISO = "ercot"

# ISO = "ercot_prt_crcl"
# ISO = "ercot"
# ISO_name = "ercot"

# ISO = "miso_1202"
# ISO = "miso"
# ISO_name = "miso"
# ISO = "pjm"
# ISO_name = "pjm"

# ISO = "spp"
# ISO_name = "spp"
ISO = "miso"
ISO_name = "miso"


if ISO == "miso":
    # schema = "miso_20241023T1104Z"   #old Miso
    schema = "miso_20241126T0106Z"
elif ISO == "ercot":
    schema = "ercot_20240909T1130Z"
    # schema = "ercot_20240909T1135Z"
elif ISO == "spp":
    schema = "spp_20241108T0106Z"
elif ISO == "pjm":
    schema = "pjm_20250930T1002Z"

plt.rcParams["figure.figsize"] = (15, 7)


def _get_auth():
    return tuple(os.environ["SELF"].split(":"))


AUTH = _get_auth()


def _get_dfm(url):
    resp = requests.get(url, auth=AUTH)

    if resp.status_code != 200:
        print(resp.text)
        resp.raise_for_status()

    dfm = pd.read_csv(io.StringIO(resp.text))

    return dfm


MARKET_DATA = {
    "spp": {
        "MARKET": "rt",
        "COLLECTION": "spp-se",
        "RUN": "spp",
        "STREAM_UID": "load_100_wind_100_sol_100_ng_100_iir",
    },
    "miso": {
        "MARKET": "rt",
        "COLLECTION": "miso-se",
        "RUN": "miso",
        "STREAM_UID": "load_100_wind_100_sol_100_ng_100_iir",
    },
    "ercot": {
        "MARKET": "rt",
        "COLLECTION": "ercot-rt-se",
        "RUN": "ercot",
        "STREAM_UID": "load_100_wind_100_sol_100_ng_100_iir",
    },
    "ercot_prt_crcl": {
        "MARKET": "rt",
        "COLLECTION": "ercot-rt-se",
        "RUN": "ercot_prt_crcl",
        "STREAM_UID": "load_100_wind_100_sol_100_ng_100_iir",
    },
    "miso_1202": {
        "MARKET": "rt",
        "COLLECTION": "miso-se",
        "RUN": "miso_1202",
        "STREAM_UID": "load_100_wind_100_sol_100_ng_100_iir",
    },
    "pjm": {
        "MARKET": "rt",
        "COLLECTION": "miso-se.xt",
        "RUN": "pjm",
        "STREAM_UID": "load_100_wind_100_sol_100_ng_100_iir",
    },

}


MARKET = MARKET_DATA[ISO]["MARKET"]
COLLECTION = MARKET_DATA[ISO]["COLLECTION"]
RUN = MARKET_DATA[ISO]["RUN"]
STREAM_UID = MARKET_DATA[ISO]["STREAM_UID"]

# ISO = "spp"
# MARKET = "rt"
# COLLECTION = "spp-se"
# RUN = "spp"
# STREAM_UID = "load_100_wind_100_sol_100_ng_100_iir"

ISO_TZ = {
    "spp": "US/Central",
    "miso": "Etc/GMT+5",
    "ercot": "US/Central",
    "pjm": "US/Eastern",
}


def find_optimal_predictor(df_hr):

    # Create the splines# Create the splines

    df_hr["spline1"] = df_hr.apply(
        lambda x: interp.UnivariateSpline(
            np.array([50, 70, 80, 90, 100]),
            np.array(
                [0, x[("cond_70", "")], x[("cond_80", "")], x[("cond_90", "")], 1]
            ),
            k=1,
            s=0,
        ),
        axis=1,
    )
    df_hr["spline2"] = df_hr.apply(
        lambda x: interp.UnivariateSpline(
            np.array([50, 70, 80, 90, 100]),
            np.array(
                [
                    x[("Median % Rating", "During Binding Event")],
                    x[("70_hit", "% Hours Hit")],
                    x[("80_hit", "% Hours Hit")],
                    x[("90_hit", "% Hours Hit")],
                    0,
                ]
            ),
            k=1,
            s=0,
        ),
        axis=1,
    )

    if False:
        # FOR SPP and MISO: Define the function to find the intersection for each row
        def find_intersection(x, row):
            result = row["spline1"].apply(lambda f: f(x)) - row["spline2"].apply(
                lambda f: f(x)
            )

            return result

    else:

        def find_intersection(x, row):
            return row["spline1"].item()(x) - row["spline2"].item()(x)

    # def find_value(x, row):
    #     # return row["spline1"].apply(lambda f: f([x]))
    #     return row["spline1"].apply(lambda f: f(x.item()))

    def find_value(x, row):
        return row["spline1"].item()(x)[0]

    # Apply the function to each row to find the intersection points
    # df_hr["intersection_x"] = df_hr.apply(
    #     lambda row: opt.brentq(find_intersection, 50, 100, args=(row,)),
    #     axis=1,
    # )

    if False:
        # the following works for SPP and MISO
        df_hr["intersection_x"] = df_hr.apply(
            lambda row: (
                opt.brentq(find_intersection, 50, 100, args=(row,))
                # opt.brentq(find_intersection, 0, 100, args=(row,))
                if np.sign(find_intersection(50, row).item())
                != np.sign(find_intersection(100, row).item())
                else np.nan
            ),
            axis=1,
        ).round(2)
    else:
        # Apply the function to each row to find the intersection points
        # df_hr["intersection_x"] = df_hr.apply(
        #     lambda row: (
        #         opt.brentq(find_intersection, 50, 100, args=(row,))
        #         if np.sign(find_intersection(50, row))
        #         != np.sign(find_intersection(100, row))
        #         else np.nan
        #     ),
        #     axis=1,
        # ).round(2)

        df_hr["intersection_x"] = df_hr.apply(
            lambda row: (
                (
                    opt.brentq(find_intersection, 50, 100, args=(row,))
                    if np.sign(find_intersection(50, row)).item()
                    != np.sign(find_intersection(100, row)).item()
                    else np.nan
                )
                if (not np.isnan(row["spline1"].item()(50)))
                and (not np.isnan(row["spline1"].item()(100)))
                else np.nan
            ),
            axis=1,
        ).round(2)

        def safe_intersection_y(row):
            try:
                intersection_x = row["intersection_x"]
                # Handle case where intersection_x might be a Series or scalar
                if hasattr(intersection_x, 'iloc'):
                    intersection_x = intersection_x.iloc[0]
                
                if pd.notna(intersection_x):
                    return find_value(intersection_x, row)
                else:
                    return np.nan
            except:
                return np.nan
        
        df_hr["intersection_y"] = df_hr.apply(safe_intersection_y, axis=1).round(2)
    
    # Set both intersection_x and intersection_y to None when either is NaN
    df_hr.loc[df_hr.intersection_x.isna() | df_hr.intersection_y.isna(), "intersection_x"] = None
    df_hr.loc[df_hr.intersection_x.isna() | df_hr.intersection_y.isna(), "intersection_y"] = None
    
    df_hr.loc[df_hr.intersection_x < 0.01, "intersection_y"] = None
    df_hr.loc[df_hr.intersection_x < 0.01, "intersection_x"] = None
    df_hr.loc[df_hr.intersection_y < 0.01, "intersection_x"] = None
    df_hr.loc[df_hr.intersection_y < 0.01, "intersection_y"] = None


# # 1) Get binding constraints
# df_bc = _get_dfm(
#     f"https://api1.marginalunit.com/constraintdb/{ISO_name}/binding_constraints?start_datetime=2024-01-01T00:00:00-05:00&end_datetime=2025-03-01T00:00:00-05:00&market={MARKET}"
# )


dfs = []


# contingency_uid
def analyze_bindings(df):

    df["flow_rating_ratio"] = df["fcst_constraint_flow"] / df["rating"]

    # Group by 'monitored_uid' and 'contingency_uid'
    grouped = df.groupby(["monitored_uid", "contingency_uid"])

    # Calculate the sum of 'is_binding' for each group
    num_observed_bindings = (
        grouped["is_binding"].sum().reset_index(name="num_observed_bindings")
    )

    # Merge the num_observed_bindings back to the original DataFrame
    df = df.merge(num_observed_bindings, on=["monitored_uid", "contingency_uid"])

    df = df[df["num_observed_bindings"] >= 0].copy()

    grouped = df.groupby(["monitored_uid", "contingency_uid"])

    def process_group(group):
        group = group.sort_values(by="flow_rating_ratio", ascending=False)
        group["Precision"] = group.apply(
            lambda row: (
                group[
                    (group["is_binding"] == True)
                    & (group["flow_rating_ratio"] >= row["flow_rating_ratio"])
                ].shape[0]
                / group[group["flow_rating_ratio"] >= row["flow_rating_ratio"]].shape[0]
                if row["is_binding"]
                else 0
            ),
            axis=1,
        )
        group["Recall"] = group.apply(
            lambda row: (
                group[
                    (group["is_binding"] == True)
                    & (group["flow_rating_ratio"] >= row["flow_rating_ratio"])
                ].shape[0]
                / row["num_observed_bindings"]
                if row["is_binding"]
                else 0
            ),
            axis=1,
        )
        # find the candidate for confidence for each row. It is the min of the max between FP and FN
        group["confidence"] = group[["Precision", "Recall"]].min(axis=1)

        # find the flow_rating_ratio where the confidence is the highest
        group["Predictor"] = group.loc[group["confidence"].idxmax()][
            "flow_rating_ratio"
        ]

        # find the flow_rating_ratio where the Precision is the highest
        group["confidence"] = group["confidence"].max()

        res = group[
            [
                "monitored_uid",
                "contingency_uid",
                "Predictor",
                "confidence",
                # "num_observed_bindings",
            ]
        ].drop_duplicates()

        return res

    result = grouped.apply(process_group).reset_index(drop=True)

    return result


# Function to convert 'case' to 'str_timestamp'
def convert_case_to_timestamp(case):
    if ISO in ["miso", "pjm"]:
        date_str = case.split("_")[2]
        time_str = case.split("_")[2][-4:-2]
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str}:00:00"
    elif ISO == "ercot":
        date_str = case.split("_")[3]
        time_str = case.split("_")[4][-2:]
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str}:00:00"


# contingency_uid
def analyze_bindings_OLD(df):

    df["flow_rating_ratio"] = df["fcst_constraint_flow"] / df["rating"]

    # Group by 'monitored_uid' and 'contingency_uid'
    grouped = df.groupby(["monitored_uid", "contingency_uid"])

    # Calculate the sum of 'is_binding' for each group
    num_observed_bindings = (
        grouped["is_binding"].sum().reset_index(name="num_observed_bindings")
    )

    # Merge the num_observed_bindings back to the original DataFrame
    df = df.merge(num_observed_bindings, on=["monitored_uid", "contingency_uid"])

    df = df[df["num_observed_bindings"] > 2].copy()

    grouped = df.groupby(["monitored_uid", "contingency_uid"])

    # Find the 'num_observed_bindings'-th largest value in 'flow_rating_ratio' for each group
    def nth_largest(group):
        n = group["num_observed_bindings"].iloc[0]
        if group.contingency_uid.any() == "DBIGKEN5":
            print(18)
        if n == 0:
            return None
        if len(group) >= n:
            return group.nlargest(n, "flow_rating_ratio").iloc[-1]["flow_rating_ratio"]
        else:
            return None

    nth_largest_value = grouped.apply(nth_largest).reset_index(name="nth_largest_value")
    df = df.merge(nth_largest_value, on=["monitored_uid", "contingency_uid"])
    grouped = df.groupby(["monitored_uid", "contingency_uid"])

    def sum_first_n_bindings(group):
        # Sort the group by 'flow_rating_ratio'
        sorted_group = group.sort_values(by="flow_rating_ratio", ascending=False)
        # Get the number of observed bindings
        n = sorted_group["num_observed_bindings"].iloc[0]
        # Sum the first 'n' entries in 'is_binding'
        return sorted_group.head(n)["is_binding"].sum()

    # Find the sum of the first 'num_observed_bindings' entries in 'is_binding' for each pair
    # sum_first_n_bindings = grouped.apply(
    #     lambda x: x.sort_values(by="flow_rating_ratio")
    #     .head(x["num_observed_bindings"].iloc[0])["is_binding"]
    #     .sum()
    # ).reset_index(name="sum_first_n_bindings")
    sum_first_n_bindings = grouped.apply(sum_first_n_bindings).reset_index(
        name="sum_first_n_bindings"
    )

    # Merge the sum_first_n_bindings back to the original DataFrame
    df = df.merge(sum_first_n_bindings, on=["monitored_uid", "contingency_uid"])

    grouped = df.groupby(["monitored_uid", "contingency_uid"])

    # group df by 'monitored_uid' and 'contingency_uid', and for each group add a column nth_largest_value that was calculated above and another column which is sum_first_n_bindings / num_observed_bindings

    df["confidence"] = df["sum_first_n_bindings"] / df["num_observed_bindings"]

    grouped = (
        df.groupby(["monitored_uid", "contingency_uid"])
        .agg(
            {
                "confidence": "first",
                "nth_largest_value": "first",
                "sum_first_n_bindings": "size",
                "num_observed_bindings": "first",
            }
        )
        .reset_index()
    )

    # rename the column nth_largest_value to 'Predictor'
    grouped = grouped.rename(columns={"nth_largest_value": "Predictor"})
    grouped = grouped.rename(columns={"sum_first_n_bindings": "num_of_hours"})

    grouped.loc[grouped["confidence"] == 0, "confidence"] = (
        grouped["num_observed_bindings"] / grouped["num_of_hours"]
    )
    grouped["confidence"] = grouped["confidence"].round(2)

    return grouped


if SAVE_DATA:

    # 1) Get binding constraints
    # df_bc = _get_dfm(
    #     f"https://api1.marginalunit.com/constraintdb/{ISO_name}/binding_constraints?start_datetime=2024-01-01T00:00:00-05:00&end_datetime=2025-03-01T00:00:00-05:00&market={MARKET}"
    # )
    df_bc = _get_dfm(
        f"https://api1.marginalunit.com/constraintdb/{ISO_name}/binding_constraints?start_datetime={FROM_DATETIME}&end_datetime={TO_DATETIME}&market={MARKET}"
    )
    

    for ind, row in enumerate(df_bc.itertuples()):

        # if ind > 11:
        #     break

        print(f"{ind}/{len(df_bc)}")

        monitored_uid = row.monitored_uid
        contingency_uid = row.contingency_uid

        # if not contingency_uid == "AMI34012":
        #     continue

        print(f"{monitored_uid} flo {contingency_uid}")

        try:
            # actual Shadow price
            # dft = _get_dfm(
            #     f"https://api1.marginalunit.com/constraintdb/{ISO_name}/{MARKET}/timeseries?"
            #     f"monitored_uid={monitored_uid}&contingency_uid={contingency_uid}&"
            #     f"start_datetime={FROM_DATETIME}&end_datetime=2025-03-01T00:00:00-05:00"
            # )
            dft = _get_dfm(
                f"https://api1.marginalunit.com/constraintdb/{ISO_name}/{MARKET}/timeseries?"
                f"monitored_uid={monitored_uid}&contingency_uid={contingency_uid}&"
                f"start_datetime={FROM_DATETIME}&end_datetime={TO_DATETIME}&market={MARKET}"
            )
            

            dft["period"] = pd.to_datetime(dft.period, utc=True).dt.tz_convert(
                ISO_TZ[ISO_name]
            )
            dft = dft.set_index("period")

            # MUSE monitored flow (seems to not be used)
            df_r = _get_dfm(
                f"https://api1.marginalunit.com/reflow/{COLLECTION}/constraint/flow?"
                f"monitored_uid={monitored_uid}&contingency_uid={contingency_uid}"
            )
            df_r["timestamp"] = pd.to_datetime(df_r.timestamp, utc=True).dt.tz_convert(
                ISO_TZ[ISO_name]
            )
            df_r = df_r.set_index("timestamp")

            #### SNOWFLAKE QUERY INSTEAD OF API#####

            # Forecast flow (Constraint Flow, Monitored Flow)
            # https://github.com/enverus-pr/mu-placebo-api/blob/master/placebo_api/db/snowflake/constraint.py#L225-L238: def get_lookahead_timeseries() in mu-placebo-api/placebo-api/db/snowflake/constraint.py
            if ISO == "ercot":
                query = f"""
                SELECT
                ctf.timestamp, ctf.monitored_uid, ctf.contingency_uid,
                ctf.monitored_branch_flow_ct, ctf.constraint_flow,
                ctf.as_of, ctf.stream_uid, ct."rating", ctf.as_of as ref_datetime
                FROM "constraint_flow_performance" ctf
                INNER JOIN "constraint" ct
                ON ct."monitored_uid" = ctf.monitored_uid
                AND ct."contingency_uid" = ctf.contingency_uid

                WHERE monitored_uid = '{monitored_uid}'
                AND contingency_uid = '{contingency_uid}'
                AND days_prior = 1
                AND stream_uid = '{STREAM_UID}'
                AND DATE(AS_OF) >= '{FROM_DATETIME[:10]}'
                AND REF_HOUR = 4
                """

                # cnx_string = f'snowflake://MICHAEL.SIMANTOV@DRILLINGINFO.COM@enverus_pr.us-east-1/"pr-forecast"/"ercot_20240909T1130Z"?privatekeyfile=./rsa_key.p8'
                cnx_string = f'snowflake://MICHAEL.SIMANTOV@DRILLINGINFO.COM@enverus_pr.us-east-1/"pr-forecast"/"{schema}"?privatekeyfile=./rsa_key.p8'
                cnx_params = snowflake_utils.ConnectionParams.from_cnx_string(
                    cnx_string
                )
                cnx = cnx_params.build()
                df_fcst = snowflake_utils.execute_and_fetch(cnx, query)

            ##########
            else:
                # Forecast using API:
                df_fcst = _get_dfm(
                    f"https://api1.marginalunit.com/pr-forecast/{RUN}/constraint/lookahead_timeseries?"
                    f"monitored_uid={monitored_uid}&contingency_uid={contingency_uid}&days_prior=1&stream_uid={STREAM_UID}&from_datetime={FROM_DATETIME}"
                )

            if ISO_name == "spp":
                df_rating_fcst = _get_dfm(
                    f"https://api1.marginalunit.com/misc-data/{RUN}/flowgate_ratings?"
                    f"monitored_uid={monitored_uid}&contingency_uid={contingency_uid}&days_prior=1&stream_uid={STREAM_UID}&from_datetime={FROM_DATETIME}"
                )
                df_rating_fcst.rename(columns={"emergency_rating": "dynamic_rating"})

            elif ISO_name in ["miso", "pjm"]:

                branch_name = monitored_uid.split(",")[0]
                branch_name = branch_name.replace(" ", "%20")
                # df_rating_fcst_example = _get_dfm(https://api1.marginalunit.com/reflow/miso-se/branch?name=08CLOVDL%20BK1_138   #from Steven

                df_rating_fcst = _get_dfm(
                    f"https://api1.marginalunit.com/reflow/miso-se/branch?name={branch_name}&columns=rate_b,case"
                )
                df_rating_fcst.rename(
                    columns={"rate_b": "dynamic_rating"}, inplace=True
                )

            elif ISO_name == "ercot":

                branch_name = monitored_uid.split(",")[0]
                branch_name = branch_name.replace(" ", "%20")
                # df_rating_fcst_example = _get_dfm(https://api1.marginalunit.com/reflow/miso-se/branch?name=08CLOVDL%20BK1_138   #from Steven

                df_rating_fcst = _get_dfm(
                    # f"https://api1.marginalunit.com/reflow/ercot-rt-se/branch?name={branch_name}&columns=rate_b,case"
                    f"https://api1.marginalunit.com/reflow/ercot-dam-se/branch?name={branch_name}&columns=rate_b,case"
                )
                df_rating_fcst.rename(
                    columns={"rate_b": "dynamic_rating"}, inplace=True
                )

            # https://api1.marginalunit.com/misc-data/spp/flowgate_ratings?monitored_uid=OAHE+OAHESULLY23_1+1+LN,OAHE,230.0,SULLYBT,230.0&contingency_uid=WAUE:LELANDO+FTTHOMP:345:3:14
            # https://api1.marginalunit.com/misc-data/spp/flowgate_ratings?monitored_uid=monitored_uid=OAHE OAHESULLY23_1 1 LN,OAHE,230.0,SULLYBT,230.0&contingency_uid=WAUE:LELANDO FTTHOMP:345:3:14

            # sleep for 0.1 seconds
            time.sleep(2)

            df_fcst["timestamp"] = pd.to_datetime(
                df_fcst.timestamp, utc=True
            ).dt.tz_convert(ISO_TZ[ISO_name])

            if ISO_name in ["spp"]:
                df_fcst["str_timestamp"] = df_fcst["timestamp"].apply(
                    lambda x: str(x)[:19]
                )
                merged___ = pd.merge(
                    df_fcst,
                    df_rating_fcst,
                    left_on="str_timestamp",
                    right_on="period",
                    how="left",
                )

                merged___ = merged___.set_index("timestamp")
                dfm = pd.merge(
                    merged___[
                        [
                            "monitored_uid",
                            "contingency_uid",
                            "constraint_flow",
                            "dynamic_rating",
                        ]
                    ].rename(
                        columns={
                            "constraint_flow": "fcst_constraint_flow",
                            "dynamic_rating": "rating",
                        }
                    ),
                    df_r,
                    left_index=True,
                    right_index=True,
                    how="left",
                )
            elif ISO_name in ["miso", "pjm", "ercot"]:
                df_fcst["str_timestamp"] = df_fcst["timestamp"].apply(
                    lambda x: str(x)[:19]
                )

                df_rating_fcst["str_timestamp"] = df_rating_fcst["case"].apply(
                    convert_case_to_timestamp
                )

                merged___ = pd.merge(
                    df_fcst,
                    df_rating_fcst,
                    left_on="str_timestamp",
                    right_on="str_timestamp",
                    how="left",
                )
                # merged___["dynamic_rating"].fillna(method="ffill", inplace=True)   #will yield warning in the future
                merged___["dynamic_rating"] = merged___["dynamic_rating"].ffill()

                merged___ = merged___.set_index("timestamp")
                dfm = pd.merge(
                    merged___[
                        [
                            "monitored_uid",
                            "contingency_uid",
                            "constraint_flow",
                            "dynamic_rating",
                        ]
                    ].rename(
                        columns={
                            "constraint_flow": "fcst_constraint_flow",
                            "dynamic_rating": "rating",
                        }
                    ),
                    df_r,
                    left_index=True,
                    right_index=True,
                    how="left",
                )

            else:
                # not supposed to be here!
                zzz
                # df_fcst = df_fcst.set_index("timestamp")

                # dfm = pd.merge(
                #     df_fcst[
                #         [
                #             "monitored_uid",
                #             "contingency_uid",
                #             "constraint_flow",
                #             "rating",
                #         ]
                #     ].rename(columns={"constraint_flow": "fcst_constraint_flow"}),
                #     df_r,
                #     left_index=True,
                #     right_index=True,
                #     how="left",
                # )

            dfm = pd.merge(
                dfm,
                dft[["shadow_price"]],
                left_index=True,
                right_index=True,
                how="left",
            )

            # remove the rows with missing values in the columns 'fcst_constraint_flow', 'rating', 'shadow_price'
            # dfm = dfm.dropna(subset=["constraint_flow", "rating", "shadow_price"])
            dfm = dfm.dropna(subset=["rating"])

            if USE_PERCENTILE:
                dfm["pct_rating"] = dfm.fcst_constraint_flow.rank(pct=True)

            dfs.append(dfm)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Issue: {str(e)}")

    dff = pd.concat(dfs)
    dff = dff.dropna(subset=["fcst_constraint_flow", "rating"])
    if not USE_PERCENTILE:
        dff["pct_rating"] = dff.fcst_constraint_flow / dff.rating

    dff["70_hit"] = dff.pct_rating >= 0.7
    dff["80_hit"] = dff.pct_rating >= 0.8
    dff["90_hit"] = dff.pct_rating >= 0.9

    # dff = pd.concat(dfs)
    dff["is_binding"] = dff.shadow_price >= 50

    today_date = datetime.today().strftime("%Y-%m-%d")
    # Save the DataFrame to a Parquet file
    if USE_PERCENTILE:
        dff.to_parquet(f"{ISO}_aggregated_data_2024_percentile.parquet", index=True)
    else:
        dff.to_parquet(f"{ISO}_aggregated_data_{today_date}_flow.parquet", index=True)
        # dff.to_parquet(f"{ISO}_aggregated_data_2024_flow.parquet", index=True)

else:
    # Read the DataFrame from the Parquet file
    today_date = datetime.today().strftime("%Y-%m-%d")
    # today_date = "2025-10-16"
    if USE_PERCENTILE:
        dff = pd.read_parquet(f"{ISO}_aggregated_data_2024_percentile.parquet")
    else:
        dff = pd.read_parquet(f"{ISO}_aggregated_data_{today_date}_flow.parquet")
        # dff = pd.read_parquet(f"{ISO}_aggregated_data_2025-02-25_flow.parquet")

    merged_df_hr = pd.DataFrame()
    merged_df_all_data = pd.DataFrame()

    if True:
        # for m in list(range(5, 12)):

        #     dff_c = (
        #         dff[(dff.index >= f"2024-{m}-01") & (dff.index < f"2024-{m+1}-01")]
        #         .reset_index()
        #         .copy()
        #     )
        #     if len(dff_c) == 0:
        #         continue

        # if USE_PERCENTILE:  # zzz
        #     dff_copy = dff.copy()
        #     dff["pct_rating"] = dff.pct_rating.rank(pct=True)

        # dff_c = (
        #     dff[(dff.index >= f"2024-03-01") & (dff.index < f"2024-{12}-01")]
        #     .reset_index()
        #     .copy()
        # )

        dff_c = dff.reset_index().copy()
        if DELETE_CONTINGENCY:
            dff_c.contingency_uid = 'NULL_CONTINGENCY'

        # Michael added Jan 16
        Predictor_Confidence = analyze_bindings(dff_c)

        dff_p = (
            dff_c.pivot_table(
                index=["monitored_uid", "contingency_uid"],
                columns="is_binding",
                values="pct_rating",
                aggfunc="median",
            )
            .sort_values(True, ascending=False)
            .round(2)
        )

        # merge Predictor_Confidence into dff_p
        # dff_p = dff_p.merge(
        #     Predictor_Confidence, on=["monitored_uid", "contingency_uid"], how="left"
        # )

        try:
            dff_p_std = (
                dff_c.pivot_table(
                    index=["monitored_uid", "contingency_uid"],
                    columns="is_binding",
                    values="pct_rating",
                    aggfunc="std",
                )
                .sort_values(True, ascending=False)
                .round(2)
            )
        except:
            print(18)

        dff_p.columns = pd.MultiIndex.from_tuples(
            [
                ("Median % Rating", "During Not Binding Time"),
                ("Median % Rating", "During Binding Event"),
                # "Predictor",
                # "confidence",
            ]
        )
        dff_p_std.columns = pd.MultiIndex.from_tuples(
            [
                ("Median % Rating_std", "During Not Binding Time"),
                ("Median % Rating_std", "During Binding Event"),
            ]
        )

        df_hr = (
            dff_c[dff_c.shadow_price >= 50]
            .groupby(["monitored_uid", "contingency_uid"])
            .agg(
                {
                    "70_hit": ["mean", "sum"],
                    "80_hit": ["mean", "sum"],
                    "90_hit": ["mean", "sum"],
                    "shadow_price": ["sum", "count"],
                }
            )
            .sort_values(("70_hit", "sum"), ascending=False)
        ).round(2)
        df_hr = df_hr.rename(columns={"mean": "% Hours Hit"})
        df_hr = df_hr.rename(columns={"sum": "num"})
        df_hr = pd.merge(df_hr, dff_p, left_index=True, right_index=True, how="left")
        df_hr = pd.merge(
            df_hr, dff_p_std, left_index=True, right_index=True, how="left"
        )

        if True:  # printing for debugging
            df_hr[
                (df_hr[("shadow_price", "num")] > 1)
                & (df_hr[("70_hit", "% Hours Hit")] > 0)
            ][("70_hit", "% Hours Hit")].median()
            df_hr[
                (df_hr[("shadow_price", "num")] > 1)
                & (df_hr[("80_hit", "% Hours Hit")] > 0)
            ][("80_hit", "% Hours Hit")].median()
            df_hr[
                (df_hr[("shadow_price", "num")] > 1)
                & (df_hr[("90_hit", "% Hours Hit")] > 0)
            ][("90_hit", "% Hours Hit")].median()

        df_hr["T-test"] = (
            df_hr[("Median % Rating", "During Binding Event")]
            - df_hr[("Median % Rating", "During Not Binding Time")]
        ) / df_hr[("Median % Rating_std", "During Binding Event")]

        df_hr.drop(
            ("Median % Rating_std", "During Not Binding Time"), axis=1, inplace=True
        )
        df_hr.drop(
            ("Median % Rating_std", "During Binding Event"), axis=1, inplace=True
        )

        print(18)

        # calculate the Bayesian nominator (probability of observed hit):
        # df_hr["num_observed_hit"] = df_hr[("shadow_price", "count")]
        # df_hr.drop(("shadow_price", "count"), axis=1, inplace=True)

        # convert df_hr['T-test'] to p-value
        import scipy.stats as stats

        # Calculate the degrees of freedom (df)
        # This is a simplified example; you may need to adjust based on your specific data
        degrees_of_freedom = df_hr[("shadow_price", "count")] - 1

        # Convert T-test to p-value
        df_hr["p-value"] = 2 * (
            1 - stats.t.cdf(abs(df_hr["T-test"]), degrees_of_freedom)
        ).round(2)
        df_hr.drop("T-test", axis=1, inplace=True)

        # calculate the Bayesian denominator (probability of the flow to reach 70's, 80's or 90's of the rating):
        df_hr_general = (
            dff_c.groupby(["monitored_uid", "contingency_uid"])
            .agg(
                {
                    "70_hit": ["mean", "sum"],
                    "80_hit": ["mean", "sum"],
                    "90_hit": ["mean", "sum"],
                    "shadow_price": ["size"],
                }
            )
            .rename(columns={"shadow_price": "num_hours"})
            .sort_values(("70_hit", "sum"), ascending=False)
            .round(2)
        )
        df_hr_general = df_hr_general.rename(columns={"mean": "% Hours Hit"})
        df_hr_general = df_hr_general.rename(columns={"sum": "num"})
        df_hr_general = df_hr_general.rename(columns={"70_hit": "70_hit_all"})
        df_hr_general = df_hr_general.rename(columns={"80_hit": "80_hit_all"})
        df_hr_general = df_hr_general.rename(columns={"90_hit": "90_hit_all"})
        df_hr = pd.merge(
            df_hr, df_hr_general, left_index=True, right_index=True, how="left"
        )

        # P(A|B) = P(A and B) / P(B)
        df_hr["cond_70"] = (
            df_hr[("70_hit", "% Hours Hit")]
            * (df_hr[("shadow_price", "count")] / df_hr[("num_hours", "size")])
            / df_hr[("70_hit_all", "% Hours Hit")]
        ).round(2)
        df_hr["cond_80"] = (
            df_hr[("80_hit", "% Hours Hit")]
            * (df_hr[("shadow_price", "count")] / df_hr[("num_hours", "size")])
            / df_hr[("80_hit_all", "% Hours Hit")]
        ).round(2)
        df_hr["cond_90"] = (
            df_hr[("90_hit", "% Hours Hit")]
            * (df_hr[("shadow_price", "count")] / df_hr[("num_hours", "size")])
            / df_hr[("90_hit_all", "% Hours Hit")]
        ).round(2)

        df_hr_to_save = df_hr.copy()
        df_hr_to_save.rename(columns={"70_hit": "Binding: 70% Hit"}, inplace=True)
        df_hr_to_save.rename(columns={"80_hit": "Binding: 80% Hit"}, inplace=True)
        df_hr_to_save.rename(columns={"90_hit": "Binding: 90% Hit"}, inplace=True)

        df_hr.drop(("70_hit_all", "% Hours Hit"), axis=1, inplace=True)
        df_hr.drop(("80_hit_all", "% Hours Hit"), axis=1, inplace=True)
        df_hr.drop(("90_hit_all", "% Hours Hit"), axis=1, inplace=True)
        df_hr.drop(("70_hit_all", "num"), axis=1, inplace=True)
        df_hr.drop(("80_hit_all", "num"), axis=1, inplace=True)
        df_hr.drop(("90_hit_all", "num"), axis=1, inplace=True)

        # Separate table:
        df_cond = (
            df_hr[[("cond_70", ""), ("cond_80", ""), ("cond_90", "")]]
            .copy()
            .sort_values(("cond_70", ""), ascending=False)
        )

        df_hr[("Shadow Price", "Total")] = df_hr[("shadow_price", "num")]
        df_hr[("Shadow Price", "Num Hours")] = df_hr[("shadow_price", "count")]
        df_hr.drop(("shadow_price", "num"), axis=1, inplace=True)
        df_hr.drop(("shadow_price", "count"), axis=1, inplace=True)
        df_hr.drop(("num_hours", "size"), axis=1, inplace=True)
        df_hr[("cond_70", "")] = df_hr[("cond_70", "")].replace(
            [np.inf, -np.inf], np.nan
        )
        df_hr[("cond_80", "")] = df_hr[("cond_80", "")].replace(
            [np.inf, -np.inf], np.nan
        )
        df_hr[("cond_90", "")] = df_hr[("cond_90", "")].replace(
            [np.inf, -np.inf], np.nan
        )
        df_hr = df_hr.sort_values(["cond_90"], ascending=False)
        df_hr = df_hr.rename(columns={"num": "Count of Binding Hours"})

        # if USE_PERCENTILE:
        #     df_hr.to_csv(
        #         f"conditional_monthly_2024_{ISO}_percentile.csv", index=True
        #     )  # Save the DataFrame to a CSV file
        # else:
        #     df_hr.to_csv(
        #         f"conditional_monthly_2024_{ISO}_flow.csv", index=True
        #     )  # Save the DataFrame to a CSV file

        # for each row of df_hr run the function func()
        dd = find_optimal_predictor(df_hr)

        df_hr.rename(
            columns={"intersection_x": "Predictor", "intersection_y": "F1-Score"},
            inplace=True,
        )

        df_all_data = df_hr.copy()

        df_hr = df_hr[
            [
                ("Median % Rating", "During Not Binding Time"),
                ("Median % Rating", "During Binding Event"),
                ("p-value", ""),
                ("Shadow Price", "Total"),
                ("Shadow Price", "Num Hours"),
                ("Predictor", ""),
                ("F1-Score", ""),
            ]
        ]

        m = "all"
        df_hr.columns = pd.MultiIndex.from_tuples(
            [
                ("Median % Rating", "During Not Binding Time", f"{m}/2024"),
                ("Median % Rating", "During Binding Event", f"{m}/2024"),
                ("p-value", "", f"{m}/2024"),
                ("Shadow Price", "Total", f"{m}/2024"),
                ("Shadow Price", "Num Hours", f"{m}/2024"),
                ("Predictor", "", f"{m}/2024"),
                ("F1-Score", "", f"{m}/2024"),
            ]
        )

        # merged_df_hr = pd.concat([merged_df_hr, df_hr], axis=0)

        # Merge the current df_hr with the merged_df_hr
        if merged_df_hr.empty:
            merged_df_hr = df_hr
            merged_df_all_data = df_all_data
        else:
            merged_df_hr = merged_df_hr.merge(
                df_hr, left_index=True, right_index=True, how="outer"
            )
            merged_df_all_data = merged_df_all_data.merge(
                df_all_data, left_index=True, right_index=True, how="outer"
            )
        # merged_df_hr = merged_df_hr[]

    if not merged_df_hr.empty:

        # add a column to merged_df_hr that is the sum of the 'Shadow Price' ("Shadow Price", "Total") columns over all the months
        merged_df_hr[("Shadow Price", "Total", "Total")] = merged_df_hr[
            ("Shadow Price", "Total")
        ].sum(axis=1)
        merged_df_hr[("Shadow Price", "Num Hours", "Total")] = merged_df_hr[
            ("Shadow Price", "Num Hours")
        ].sum(axis=1)

        merged_df_hr.sort_values(
            ("Shadow Price", "Total", "Total"), ascending=False, inplace=True
        )

        # Sort the columns of the merged DataFrame in alphabetical order
        merged_df_hr = merged_df_hr.sort_index(axis=1)

        Predictor_Confidence.set_index(
            ["monitored_uid", "contingency_uid"], inplace=True
        )

        Predictor_Confidence.columns = pd.MultiIndex.from_tuples(
            [
                ("Statistics", "Predictor", ""),
                ("Statistics", "Confidence", ""),
                # "Predictor",
                # "confidence",
            ]
        )

        # # merge Predictor_Confidence into merged_df_hr
        merged_df_hr = merged_df_hr.merge(
            Predictor_Confidence, left_index=True, right_index=True, how="left"
        )

        # remove unwanted columns
        f1_columns = [col for col in merged_df_hr.columns if col[0] in ["F1-Score", "Predictor"]]
        merged_df_hr = merged_df_hr.drop(columns=f1_columns)
        merged_df_hr.columns = merged_df_hr.columns.droplevel(2)


        today_date = datetime.today().strftime("%Y-%m-%d")
        merged_df_hr.to_csv(
            f"Forecast_performace_evaluation_{ISO}_{today_date}.csv", index=True
        )  # Save the DataFrame to a CSV file
        merged_df_all_data.to_csv(
            f"DEBUGGING_DATA_Forecast_performace_evaluation_{ISO}_{today_date}.csv",
            index=True,
        )  # Save the DataFrame to a CSV file

print(18)
