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

# from sklearn.metrics import r2_score

SAVE_DATA = False
USE_PERCENTILE = False

# ISO = "spp"
ISO = "miso"
# ISO = "ercot"

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

    # Create the splines
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

    # Define the function to find the intersection for each row
    def find_intersection(x, row):
        result = row["spline1"].apply(lambda f: f(x)) - row["spline2"].apply(
            lambda f: f(x)
        )

        return result

    def find_value(x, row):
        # return row["spline1"].apply(lambda f: f([x]))
        return row["spline1"].apply(lambda f: f(x.item()))

    # Apply the function to each row to find the intersection points
    # df_hr["intersection_x"] = df_hr.apply(
    #     lambda row: opt.brentq(find_intersection, 50, 100, args=(row,)),
    #     axis=1,
    # )
    df_hr["intersection_x"] = df_hr.apply(
        lambda row: (
            opt.brentq(find_intersection, 50, 100, args=(row,))
            if np.sign(find_intersection(50, row).item())
            != np.sign(find_intersection(100, row).item())
            else np.nan
        ),
        axis=1,
    ).round(2)

    df_hr["intersection_y"] = df_hr.apply(
        lambda row: find_value(row["intersection_x"], row),
        axis=1,
    ).round(2)

    df_hr.loc[df_hr.intersection_x < 0.01, "intersection_y"] = None
    df_hr.loc[df_hr.intersection_x < 0.01, "intersection_x"] = None
    df_hr.loc[df_hr.intersection_y < 0.01, "intersection_x"] = None
    df_hr.loc[df_hr.intersection_y < 0.01, "intersection_y"] = None


# 1) Get binding constraints
df_bc = _get_dfm(
    f"https://api1.marginalunit.com/constraintdb/{ISO}/binding_constraints?start_datetime=2024-05-01T00:00:00-05:00&end_datetime=2025-01-01T00:00:00-05:00&market={MARKET}"
)
FROM_DATETIME = "2024-05-01T00:00:00-05:00"

dfs = []


if SAVE_DATA:
    for ind, row in enumerate(df_bc.itertuples()):

        # if ind > 2:
        #     break

        print(f"{ind}/{len(df_bc)}")

        monitored_uid = row.monitored_uid
        contingency_uid = row.contingency_uid

        print(f"{monitored_uid} flo {contingency_uid}")

        try:
            # Shadow price
            dft = _get_dfm(
                f"https://api1.marginalunit.com/constraintdb/{ISO}/{MARKET}/timeseries?"
                f"monitored_uid={monitored_uid}&contingency_uid={contingency_uid}&"
                f"start_datetime={FROM_DATETIME}&end_datetime=2025-01-01T00:00:00-05:00"
            )
            dft["period"] = pd.to_datetime(dft.period, utc=True).dt.tz_convert(
                ISO_TZ[ISO]
            )
            dft = dft.set_index("period")

            # actual
            df_r = _get_dfm(
                f"https://api1.marginalunit.com/reflow/{COLLECTION}/constraint/flow?"
                f"monitored_uid={monitored_uid}&contingency_uid={contingency_uid}"
            )
            df_r["timestamp"] = pd.to_datetime(df_r.timestamp, utc=True).dt.tz_convert(
                ISO_TZ[ISO]
            )
            df_r = df_r.set_index("timestamp")

            # Forecast
            df_fcst = _get_dfm(
                f"https://api1.marginalunit.com/pr-forecast/{RUN}/constraint/lookahead_timeseries?"
                f"monitored_uid={monitored_uid}&contingency_uid={contingency_uid}&days_prior=1&stream_uid={STREAM_UID}&from_datetime={FROM_DATETIME}"
            )
            df_fcst["timestamp"] = pd.to_datetime(
                df_fcst.timestamp, utc=True
            ).dt.tz_convert(ISO_TZ[ISO])
            df_fcst = df_fcst.set_index("timestamp")

            dfm = pd.merge(
                df_fcst[
                    ["monitored_uid", "contingency_uid", "constraint_flow", "rating"]
                ].rename(columns={"constraint_flow": "fcst_constraint_flow"}),
                df_r,
                left_index=True,
                right_index=True,
                how="left",
            )

            dfm = pd.merge(
                dfm,
                dft[["shadow_price"]],
                left_index=True,
                right_index=True,
                how="left",
            )

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

    # Save the DataFrame to a Parquet file
    if USE_PERCENTILE:
        dff.to_parquet(f"{ISO}_aggregated_data_2024_percentile.parquet", index=True)
    else:
        dff.to_parquet(f"{ISO}_aggregated_data_2024_flow.parquet", index=True)

else:
    # Read the DataFrame from the Parquet file
    if USE_PERCENTILE:
        dff = pd.read_parquet(f"{ISO}_aggregated_data_2024_percentile.parquet")
    else:
        dff = pd.read_parquet(f"{ISO}_aggregated_data_2024_flow.parquet")

    merged_df_hr = pd.DataFrame()

    for m in list(range(5, 11)):

        dff_c = (
            dff[(dff.index >= f"2024-{m}-01") & (dff.index < f"2024-{m+1}-01")]
            .reset_index()
            .copy()
        )
        if len(dff_c) == 0:
            continue

        # if USE_PERCENTILE:  # zzz
        #     dff_copy = dff.copy()
        #     dff["pct_rating"] = dff.pct_rating.rank(pct=True)

        # dff_c = (
        #     dff[(dff.index >= f"2024-{1}-01") & (dff.index < f"2024-{12}-01")]
        #     .reset_index()
        #     .copy()
        # )

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
            continue

        dff_p.columns = pd.MultiIndex.from_tuples(
            [
                ("Median % Rating", "During Not Binding Time"),
                ("Median % Rating", "During Binding Event"),
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

        # df_hr_to_save.to_csv(
        #     f"conditional_monthly_2024_percentile.csv", index=True
        # )  # Save the DataFrame to a Parquet file

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

        # df_hr.to_parquet(
        #     f"conditional_monthly_2024_{m}.parquet", index=True
        # )  # Save the DataFrame to a Parquet file

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

        if USE_PERCENTILE:
            df_hr.to_csv(
                f"conditional_monthly_2024_{ISO}_percentile.csv", index=True
            )  # Save the DataFrame to a CSV file
        else:
            df_hr.to_csv(
                f"conditional_monthly_2024_{ISO}_flow.csv", index=True
            )  # Save the DataFrame to a CSV file

        # for each row of df_hr run the function func()
        dd = find_optimal_predictor(df_hr)

        df_hr.rename(
            columns={"intersection_x": "Predictor", "intersection_y": "F1-Score"},
            inplace=True,
        )

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
        else:
            merged_df_hr = merged_df_hr.merge(
                df_hr, left_index=True, right_index=True, how="outer"
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

        merged_df_hr.to_csv(
            f"Forecast_performace_evaluation_{ISO}_2024.csv", index=True
        )  # Save the DataFrame to a CSV file

if False:

    # Support to the statements:
    # S1:
    # df_cond[~df_cond[("cond_70", "")].isna()][("cond_90", "")].median()

    # S2: this is an example and not the real thing
    # df_percentile = df_cond[~df_cond[("cond_70", "")].isna()].copy() * 1.2
    # df_percentile.rename(columns={"cond_70": "percentile_70"}, inplace=True)
    # df_percentile.rename(columns={"cond_80": "percentile_80"}, inplace=True)
    # df_percentile.rename(columns={"cond_90": "percentile_90"}, inplace=True)
    # df_percentile[7:12]

    # conditional probability (Recall): What is the probability of congestion given that the shadow price is above X% if rating?
    df_hr.iloc[:50]["Median % Rating"].plot(kind="bar", grid=True)
    plt.title(
        f"Conditional Probability of Flow as % of Rating Given Congestion for {ISO} in {m}/2024",
        fontsize=24,
    )
    plt.legend(["Not Binding", "During Binding Event"], fontsize=20)

    plt.figure()
    df_hr.iloc[:500].sort_values(["p-value"], ascending=False)["p-value"].plot(
        kind="bar", grid=True
    )
    plt.title(
        "P-Value of Binding and Non-Binding Separation (small value is good)",
        fontsize=24,
    )

    # conditional probability (Precision): What is the probability of congestion given that the shadow price is above X% if rating?
    plt.figure()
    df_cond.iloc[:50][[("cond_70", ""), ("cond_80", ""), ("cond_90", "")]].plot(
        kind="bar", grid=True
    )
    plt.title(
        f"Conditional Probability of Congestion Given % of Rating, {ISO}, {m}/2024",
        fontsize=24,
    )
    plt.legend(["70% Hit", "80% Hit", "90% Hit"], fontsize=20)

    plt.figure()
    df_hr[df_hr[("shadow_price", "count")] > 1].sort_values(
        [("cond_70")], ascending=False
    )[("cond_70")].plot()
    df_hr[df_hr[("shadow_price", "count")] > 1].sort_values(
        [("cond_80")], ascending=False
    )[("cond_80")].plot()
    df_hr[df_hr[("shadow_price", "count")] > 1].sort_values(
        [("cond_90")], ascending=False
    )[("cond_90")].plot()
    plt.title(
        f"Conditional Probability of Congestion Given % of Rating, {ISO}, 2024",
        fontsize=24,
    )
    plt.legend(["70% Hit", "80% Hit", "90% Hit"], fontsize=20)
    plt.show()

    if False:
        import matplotlib.pyplot as plt

        # Plot fcst_constraint_flow and constraint_flow on the primary y-axis
        fig, ax1 = plt.subplots()

        # Plot fcst_constraint_flow and constraint_flow with solid lines
        dfm[
            [
                "fcst_constraint_flow",
                "constraint_flow",
                "rating",
            ]
        ].plot(ax=ax1, style="-", grid=True)

        # Set labels and legends for primary y-axis
        ax1.set_ylabel("Constraint Flow")
        ax1.legend(["fcst_constraint_flow"], loc="upper left")

        # Create a secondary y-axis for shadow_price
        ax2 = ax1.twinx()
        dfm["shadow_price"].fillna(0.0).plot(ax=ax2, style=":", color="purple")

        # Set label and legend for secondary y-axis
        ax2.set_ylabel("Shadow Price")
        ax2.legend(["shadow_price"], loc="upper right")

        # Show plot
        plt.title(f"{monitored_uid} flo {contingency_uid}")
        plt.show()
