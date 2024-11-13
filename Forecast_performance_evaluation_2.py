import pandas as pd
import matplotlib.pyplot as plt
import os
import io
import requests
import json
from datetime import datetime
import os

# from sklearn.metrics import r2_score

USE_SAVED_DATA = False

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


ISO = "spp"
MARKET = "rt"
COLLECTION = "spp-se"
RUN = "spp"
STREAM_UID = "load_100_wind_100_sol_100_ng_100_iir"

ISO_TZ = {"spp": "US/Central"}
# 1) Get binding constraints
df_bc = _get_dfm(
    f"https://api1.marginalunit.com/constraintdb/{ISO}/binding_constraints?start_datetime=2024-05-01T00:00:00-05:00&end_datetime=2025-01-01T00:00:00-05:00&market={MARKET}"
)
FROM_DATETIME = "2024-05-01T00:00:00-05:00"

dfs = []

if USE_SAVED_DATA:
    for ind, row in enumerate(df_bc.itertuples()):

        # if ind > 1:
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

            dfs.append(dfm)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Issue: {str(e)}")

    dff = pd.concat(dfs)
    dff = dff.dropna(subset=["fcst_constraint_flow", "rating"])
    dff["pct_rating"] = dff.fcst_constraint_flow / dff.rating
    dff["70_hit"] = dff.pct_rating >= 0.7
    dff["80_hit"] = dff.pct_rating >= 0.8
    dff["90_hit"] = dff.pct_rating >= 0.9

    # dff = pd.concat(dfs)
    dff["is_binding"] = dff.shadow_price >= 50

    # Save the DataFrame to a Parquet file
    dff.to_parquet(f"{ISO}_aggregated_data_2024.parquet", index=True)

else:
    # Read the DataFrame from the Parquet file
    dff = pd.read_parquet(f"{ISO}_aggregated_data_2024.parquet")

for m in list(range(5, 11)):

    # ("MIDWAY5 11303 A LN,MRYVL_SJ,161.0,MIDWAY5,161.0","LN COOPER - ST_JOE  345KV NPPD")

    dff_c = (
        dff[(dff.index >= f"2024-{m}-01") & (dff.index < f"2024-{m+1}-01")]
        .reset_index()
        .copy()
    )

    dff_p = dff_c.pivot_table(
        index=["monitored_uid", "contingency_uid"],
        columns="is_binding",
        values="pct_rating",
        aggfunc="median",
    ).sort_values(True, ascending=False)

    dff_p_std = dff_c.pivot_table(
        index=["monitored_uid", "contingency_uid"],
        columns="is_binding",
        values="pct_rating",
        aggfunc="std",
    ).sort_values(True, ascending=False)

    dff_p.columns = pd.MultiIndex.from_tuples(
        [("median_pct_rating", "not_binding"), ("median_pct_rating", "binding")]
    )
    dff_p_std.columns = pd.MultiIndex.from_tuples(
        [("median_pct_rating_std", "not_binding"), ("median_pct_rating_std", "binding")]
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
    )
    df_hr = df_hr.rename(columns={"mean": "% Hours Hit"})
    df_hr = df_hr.rename(columns={"sum": "num"})
    df_hr = pd.merge(df_hr, dff_p, left_index=True, right_index=True, how="left")
    df_hr = pd.merge(df_hr, dff_p_std, left_index=True, right_index=True, how="left")
    df_hr["T-test"] = (
        df_hr[("median_pct_rating", "binding")]
        - df_hr[("median_pct_rating", "not_binding")]
    ) / df_hr[("median_pct_rating_std", "binding")]

    # calculate the Bayesian nominator (probability of observed hit):
    df_hr["num_observed_hit"] = df_hr[("shadow_price", "count")]
    df_hr.drop(("shadow_price", "count"), axis=1, inplace=True)

    # convert df_hr['T-test'] to p-value
    import scipy.stats as stats

    # Calculate the degrees of freedom (df)
    # This is a simplified example; you may need to adjust based on your specific data
    degrees_of_freedom = df_hr["num_observed_hit"] - 1

    # Convert T-test to p-value
    df_hr["p-value"] = 2 * (1 - stats.t.cdf(abs(df_hr["T-test"]), degrees_of_freedom))
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
    )
    df_hr_general = df_hr_general.rename(columns={"mean": "% Hours Hit"})
    df_hr_general = df_hr_general.rename(columns={"sum": "num"})
    df_hr_general = df_hr_general.rename(columns={"70_hit": "70_hit_general"})
    df_hr_general = df_hr_general.rename(columns={"80_hit": "80_hit_general"})
    df_hr_general = df_hr_general.rename(columns={"90_hit": "90_hit_general"})
    df_hr = pd.merge(
        df_hr, df_hr_general, left_index=True, right_index=True, how="left"
    )

    df_hr["cond_70"] = (
        df_hr[("70_hit", "% Hours Hit")]
        * (df_hr[("num_observed_hit", "")] / df_hr[("num_hours", "size")])
        / df_hr[("70_hit_general", "% Hours Hit")]
    )  # P(A|B) = P(A and B) / P(B)
    df_hr["cond_80"] = (
        df_hr[("80_hit", "% Hours Hit")]
        * (df_hr[("num_observed_hit", "")] / df_hr[("num_hours", "size")])
        / df_hr[("80_hit_general", "% Hours Hit")]
    )  # P(A|B) = P(A and B) / P(B)
    df_hr["cond_90"] = (
        df_hr[("90_hit", "% Hours Hit")]
        * (df_hr[("num_observed_hit", "")] / df_hr[("num_hours", "size")])
        / df_hr[("90_hit_general", "% Hours Hit")]
    )  # P(A|B) = P(A and B) / P(B)

    # Separate table:
    df_cond = (
        df_hr[[("cond_70", ""), ("cond_80", ""), ("cond_90", "")]]
        .copy()
        .sort_values(("cond_70", ""), ascending=False)
    )

    df_hr.to_parquet(
        f"conditional_monthly_2024_{m}.parquet", index=True
    )  # Save the DataFrame to a Parquet file


# conditional probability (Recall): What is the probability of congestion given that the shadow price is above X% if rating?
df_hr.iloc[:50]["median_pct_rating"].plot(kind="bar", grid=True)
plt.title(
    f"Conditional Probability of Flow as % of Rating Given Congestion for {ISO} in {m}/2024",
    fontsize=24,
)
plt.legend(["Not Binding", "Binding"], fontsize=20)

plt.figure()
df_hr.iloc[:500].sort_values(["p-value"], ascending=False)["p-value"].plot(
    kind="bar", grid=True
)
plt.title(
    "P-Value of Binding and Non-Binding Separation (small value is good)", fontsize=24
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
df_hr[df_hr["num_observed_hit"] > 1].sort_values([("cond_70")], ascending=False)[
    ("cond_70")
].plot()
df_hr[df_hr["num_observed_hit"] > 1].sort_values([("cond_80")], ascending=False)[
    ("cond_80")
].plot()
df_hr[df_hr["num_observed_hit"] > 1].sort_values([("cond_90")], ascending=False)[
    ("cond_90")
].plot()
plt.title(
    f"Conditional Probability of Congestion Given % of Rating, {ISO}, {m}/2024",
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
