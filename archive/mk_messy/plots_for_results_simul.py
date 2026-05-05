import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

summary_path = Path(r"C:/Works/Sensors/Sensors/Cooling Summaries")


files = {
    "cluster_1": "cluster_1_representatives_hourly_fromexcel_summer_summary.csv",
    "cluster_2": "cluster_2_representatives_hourly_fromexcel_summer_summary.csv",
    "cluster_3": "cluster_3_representatives_hourly_fromexcel_summer_summary.csv",
    "cluster_4": "cluster_4_representatives_hourly_fromexcel_summer_summary.csv",
    "suburban":  "ARPAV_suburban_summer_summary.csv",
    "rural":     "ARPAV_rural_summer_summary.csv"

}

colors = {
    "cluster_1": "0.2",
    "cluster_2": "0.35",
    "cluster_3": "0.55",
    "cluster_4": "0.75",
    "suburban":  "#7ED957",
    "rural":     "#006400"

}

months = [5, 6, 7, 8, 9, 10]
month_labels = ["May", "Jun", "Jul", "Aug", "Sep", "Oct"]

monthly_values = {name: [] for name in files}

for name, filename in files.items():
    df = pd.read_csv(summary_path / filename, sep=";")
    df["Time"] = pd.to_datetime(df["Time"])

    df["Total cooling load [kW]"] = pd.to_numeric(
        df["Total cooling load [kW]"], errors="coerce"
    ).fillna(0.0)

    df["Month"] = df["Time"].dt.month

    monthly_sum = (
        df.groupby("Month")["Total cooling load [kW]"]
        .sum()
        .reindex(months)
        .fillna(0.0)
        / 1000.0
    )

    monthly_values[name] = monthly_sum.values

x = np.arange(len(months))
width = 0.12
n = len(files)

plt.figure(figsize=(13,6))

for i, (name, values) in enumerate(monthly_values.items()):
    plt.bar(x + i * width, values, width, color=colors[name], label=name)

plt.xticks(x + (n * width) / 2, month_labels, fontsize=12)
plt.ylabel("Load [MWh]", fontsize=13)
plt.title("Monthly Total", fontsize=14)
plt.legend(title="Scenario")

plt.tight_layout()
plt.show()



files_ordered = {
    "cluster_1": "cluster_1_representatives_hourly_fromexcel_summer_summary.csv",
    "cluster_2": "cluster_2_representatives_hourly_fromexcel_summer_summary.csv",
    "cluster_3": "cluster_3_representatives_hourly_fromexcel_summer_summary.csv",
    "cluster_4": "cluster_4_representatives_hourly_fromexcel_summer_summary.csv",
    "suburban": "ARPAV_suburban_summer_summary.csv",
    "rural":    "ARPAV_rural_summer_summary.csv",
}

colors = {
    "cluster_1": "0.2",
    "cluster_2": "0.35",
    "cluster_3": "0.55",
    "cluster_4": "0.75",
    "suburban":  "#7ED957",
    "rural":     "#006400",
}

months = [5, 6, 7, 8, 9, 10]
month_labels = ["May", "Jun", "Jul", "Aug", "Sep", "Oct"]

sens_monthly = {}
lat_monthly = {}

for name, filename in files_ordered.items():
    df = pd.read_csv(summary_path / filename, sep=";")
    df["Time"] = pd.to_datetime(df["Time"])

    sens = pd.to_numeric(df["Total sensible load [kW]"], errors="coerce").fillna(0.0)
    lat = pd.to_numeric(df["Total latent load [kW]"], errors="coerce").fillna(0.0)

    df["Month"] = df["Time"].dt.month

    sens_m = (
        df.groupby("Month")["Total sensible load [kW]"]
        .sum()
        .reindex(months)
        .fillna(0.0)
        / 1000.0
    )
    lat_m = (
        df.groupby("Month")["Total latent load [kW]"]
        .sum()
        .reindex(months)
        .fillna(0.0)
        / 1000.0
    )

    sens_monthly[name] = sens_m.values
    lat_monthly[name] = lat_m.values

x = np.arange(len(months))
width = 0.12
n = len(files_ordered)

plt.figure(figsize=(13, 6))

for i, name in enumerate(files_ordered.keys()):
    lat_vals = lat_monthly[name]
    sens_vals = sens_monthly[name]
    xpos = x + i * width

    plt.bar(xpos, lat_vals, width, color=colors[name], alpha=0.5, label=None)
    plt.bar(xpos, sens_vals, width, bottom=lat_vals, color=colors[name], label=name)

handles, labels = plt.gca().get_legend_handles_labels()
_, idx = np.unique(labels, return_index=True)
handles = [handles[i] for i in idx]
labels = [labels[i] for i in idx]

plt.xticks(x + (n * width) / 2, month_labels, fontsize=12)
plt.ylabel("Load [MWh]", fontsize=13)
plt.legend(handles, labels, title="Scenario")

plt.tight_layout()
plt.show()


files = {
    "cluster_1": "cluster_1_representatives_hourly_fromexcel_summer_summary.csv",
    "cluster_2": "cluster_2_representatives_hourly_fromexcel_summer_summary.csv",
    "cluster_3": "cluster_3_representatives_hourly_fromexcel_summer_summary.csv",
    "cluster_4": "cluster_4_representatives_hourly_fromexcel_summer_summary.csv",
    "suburban":  "ARPAV_suburban_summer_summary.csv",
    "rural":     "ARPAV_rural_summer_summary.csv",
}

peak_days = []

for scenario, fname in files.items():
    df = pd.read_csv(summary_path / fname, sep=";")
    df["Time"] = pd.to_datetime(df["Time"])

    df["Total cooling load [kW]"] = pd.to_numeric(
        df["Total cooling load [kW]"], errors="coerce"
    ).fillna(0.0)

    df["Date"] = df["Time"].dt.date

    daily = (
        df.groupby("Date")["Total cooling load [kW]"]
        .sum()
        .reset_index(name="Daily_Load_kWh")
    )

    peak = daily.loc[daily["Daily_Load_kWh"].idxmax()]

    peak_days.append({
        "Scenario": scenario,
        "Peak day": peak["Date"],
        "Daily Cooling Load [MWh]": peak["Daily_Load_kWh"] / 1000.0
    })

peak_df = pd.DataFrame(peak_days)
print(peak_df)



import matplotlib.dates as mdates


files = {
    "cluster_1": "cluster_1_representatives_hourly_fromexcel_summer_summary.csv",
    "cluster_2": "cluster_2_representatives_hourly_fromexcel_summer_summary.csv",
    "cluster_3": "cluster_3_representatives_hourly_fromexcel_summer_summary.csv",
    "cluster_4": "cluster_4_representatives_hourly_fromexcel_summer_summary.csv",
    "suburban":  "ARPAV_suburban_summer_summary.csv",
    "rural":     "ARPAV_rural_summer_summary.csv",
}

target_day = pd.Timestamp("2005-07-12")

plt.figure(figsize=(12, 6))

for name, fname in files.items():
    df = pd.read_csv(summary_path / fname, sep=";")
    df["Time"] = pd.to_datetime(df["Time"])
    df["ConditioningElectricity [kW]"] = pd.to_numeric(
        df["ConditioningElectricity [kW]"], errors="coerce"
    ).fillna(0.0)

    mask = df["Time"].dt.date == target_day.date()
    sub = df.loc[mask]

    plt.plot(sub["Time"], sub["ConditioningElectricity [kW]"], label=name)

ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlabel("Hour")
plt.ylabel("Conditioning electricity [kW]")
plt.legend(title="Scenario")
plt.tight_layout()
plt.show()