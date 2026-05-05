# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 12:30:48 2025

@author: khajmoh18975
"""

import re
import pandas as pd
from pathlib import Path

FOLDER = Path("./2020Analysis")  
SHEET_NAME = "Foglio3"
RURAL_COL = "T_rural (A station)"
EU_COL = "T_EUReCA"
DATE_COL = None 

files = []
for p in FOLDER.glob("T_urban Heat island*.xlsx"):
    name = p.name
    m = re.search(r"v(\d+)\.xlsx$", name, flags=re.IGNORECASE)
    if m:
        v = int(m.group(1))
    else:
        v = 1
    files.append((v, p))

files = sorted([fp for fp in files if 1 <= fp[0] <= 5], key=lambda x: x[0])

if not files:
    raise FileNotFoundError("No matching Excel files found.")

v1_path = None
for v, p in files:
    if v == 1:
        v1_path = p
        break
if v1_path is None:
    raise FileNotFoundError("Version 1 file not found (the one without 'v' in its name).")

def load_sheet(path, version=None):
    print(path)
    df = pd.read_excel(path, sheet_name=SHEET_NAME)
    # infer datetime column if user left DATE_COL=None
    if DATE_COL is None:
        # assume first column is Datetime
        dt_col = df.columns[0]
    else:
        dt_col = DATE_COL
    df = df.rename(columns={dt_col: "Datetime"})
    # ensure datetime dtype
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce", dayfirst=True)
    # keep only needed columns if present
    cols = ["Datetime"]
    if RURAL_COL in df.columns:
        cols.append(RURAL_COL)
    if EU_COL in df.columns:
        # rename EU column by version if provided
        if version is not None:
            df = df.rename(columns={EU_COL: f"T_EUReCA_v{version}"})
            cols.append(f"T_EUReCA_v{version}")
        else:
            cols.append(EU_COL)
    return df[cols].dropna(subset=["Datetime"])

base = load_sheet(v1_path, version=1)

base = base.sort_values("Datetime").drop_duplicates(subset=["Datetime"], keep="first")

out = base[["Datetime", RURAL_COL]].copy()

for v, p in files:
    dfv = load_sheet(p, version=v)
    eu_col_v = f"T_EUReCA_v{v}"
    if eu_col_v in dfv.columns:
        out = out.merge(dfv[["Datetime", eu_col_v]],
                        on="Datetime", how="outer")

eu_cols = [c for c in out.columns if c.startswith("T_EUReCA_v")]
eu_cols_sorted = sorted(eu_cols, key=lambda x: int(x.split("_v")[-1]))
out = out[["Datetime", RURAL_COL] + eu_cols_sorted].sort_values("Datetime").reset_index(drop=True)

out.to_excel("T_urban_Heat_island_MERGED.xlsx", index=False)
print("Saved: T_urban_Heat_island_MERGED.xlsx")


import matplotlib.pyplot as plt
from pathlib import Path

file_path = Path("T_urban_Heat_island_MERGED.xlsx")
df = pd.read_excel(file_path)

df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce", dayfirst=True)
df = df.dropna(subset=["Datetime"])

df["Month"] = df["Datetime"].dt.month

eu_cols = [c for c in df.columns if c.startswith("T_EUReCA_v")]
rural_col = "T_rural (A station)"

diff_dfs = []
for col in eu_cols:
    tmp = df[["Month", col, rural_col]].copy()
    tmp["Difference"] = tmp[col] - tmp[rural_col]
    tmp["Version"] = col.split("_v")[-1]
    diff_dfs.append(tmp[["Month", "Difference", "Version"]])

diff_all = pd.concat(diff_dfs)

plt.figure(figsize=(12, 6))
month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

positions = []
data = []
labels = []
pos = 1
for v in sorted(diff_all["Version"].unique(), key=int):
    for m in range(1, 13):
        vals = diff_all[(diff_all["Version"] == v) & (diff_all["Month"] == m)]["Difference"]
        if not vals.empty:
            data.append(vals)
            positions.append(pos)
            labels.append(f"{month_labels[m-1]}\n(v{v})")
            pos += 1
    pos += 1  
plt.boxplot(data, positions=positions, patch_artist=True)
plt.xticks(positions, labels, rotation=90)
plt.xlabel("Month and Version")
plt.ylabel("Temperature Difference (°C)")
plt.title("Monthly Distribution of (T_EUReCA_v# - T_rural)")

plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()



import re

DATA_DIR = Path("./cluster_exports")
CLUSTER_PATTERN = "cluster_*_representatives_hourly.*sx"
EPW_FILE = DATA_DIR / "ITA_Venezia-Tessera.161050_IGDG__arpav_rural.epw"

TIME_COL = "time"
DB_COL = "db_temp"

def to_naive_datetime(s):
    dt = pd.to_datetime(s, errors="coerce")
    if hasattr(dt.dt, "tz"):
        dt = dt.dt.tz_localize(None)
    return dt

def load_cluster_excel(path: Path):
    df = pd.read_excel(path)
    df = df[[TIME_COL, DB_COL]].copy()
    df.rename(columns={TIME_COL: "Datetime", DB_COL: "T_cluster"}, inplace=True)
    df["Datetime"] = to_naive_datetime(df["Datetime"])
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime").drop_duplicates("Datetime")
    m = re.search(r"cluster_(\d+)_representative_hourly", path.name, flags=re.IGNORECASE)
    cluster_name = f"cluster_{m.group(1)}" if m else path.stem
    df["cluster_name"] = cluster_name
    return df

def load_epw(path: Path):
    epw = pd.read_csv(path, header=None, skiprows=8)
    epw = epw[[0,1,2,3,4,6]].copy()
    epw.columns = ["Year","Month","Day","Hour","Minute","T_rural"]
    epw["Year"] = 2024
    dt = pd.to_datetime(epw[["Year","Month","Day","Hour","Minute"]], errors="coerce") - pd.to_timedelta(1, unit="h")
    epw["Datetime"] = dt
    return epw[["Datetime","T_rural"]].dropna().sort_values("Datetime")

rural = load_epw(EPW_FILE)

cluster_files = sorted(DATA_DIR.glob(CLUSTER_PATTERN))
cluster_wide = rural.copy()

for f in cluster_files:
    dfc = load_cluster_excel(f)
    wide = dfc.pivot_table(index="Datetime", columns="cluster_name", values="T_cluster", aggfunc="first")
    cluster_wide = cluster_wide.merge(wide, left_on="Datetime", right_index=True, how="outer")

cols = ["Datetime","T_rural"] + sorted([c for c in cluster_wide.columns if c not in ["Datetime","T_rural"]])
cluster_wide = cluster_wide[cols].sort_values("Datetime").reset_index(drop=True)

cluster_wide.set_index("Datetime", inplace=True)
cluster_wide = cluster_wide.asfreq("H")
cluster_wide.reset_index(inplace=True)

cluster_cols = [c for c in cluster_wide.columns if c not in ["Datetime","T_rural"]]
long = cluster_wide.melt(id_vars=["Datetime","T_rural"], value_vars=cluster_cols, var_name="Cluster", value_name="T_cluster")
long["Difference"] = long["T_cluster"] - long["T_rural"]
long["Month"] = long["Datetime"].dt.month
long = long.dropna(subset=["Difference", "Month"])

months = sorted(long["Month"].unique())
clusters = sorted(long["Cluster"].dropna().unique())

data = []
positions = []
labels = []
pos = 1.0
stride = len(clusters) + 1
gap = 1.0

for m in months:
    month_data = long[long["Month"] == m]
    for i, cl in enumerate(clusters):
        vals = month_data[month_data["Cluster"] == cl]["Difference"].dropna()
        data.append(vals.values)
        positions.append(pos + i)
        labels.append(f"{m}-{cl}")
    pos += stride + gap

plt.figure(figsize=(18, 8))
plt.boxplot(data, positions=positions, patch_artist=True, showfliers=False)
plt.xticks(positions, labels, rotation=90)
plt.xlabel("Month-Cluster")
plt.ylabel("ΔT = T_cluster − T_rural (°C)")
plt.title("Monthly Boxplots of Cluster–Rural Temperature Differences (ARPAV Year=2024)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

cluster_wide.to_excel("CLUSTERS_vs_RURAL_merged_hourly.xlsx", index=False)
print("Saved: CLUSTERS_vs_RURAL_merged_hourly.xlsx")



import re

DATA_DIR = Path("./cluster_exports")
CLUSTER_PATTERN = "cluster_*_representative_summer.*sx"
EPW_FILE = DATA_DIR / "ITA_Venezia-Tessera.161050_IGDG__arpav_rural.epw"

TIME_COL = "time"
DB_COL = "db_temp"

def to_naive_datetime(s):
    dt = pd.to_datetime(s, errors="coerce")
    if hasattr(dt.dt, "tz"):
        dt = dt.dt.tz_localize(None)
    return dt

def load_cluster_excel(path: Path):
    df = pd.read_excel(path)
    df = df[[TIME_COL, DB_COL]].copy()
    df.rename(columns={TIME_COL: "Datetime", DB_COL: "T_cluster"}, inplace=True)
    df["Datetime"] = to_naive_datetime(df["Datetime"])
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime").drop_duplicates("Datetime")
    m = re.search(r"cluster_(\d+)_representative_hourly", path.name, flags=re.IGNORECASE)
    cluster_name = f"cluster_{m.group(1)}" if m else path.stem
    df["cluster_name"] = cluster_name
    return df

def load_epw(path: Path):
    epw = pd.read_csv(path, header=None, skiprows=8)
    epw = epw[[0,1,2,3,4,6]].copy()
    epw.columns = ["Year","Month","Day","Hour","Minute","T_rural"]
    epw["Year"] = 2024
    dt = pd.to_datetime(epw[["Year","Month","Day","Hour","Minute"]], errors="coerce") - pd.to_timedelta(1, unit="h")
    epw["Datetime"] = dt
    return epw[["Datetime","T_rural"]].dropna().sort_values("Datetime")

rural = load_epw(EPW_FILE)

cluster_files = sorted(DATA_DIR.glob(CLUSTER_PATTERN))
cluster_wide = rural.copy()

for f in cluster_files:
    dfc = load_cluster_excel(f)
    wide = dfc.pivot_table(index="Datetime", columns="cluster_name", values="T_cluster", aggfunc="first")
    cluster_wide = cluster_wide.merge(wide, left_on="Datetime", right_index=True, how="outer")

cols = ["Datetime","T_rural"] + sorted([c for c in cluster_wide.columns if c not in ["Datetime","T_rural"]])
cluster_wide = cluster_wide[cols].sort_values("Datetime").reset_index(drop=True)

cluster_wide.set_index("Datetime", inplace=True)
cluster_wide = cluster_wide.asfreq("H")
cluster_wide.reset_index(inplace=True)

cluster_cols = [c for c in cluster_wide.columns if c not in ["Datetime","T_rural"]]
long = cluster_wide.melt(id_vars=["Datetime","T_rural"], value_vars=cluster_cols, var_name="Cluster", value_name="T_cluster")
long["Difference"] = long["T_cluster"] - long["T_rural"]
long["Month"] = long["Datetime"].dt.month
long = long.dropna(subset=["Difference", "Month"])

months = sorted(long["Month"].unique())
clusters = sorted(long["Cluster"].dropna().unique())

data = []
positions = []
labels = []
pos = 1.0
stride = len(clusters) + 1
gap = 1.0

for m in months:
    month_data = long[long["Month"] == m]
    for i, cl in enumerate(clusters):
        vals = month_data[month_data["Cluster"] == cl]["Difference"].dropna()
        data.append(vals.values)
        positions.append(pos + i)
        labels.append(f"{m}-{cl}")
    pos += stride + gap

plt.figure(figsize=(18, 8))
plt.boxplot(data, positions=positions, patch_artist=True, showfliers=False)
plt.xticks(positions, labels, rotation=90)
plt.xlabel("Month-Cluster")
plt.ylabel("ΔT = T_cluster − T_rural (°C)")
plt.title("Monthly Boxplots of Cluster–Rural Temperature Differences (ARPAV Year=2024)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

cluster_wide.to_excel("CLUSTERS_vs_RURAL_merged_all.xlsx", index=False)
print("Saved: CLUSTERS_vs_RURAL_merged_hourly.xlsx")

