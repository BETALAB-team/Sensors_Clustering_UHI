import pandas as pd 
import numpy as np

#inputs
reliability_index_threshold = 3


#data
sensor_data = pd.read_pickle("sensors_data_by_name_preprocessed.pickle")
reliability_data = pd.read_pickle("reliability_table.pkl")


grouped = {name:group for name, group in reliability_data.groupby(reliability_data.index)}
filtered_sensors_dict = {
    name: df
    for name, df in sensor_data.items()
    if name.startswith("T")
    and name in reliability_data.index
    and reliability_data.loc[name, "reliability_index"] >= reliability_index_threshold
}
#%%
def build_wide_tables(sensor_dict):
    temp_series, hum_series = [], []
    for name, df in sensor_dict.items():
        idx = pd.to_datetime(df["index_new"], errors="coerce") if "index_new" in df.columns else pd.to_datetime(df.index, errors="coerce")
        df = df.copy()
        df.index = idx
        df = df[~df.index.isna()].sort_index()
        df = df[~df.index.duplicated(keep="last")]
        if "Temperature" in df.columns:
            temp_series.append(pd.to_numeric(df["Temperature"], errors="coerce").rename(name))
        if "Humidity" in df.columns:
            hum_series.append(pd.to_numeric(df["Humidity"], errors="coerce").rename(name))
    df_temp = pd.concat(temp_series, axis=1, join="outer").sort_index() if temp_series else pd.DataFrame()
    df_hum  = pd.concat(hum_series,  axis=1, join="outer").sort_index() if hum_series  else pd.DataFrame()
    return df_temp, df_hum
#%%
def resample_10min(df):
    if df.empty:
        return df
    start = df.index.min().floor("10min")
    end = df.index.max().ceil("10min")
    target_idx = pd.date_range(start, end, freq="10min", tz=getattr(df.index, "tz", None))
    out = pd.DataFrame(index=target_idx)

    for col in df.columns:
        s = df[col].astype(float)
        if s.dropna().empty:
            out[col] = np.nan
            continue

        union_idx = s.index.union(target_idx)
        s_union = s.reindex(union_idx).interpolate(method="time", limit_direction="both")
        s_interp = s_union.reindex(target_idx)

        t_valid = s.dropna().index.view("int64")
        t_target = target_idx.view("int64")

        pos = np.searchsorted(t_valid, t_target, side="left")
        left = np.clip(pos - 1, 0, len(t_valid) - 1)
        right = np.clip(pos, 0, len(t_valid) - 1)
        d_left = np.abs(t_target - t_valid[left])
        d_right = np.abs(t_valid[right] - t_target)
        d_min = np.minimum(d_left, d_right)

        mask = d_min <= pd.Timedelta("30min").value
        s_interp[~mask] = np.nan

        out[col] = s_interp
    return out

#%%
def wet_bulb_temp(dry_bulb_temp, RH):
    """
    Stull (2011)
    """
    T, RH = dry_bulb_temp.align(RH, join = "outer")
    T = T.astype(float)
    RH= RH.astype(float) 
    Tw = T * np.arctan(0.151977*np.sqrt(RH+8.313659))\
        + np.arctan(T+RH)\
        - np.arctan(RH-1.676331)\
        + 0.00391838*(RH**1.5)*np.arctan(0.023101*RH)\
        - 4.686035
    return Tw.where(~(T.isna() | RH.isna()))
df_temperature, df_humidity = build_wide_tables(filtered_sensors_dict)
df_temperature_10min = resample_10min(df_temperature)
df_humidity_10min = resample_10min(df_humidity)

df_db_temp = df_temperature_10min.copy()
df_wb_temp = wet_bulb_temp (df_db_temp, df_humidity_10min)


#%%
df_db_temp = df_db_temp.drop(columns=["T91"], errors="ignore")
df_wb_temp = df_wb_temp.drop(columns=["T91"], errors="ignore")
common_idx = df_db_temp.index.intersection(df_wb_temp.index)
sensors = df_db_temp.columns.intersection(df_wb_temp.columns)

data = []
for sensor in sensors: 
    db = df_db_temp.loc[common_idx, sensor].to_numpy()
    wb = df_wb_temp.loc[common_idx, sensor].to_numpy()
    X = np.vstack([db, wb]).T
    data.append(X)
data = np.array(data)

#%%

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def prep(df_db, df_wb, freq="30min"):
    idx=df_db.index.intersection(df_wb.index)
    db=df_db.loc[idx].astype("float32").resample(freq).mean()
    wb=df_wb.loc[idx].astype("float32").resample(freq).mean()
    sens=db.columns.intersection(wb.columns)
    db,wb=db[sens],wb[sens]
    db=db.interpolate(limit_direction="both")
    wb=wb.interpolate(limit_direction="both")
    return db,wb,sens

def daily_amp(x,step_per_day):
    g=x.groupby(np.arange(len(x))//step_per_day)
    return (g.max()-g.min()).mean()

def feat_matrix(db,wb,freq="30min"):
    step={"10min":144,"15min":96,"20min":72,"30min":48,"60min":24}[freq]
    F=[]
    for s in db.columns:
        x1=db[s].values; x2=wb[s].values; d=x1-x2
        m1=x1.mean(); sd1=x1.std(); m2=x2.mean(); sd2=x2.std(); md=d.mean(); sdd=d.std()
        p5=np.nanpercentile(x1,5); p95=np.nanpercentile(x1,95)
        ac1=np.corrcoef(x1[:-1],x1[1:])[0,1] if len(x1)>1 else 0.0
        amp1=daily_amp(pd.Series(x1),step); amp2=daily_amp(pd.Series(x2),step)
        X=np.vstack([x1,x2]).T - np.mean(np.vstack([x1,x2]).T,axis=0,keepdims=True)
        spec=np.fft.rfft(X,axis=0); mag=np.abs(spec)
        h1_db=mag[1,0]/(mag[:,0].sum()+1e-9); h1_wb=mag[1,1]/(mag[:,1].sum()+1e-9)
        F.append([m1,sd1,m2,sd2,md,sdd,p5,p95,ac1,amp1,amp2,h1_db,h1_wb])
    cols=["m_db","sd_db","m_wb","sd_wb","m_delta","sd_delta","p5_db","p95_db","ac1_db","amp_db","amp_wb","h1_db","h1_wb"]
    return pd.DataFrame(F,index=db.columns,columns=cols)

db,wb,sensors=prep(df_db_temp, df_wb_temp, freq="30min")
X=feat_matrix(db,wb,freq="30min")
Z=StandardScaler().fit_transform(X.values)

Ks=range(2,10); inertia=[]
for k in Ks:
    km=KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z)
    inertia.append(km.inertia_)
plt.plot(list(Ks), inertia, marker="o"); plt.xlabel("K"); plt.ylabel("Inertia"); plt.title("Elbow (feature-based)"); plt.grid(True); plt.show()

k_opt=4
km=KMeans(n_clusters=k_opt, n_init=20, random_state=0).fit(Z)
cluster_map=pd.Series(km.labels_, index=X.index, name="cluster")


def plot_cluster_profiles(db, wb, cluster_map, freq="30min"):
    step = {"10min":144, "15min":96, "20min":72, "30min":48, "60min":24}[freq]
    n_clusters = cluster_map.nunique()

    # Compute average daily profiles at desired frequency
    def daily_mean(df):
        step_sec = pd.to_timedelta(freq).seconds
        seconds_in_day = 24 * 3600
        day_pos = (df.index.hour * 3600 + df.index.minute * 60 + df.index.second)
        bins = np.arange(0, seconds_in_day + step_sec, step_sec)
        labels = bins[:-1] / 3600
        df["__bin__"] = pd.cut(day_pos, bins=bins, labels=labels, include_lowest=True)
        out = df.groupby("__bin__").mean(numeric_only=True)
        out.index = out.index.astype(float)
        df.drop(columns="__bin__", inplace=True, errors="ignore")
        return out

    for k in range(n_clusters):
        members = cluster_map[cluster_map == k].index
        db_sel = db[members]
        wb_sel = wb[members]

        db_day = daily_mean(db_sel)
        wb_day = daily_mean(wb_sel)

        t = db_day.index.values

        plt.figure(figsize=(8,4))
        for s in members:
            plt.plot(t, db_day[s], color="lightcoral", alpha=0.4, lw=0.6)
            plt.plot(t, wb_day[s], color="lightblue", alpha=0.4, lw=0.6)

        plt.plot(t, db_day.mean(axis=1), color="red", lw=2, label="DB mean")
        plt.plot(t, wb_day.mean(axis=1), color="blue", lw=2, label="WB mean")

        plt.title(f"Cluster {k+1}")
        plt.xlabel("Hour of Day")
        plt.ylabel("Temperature [°C]")
        plt.xticks(np.arange(0, 25, 4))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# usage
plot_cluster_profiles(db, wb, cluster_map, freq="30min")



#%%
def daily_representatives(db, wb, cluster_map, freq="30min"):
    db = db.resample(freq).mean()
    wb = wb.resample(freq).mean()
    clusters = sorted(cluster_map.unique())
    results = {}

    for k in clusters:
        members = cluster_map[cluster_map == k].index.intersection(db.columns).intersection(wb.columns)
        if len(members) == 0:
            results[k] = pd.DataFrame(columns=["day", "sensor", "distance"])
            continue

        db_c = db[members]
        wb_c = wb[members]

        mean_db = db_c.mean(axis=1)
        mean_wb = wb_c.mean(axis=1)

        dist_df = pd.DataFrame(index=db.index, columns=members, dtype=float)

        for s in members:
            d = np.sqrt((db_c[s] - mean_db)**2 + (wb_c[s] - mean_wb)**2)
            dist_df[s] = d

        daily_sum = dist_df.groupby(dist_df.index.date).sum()
        best_sensors = daily_sum.idxmin(axis=1)
        best_distances = daily_sum.min(axis=1)

        results[k] = pd.DataFrame({
            "day": daily_sum.index,
            "sensor": best_sensors.values,
            "distance": best_distances.values
        })

    return results


reps = daily_representatives(db, wb, cluster_map, freq="30min")

#%%
import numpy as np
import pandas as pd
from pathlib import Path

def _to_naive(df):
    if getattr(df.index, "tz", None) is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df

def export_from_reps(db, wb, rh, reps, out_freq="1H", out_dir="."):
    db = _to_naive(db)
    wb = _to_naive(wb)
    rh = _to_naive(rh)
    common = db.columns.intersection(wb.columns).intersection(rh.columns)
    db = db[common]
    wb = wb[common]
    rh = rh[common]
    db_h = db.resample(out_freq).mean()
    wb_h = wb.resample(out_freq).mean()
    rh_h = rh.resample(out_freq).mean()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for k, rep in reps.items():
        if rep is None or len(rep) == 0:
            out_df = pd.DataFrame(columns=["sensor","time","db_temp","wb_temp","rh","day"])
        else:
            rows = []
            for _, r in rep.iterrows():
                s = r["sensor"]
                if pd.isna(s) or s not in common:
                    continue
                day = pd.to_datetime(r["day"]).date()
                start = pd.Timestamp(day)
                end = start + pd.Timedelta(days=1)
                idx = db_h.loc[start:end - pd.Timedelta(seconds=1)].index
                if len(idx) == 0:
                    continue
                part = pd.DataFrame({
                    "sensor": s,
                    "time": idx,
                    "db_temp": db_h.loc[start:end - pd.Timedelta(seconds=1), s].to_numpy(),
                    "wb_temp": wb_h.loc[start:end - pd.Timedelta(seconds=1), s].to_numpy(),
                    "rh": rh_h.loc[start:end - pd.Timedelta(seconds=1), s].to_numpy(),
                    "day": day
                })
                rows.append(part)
            out_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["sensor","time","db_temp","wb_temp","rh","day"])
        out_path = out_dir / f"cluster_{int(k)+1}_representatives_hourly.xlsx"
        with pd.ExcelWriter(out_path, engine="openpyxl") as xlw:
            out_df.to_excel(xlw, index=False, sheet_name="hourly")



drop_cols = ["T91"]
db = df_db_temp.drop(columns=drop_cols, errors="ignore")
wb = df_wb_temp.drop(columns=drop_cols, errors="ignore")
rh = df_humidity_10min.drop(columns=drop_cols, errors="ignore")

# Make all time indices timezone-naive (important!)
for df in [db, wb, rh]:
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

# --- 2. Compute daily representative sensors ---
reps = daily_representatives(db, wb, cluster_map, freq="10min")

# --- 3. Export one Excel per cluster (hourly data of daily reps) ---
export_from_reps(
    db=db,
    wb=wb,
    rh=rh,
    reps=reps,
    out_freq="1H",
    out_dir="./cluster_exports"
)

import numpy as np
import pandas as pd
from pathlib import Path

def _to_naive(df):
    if getattr(df.index, "tz", None) is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df

def _align(db, wb, rh):
    db = _to_naive(db); wb = _to_naive(wb); rh = _to_naive(rh)
    cols = db.columns.intersection(wb.columns).intersection(rh.columns)
    return db[cols], wb[cols], rh[cols], cols

def _mask_summer(df):
    return df[(df.index.month >= 6) & (df.index.month <= 8)]

def representative_per_cluster(db, wb, cluster_map, freq="10min", use_summer_only=False):
    if use_summer_only:
        db = _mask_summer(db)
        wb = _mask_summer(wb)
    db = db.resample(freq).mean()
    wb = wb.resample(freq).mean()
    reps = {}
    for k in sorted(cluster_map.unique()):
        members = cluster_map[cluster_map == k].index.intersection(db.columns).intersection(wb.columns)
        if len(members) == 0:
            reps[k] = None
            continue
        db_c = db[members]; wb_c = wb[members]
        m_db = db_c.mean(axis=1); m_wb = wb_c.mean(axis=1)
        scores = {}
        for s in members:
            d = np.sqrt((db_c[s] - m_db)**2 + (wb_c[s] - m_wb)**2)
            scores[s] = d.dropna().sum()
        reps[k] = min(scores, key=scores.get) if len(scores) else None
    return reps

def export_representatives_fullyear(db, wb, rh, reps, out_freq="1H", out_dir=".", tag="all"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    db_h = db.resample(out_freq).mean()
    wb_h = wb.resample(out_freq).mean()
    rh_h = rh.resample(out_freq).mean()
    for k, s in reps.items():
        if s is None or s not in db_h.columns:
            out_df = pd.DataFrame(columns=["sensor","time","db_temp","wb_temp","rh"])
        else:
            idx = db_h.index.intersection(wb_h.index).intersection(rh_h.index)
            out_df = pd.DataFrame({
                "sensor": s,
                "time": idx,
                "db_temp": db_h[s].reindex(idx).to_numpy(),
                "wb_temp": wb_h[s].reindex(idx).to_numpy(),
                "rh": rh_h[s].reindex(idx).to_numpy()
            })
        out_path = out_dir / f"cluster_{int(k)+1}_representative_{tag}.xlsx"
        with pd.ExcelWriter(out_path, engine="openpyxl") as xlw:
            out_df.to_excel(xlw, index=False, sheet_name="hourly")

# --- usage ---
drop_cols = ["T91"]
db = df_db_temp.drop(columns=drop_cols, errors="ignore")
wb = df_wb_temp.drop(columns=drop_cols, errors="ignore")
rh = df_humidity_10min.drop(columns=drop_cols, errors="ignore")

db, wb, rh, _ = _align(db, wb, rh)

# 1️⃣ Representative per cluster (all-year selection)
reps_all = representative_per_cluster(db, wb, cluster_map, freq="10min", use_summer_only=False)
export_representatives_fullyear(db, wb, rh, reps_all, out_freq="1H", out_dir="./cluster_exports", tag="all")

# 2️⃣ Representative per cluster (chosen using summer data, but exported full-year)
reps_summer = representative_per_cluster(db, wb, cluster_map, freq="10min", use_summer_only=True)
export_representatives_fullyear(db, wb, rh, reps_summer, out_freq="1H", out_dir="./cluster_exports", tag="summer")



#%%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_representative_excels(folder, tag):
    files = sorted(Path(folder).glob(f"cluster_*_representative_{tag}.xlsx"))
    dfs = []
    for f in files:
        k = int(f.stem.split("_")[1])
        df = pd.read_excel(f)
        df["cluster"] = k
        df["time"] = pd.to_datetime(df["time"])
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No representative Excel files found for tag '{tag}' in {folder}")
    return pd.concat(dfs, ignore_index=True)

def load_daily_excels(folder):
    files = sorted(Path(folder).glob("cluster_*_representatives_hourly.xlsx"))
    dfs = []
    for f in files:
        k = int(f.stem.split("_")[1])
        df = pd.read_excel(f)
        df["cluster"] = k
        df["time"] = pd.to_datetime(df["time"])
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No daily representative Excel files found in {folder}")
    return pd.concat(dfs, ignore_index=True)

def monthly_max_daily_boxplot(df, tag):
    df = df.copy()
    df["day"] = df["time"].dt.date
    df["month"] = df["time"].dt.month
    df["month_name"] = df["time"].dt.strftime("%b")

    # Compute daily maximum DB temp per cluster
    df_dailymax = df.groupby(["cluster", "day"], as_index=False)["db_temp"].max()
    df_dailymax["month_name"] = pd.to_datetime(df_dailymax["day"]).dt.strftime("%b")
    df_dailymax["month"] = pd.to_datetime(df_dailymax["day"]).dt.month

    order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    plt.figure(figsize=(12,6))
    sns.boxplot(data=df_dailymax, x="month_name", y="db_temp", hue="cluster",
                order=order, palette="Set2", showfliers=False)
    plt.title(f"Monthly Boxplot of Daily Max DB Temperature per Cluster ({tag})")
    plt.xlabel("Month")
    plt.ylabel("Daily Max DB Temperature [°C]")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

folder = Path("./cluster_exports")

df_all = load_representative_excels(folder, tag="all")
monthly_max_daily_boxplot(df_all, "All-time Representative")

df_summer = load_representative_excels(folder, tag="summer")
monthly_max_daily_boxplot(df_summer, "Summer Representative (All Months)")

df_daily = load_daily_excels(folder)
monthly_max_daily_boxplot(df_daily, "Daily Representative")



#%%
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import timedelta

def load_daily_excels(folder):
    files = sorted(Path(folder).glob("cluster_*_representatives_hourly.xlsx"))
    dfs = []
    for f in files:
        k = int(f.stem.split("_")[1])
        df = pd.read_excel(f)
        df["cluster"] = k
        df["time"] = pd.to_datetime(df["time"])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else None

def load_representative_excels(folder, tag):
    files = sorted(Path(folder).glob(f"cluster_*_representative_{tag}.xlsx"))
    dfs = []
    for f in files:
        k = int(f.stem.split("_")[1])
        df = pd.read_excel(f)
        df["cluster"] = k
        df["time"] = pd.to_datetime(df["time"])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else None

def plot_hottest_week_overlay(df, title_suffix=""):
    df = df.copy()
    df["date"] = df["time"].dt.date

    # find hottest day per cluster
    hottest = (
        df.groupby(["cluster", "date"])["db_temp"]
        .max()
        .reset_index()
        .sort_values(["cluster", "db_temp"], ascending=[True, False])
        .groupby("cluster")
        .first()
    )

    # determine common week range (min Monday, max Sunday)
    hot_days = pd.to_datetime(hottest["date"]).dt.date
    week_starts = [d - timedelta(days=d.weekday()) for d in hot_days]
    week_start = min(week_starts)
    week_end = week_start + timedelta(days=7)

    colors = plt.cm.tab10.colors
    plt.figure(figsize=(10,5))

    for i, (k, row) in enumerate(hottest.iterrows()):
        hot_day = pd.to_datetime(row["date"]).date()
        sub = df[(df["cluster"] == k) &
                 (df["time"] >= pd.Timestamp(week_start)) &
                 (df["time"] < pd.Timestamp(week_end))].copy()
        sub["t_rel"] = (sub["time"] - pd.Timestamp(week_start)).dt.total_seconds() / 3600
        plt.plot(sub["t_rel"], sub["db_temp"], lw=2, color=colors[i % len(colors)], label=f"Cluster {k} DB")
        plt.plot(sub["t_rel"], sub["wb_temp"], lw=1.2, ls="--", alpha=0.8, color=colors[i % len(colors)], label=f"Cluster {k} WB")
        plt.text(160, sub["db_temp"].max(), f"({hot_day})", color=colors[i % len(colors)], fontsize=9)

    plt.title(f"Hottest Week Overlay — {title_suffix}")
    plt.xlabel("Hours since Monday 00:00 [h]")
    plt.ylabel("Temperature [°C]")
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# === usage ===
folder = Path("./cluster_exports")

# 1️⃣ Daily representatives
df_daily = load_daily_excels(folder)
plot_hottest_week_overlay(df_daily, "Daily Representatives")

# 2️⃣ Summer representatives
df_summer = load_representative_excels(folder, "summer")
plot_hottest_week_overlay(df_summer, "Summer Representatives")
#%%
import pandas as pd
from pathlib import Path

def load_daily_excels(folder):
    files = sorted(Path(folder).glob("cluster_*_representatives_hourly.xlsx"))
    dfs = []
    for f in files:
        k = int(f.stem.split("_")[1])
        df = pd.read_excel(f)
        df["cluster"] = k
        df["time"] = pd.to_datetime(df["time"])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else None

def load_representative_excels(folder, tag="all"):
    files = sorted(Path(folder).glob(f"cluster_*_representative_{tag}.xlsx"))
    dfs = []
    for f in files:
        k = int(f.stem.split("_")[1])
        df = pd.read_excel(f)
        df["cluster"] = k
        df["time"] = pd.to_datetime(df["time"])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else None

def monthly_cluster_summary(df):
    df = df.copy()
    df["month"] = df["time"].dt.month
    df["month_name"] = df["time"].dt.strftime("%b")

    mean_df = df.groupby(["month_name","cluster"])["db_temp"].mean().unstack()
    max_df  = df.groupby(["month_name","cluster"])["db_temp"].max().unstack()
    min_df  = df.groupby(["month_name","cluster"])["db_temp"].min().unstack()

    # Reorder months chronologically
    order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    mean_df = mean_df.reindex(order)
    max_df  = max_df.reindex(order)
    min_df  = min_df.reindex(order)

    return mean_df, max_df, min_df

# === usage ===
folder = Path("./cluster_exports")

# 1️⃣ Daily representatives
df_daily = load_daily_excels(folder)
mean_daily, max_daily, min_daily = monthly_cluster_summary(df_daily)

# 2️⃣ Hourly representatives (e.g. “all”)
df_hourly = load_representative_excels(folder, tag="all")
mean_hourly, max_hourly, min_hourly = monthly_cluster_summary(df_hourly)

# === display or export ===
print("---- DAILY REPRESENTATIVES ----")
print("\nMean:\n", mean_daily.round(2))
print("\nMax:\n", max_daily.round(2))
print("\nMin:\n", min_daily.round(2))

print("\n---- HOURLY REPRESENTATIVES ----")
print("\nMean:\n", mean_hourly.round(2))
print("\nMax:\n", max_hourly.round(2))
print("\nMin:\n", min_hourly.round(2))

# optional export
mean_daily.to_excel("mean_daily.xlsx")
max_daily.to_excel("max_daily.xlsx")
min_daily.to_excel("min_daily.xlsx")
mean_hourly.to_excel("mean_hourly.xlsx")
max_hourly.to_excel("max_hourly.xlsx")
min_hourly.to_excel("min_hourly.xlsx")


#%%
import geopandas as gpd
import pandas as pd

# Load your GeoJSON file
gdf = gpd.read_file("location_with_monthly_stats.geojson")



# Convert to DataFrame for merging
df_cluster = cluster_map.rename("CLUSTERLABEL").reset_index().rename(columns={"index": "name"})

# Merge into GeoDataFrame based on 'name'
gdf = gdf.merge(df_cluster, on="name", how="left")

# Save back to file
gdf.to_file("your_file_with_clusters.geojson", driver="GeoJSON")
