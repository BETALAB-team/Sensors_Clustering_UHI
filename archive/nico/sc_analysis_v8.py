# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 12:36:06 2025

@author: borgnic12709
"""

#%% Moduels and Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
import pathlib
import pickle
import copy
import seaborn as sns
import osmnx as ox
from astral import LocationInfo
from astral.sun import sun
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import scipy.stats as st

#%% Default sns settings:
sns.set_theme(rc={'figure.figsize':(16,9)})
sns.set(context = 'paper', style = 'whitegrid', font_scale = 1.25)

#%% Configuration
TIME_RESAMPLE = '60min'
time_period_clustering = ['2024-06-15', '2024-09-15']
CLUSTER_RANGE = range(2, 8)
RANDOM_STATE = 42
DTW_GAMMA = 0.01
CITY = LocationInfo("Padua", "Italy", "Europe/Rome", 45.4064, 11.8768)
THRESHOLD_PERC = 65
POSTPROCESS_MODE = 'fixed'  # 'threshold' or 'fixed' (we will keep both but fixed preferred)
FINAL_CLUSTERS = 4  # final number of clusters when using fixed postprocess
WINDOW_HOURS = 2  # hours around peak to keep per day (±WINDOW_HOURS)
FIGURE_DIR_NAME = 'analysis_v8'

#%% Paths and filenames
code_dir_path = pathlib.Path.cwd()
project_dir_path = code_dir_path.parent
input_dir_path = code_dir_path.joinpath('1 INPUT')
output_dir_path = code_dir_path.joinpath('2 OUTPUT')
figures_dir_path = code_dir_path.joinpath('3 FIGURES')

# New figures dir path:
figures_dir_path = figures_dir_path.joinpath(FIGURE_DIR_NAME)

# filenames expected in input folder
reliability_table_name = 'reliability_table.pkl'
sensors_data_by_name_name = 'sensors_data_by_name_preprocessed.pickle'
sensors_locations_name = 'sensors_location_complete.geojson'

# create output folders if missing
for p in [output_dir_path, figures_dir_path]:
    p.mkdir(parents=True, exist_ok=True)

#%% Helper functions

def compute_sun_times(dates):
    sun_times = {}
    for d in dates:
        s = sun(CITY.observer, date=d, tzinfo=CITY.timezone)
        sun_times[d] = (s['sunrise'], s['sunset'])
    return sun_times


def isolate_daylight(temp_df, rh_df=None):
    """
    Make the input DataFrames timezone-aware and return only timestamps between sunrise and
    sunset for each day. This version handles DST transitions (nonexistent/ambiguous times).
    """
    temp = temp_df.copy()
    # ensure DatetimeIndex
    if not isinstance(temp.index, pd.DatetimeIndex):
        temp.index = pd.to_datetime(temp.index)

    # Helper to safely localize or convert index to target tz
    def _ensure_tz(idx, tz):
        # idx: DatetimeIndex (naive or tz-aware)
        if idx.tz is not None:
            # already tz-aware: convert
            try:
                return idx.tz_convert(tz)
            except Exception:
                return idx.tz_convert(tz)
        # naive index: try to localize handling DST issues
        try:
            return idx.tz_localize(tz, ambiguous='infer', nonexistent='shift_forward')
        except Exception:
            try:
                return idx.tz_localize(tz, ambiguous='infer', nonexistent='shift_backward')
            except Exception:
                # last resort: coerce nonexistent times to NaT then drop them
                return idx.tz_localize(tz, ambiguous='infer', nonexistent='NaT')

    # apply tz handling
    temp.index = _ensure_tz(temp.index, CITY.timezone)

    dates = np.unique(temp.index.date)
    sun_times = compute_sun_times(dates)
    mask = pd.Series(False, index=temp.index)
    for d in dates:
        start, end = sun_times[d]
        mask |= ((temp.index >= start) & (temp.index <= end))
    temp_day = temp.loc[mask]

    rh_day = None
    if rh_df is not None:
        rh = rh_df.copy()
        if not isinstance(rh.index, pd.DatetimeIndex):
            rh.index = pd.to_datetime(rh.index)
        rh.index = _ensure_tz(rh.index, CITY.timezone)
        # Reindex rh to match temp_day timestamps (drop missing)
        rh_day = rh.reindex(temp_day.index).dropna(axis=0, how='all')
    return temp_day, rh_day, sun_times


def isolate_peak_windows(temp_day, window_hours=WINDOW_HOURS):
    """
    For each day, find the datetime of the mean-field peak and return a DataFrame that
    concatenates for each day the window [peak - window_hours, peak + window_hours].
    The returned DataFrame keeps original timestamps (may have gaps between days).
    """
    frames = []
    dates = np.unique(temp_day.index.date)
    for d in dates:
        mask = temp_day.index.date == d
        df_d = temp_day.loc[mask]
        if df_d.shape[0] == 0:
            continue
        mean_series = df_d.mean(axis=1)
        peak_time = mean_series.idxmax()
        start = peak_time - pd.Timedelta(hours=window_hours)
        end = peak_time + pd.Timedelta(hours=window_hours)
        win = df_d.loc[(df_d.index >= start) & (df_d.index <= end)]
        # only keep windows with at least 1 observation
        if win.shape[0] > 0:
            # Add a day marker column to assist later concatenation if needed
            win['_day'] = pd.to_datetime(d)
            frames.append(win)
    if not frames:
        raise ValueError('No peak windows found in the period provided')
    temp_peak_windows = pd.concat(frames)
    return temp_peak_windows


def build_concatenated_peak_series(temp_peak_windows):
    """
    Build one long time series per sensor by concatenating all the daily peak windows in chronological order.
    Returns a DataFrame sensors x timepoints (index: integer sequence) where columns are sensor names.
    This version is robust to tz-aware DatetimeIndex and avoids sorting by the index object directly.
    """
    # Reset index so the timestamp becomes a regular column we can sort by
    df_reset = temp_peak_windows.reset_index()
    # first column after reset_index() is the timestamp column (its name may vary)
    time_col = df_reset.columns[0]
    # Sort by day marker and timestamp
    df_sorted = df_reset.sort_values(['_day', time_col])
    # Drop helper columns (_day and the timestamp) to keep only sensor columns
    values = df_sorted.drop(columns=['_day', time_col])
    # Reset integer index for concatenation
    values_reset = values.reset_index(drop=True)
    # transpose to sensors x timepoints
    series_per_sensor = values_reset.T
    return series_per_sensor


def preprocess_for_clustering(series_df, preserve_magnitude=True):
    """
    Convert DataFrame (sensors x timepoints) to numpy array for clustering.
    If preserve_magnitude is True, do NOT z-score per series; instead apply optional global min-max scaling
    to keep absolute differences between series.
    Returns arr shape (n_sensors, timepoints, 1) and sensor_names list.
    """
    # drop sensors with any NaNs (simpler) or interpolate
    series_df_clean = series_df.dropna(axis=0, how='any')
    sensor_names = list(series_df_clean.index)
    arr = series_df_clean.values[:, :]
    # choose scaling strategy
    if preserve_magnitude:
        # global min-max to [0,1] preserving relative differences
        global_min = np.nanmin(arr)
        global_max = np.nanmax(arr)
        if global_max - global_min > 0:
            arr_scaled = (arr - global_min) / (global_max - global_min)
        else:
            arr_scaled = arr.copy()
    else:
        # sample-wise mean-variance scaling (not preserving magnitude)
        scaler = TimeSeriesScalerMeanVariance()
        arr_scaled = scaler.fit_transform(arr[:, :, np.newaxis])[:, :, 0]
    # reshape for tslearn (n_ts, sz, 1)
    arr_tslearn = arr_scaled[:, :, np.newaxis]
    return arr_tslearn, sensor_names


def run_time_series_clustering(arr, n_clusters, method='euclidean'):
    """Return labels and cluster centers (centers may be None for some methods)"""
    if method == 'euclidean':
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric='euclidean', random_state=RANDOM_STATE)
        labels = model.fit_predict(arr)
        centers = model.cluster_centers_
    elif method == 'softdtw':
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric='softdtw', metric_params={'gamma': DTW_GAMMA}, random_state=RANDOM_STATE)
        labels = model.fit_predict(arr)
        centers = model.cluster_centers_
    elif method == 'kmeans_flat':
        # fallback: classical KMeans on flattened series (preserves magnitude if arr not standardized)
        flat = arr.reshape(arr.shape[0], -1)
        km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
        labels = km.fit_predict(flat)
        centers = None
    else:
        raise ValueError('Unsupported clustering method')
    return labels, centers


def compute_kpis(arr, labels, sensor_names, temp_peak_df, coords_df):
    """
    Compute a set of KPIs focused on magnitude and trend for the provided clustering.
    arr: (n_sensors, timepoints, 1)
    labels: array length n_sensors
    temp_peak_df: DataFrame sensors x timepoints with original °C values (not scaled)
    coords_df: DataFrame with sensor, latitude, longitude
    Returns dict of KPIs.
    """
    n = arr.shape[0]
    flat = arr.reshape(n, -1)
    kpis = {}
    # Basic cluster quality
    try:
        sil = silhouette_score(flat, labels)
    except Exception:
        sil = np.nan
    try:
        db = davies_bouldin_score(flat, labels)
    except Exception:
        db = np.nan
    try:
        ch = calinski_harabasz_score(flat, labels)
    except Exception:
        ch = np.nan
    kpis['silhouette'] = sil
    kpis['davies_bouldin'] = db
    kpis['calinski_harabasz'] = ch

    # Ensure temp_peak_df is oriented sensors x timepoints
    # If sensors are columns instead, transpose
    if list(temp_peak_df.index) == list(range(len(temp_peak_df.index))):
        # index looks numeric, try to detect if sensors are columns
        if all(s in temp_peak_df.columns for s in sensor_names):
            peak_df = temp_peak_df
            orient = 'columns'
        else:
            peak_df = temp_peak_df.T
            orient = 'rows'
    else:
        # prefer rows = sensors
        if all(s in temp_peak_df.index for s in sensor_names):
            peak_df = temp_peak_df
            orient = 'rows'
        elif all(s in temp_peak_df.columns for s in sensor_names):
            peak_df = temp_peak_df
            orient = 'columns'
        else:
            # fallback: try to transpose if that helps
            if all(s in temp_peak_df.T.index for s in sensor_names):
                peak_df = temp_peak_df.T
                orient = 'rows'
            else:
                raise KeyError('Sensor names do not match temp_peak_df rows or columns')

    # Now peak_df has sensors either in index (rows) or columns
    # We will always work with sensors as rows for convenience
    if orient == 'columns':
        peak_df = peak_df.T

    # Within-cluster magnitude statistics (using original °C values)
    amps = []
    stds = []
    for cid in np.unique(labels):
        members = [sensor_names[i] for i in range(len(sensor_names)) if labels[i] == cid]
        # keep only members present in peak_df
        members_present = [m for m in members if m in peak_df.index]
        if not members_present:
            continue
        dfc = peak_df.loc[members_present]
        # amplitude per sensor across time: max - min, then average across members
        per_sensor_amp = dfc.max(axis=1) - dfc.min(axis=1)
        amps.append(per_sensor_amp.mean())
        # intra-cluster std: sensor-wise std averaged
        per_sensor_std = dfc.std(axis=1)
        stds.append(per_sensor_std.mean())

    kpis['mean_peak_amplitude'] = np.nanmean(amps) if amps else np.nan
    kpis['mean_intra_std'] = np.nanmean(stds) if stds else np.nan

    # Spatial coherence: mean pairwise distance within clusters
    # Align coords to sensor_names present in arr
    coords_available = coords_df.set_index('name')
    sensor_names_present = [s for s in sensor_names if s in coords_available.index]
    if len(sensor_names_present) < len(sensor_names):
        # warn if some sensors missing
        missing = set(sensor_names) - set(sensor_names_present)
        print(f"Warning: {len(missing)} sensors missing coordinates: {sorted(list(missing))[:5]}...")
    coords = coords_available.loc[sensor_names_present][['Latitude', 'Longitude']].values
    coords = coords.astype(float)
    if coords.shape[0] >= 2:
        dist_mat = squareform(pdist(coords))
    else:
        dist_mat = np.array([[0]])
    intra_dists = []
    # Build index mapping for coordinates
    coord_idx = {s: i for i, s in enumerate(sensor_names_present)}
    for i in range(len(sensor_names)):
        s1 = sensor_names[i]
        if s1 not in coord_idx:
            continue
        for j in range(i+1, len(sensor_names)):
            s2 = sensor_names[j]
            if s2 not in coord_idx:
                continue
            if labels[i] == labels[j]:
                intra_dists.append(dist_mat[coord_idx[s1], coord_idx[s2]])
    kpis['mean_intra_distance'] = np.nanmean(intra_dists) if intra_dists else np.nan

    # ANOVA across clusters on all values per cluster
    groups = []
    for cid in np.unique(labels):
        members = [sensor_names[i] for i in range(len(sensor_names)) if labels[i] == cid]
        members_present = [m for m in members if m in peak_df.index]
        if not members_present:
            continue
        dfc = peak_df.loc[members_present]
        groups.append(dfc.values.flatten())
    if len(groups) > 1:
        try:
            F, p = st.f_oneway(*groups)
        except Exception:
            F, p = np.nan, np.nan
    else:
        F, p = np.nan, np.nan
    kpis['anova_F'] = F
    kpis['anova_p'] = p
    return kpis


def plot_kpis(kpi_results, out_dir):
    df = pd.DataFrame(kpi_results).T
    # plot silhouettes, DB, CH side by side
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax2 = ax1.twinx()
    df['silhouette'].plot(kind='bar', ax=ax1, position=0, width=0.3, color='tab:blue', label='Silhouette')
    df['davies_bouldin'].plot(kind='bar', ax=ax1, position=1, width=0.3, color='tab:orange', label='DBI')
    df['calinski_harabasz'].plot(kind='bar', ax=ax2, position=2, width=0.3, color='tab:green', label='CHI')
    ax1.set_ylabel('Silhouette / DBI')
    ax2.set_ylabel('Calinski-Harabasz')
    ax1.set_xticklabels(df.index, rotation=0)
    plt.title('Clustering quality indices')
    fig.tight_layout()
    fig.savefig(out_dir / 'kpis_indices.png', dpi=300)
    plt.show()

    # amplitude & std
    fig, ax = plt.subplots(figsize=(10,5))
    df[['mean_peak_amplitude', 'mean_intra_std']].plot(kind='bar', ax=ax)
    ax.set_ylabel('°C')
    ax.set_xticklabels(df.index, rotation=0)
    plt.title('Amplitude and intra-cluster std')
    fig.tight_layout(); fig.savefig(out_dir / 'kpis_amp_std.png', dpi=300); plt.show()

    # spatial
    fig, ax = plt.subplots(figsize=(10,5))
    df['mean_intra_distance'].plot(kind='bar', ax=ax, color='tab:purple')
    ax.set_ylabel('Distance (deg)')
    ax.set_xticklabels(df.index, rotation=0)
    plt.title('Mean intra-cluster spatial distance')
    fig.tight_layout(); fig.savefig(out_dir / 'kpis_spatial.png', dpi=300); plt.show()

#%% Main workflow
if __name__ == '__main__':
        # --- Load inputs ---
    import pickle, importlib, sys
    
    # Light shim so pickle can resolve old module paths
    def _install_numpy_core_shims():
        try:
            sys.modules['numpy._core'] = importlib.import_module('numpy.core')
            sys.modules['numpy._core.numeric'] = importlib.import_module('numpy.core.numeric')
        except Exception:
            pass
    
    _install_numpy_core_shims()
    
    with open(input_dir_path.joinpath(reliability_table_name),"rb") as f:
        reliability_table = pickle.load(f)
    # with open(input_dir_path.joinpath(reliability_table_name), 'rb') as f:
    #     reliability_table = pickle.load(f)
    with open(input_dir_path.joinpath(sensors_data_by_name_name), 'rb') as f:
        sensors_data = pickle.load(f)
    # Load sensors location geojson
    sensors_loc_gdf = gpd.read_file(input_dir_path.joinpath(sensors_locations_name))

    # Keep only reliable sensors (example indices)
    rel_index_acc = [3]
    sensors_data_rel = {}
    for ind in rel_index_acc:
        for sensor in sensors_data.keys():
            try:
                if reliability_table.loc[sensor, 'reliability_index'] == ind:
                    sensors_data_rel[sensor] = sensors_data[sensor]
            except Exception:
                pass

    # Build a unified hourly temperature DataFrame: index=datetime, columns=sensors
    df_list = []
    for sensor, df in sensors_data_rel.items():
        try:
            s = df['Temperature'].tz_localize(None)
            s.name = sensor
            df_list.append(s)
        except Exception:
            continue
    if not df_list:
        raise ValueError('No temperature series found')
    temp_df = pd.concat(df_list, axis=1)
    # resample hourly
    temp_df = temp_df.resample(TIME_RESAMPLE).mean()
    
    # Select day based on desired period:
    temp_df = temp_df.loc[time_period_clustering[0]:time_period_clustering[1]]

    # Isolate daylight and peak windows
    temp_day, rh_day, sun_times = isolate_daylight(temp_df, None)
    temp_peak_windows = isolate_peak_windows(temp_day, window_hours=WINDOW_HOURS)
    
    # Fill nan:
    temp_peak_windows = temp_peak_windows.fillna(temp_peak_windows.mean())
    # Drop nan columns:
    temp_peak_windows = temp_peak_windows.dropna(axis=1, how='any')

    # Build concatenated peak-time series per sensor
    series_per_sensor = build_concatenated_peak_series(temp_peak_windows)  # sensors x timepoints

    # Preprocess for clustering (preserve magnitude)
    arr_ts, sensor_names = preprocess_for_clustering(series_per_sensor, preserve_magnitude=True)

    #%% Choose clustering methods to compare
    methods = [ 'kmeans_flat']
    cluster_results = {}
    kpi_results = {}

    for method in methods:
        labels, centers = run_time_series_clustering(arr_ts, 3, method=method)
        cluster_results[method] = {'labels': labels, 'centers': centers}
        # Compute KPIs using original °C windows (series_per_sensor DataFrame)
        kpis = compute_kpis(arr_ts, labels, sensor_names, series_per_sensor, sensors_loc_gdf)
        kpi_results[method] = kpis
        print(f"Method {method} KPIs:", kpis)
        # Plot clustered time series
        fig, axs = plt.subplots(1, FINAL_CLUSTERS, figsize=(4*FINAL_CLUSTERS, 4), sharey=True)
        ts_len = arr_ts.shape[1]
        time_axis = np.arange(ts_len)
        for cid in range(FINAL_CLUSTERS):
            ax = axs[cid]
            idx = np.where(labels == cid)[0]
            for i in idx:
                ax.plot(time_axis, arr_ts[i,:,0], color='gray', alpha=0.4)
            if centers is not None:
                center = centers[cid].ravel()
                if len(center) != ts_len:
                    center = center[:ts_len]
                ax.plot(time_axis, center, color='red', linewidth=2)
            ax.set_title(f'Cluster {cid} (n={len(idx)})')
            ax.set_xlabel('Time-step (concatenated windows)')
        plt.suptitle(f'Time-series clusters ({method})')
        plt.tight_layout(); plt.savefig(figures_dir_path/f'clusters_timeseries_{method}.png', dpi=300); plt.show(); plt.close()

        # Map plot: attach labels to sensor locations and save geojson
        sensors_loc = sensors_loc_gdf.set_index('name').loc[sensor_names].copy()
        sensors_loc['cluster'] = [int(l) for l in labels]
        gdf = sensors_loc.reset_index()
        out_geo = output_dir_path / f'sensors_clusters_{method}.geojson'
        gdf.to_file(out_geo, driver='GeoJSON')
        print(f'Saved clusters geojson: {out_geo}')

        # Plot on city map
        fig, ax = plt.subplots(figsize=(10,10))
        boundary = ox.geocode_to_gdf('Padova, Italy').to_crs(epsg=3857)
        sensors_plot = gdf.to_crs(epsg=3857)
        boundary.plot(ax=ax, facecolor='none', edgecolor='black')
        sensors_plot.plot(ax=ax, column='cluster', categorical=True, legend=True, markersize=80)
        try:
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception:
            pass
        ax.set_axis_off()
        plt.title(f'Sensor clusters ({method})')
        plt.tight_layout(); plt.savefig(figures_dir_path/f'map_clusters_{method}.png', dpi=300); plt.show(); plt.close()

    # Save KPIs dictionary
    with open(output_dir_path/'kpi_results_peaks.pkl', 'wb') as f:
        pickle.dump(kpi_results, f)
    # Plot KPIs summary
    plot_kpis(kpi_results, figures_dir_path)

    print('Processing complete.')
